"""
Backtesting engine for the GPT Trading System.
Enables historical strategy testing with full integration of existing components.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path
import json

from core.domain.enums.mt5_enums import TimeFrame
from core.domain.models import (
    RiskClass, TradingSignal, Trade, MarketData, Candle, SignalType, 
    TradeStatus, TradeResult, create_trade_id
)
from core.domain.exceptions import BacktestingError, ErrorContext
from core.services.signal_service import SignalService
from core.services.offline_validator import OfflineSignalValidator
from core.infrastructure.mt5.data_provider import MT5DataProvider
from core.utils.chart_utils import ChartGenerator
from config.settings import TradingSettings
from core.infrastructure.database.backtest_repository import BacktestRepository

logger = logging.getLogger(__name__)


class BacktestMode(Enum):
    """Backtesting modes"""
    FULL = "full"  # Run all components including GPT
    OFFLINE_ONLY = "offline_only"  # Only offline validation
    HISTORICAL = "historical"  # Use historical signals


@dataclass
class BacktestConfig:
    """Configuration for backtesting run"""
    start_date: datetime
    end_date: datetime
    symbols: List[str]
    timeframe: str = "H1"
    initial_balance: float = 10000.0
    risk_per_trade: float = 0.015  # 1.5%
    max_open_trades: int = 5
    commission_per_lot: float = 5.0
    slippage_points: int = 1
    mode: BacktestMode = BacktestMode.OFFLINE_ONLY
    use_spread_filter: bool = True
    use_news_filter: bool = True
    save_results: bool = True
    results_dir: str = "backtest_results"


@dataclass
class BacktestTrade:
    """Represents a trade in backtesting"""
    signal: TradingSignal
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    lot_size: float = 0.01
    commission: float = 0.0
    slippage: float = 0.0
    pnl: float = 0.0
    pnl_points: float = 0.0
    max_drawdown: float = 0.0
    max_profit: float = 0.0
    result: Optional[TradeResult] = None
    exit_reason: Optional[str] = None
    bars_held: int = 0
    risk_reward_achieved: float = 0.0


@dataclass
class BacktestResults:
    """Comprehensive backtesting results"""
    config: BacktestConfig
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    
    # Performance metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    total_pnl: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    average_win: float = 0.0
    average_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    
    average_bars_held: float = 0.0
    average_rr_achieved: float = 0.0
    
    # Additional statistics
    statistics: Dict[str, Any] = field(default_factory=dict)


class BacktestDataProvider:
    """Provides historical data for backtesting"""
    
    def __init__(self, data_provider: MT5DataProvider):
        self.data_provider = data_provider
        self._cache = {}
    
    async def get_historical_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime,
        timeframe: str = "H1"
    ) -> pd.DataFrame:
        """Get historical data for backtesting period"""
        cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Convert string timeframe to TimeFrame enum
        timeframe_enum = TimeFrame[timeframe]
        
        # Calculate bars needed with extra buffer
        hours_needed = int((end_date - start_date).total_seconds() / 3600)
        bars_needed = hours_needed + 200  # Add 200 bars buffer for indicators
        
        # We need to request data from current time, not from the specific date
        # MT5 copy_rates_from_pos gets data counting backwards from current time
        print(f"Requesting {bars_needed} bars for {symbol}")
        
        try:
            market_data = await self.data_provider.get_market_data(
                symbol=symbol,
                timeframe=timeframe_enum,
                bars=bars_needed + 1000  # Request extra to ensure we cover the date range
            )
            
            if not market_data.candles:
                print(f"No candles received for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = self._market_data_to_dataframe(market_data)
            
            print(f"Received data from {df.index.min()} to {df.index.max()}")
            
            # Filter by date range
            df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]
            
            if df_filtered.empty:
                print(f"No data in requested range {start_date} to {end_date}")
                print(f"Available data range: {df.index.min()} to {df.index.max()}")
            else:
                print(f"Filtered to {len(df_filtered)} bars for backtesting")
            
            self._cache[cache_key] = df_filtered
            return df_filtered
            
        except Exception as e:
            print(f"Error getting historical data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def _market_data_to_dataframe(self, market_data: MarketData) -> pd.DataFrame:
        """Convert MarketData to DataFrame"""
        data = []
        for candle in market_data.candles:
            data.append({
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume,
                'spread': candle.spread,
                'ema50': candle.ema50,
                'ema200': candle.ema200,
                'rsi14': candle.rsi14,
                'atr14': candle.atr14
            })
        
        df = pd.DataFrame(data)
        df.index = pd.DatetimeIndex([c.timestamp for c in market_data.candles])
        return df


class BacktestSignalGenerator:
    """Generates signals in backtesting mode"""
    
    def __init__(
        self, 
        signal_service: Optional[SignalService] = None,
        offline_validator: Optional[OfflineSignalValidator] = None,
        mode: BacktestMode = BacktestMode.OFFLINE_ONLY
    ):
        self.signal_service = signal_service
        self.offline_validator = offline_validator or OfflineSignalValidator()
        self.mode = mode
    
    async def generate_signal(
        self, 
        market_data: MarketData,
        historical_context: Optional[Dict] = None
    ) -> TradingSignal:
        """Generate signal based on backtest mode"""
        
        if self.mode == BacktestMode.FULL and self.signal_service:
            # Full signal generation with GPT
            return await self.signal_service.generate_signal(market_data.symbol)
        
        elif self.mode == BacktestMode.OFFLINE_ONLY:
            # Use only offline validation
            validation_result = await self.offline_validator.validate_market_data(
                market_data,
                min_score_threshold=0.6
            )
            
            # Create signal based on validation
            return self._create_signal_from_validation(market_data, validation_result)
        
        else:
            # Historical mode - would load from database
            return self._create_default_signal(market_data)
    
    def _create_signal_from_validation(
        self, 
        market_data: MarketData,
        validation_result: Dict
    ) -> TradingSignal:
        """Create signal from validation results with improved logic"""
        score = validation_result['validation_summary']['weighted_score']
        
        if score < 0.65:  # Raised threshold
            return TradingSignal(
                symbol=market_data.symbol,
                signal=SignalType.WAIT,
                reason=f"Validation score too low: {score:.2f}",
                risk_class=RiskClass.C
            )
        
        # Get recent candles for VSA analysis
        candles = market_data.candles[-20:]
        if len(candles) < 20:
            return self._create_wait_signal(market_data.symbol, "Insufficient data")
        
        latest = candles[-1]
        prev = candles[-2]
        
        # Calculate volume metrics
        avg_volume = sum(c.volume for c in candles[:-1]) / len(candles[:-1])
        volume_ratio = latest.volume / avg_volume if avg_volume > 0 else 1
        
        # VSA Pattern Detection
        buy_signal = False
        sell_signal = False
        
        # 1. Check for Stopping Volume (bullish)
        if (prev.close < prev.open and  # Down bar
            prev.volume > avg_volume * 1.5 and  # High volume
            latest.close > latest.open and  # Up bar
            latest.close > prev.close):  # Reversal
            buy_signal = True
            reason = "Stopping Volume pattern"
        
        # 2. Check for No Supply (bullish)
        elif (latest.close < latest.open and  # Down bar
            latest.volume < avg_volume * 0.5 and  # Low volume
            latest.close > latest.low):  # Close off lows
            # Need confirmation
            if candles[-3].close < candles[-3].open:  # Previous trend was down
                buy_signal = True
                reason = "No Supply pattern"
        
        # 3. Check for Upthrust (bearish)
        if (latest.high > max(c.high for c in candles[-10:-1]) and  # New high
            latest.close < latest.open and  # But closed down
            latest.close < (latest.high + latest.low) / 2 and  # Closed in lower half
            volume_ratio > 1.3):  # With volume
            sell_signal = True
            reason = "Upthrust pattern"
        
        # 4. Check for No Demand (bearish)
        elif (latest.close > latest.open and  # Up bar
            latest.volume < avg_volume * 0.5 and  # Low volume
            latest.close < latest.high):  # Close off highs
            if candles[-3].close > candles[-3].open:  # Previous trend was up
                sell_signal = True
                reason = "No Demand pattern"
        
        # Additional filters
        if buy_signal:
            # Check trend alignment
            if latest.ema50 and latest.ema200:
                if latest.ema50 < latest.ema200:  # Against major trend
                    if score < 0.8:  # Only take if very high score
                        buy_signal = False
            
            # Check RSI
            if latest.rsi14 and latest.rsi14 > 70:
                buy_signal = False  # Overbought
        
        if sell_signal:
            # Check trend alignment
            if latest.ema50 and latest.ema200:
                if latest.ema50 > latest.ema200:  # Against major trend
                    if score < 0.8:
                        sell_signal = False
            
            # Check RSI
            if latest.rsi14 and latest.rsi14 < 30:
                sell_signal = False  # Oversold
        
        # Generate signal
        if buy_signal and score > 0.7:
            return self._create_buy_signal_improved(market_data, score, reason)
        elif sell_signal and score > 0.7:
            return self._create_sell_signal_improved(market_data, score, reason)
        
        return self._create_wait_signal(market_data.symbol, "No clear VSA pattern")

    def _create_buy_signal_improved(self, market_data: MarketData, score: float, reason: str) -> TradingSignal:
        """Create improved buy signal with better SL/TP"""
        latest = market_data.latest_candle
        candles = market_data.candles[-20:]
        
        # Find recent swing low for stop loss
        recent_low = min(c.low for c in candles[-10:])
        atr = latest.atr14 or 0.0001
        
        entry = latest.close
        stop_loss = min(recent_low - (0.5 * atr), entry - (2 * atr))
        
        # Dynamic TP based on R:R
        risk = entry - stop_loss
        take_profit = entry + (risk * 3)  # Target 3:1 RR
        
        return TradingSignal(
            symbol=market_data.symbol,
            signal=SignalType.BUY,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=3.0,
            risk_class=RiskClass.A if score > 0.85 else RiskClass.B,
            reason=f"{reason}, validation score: {score:.2f}"
        )
    
    def _create_sell_signal_improved(self, market_data: MarketData, score: float, reason: str) -> TradingSignal:
        """Create improved sell signal with better SL/TP"""
        latest = market_data.latest_candle
        candles = market_data.candles[-20:]
        
        # Find recent swing high for stop loss
        recent_high = max(c.high for c in candles[-10:])
        atr = latest.atr14 or 0.0001
        
        entry = latest.close
        stop_loss = max(recent_high + (0.5 * atr), entry + (2 * atr))
        
        # Dynamic TP based on R:R
        risk = stop_loss - entry
        take_profit = entry - (risk * 3)  # Target 3:1 RR
        
        return TradingSignal(
            symbol=market_data.symbol,
            signal=SignalType.SELL,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=3.0,
            risk_class=RiskClass.A if score > 0.85 else RiskClass.B,
            reason=f"{reason}, validation score: {score:.2f}"
        )

    def _create_buy_signal(self, market_data: MarketData, score: float) -> TradingSignal:
        """Create buy signal"""
        latest = market_data.latest_candle
        atr = latest.atr14 or 0.0001
        
        entry = latest.close
        stop_loss = entry - (2 * atr)
        take_profit = entry + (4 * atr)
        
        return TradingSignal(
            symbol=market_data.symbol,
            signal=SignalType.BUY,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=2.0,
            risk_class=RiskClass.A if score > 0.8 else RiskClass.B,
            reason=f"Bullish momentum breakout, validation score: {score:.2f}"
        )
    
    def _create_sell_signal(self, market_data: MarketData, score: float) -> TradingSignal:
        """Create sell signal"""
        latest = market_data.latest_candle
        atr = latest.atr14 or 0.0001
        
        entry = latest.close
        stop_loss = entry + (2 * atr)
        take_profit = entry - (4 * atr)
        
        return TradingSignal(
            symbol=market_data.symbol,
            signal=SignalType.SELL,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=2.0,
            risk_class=RiskClass.A if score > 0.8 else RiskClass.B,
            reason=f"Bearish momentum breakout, validation score: {score:.2f}"
        )
    
    def _create_wait_signal(self, symbol: str, reason: str) -> TradingSignal:
        """Create wait signal"""
        return TradingSignal(
            symbol=symbol,
            signal=SignalType.WAIT,
            reason=reason,
            risk_class=RiskClass.C
        )
    
    def _create_default_signal(self, market_data: MarketData) -> TradingSignal:
        """Create default signal for historical mode"""
        return self._create_wait_signal(market_data.symbol, "Historical mode")


class BacktestExecutor:
    """Executes trades in backtesting"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.balance = config.initial_balance
        self.equity = config.initial_balance
        self.open_trades: Dict[str, BacktestTrade] = {}
        self.closed_trades: List[BacktestTrade] = []
        self.equity_curve = [config.initial_balance]
    
    def can_open_trade(self) -> bool:
        """Check if we can open a new trade"""
        return len(self.open_trades) < self.config.max_open_trades
    
    def calculate_lot_size(
        self, 
        signal: TradingSignal,
        current_price: float
    ) -> float:
        """Calculate lot size based on risk management"""
        if not signal.stop_loss:
            return 0.01  # Default minimum
        
        risk_amount = self.balance * self.config.risk_per_trade
        stop_distance = abs(current_price - signal.stop_loss)
        
        if stop_distance == 0:
            return 0.01
        
        # Simplified calculation (would need symbol specs in real implementation)
        lot_size = risk_amount / (stop_distance * 100000)  # Assuming standard lot
        
        return max(0.01, round(lot_size, 2))
    
    def open_trade(
        self, 
        signal: TradingSignal,
        current_bar: pd.Series,
        current_time: datetime
    ) -> Optional[BacktestTrade]:
        """Open a new trade"""
        if not self.can_open_trade():
            return None
        
        # Calculate entry with slippage
        slippage = self.config.slippage_points * 0.00001
        
        if signal.signal == SignalType.BUY:
            entry_price = current_bar['close'] + slippage
        else:
            entry_price = current_bar['close'] - slippage
        
        # Calculate lot size
        lot_size = self.calculate_lot_size(signal, entry_price)
        
        # Calculate commission
        commission = self.config.commission_per_lot * lot_size
        
        # Create trade
        trade = BacktestTrade(
            signal=signal,
            entry_price=entry_price,
            entry_time=current_time,
            lot_size=lot_size,
            commission=commission,
            slippage=slippage * 100000 * lot_size  # Convert to money
        )
        
        # Store trade
        self.open_trades[signal.symbol] = trade
        
        # Update balance for commission
        self.balance -= commission
        
        logger.debug(f"Opened {signal.signal.value} trade for {signal.symbol} at {entry_price}")
        
        return trade
    
    def update_trades(self, current_bars: Dict[str, pd.Series], current_time: datetime):
        """Update all open trades with current prices"""
        # Create a copy of the dictionary keys to avoid modification during iteration
        symbols_to_check = list(self.open_trades.keys())
        
        for symbol in symbols_to_check:
            if symbol not in current_bars:
                continue
            
            # Check if trade still exists (might have been closed already)
            if symbol not in self.open_trades:
                continue
                
            trade = self.open_trades[symbol]
            current_bar = current_bars[symbol]
            
            # Check if trade has exceeded max duration
            if trade.bars_held > 20:  # Maximum 20 hours
                self._close_trade(
                    trade, 
                    current_bar['close'], 
                    current_time, 
                    "Max Duration"
                )
                continue  # Skip further checks for this trade
            
            # Check stop loss
            if self._check_stop_loss(trade, current_bar):
                self._close_trade(trade, trade.signal.stop_loss, current_time, "Stop Loss")
                continue  # Skip further checks for this trade
            
            # Check take profit
            elif self._check_take_profit(trade, current_bar):
                self._close_trade(trade, trade.signal.take_profit, current_time, "Take Profit")
                continue  # Skip further checks for this trade
            
            else:
                # Update trade metrics
                self._update_trade_metrics(trade, current_bar)
                trade.bars_held += 1
    
    def _check_stop_loss(self, trade: BacktestTrade, current_bar: pd.Series) -> bool:
        """Check if stop loss is hit"""
        if not trade.signal.stop_loss:
            return False
        
        if trade.signal.signal == SignalType.BUY:
            return current_bar['low'] <= trade.signal.stop_loss
        else:
            return current_bar['high'] >= trade.signal.stop_loss
    
    def _check_take_profit(self, trade: BacktestTrade, current_bar: pd.Series) -> bool:
        """Check if take profit is hit"""
        if not trade.signal.take_profit:
            return False
        
        if trade.signal.signal == SignalType.BUY:
            return current_bar['high'] >= trade.signal.take_profit
        else:
            return current_bar['low'] <= trade.signal.take_profit
    
    def _update_trade_metrics(self, trade: BacktestTrade, current_bar: pd.Series):
        """Update trade metrics"""
        current_price = current_bar['close']
        
        if trade.signal.signal == SignalType.BUY:
            points_profit = current_price - trade.entry_price
        else:
            points_profit = trade.entry_price - current_price
        
        # Update max profit/drawdown
        trade.max_profit = max(trade.max_profit, points_profit)
        trade.max_drawdown = min(trade.max_drawdown, points_profit)
    
    def _close_trade(
        self, 
        trade: BacktestTrade,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str
    ):
        """Close a trade"""
        trade.exit_price = exit_price
        trade.exit_time = exit_time
        trade.exit_reason = exit_reason
        
        # Add slippage to exit
        slippage = self.config.slippage_points * 0.00001
        if trade.signal.signal == SignalType.BUY:
            trade.exit_price -= slippage
        else:
            trade.exit_price += slippage
        
        # Calculate P&L
        if trade.signal.signal == SignalType.BUY:
            trade.pnl_points = trade.exit_price - trade.entry_price
        else:
            trade.pnl_points = trade.entry_price - trade.exit_price
        
        trade.pnl = (trade.pnl_points * 100000 * trade.lot_size) - trade.commission
        
        # Determine result
        if trade.pnl > 0:
            trade.result = TradeResult.WIN
        elif trade.pnl < 0:
            trade.result = TradeResult.LOSS
        else:
            trade.result = TradeResult.BREAKEVEN
        
        # Calculate risk/reward achieved
        if trade.signal.stop_loss:
            risk_points = abs(trade.entry_price - trade.signal.stop_loss)
            if risk_points > 0:
                trade.risk_reward_achieved = trade.pnl_points / risk_points
        
        # Update balance
        self.balance += trade.pnl
        
        # Move to closed trades - Check if trade exists before deleting
        symbol = trade.signal.symbol
        if symbol in self.open_trades:
            del self.open_trades[symbol]
        else:
            logger.warning(f"Trade for {symbol} not found in open_trades when closing")
        
        self.closed_trades.append(trade)
        
        logger.debug(
            f"Closed {trade.signal.signal.value} trade for {symbol} "
            f"at {exit_price} ({exit_reason}), P&L: {trade.pnl:.2f}"
        )
    
    def update_equity(self, current_prices: Optional[Dict[str, float]] = None):
        """Update equity curve"""
        # Calculate open equity
        open_equity = 0
        
        if current_prices:
            for symbol, trade in self.open_trades.items():
                if symbol in current_prices:
                    current_price = current_prices[symbol]
                    
                    if trade.signal.signal == SignalType.BUY:
                        current_pnl = (current_price - trade.entry_price) * 100000 * trade.lot_size
                    else:
                        current_pnl = (trade.entry_price - current_price) * 100000 * trade.lot_size
                    
                    open_equity += current_pnl
        
        self.equity = self.balance + open_equity
        self.equity_curve.append(self.equity)


class BacktestAnalyzer:
    """Analyzes backtesting results"""
    
    def analyze_results(
        self, 
        config: BacktestConfig,
        trades: List[BacktestTrade],
        equity_curve: List[float]
    ) -> BacktestResults:
        """Analyze backtesting results"""
        results = BacktestResults(
            config=config,
            trades=trades,
            equity_curve=equity_curve
        )
        
        if not trades:
            return results
        
        # Basic statistics
        results.total_trades = len(trades)
        results.winning_trades = sum(1 for t in trades if t.result == TradeResult.WIN)
        results.losing_trades = sum(1 for t in trades if t.result == TradeResult.LOSS)
        results.win_rate = results.winning_trades / results.total_trades if results.total_trades > 0 else 0
        
        # P&L statistics
        results.total_pnl = sum(t.pnl for t in trades)
        results.total_return = (results.total_pnl / config.initial_balance) * 100
        
        # Win/Loss statistics
        wins = [t.pnl for t in trades if t.result == TradeResult.WIN]
        losses = [t.pnl for t in trades if t.result == TradeResult.LOSS]
        
        results.average_win = np.mean(wins) if wins else 0
        results.average_loss = np.mean(losses) if losses else 0
        
        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        results.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Expectancy
        results.expectancy = results.total_pnl / results.total_trades if results.total_trades > 0 else 0
        
        # Drawdown analysis
        results.max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        # Risk metrics
        results.sharpe_ratio = self._calculate_sharpe_ratio(equity_curve)
        results.sortino_ratio = self._calculate_sortino_ratio(equity_curve)
        results.calmar_ratio = self._calculate_calmar_ratio(results.total_return, results.max_drawdown)
        
        # Trade statistics
        results.average_bars_held = np.mean([t.bars_held for t in trades])
        results.average_rr_achieved = np.mean([t.risk_reward_achieved for t in trades if t.risk_reward_achieved > 0])
        
        # Additional statistics by various dimensions
        results.statistics = self._calculate_detailed_statistics(trades)
        
        return results
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown percentage"""
        if len(equity_curve) < 2:
            return 0.0
        
        peak = equity_curve[0]
        max_dd = 0.0
        
        for value in equity_curve[1:]:
            if value > peak:
                peak = value
            
            dd = (peak - value) / peak * 100
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _calculate_sharpe_ratio(self, equity_curve: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(equity_curve) < 2:
            return 0.0
        
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    
    def _calculate_sortino_ratio(self, equity_curve: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        if len(equity_curve) < 2:
            return 0.0
        
        returns = np.diff(equity_curve) / equity_curve[:-1]
        excess_returns = returns - (risk_free_rate / 252)
        
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return 0.0
        
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return 0.0
        
        return np.sqrt(252) * np.mean(excess_returns) / downside_std
    
    def _calculate_calmar_ratio(self, total_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        if max_drawdown == 0:
            return 0.0
        
        return total_return / max_drawdown
    
    def _calculate_detailed_statistics(self, trades: List[BacktestTrade]) -> Dict[str, Any]:
        """Calculate detailed statistics by various dimensions"""
        stats = {}
        
        # By symbol
        symbol_stats = {}
        for trade in trades:
            symbol = trade.signal.symbol
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {
                    'trades': 0,
                    'wins': 0,
                    'pnl': 0.0,
                    'win_rate': 0.0
                }
            
            symbol_stats[symbol]['trades'] += 1
            symbol_stats[symbol]['pnl'] += trade.pnl
            if trade.result == TradeResult.WIN:
                symbol_stats[symbol]['wins'] += 1
        
        for symbol, data in symbol_stats.items():
            data['win_rate'] = data['wins'] / data['trades'] if data['trades'] > 0 else 0
        
        stats['by_symbol'] = symbol_stats
        
        # By day of week
        day_stats = {}
        for trade in trades:
            day = trade.entry_time.strftime('%A')
            if day not in day_stats:
                day_stats[day] = {'trades': 0, 'pnl': 0.0}
            
            day_stats[day]['trades'] += 1
            day_stats[day]['pnl'] += trade.pnl
        
        stats['by_day'] = day_stats
        
        # By hour
        hour_stats = {}
        for trade in trades:
            hour = trade.entry_time.hour
            if hour not in hour_stats:
                hour_stats[hour] = {'trades': 0, 'pnl': 0.0}
            
            hour_stats[hour]['trades'] += 1
            hour_stats[hour]['pnl'] += trade.pnl
        
        stats['by_hour'] = hour_stats
        
        # By risk class
        risk_stats = {}
        for trade in trades:
            risk_class = trade.signal.risk_class.value
            if risk_class not in risk_stats:
                risk_stats[risk_class] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
            
            risk_stats[risk_class]['trades'] += 1
            risk_stats[risk_class]['pnl'] += trade.pnl
            if trade.result == TradeResult.WIN:
                risk_stats[risk_class]['wins'] += 1
        
        stats['by_risk_class'] = risk_stats
        
        return stats


class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(
        self,
        data_provider: MT5DataProvider,
        signal_generator: Optional[BacktestSignalGenerator] = None,
        chart_generator: Optional[ChartGenerator] = None
    ):
        self.data_provider = BacktestDataProvider(data_provider)
        self.signal_generator = signal_generator or BacktestSignalGenerator()
        self.chart_generator = chart_generator
        self.analyzer = BacktestAnalyzer()
    
    async def run_backtest(self, config: BacktestConfig) -> BacktestResults:
        """Run backtest with given configuration"""
        logger.info(f"Starting backtest from {config.start_date} to {config.end_date}")
        
        # Initialize executor
        executor = BacktestExecutor(config)
        
        # Store all symbol data first
        all_data = {}
        for symbol in config.symbols:
            logger.info(f"Loading data for {symbol}")
            df = await self.data_provider.get_historical_data(
                symbol, 
                config.start_date,
                config.end_date,
                config.timeframe
            )
            if not df.empty:
                all_data[symbol] = df
            else:
                logger.warning(f"No data available for {symbol}")
        
        if not all_data:
            logger.error("No data available for any symbols")
            return BacktestResults(config=config)
        
        # Find common dates across all symbols
        all_dates = None
        for symbol, df in all_data.items():
            if all_dates is None:
                all_dates = set(df.index)
            else:
                all_dates = all_dates.intersection(set(df.index))
        
        all_dates = sorted(list(all_dates))
        
        # Process each date
        for date_idx, current_date in enumerate(all_dates[20:], 20):  # Start from bar 20 for indicators
            # Prepare current bars for all symbols
            current_bars = {}
            current_prices = {}
            
            for symbol, df in all_data.items():
                if current_date in df.index:
                    current_bars[symbol] = df.loc[current_date]
                    current_prices[symbol] = df.loc[current_date]['close']
            
            # Process each symbol
            for symbol in config.symbols:
                if symbol not in all_data or current_date not in all_data[symbol].index:
                    continue
                
                df = all_data[symbol]
                date_position = df.index.get_loc(current_date)
                
                # Check for open trades
                if symbol not in executor.open_trades:
                    # Create market data for signal generation
                    market_data = self._create_market_data(
                        symbol,
                        df.iloc[max(0, date_position-100):date_position+1]
                    )
                    
                    # Generate signal
                    signal = await self.signal_generator.generate_signal(market_data)
                    
                    # Execute if actionable
                    if signal.is_actionable:
                        executor.open_trade(signal, df.loc[current_date], current_date)
            
            # Update all trades with current prices
            executor.update_trades(current_bars, current_date)
            
            # Update equity with current prices
            executor.update_equity(current_prices)
        
        # Close any remaining open trades
        for symbol, trade in list(executor.open_trades.items()):
            if symbol in all_data and len(all_data[symbol]) > 0:
                last_price = all_data[symbol].iloc[-1]['close']
                executor._close_trade(
                    trade,
                    last_price,
                    config.end_date,
                    "Backtest End"
                )
        
        # Analyze results
        results = self.analyzer.analyze_results(
            config,
            executor.closed_trades,
            executor.equity_curve
        )
        
        # Save results if requested
        if config.save_results:
            await self._save_results(results)
        
        logger.info(f"Backtest complete: {results.total_trades} trades, "
                f"{results.win_rate:.1%} win rate, "
                f"total return: {results.total_return:.2f}%")
        
        return results

    def _create_market_data(self, symbol: str, df: pd.DataFrame) -> MarketData:
        """Create MarketData object from DataFrame"""
        candles = []
        
        for idx, row in df.iterrows():
            candle = Candle(
                timestamp=idx,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=int(row['volume']),
                spread=row.get('spread'),
                ema50=row.get('ema50'),
                ema200=row.get('ema200'),
                rsi14=row.get('rsi14'),
                atr14=row.get('atr14')
            )
            candles.append(candle)
        
        return MarketData(
            symbol=symbol,
            timeframe="H1",
            candles=candles
        )

    async def _save_results(self, results: BacktestResults):
        """Save backtest results to file"""
        results_dir = Path(results.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        symbols_str = "_".join(results.config.symbols[:3])  # First 3 symbols
        filename = f"backtest_{symbols_str}_{timestamp}.json"
        
        # Prepare results for JSON serialization
        results_dict = {
            'config': {
                'start_date': results.config.start_date.isoformat(),
                'end_date': results.config.end_date.isoformat(),
                'symbols': results.config.symbols,
                'initial_balance': results.config.initial_balance,
                'risk_per_trade': results.config.risk_per_trade,
                'mode': results.config.mode.value
            },
            'summary': {
                'total_trades': results.total_trades,
                'winning_trades': results.winning_trades,
                'losing_trades': results.losing_trades,
                'win_rate': results.win_rate,
                'total_pnl': results.total_pnl,
                'total_return': results.total_return,
                'max_drawdown': results.max_drawdown,
                'sharpe_ratio': results.sharpe_ratio,
                'sortino_ratio': results.sortino_ratio,
                'calmar_ratio': results.calmar_ratio,
                'profit_factor': results.profit_factor,
                'expectancy': results.expectancy,
                'average_win': results.average_win,
                'average_loss': results.average_loss,
                'average_bars_held': results.average_bars_held,
                'average_rr_achieved': results.average_rr_achieved
            },
            'trades': [
                {
                    'symbol': trade.signal.symbol,
                    'signal': trade.signal.signal.value,
                    'entry_time': trade.entry_time.isoformat(),
                    'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'pnl': trade.pnl,
                    'pnl_points': trade.pnl_points,
                    'result': trade.result.value if trade.result else None,
                    'exit_reason': trade.exit_reason,
                    'bars_held': trade.bars_held,
                    'risk_reward_achieved': trade.risk_reward_achieved
                }
                for trade in results.trades
            ],
            'equity_curve': results.equity_curve,
            'statistics': results.statistics
        }
        
        # Save to JSON file
        filepath = results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
        
        # Also save a summary report
        await self._save_summary_report(results, results_dir / f"report_{symbols_str}_{timestamp}.txt")

        # NEW: Save to database for ML
        try:            
            repo = BacktestRepository(self.db_path)  # You'll need to pass db_path
            run_id = repo.save_backtest_run(results)
            
            logger.info(f"Backtest results saved to database with ID: {run_id}")
            
            # Also store the run_id in the JSON for reference
            results_dict['database_run_id'] = run_id
            
            # Re-save JSON with database reference
            with open(filepath, 'w') as f:
                json.dump(results_dict, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save backtest results to database: {e}")
            # Don't fail the whole backtest if DB save fails

    async def _save_summary_report(self, results: BacktestResults, filepath: Path):
        """Save human-readable summary report"""
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BACKTEST RESULTS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # Configuration
            f.write("Configuration:\n")
            f.write(f"  Period: {results.config.start_date.date()} to {results.config.end_date.date()}\n")
            f.write(f"  Symbols: {', '.join(results.config.symbols)}\n")
            f.write(f"  Initial Balance: ${results.config.initial_balance:,.2f}\n")
            f.write(f"  Risk per Trade: {results.config.risk_per_trade*100:.1f}%\n")
            f.write(f"  Mode: {results.config.mode.value}\n\n")
            
            # Performance Summary
            f.write("Performance Summary:\n")
            f.write(f"  Total Trades: {results.total_trades}\n")
            f.write(f"  Winning Trades: {results.winning_trades}\n")
            f.write(f"  Losing Trades: {results.losing_trades}\n")
            f.write(f"  Win Rate: {results.win_rate:.1%}\n")
            f.write(f"  Total P&L: ${results.total_pnl:,.2f}\n")
            f.write(f"  Total Return: {results.total_return:.2f}%\n")
            f.write(f"  Final Balance: ${results.equity_curve[-1]:,.2f}\n\n")
            
            # Risk Metrics
            f.write("Risk Metrics:\n")
            f.write(f"  Max Drawdown: {results.max_drawdown:.2f}%\n")
            f.write(f"  Sharpe Ratio: {results.sharpe_ratio:.2f}\n")
            f.write(f"  Sortino Ratio: {results.sortino_ratio:.2f}\n")
            f.write(f"  Calmar Ratio: {results.calmar_ratio:.2f}\n\n")
            
            # Trade Statistics
            f.write("Trade Statistics:\n")
            f.write(f"  Average Win: ${results.average_win:,.2f}\n")
            f.write(f"  Average Loss: ${results.average_loss:,.2f}\n")
            f.write(f"  Profit Factor: {results.profit_factor:.2f}\n")
            f.write(f"  Expectancy: ${results.expectancy:,.2f}\n")
            f.write(f"  Avg Bars Held: {results.average_bars_held:.1f}\n")
            f.write(f"  Avg RR Achieved: {results.average_rr_achieved:.2f}\n\n")
            
            # By Symbol Performance
            if 'by_symbol' in results.statistics:
                f.write("Performance by Symbol:\n")
                for symbol, stats in results.statistics['by_symbol'].items():
                    f.write(f"  {symbol}:\n")
                    f.write(f"    Trades: {stats['trades']}\n")
                    f.write(f"    Win Rate: {stats['win_rate']:.1%}\n")
                    f.write(f"    P&L: ${stats['pnl']:,.2f}\n")
            
            f.write("\n" + "=" * 80 + "\n")


class BacktestReportGenerator:
    """Generates detailed backtest reports and visualizations"""
    
    def __init__(self, chart_generator: Optional[ChartGenerator] = None):
        self.chart_generator = chart_generator
    
    async def generate_report(self, results: BacktestResults, output_dir: str):
        """Generate comprehensive backtest report with charts"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate equity curve chart
        await self._plot_equity_curve(results, output_path / "equity_curve.png")
        
        # Generate drawdown chart
        await self._plot_drawdown(results, output_path / "drawdown.png")
        
        # Generate monthly returns heatmap
        await self._plot_monthly_returns(results, output_path / "monthly_returns.png")
        
        # Generate trade distribution charts
        await self._plot_trade_distribution(results, output_path / "trade_distribution.png")
        
        # Generate HTML report
        await self._generate_html_report(results, output_path / "report.html")
    
    async def _plot_equity_curve(self, results: BacktestResults, filepath: Path):
        """Plot equity curve"""
        if not self.chart_generator:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create x-axis (assuming daily data)
            dates = pd.date_range(
                start=results.config.start_date,
                end=results.config.end_date,
                periods=len(results.equity_curve)
            )
            
            ax.plot(dates, results.equity_curve, linewidth=2)
            ax.set_title('Equity Curve', fontsize=16)
            ax.set_xlabel('Date')
            ax.set_ylabel('Equity ($)')
            ax.grid(True, alpha=0.3)
            
            # Add horizontal line for initial balance
            ax.axhline(
                y=results.config.initial_balance,
                color='red',
                linestyle='--',
                alpha=0.5,
                label='Initial Balance'
            )
            
            ax.legend()
            plt.tight_layout()
            plt.savefig(filepath, dpi=100)
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to plot equity curve: {e}")
    
    async def _plot_drawdown(self, results: BacktestResults, filepath: Path):
        """Plot drawdown chart"""
        if not self.chart_generator:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Calculate drawdown series
            equity = np.array(results.equity_curve)
            running_max = np.maximum.accumulate(equity)
            drawdown = (equity - running_max) / running_max * 100
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            dates = pd.date_range(
                start=results.config.start_date,
                end=results.config.end_date,
                periods=len(drawdown)
            )
            
            ax.fill_between(dates, drawdown, 0, color='red', alpha=0.3)
            ax.plot(dates, drawdown, color='red', linewidth=1)
            ax.set_title('Drawdown', fontsize=16)
            ax.set_xlabel('Date')
            ax.set_ylabel('Drawdown (%)')
            ax.grid(True, alpha=0.3)
            
            # Add max drawdown line
            max_dd_idx = np.argmin(drawdown)
            ax.annotate(
                f'Max DD: {results.max_drawdown:.2f}%',
                xy=(dates[max_dd_idx], drawdown[max_dd_idx]),
                xytext=(dates[max_dd_idx], drawdown[max_dd_idx] - 5),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10
            )
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=100)
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to plot drawdown: {e}")
    
    async def _plot_monthly_returns(self, results: BacktestResults, filepath: Path):
        """Plot monthly returns heatmap"""
        # Implementation would create a heatmap of monthly returns
        pass
    
    async def _plot_trade_distribution(self, results: BacktestResults, filepath: Path):
        """Plot trade distribution charts"""
        if not self.chart_generator:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # P&L distribution
            pnls = [trade.pnl for trade in results.trades]
            axes[0, 0].hist(pnls, bins=30, alpha=0.7, color='blue')
            axes[0, 0].axvline(x=0, color='red', linestyle='--')
            axes[0, 0].set_title('P&L Distribution')
            axes[0, 0].set_xlabel('P&L ($)')
            axes[0, 0].set_ylabel('Frequency')
            
            # Win rate by symbol
            if 'by_symbol' in results.statistics:
                symbols = list(results.statistics['by_symbol'].keys())
                win_rates = [stats['win_rate'] for stats in results.statistics['by_symbol'].values()]
                axes[0, 1].bar(symbols, win_rates)
                axes[0, 1].set_title('Win Rate by Symbol')
                axes[0, 1].set_ylabel('Win Rate')
                axes[0, 1].set_ylim(0, 1)
            
            # P&L by hour
            if 'by_hour' in results.statistics:
                hours = sorted(results.statistics['by_hour'].keys())
                hour_pnls = [results.statistics['by_hour'][h]['pnl'] for h in hours]
                axes[1, 0].bar(hours, hour_pnls)
                axes[1, 0].set_title('P&L by Hour')
                axes[1, 0].set_xlabel('Hour (UTC)')
                axes[1, 0].set_ylabel('Total P&L ($)')
            
            # Trade duration distribution
            durations = [trade.bars_held for trade in results.trades]
            axes[1, 1].hist(durations, bins=20, alpha=0.7, color='green')
            axes[1, 1].set_title('Trade Duration Distribution')
            axes[1, 1].set_xlabel('Bars Held')
            axes[1, 1].set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=100)
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to plot trade distribution: {e}")
    
    async def _generate_html_report(self, results: BacktestResults, filepath: Path):
        """Generate HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ font-size: 24px; font-weight: bold; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .chart {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Backtest Results</h1>
            
            <h2>Configuration</h2>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Period</td><td>{results.config.start_date.date()} to {results.config.end_date.date()}</td></tr>
                <tr><td>Symbols</td><td>{', '.join(results.config.symbols)}</td></tr>
                <tr><td>Initial Balance</td><td>${results.config.initial_balance:,.2f}</td></tr>
                <tr><td>Risk per Trade</td><td>{results.config.risk_per_trade*100:.1f}%</td></tr>
            </table>
            
            <h2>Performance Summary</h2>
            <table>
                <tr>
                    <td>Total Return</td>
                    <td class="metric {'positive' if results.total_return > 0 else 'negative'}">{results.total_return:.2f}%</td>
                </tr>
                <tr>
                    <td>Win Rate</td>
                    <td class="metric">{results.win_rate:.1%}</td>
                </tr>
                <tr>
                    <td>Profit Factor</td>
                    <td class="metric">{results.profit_factor:.2f}</td>
                </tr>
                <tr>
                    <td>Sharpe Ratio</td>
                    <td class="metric">{results.sharpe_ratio:.2f}</td>
                </tr>
            </table>
            
            <div class="chart">
                <h2>Equity Curve</h2>
                <img src="equity_curve.png" width="800">
            </div>
            
            <div class="chart">
                <h2>Drawdown</h2>
                <img src="drawdown.png" width="800">
            </div>
            
            <div class="chart">
                <h2>Trade Distribution</h2>
                <img src="trade_distribution.png" width="800">
            </div>
        </body>
        </html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html_content)


# Example usage function
async def run_backtest_example():
    """Example of how to run a backtest"""
    from config.settings import get_settings
    from core.infrastructure.mt5.client import MT5Client
    from core.infrastructure.mt5.data_provider import MT5DataProvider
    
    # Initialize components
    settings = get_settings()
    mt5_client = MT5Client(settings.mt5)
    data_provider = MT5DataProvider(mt5_client)
    
    # Configure backtest
    config = BacktestConfig(
        start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 12, 31, tzinfo=timezone.utc),
        symbols=["EURUSD", "GBPUSD"],
        mode=BacktestMode.OFFLINE_ONLY,
        risk_per_trade=0.01,  # 1% risk
        initial_balance=10000
    )
    
    # Create and run backtest
    engine = BacktestEngine(data_provider)
    results = await engine.run_backtest(config)
    
    # Generate report
    report_generator = BacktestReportGenerator()
    await report_generator.generate_report(results, "backtest_results")
    
    return results


# Export main components
__all__ = [
    'BacktestEngine',
    'BacktestConfig',
    'BacktestResults',
    'BacktestMode',
    'BacktestReportGenerator',
    'run_backtest_example'
]