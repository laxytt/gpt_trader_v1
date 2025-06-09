"""
MT5 data provider for market data retrieval and technical analysis.
Handles candle data fetching, indicator calculations, and chart generation.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import pandas as pd

from core.infrastructure.data.unified_data_provider import DataRequest, UnifiedDataProvider
import numpy as np

from core.infrastructure.mt5.client import MT5Client
from core.domain.models import MarketData, Candle
from core.domain.exceptions import (
    MT5DataError, InsufficientDataError, ErrorContext
)
from core.domain.enums import MT5TimeFrame, TIMEFRAME_TO_MT5, TimeFrame
from core.utils.chart_utils import ChartGenerator
from core.utils.symbol_resolver import get_symbol_resolver


logger = logging.getLogger(__name__)


class MT5DataProvider:
    """
    MT5 data provider that delegates to UnifiedDataProvider.
    Maintains backward compatibility while using unified logic.    """
    
    def __init__(self, mt5_client: MT5Client, chart_generator: Optional[ChartGenerator] = None):
        self.mt5_client = mt5_client
        self.chart_generator = chart_generator
        self.unified_provider = UnifiedDataProvider(mt5_client)
        self.symbol_resolver = get_symbol_resolver()
        
        # Technical analysis parameters
        self.indicator_warmup_bars = 250  # Bars needed for indicator stability
        self.min_required_bars = 20      # Minimum bars for analysis
        
        # Position trading specific parameters
        self.position_trading_min_daily_bars = 100  # Minimum daily bars for position analysis
        self.position_trading_min_weekly_bars = 52  # Minimum weekly bars for trend context
    
    async def get_market_data(
        self, 
        symbol: str, 
        timeframe: TimeFrame, 
        bars: int = 100,
        include_indicators: bool = True
    ) -> MarketData:
        """
        Get market data with optional technical indicators.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for data
            bars: Number of bars to retrieve
            include_indicators: Whether to calculate technical indicators
            
        Returns:
            MarketData object with candles and indicators
            
        Raises:
            MT5DataError: If data retrieval fails
            InsufficientDataError: If not enough data available
        """

        # Resolve symbol name for broker compatibility
        resolved_symbol = self.symbol_resolver.resolve_and_enable(symbol)
        if not resolved_symbol:
            logger.warning(f"Failed to resolve symbol {symbol}, using original name")
            resolved_symbol = symbol
        elif resolved_symbol != symbol:
            logger.info(f"Resolved symbol {symbol} -> {resolved_symbol}")
            
        # Adjust bar count for position trading timeframes
        adjusted_bars = bars
        if timeframe == TimeFrame.D1 and bars < self.position_trading_min_daily_bars:
            logger.info(f"Adjusting daily bars from {bars} to {self.position_trading_min_daily_bars} for position trading")
            adjusted_bars = self.position_trading_min_daily_bars
        elif timeframe == TimeFrame.W1 and bars < self.position_trading_min_weekly_bars:
            logger.info(f"Adjusting weekly bars from {bars} to {self.position_trading_min_weekly_bars} for position trading")
            adjusted_bars = self.position_trading_min_weekly_bars
        
        # Create request for recent bars
        request = DataRequest(
            symbol=resolved_symbol,
            timeframe=timeframe,
            num_bars=adjusted_bars
        )
        
         # Use unified provider
        market_data = await self.unified_provider.get_data(request)
        
        # Indicators are already included from unified provider
        return market_data
    
    def _rates_to_dataframe(self, rates_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert MT5 rates data to pandas DataFrame"""
        df = pd.DataFrame(rates_data)
        
        # Convert time to datetime
        df['timestamp'] = pd.to_datetime(df['time'], unit='s', utc=True)
        
        # Rename tick_volume to volume for consistency
        if 'tick_volume' in df.columns:
            df = df.rename(columns={'tick_volume': 'volume'})
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise MT5DataError(f"Missing required column: {col}")
        
        # Add spread calculation if not present
        if 'spread' not in df.columns:
            df['spread'] = np.nan  # Will be filled by real spread data if available
        
        return df
    
    # Note: _calculate_indicators method removed as indicators are now calculated in UnifiedDataProvider
    
    def _dataframe_to_candles(self, df: pd.DataFrame) -> List[Candle]:
        """Convert DataFrame to list of Candle objects"""
        candles = []
        
        for _, row in df.iterrows():
            candle = Candle(
                timestamp=row['timestamp'],
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=int(row['volume']),
                spread=float(row.get('spread', 0)) if pd.notna(row.get('spread')) else None,
                
                # Technical indicators (handle NaN values)
                ema50=float(row['ema50']) if pd.notna(row.get('ema50')) else None,
                ema200=float(row['ema200']) if pd.notna(row.get('ema200')) else None,
                rsi14=float(row['rsi14']) if pd.notna(row.get('rsi14')) else None,
                atr14=float(row['atr14']) if pd.notna(row.get('atr14')) else None,
                rsi_slope=float(row['rsi_slope']) if pd.notna(row.get('rsi_slope')) else None,
                
                # Additional indicators for position trading
                ema20=float(row['ema20']) if pd.notna(row.get('ema20')) else None,
                sma50=float(row['sma50']) if pd.notna(row.get('sma50')) else None,
                sma200=float(row['sma200']) if pd.notna(row.get('sma200')) else None,
                true_range=float(row['true_range']) if pd.notna(row.get('true_range')) else None,
                atr_percentage=float(row['atr_percentage']) if pd.notna(row.get('atr_percentage')) else None,
                volume_ratio=float(row['volume_ratio']) if pd.notna(row.get('volume_ratio')) else None,
                body_size=float(row['body_size']) if pd.notna(row.get('body_size')) else None,
                upper_shadow=float(row['upper_shadow']) if pd.notna(row.get('upper_shadow')) else None,
                lower_shadow=float(row['lower_shadow']) if pd.notna(row.get('lower_shadow')) else None,
                weekly_range=float(row['weekly_range']) if pd.notna(row.get('weekly_range')) else None
            )
            candles.append(candle)
        
        return candles
    
    async def get_multi_timeframe_data(
        self, 
        symbol: str, 
        timeframes: List[TimeFrame],
        bars: int = 100
    ) -> Dict[str, MarketData]:
        """
        Get market data for multiple timeframes.
        
        Args:
            symbol: Trading symbol
            timeframes: List of timeframes
            bars: Number of bars per timeframe
            
        Returns:
            Dict mapping timeframe to MarketData
        """
        result = {}
        
        for tf in timeframes:
            try:
                market_data = await self.get_market_data(symbol, tf, bars)
                result[tf.value] = market_data
            except Exception as e:
                logger.error(f"Failed to get data for {symbol} {tf}: {e}")
                # Continue with other timeframes
        
        return result 
    
    def get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get current bid/ask prices for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with bid/ask prices or None if failed
        """
        tick = self.mt5_client.get_symbol_tick(symbol)
        if not tick:
            return None
        
        return {
            'bid': tick['bid'],
            'ask': tick['ask'],
            'last': tick.get('last', tick['bid']),
            'time': datetime.fromtimestamp(tick['time'], tz=timezone.utc)
        }
    
    def calculate_volatility_metrics(self, market_data: MarketData) -> Dict[str, float]:
        """
        Calculate volatility metrics from market data.
        
        Args:
            market_data: Market data to analyze
            
        Returns:
            Dict with volatility metrics
        """
        if not market_data.candles:
            return {}
        
        # Get ATR values
        atr_values = [c.atr14 for c in market_data.candles if c.atr14 is not None]
        
        if not atr_values:
            return {}
        
        current_atr = atr_values[-1] if atr_values else 0
        avg_atr = sum(atr_values) / len(atr_values)
        
        # Calculate volume metrics
        volumes = [c.volume for c in market_data.candles]
        avg_volume = sum(volumes) / len(volumes) if volumes else 0
        current_volume = volumes[-1] if volumes else 0
        
        return {
            'current_atr': current_atr,
            'average_atr': avg_atr,
            'atr_ratio': current_atr / avg_atr if avg_atr > 0 else 1,
            'current_volume': current_volume,
            'average_volume': avg_volume,
            'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1,
            'volatility_level': self._classify_volatility(current_atr, avg_atr)
        }
    
    def _classify_volatility(self, current_atr: float, avg_atr: float) -> str:
        """Classify volatility level based on ATR"""
        if avg_atr == 0:
            return "unknown"
        
        ratio = current_atr / avg_atr
        
        if ratio > 1.3:
            return "high"
        elif ratio < 0.7:
            return "low"
        else:
            return "medium"
    
    async def validate_symbol_data(self, symbol: str, timeframe: TimeFrame) -> bool:
        """
        Validate that symbol has sufficient data for analysis.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe to check
            
        Returns:
            bool: True if symbol has sufficient data
        """
        try:
            # Use appropriate minimum bars based on timeframe
            min_bars = self.min_required_bars
            if timeframe == TimeFrame.D1:
                min_bars = self.position_trading_min_daily_bars
            elif timeframe == TimeFrame.W1:
                min_bars = self.position_trading_min_weekly_bars
                
            market_data = await self.get_market_data(symbol, timeframe, bars=min_bars)
            return len(market_data.candles) >= min_bars
        except Exception:
            return False
    
    async def get_position_trading_data(
        self,
        symbol: str,
        include_weekly: bool = True
    ) -> Dict[str, MarketData]:
        """
        Get comprehensive data for position trading analysis.
        
        Args:
            symbol: Trading symbol
            include_weekly: Whether to include weekly timeframe data
            
        Returns:
            Dict with 'daily' and optionally 'weekly' MarketData
        """
        result = {}
        
        # Get daily data (100+ bars for proper trend analysis)
        try:
            daily_data = await self.get_market_data(
                symbol, 
                TimeFrame.D1, 
                bars=self.position_trading_min_daily_bars
            )
            result['daily'] = daily_data
            logger.info(f"Retrieved {len(daily_data.candles)} daily bars for {symbol}")
        except Exception as e:
            logger.error(f"Failed to get daily data for {symbol}: {e}")
            raise
        
        # Get weekly data if requested (52+ bars for year-long trend)
        if include_weekly:
            try:
                weekly_data = await self.get_market_data(
                    symbol,
                    TimeFrame.W1,
                    bars=self.position_trading_min_weekly_bars
                )
                result['weekly'] = weekly_data
                logger.info(f"Retrieved {len(weekly_data.candles)} weekly bars for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to get weekly data for {symbol}: {e}")
                # Don't fail if weekly data unavailable, daily is sufficient
        
        return result
    
    def calculate_position_size_metrics(
        self, 
        market_data: MarketData,
        account_balance: float,
        risk_percentage: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate position sizing metrics for longer-term trades.
        
        Args:
            market_data: Market data (preferably daily timeframe)
            account_balance: Account balance for position sizing
            risk_percentage: Risk percentage per trade (default 1%)
            
        Returns:
            Dict with position sizing recommendations
        """
        if not market_data.candles or len(market_data.candles) < 14:
            return {}
        
        # Get latest ATR and price
        latest_candle = market_data.candles[-1]
        current_price = latest_candle.close
        current_atr = latest_candle.atr14 or 0
        
        if current_atr == 0:
            return {}
        
        # Calculate risk amount
        risk_amount = account_balance * (risk_percentage / 100)
        
        # Position sizing based on ATR
        # For position trading, use 2x ATR for stop loss
        stop_distance = current_atr * 2
        position_size = risk_amount / stop_distance
        
        # Calculate position value and lots (assuming standard lot = 100,000)
        position_value = position_size * current_price
        lots = position_size / 100000
        
        return {
            'current_price': current_price,
            'current_atr': current_atr,
            'atr_percentage': (current_atr / current_price) * 100,
            'suggested_stop_distance': stop_distance,
            'suggested_stop_price_long': current_price - stop_distance,
            'suggested_stop_price_short': current_price + stop_distance,
            'position_size': position_size,
            'position_value': position_value,
            'lots': round(lots, 2),
            'risk_amount': risk_amount
        }


# Export main class
__all__ = ['MT5DataProvider']