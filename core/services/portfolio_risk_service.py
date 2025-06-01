"""
Portfolio-level risk management service.
Manages overall portfolio risk, correlation limits, and drawdown controls.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
import numpy as np
import pandas as pd

from core.domain.models import Trade, TradeStatus, TradingSignal, SignalType
from core.domain.exceptions import ServiceError, ErrorContext
from core.infrastructure.database.repositories import TradeRepository
from core.infrastructure.mt5.client import MT5Client
from core.utils.error_handling import with_error_recovery
from config.settings import TradingSettings

logger = logging.getLogger(__name__)


@dataclass
class PortfolioMetrics:
    """Current portfolio risk metrics"""
    total_exposure: float
    current_drawdown: float
    correlation_score: float
    var_95: float  # Value at Risk at 95% confidence
    sharpe_ratio: float
    open_positions: int
    risk_score: float  # Overall risk score (0-100)


@dataclass
class RiskLimits:
    """Portfolio risk limits"""
    max_total_exposure: float = 0.1  # 10% of account
    max_drawdown: float = 0.15  # 15% max drawdown
    max_correlation: float = 0.7  # Max correlation between positions
    max_positions: int = 5  # Max concurrent positions
    max_position_size: float = 0.03  # 3% per position
    min_sharpe_ratio: float = 0.5  # Minimum acceptable Sharpe
    max_var_95: float = 0.05  # 5% VaR limit


class PortfolioRiskManager:
    """
    Manages portfolio-level risk including:
    - Position sizing based on portfolio exposure
    - Correlation-based position limits
    - Drawdown-based trading restrictions
    - Value at Risk (VaR) calculations
    """
    
    def __init__(
        self,
        trade_repository: TradeRepository,
        mt5_client: MT5Client,
        trading_config: TradingSettings,
        risk_limits: Optional[RiskLimits] = None
    ):
        self.trade_repository = trade_repository
        self.mt5_client = mt5_client
        self.trading_config = trading_config
        self.risk_limits = risk_limits or RiskLimits()
        
        # Cache for correlation data
        self._correlation_cache = {}
        self._cache_expiry = datetime.now(timezone.utc)
        
    @with_error_recovery
    async def check_signal_risk(self, signal: TradingSignal) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a trading signal passes portfolio risk checks.
        
        Returns:
            Tuple of (is_allowed, risk_details)
        """
        with ErrorContext("Portfolio risk check", symbol=signal.symbol):
            # Get current portfolio metrics
            metrics = await self.calculate_portfolio_metrics()
            
            risk_checks = {
                'max_positions': self._check_position_limit(metrics),
                'max_exposure': self._check_exposure_limit(metrics, signal),
                'correlation': await self._check_correlation_limit(signal),
                'drawdown': self._check_drawdown_limit(metrics),
                'var_limit': self._check_var_limit(metrics, signal),
                'sharpe_ratio': self._check_sharpe_limit(metrics)
            }
            
            # Calculate if signal is allowed
            is_allowed = all(check['passed'] for check in risk_checks.values())
            
            # Adjust position size if needed
            suggested_size = self._calculate_risk_adjusted_size(
                signal, metrics, risk_checks
            )
            
            return is_allowed, {
                'risk_checks': risk_checks,
                'current_metrics': metrics,
                'suggested_position_size': suggested_size,
                'risk_score': metrics.risk_score
            }
    
    async def calculate_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate current portfolio risk metrics"""
        # Get open trades
        open_trades = self.trade_repository.get_trades_by_status(TradeStatus.OPEN)
        
        # Get account info
        account_info = self.mt5_client.get_account_info()
        if not account_info:
            raise ServiceError("Failed to get account information")
        
        balance = account_info['balance']
        equity = account_info['equity']
        
        # Calculate exposure
        total_exposure = sum(
            abs(trade.position_size * trade.entry_price) 
            for trade in open_trades
        ) / balance
        
        # Calculate drawdown
        current_drawdown = (balance - equity) / balance if balance > 0 else 0
        
        # Calculate correlation score
        correlation_score = await self._calculate_portfolio_correlation(open_trades)
        
        # Calculate VaR
        var_95 = await self._calculate_var(open_trades, confidence=0.95)
        
        # Calculate Sharpe ratio (rolling 30 days)
        sharpe_ratio = await self._calculate_rolling_sharpe(days=30)
        
        # Calculate overall risk score (0-100)
        risk_score = self._calculate_risk_score(
            total_exposure, current_drawdown, correlation_score, var_95
        )
        
        return PortfolioMetrics(
            total_exposure=total_exposure,
            current_drawdown=current_drawdown,
            correlation_score=correlation_score,
            var_95=var_95,
            sharpe_ratio=sharpe_ratio,
            open_positions=len(open_trades),
            risk_score=risk_score
        )
    
    def _check_position_limit(self, metrics: PortfolioMetrics) -> Dict[str, Any]:
        """Check if we're within position limits"""
        passed = metrics.open_positions < self.risk_limits.max_positions
        return {
            'passed': passed,
            'current': metrics.open_positions,
            'limit': self.risk_limits.max_positions,
            'message': f"Positions: {metrics.open_positions}/{self.risk_limits.max_positions}"
        }
    
    def _check_exposure_limit(
        self, 
        metrics: PortfolioMetrics, 
        signal: TradingSignal
    ) -> Dict[str, Any]:
        """Check if new position would exceed exposure limits"""
        # Estimate new position exposure
        estimated_position_exposure = self.trading_config.risk_per_trade_percent / 100
        new_total_exposure = metrics.total_exposure + estimated_position_exposure
        
        passed = new_total_exposure <= self.risk_limits.max_total_exposure
        return {
            'passed': passed,
            'current': metrics.total_exposure,
            'new_total': new_total_exposure,
            'limit': self.risk_limits.max_total_exposure,
            'message': f"Exposure: {new_total_exposure:.1%} / {self.risk_limits.max_total_exposure:.1%}"
        }
    
    async def _check_correlation_limit(self, signal: TradingSignal) -> Dict[str, Any]:
        """Check if new position would exceed correlation limits"""
        open_trades = self.trade_repository.get_trades_by_status(TradeStatus.OPEN)
        
        if not open_trades:
            return {
                'passed': True,
                'max_correlation': 0.0,
                'message': "No open positions to correlate with"
            }
        
        # Get correlation with existing positions
        max_correlation = 0.0
        for trade in open_trades:
            correlation = await self._get_symbol_correlation(signal.symbol, trade.symbol)
            max_correlation = max(max_correlation, abs(correlation))
        
        passed = max_correlation <= self.risk_limits.max_correlation
        return {
            'passed': passed,
            'max_correlation': max_correlation,
            'limit': self.risk_limits.max_correlation,
            'message': f"Max correlation: {max_correlation:.2f} / {self.risk_limits.max_correlation:.2f}"
        }
    
    def _check_drawdown_limit(self, metrics: PortfolioMetrics) -> Dict[str, Any]:
        """Check if current drawdown allows new trades"""
        passed = metrics.current_drawdown < self.risk_limits.max_drawdown
        return {
            'passed': passed,
            'current': metrics.current_drawdown,
            'limit': self.risk_limits.max_drawdown,
            'message': f"Drawdown: {metrics.current_drawdown:.1%} / {self.risk_limits.max_drawdown:.1%}"
        }
    
    def _check_var_limit(
        self, 
        metrics: PortfolioMetrics, 
        signal: TradingSignal
    ) -> Dict[str, Any]:
        """Check if new position would exceed VaR limits"""
        # This is simplified - in production, recalculate VaR with new position
        estimated_var_increase = 0.01  # 1% estimate
        new_var = metrics.var_95 + estimated_var_increase
        
        passed = new_var <= self.risk_limits.max_var_95
        return {
            'passed': passed,
            'current': metrics.var_95,
            'new_estimate': new_var,
            'limit': self.risk_limits.max_var_95,
            'message': f"VaR(95%): {new_var:.1%} / {self.risk_limits.max_var_95:.1%}"
        }
    
    def _check_sharpe_limit(self, metrics: PortfolioMetrics) -> Dict[str, Any]:
        """Check if portfolio Sharpe ratio is acceptable"""
        passed = metrics.sharpe_ratio >= self.risk_limits.min_sharpe_ratio
        return {
            'passed': passed,
            'current': metrics.sharpe_ratio,
            'limit': self.risk_limits.min_sharpe_ratio,
            'message': f"Sharpe: {metrics.sharpe_ratio:.2f} / {self.risk_limits.min_sharpe_ratio:.2f}"
        }
    
    def _calculate_risk_adjusted_size(
        self,
        signal: TradingSignal,
        metrics: PortfolioMetrics,
        risk_checks: Dict[str, Dict[str, Any]]
    ) -> float:
        """Calculate position size adjusted for portfolio risk"""
        base_size = self.trading_config.risk_per_trade_percent / 100
        
        # Adjust based on risk factors
        adjustments = []
        
        # Reduce size if approaching exposure limit
        if metrics.total_exposure > self.risk_limits.max_total_exposure * 0.8:
            adjustments.append(0.5)  # 50% reduction
        
        # Reduce size if high correlation
        max_corr = risk_checks.get('correlation', {}).get('max_correlation', 0)
        if max_corr > 0.5:
            adjustments.append(1 - max_corr)  # Reduce by correlation amount
        
        # Reduce size if in drawdown
        if metrics.current_drawdown > 0.1:  # 10% drawdown
            adjustments.append(0.5)  # 50% reduction
        
        # Apply adjustments
        if adjustments:
            adjustment_factor = min(adjustments)
            return base_size * adjustment_factor
        
        return base_size
    
    async def _calculate_portfolio_correlation(self, open_trades: List[Trade]) -> float:
        """Calculate average correlation between open positions"""
        if len(open_trades) < 2:
            return 0.0
        
        correlations = []
        symbols = [trade.symbol for trade in open_trades]
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                corr = await self._get_symbol_correlation(symbols[i], symbols[j])
                correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    async def _get_symbol_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols"""
        # Check cache
        cache_key = tuple(sorted([symbol1, symbol2]))
        if cache_key in self._correlation_cache and datetime.now(timezone.utc) < self._cache_expiry:
            return self._correlation_cache[cache_key]
        
        # Calculate correlation from historical data
        try:
            # This is simplified - in production, use proper data provider
            # For now, return estimated correlations
            correlation_map = {
                ('EURUSD', 'GBPUSD'): 0.7,
                ('EURUSD', 'USDCHF'): -0.8,
                ('EURUSD', 'USDJPY'): -0.3,
                ('XAUUSD', 'XAGUSD'): 0.8,
            }
            
            correlation = correlation_map.get(cache_key, 0.2)  # Default low correlation
            
            # Cache result
            self._correlation_cache[cache_key] = correlation
            self._cache_expiry = datetime.now(timezone.utc) + timedelta(hours=1)
            
            return correlation
            
        except Exception as e:
            logger.error(f"Failed to calculate correlation: {e}")
            return 0.0
    
    async def _calculate_var(self, open_trades: List[Trade], confidence: float = 0.95) -> float:
        """Calculate Value at Risk for current portfolio"""
        if not open_trades:
            return 0.0
        
        # Simplified VaR calculation
        # In production, use historical returns and proper statistical methods
        portfolio_volatility = 0.02  # 2% daily volatility estimate
        z_score = 1.645 if confidence == 0.95 else 2.33  # 95% or 99% confidence
        
        return portfolio_volatility * z_score
    
    async def _calculate_rolling_sharpe(self, days: int = 30) -> float:
        """Calculate rolling Sharpe ratio"""
        # Get historical trades
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        trades = self.trade_repository.get_trades_by_date_range(start_date, end_date)
        
        if len(trades) < 5:  # Need minimum trades
            return 0.0
        
        # Calculate daily returns
        returns = []
        for trade in trades:
            if trade.status == TradeStatus.CLOSED and trade.exit_time:
                days_held = (trade.exit_time - trade.entry_time).days or 1
                daily_return = (trade.realized_pnl / trade.entry_price) / days_held
                returns.append(daily_return)
        
        if not returns:
            return 0.0
        
        # Calculate Sharpe ratio
        returns_array = np.array(returns)
        avg_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe ratio
        return (avg_return / std_return) * np.sqrt(252)
    
    def _calculate_risk_score(
        self,
        exposure: float,
        drawdown: float,
        correlation: float,
        var: float
    ) -> float:
        """Calculate overall risk score (0-100, higher is riskier)"""
        # Weight different risk factors
        weights = {
            'exposure': 0.3,
            'drawdown': 0.3,
            'correlation': 0.2,
            'var': 0.2
        }
        
        # Normalize metrics to 0-1 scale
        exposure_score = min(exposure / self.risk_limits.max_total_exposure, 1.0)
        drawdown_score = min(drawdown / self.risk_limits.max_drawdown, 1.0)
        correlation_score = min(correlation / self.risk_limits.max_correlation, 1.0)
        var_score = min(var / self.risk_limits.max_var_95, 1.0)
        
        # Calculate weighted score
        risk_score = (
            weights['exposure'] * exposure_score +
            weights['drawdown'] * drawdown_score +
            weights['correlation'] * correlation_score +
            weights['var'] * var_score
        ) * 100
        
        return min(risk_score, 100.0)
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio risk summary"""
        metrics = await self.calculate_portfolio_metrics()
        open_trades = self.trade_repository.get_trades_by_status(TradeStatus.OPEN)
        
        return {
            'metrics': {
                'total_exposure': f"{metrics.total_exposure:.1%}",
                'current_drawdown': f"{metrics.current_drawdown:.1%}",
                'correlation_score': f"{metrics.correlation_score:.2f}",
                'var_95': f"{metrics.var_95:.1%}",
                'sharpe_ratio': f"{metrics.sharpe_ratio:.2f}",
                'open_positions': metrics.open_positions,
                'risk_score': f"{metrics.risk_score:.0f}/100"
            },
            'positions': [
                {
                    'symbol': trade.symbol,
                    'size': trade.position_size,
                    'pnl': trade.unrealized_pnl,
                    'duration': str(datetime.now(timezone.utc) - trade.entry_time)
                }
                for trade in open_trades
            ],
            'risk_limits': {
                'max_exposure': f"{self.risk_limits.max_total_exposure:.1%}",
                'max_drawdown': f"{self.risk_limits.max_drawdown:.1%}",
                'max_correlation': f"{self.risk_limits.max_correlation:.2f}",
                'max_positions': self.risk_limits.max_positions
            }
        }


# Export
__all__ = ['PortfolioRiskManager', 'PortfolioMetrics', 'RiskLimits']