"""
Position Monitor Service for managing longer-term trades
Monitors open positions with daily/weekly context for position trading
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import asyncio

from core.domain.models import Trade, TradeStatus, SignalType, MarketData
from core.domain.exceptions import ServiceError, ErrorContext
from core.infrastructure.mt5.data_provider import MT5DataProvider
from core.infrastructure.mt5.order_manager import MT5OrderManager
from core.infrastructure.database.repositories import TradeRepository
from core.services.enhanced_news_service import EnhancedNewsService
from core.infrastructure.gpt.client import GPTClient
from config.position_trading_config import PositionTradingConfig
from core.domain.enums import TimeFrame
from core.infrastructure.notifications.telegram import TelegramNotifier

logger = logging.getLogger(__name__)


@dataclass
class PositionHealth:
    """Health status of a position"""
    symbol: str
    trade_id: str
    current_pnl: float
    current_r_multiple: float
    days_held: int
    health_score: float  # 0-100
    issues: List[str]
    recommendations: List[str]
    trailing_stop_price: Optional[float] = None
    partial_exit_suggestion: Optional[float] = None


@dataclass
class PortfolioHeat:
    """Portfolio-wide risk metrics"""
    total_risk_percent: float
    position_count: int
    correlated_positions: List[Tuple[str, str, float]]  # (symbol1, symbol2, correlation)
    max_drawdown_risk: float
    recommendations: List[str]


class PositionMonitor:
    """
    Monitors open positions for position trading strategies
    Provides daily reviews, partial profit taking, and risk management
    """
    
    def __init__(
        self,
        data_provider: MT5DataProvider,
        order_manager: MT5OrderManager,
        trade_repository: TradeRepository,
        news_service: EnhancedNewsService,
        gpt_client: GPTClient,
        account_balance: float,
        telegram_notifier: Optional[TelegramNotifier] = None
    ):
        self.data_provider = data_provider
        self.order_manager = order_manager
        self.trade_repository = trade_repository
        self.news_service = news_service
        self.gpt_client = gpt_client
        self.account_balance = account_balance
        self.telegram = telegram_notifier
        
        # Position trading configuration
        self.config = PositionTradingConfig()
        
        # Monitoring state
        self.last_review_time: Dict[str, datetime] = {}
        self.position_correlations: Dict[Tuple[str, str], float] = {}
        self._correlation_cache_time: Optional[datetime] = None
    
    async def monitor_all_positions(self) -> Dict[str, PositionHealth]:
        """
        Monitor all open positions and return their health status
        
        Returns:
            Dictionary mapping trade IDs to PositionHealth objects
        """
        with ErrorContext("Position monitoring") as ctx:
            try:
                # Get all open trades
                open_trades = self.trade_repository.get_open_trades()
                if not open_trades:
                    logger.info("No open positions to monitor")
                    return {}
                
                logger.info(f"Monitoring {len(open_trades)} open positions")
                
                # Monitor each position
                position_health = {}
                for trade in open_trades:
                    try:
                        health = await self._monitor_single_position(trade)
                        position_health[trade.id] = health
                        
                        # Send alerts for critical issues
                        if health.health_score < 40:
                            await self._send_position_alert(trade, health)
                        
                    except Exception as e:
                        logger.error(f"Error monitoring position {trade.id}: {e}")
                        ctx.add_detail(f"position_{trade.id}_error", str(e))
                
                # Calculate portfolio heat
                portfolio_heat = await self._calculate_portfolio_heat(open_trades)
                ctx.add_detail("portfolio_heat", portfolio_heat.total_risk_percent)
                
                # Log summary
                avg_health = sum(h.health_score for h in position_health.values()) / len(position_health) if position_health else 0
                logger.info(f"Position monitoring complete. Average health: {avg_health:.1f}%, Portfolio heat: {portfolio_heat.total_risk_percent:.1f}%")
                
                return position_health
                
            except Exception as e:
                logger.error(f"Position monitoring failed: {e}")
                raise ServiceError(f"Failed to monitor positions: {str(e)}")
    
    async def _monitor_single_position(self, trade: Trade) -> PositionHealth:
        """Monitor a single position"""
        try:
            # Get current market data
            market_data = await self._get_position_market_data(trade.symbol)
            if not market_data:
                raise ValueError(f"No market data available for {trade.symbol}")
            
            # Calculate position metrics
            current_price = market_data['d1'].latest_candle.close
            days_held = (datetime.now(timezone.utc) - trade.entry_time).days
            
            # Calculate PnL and R-multiple
            if trade.side == SignalType.BUY:
                pnl_pips = (current_price - trade.entry_price) / self._get_pip_value(trade.symbol)
                current_r = (current_price - trade.entry_price) / (trade.entry_price - trade.stop_loss) if trade.stop_loss else 0
            else:
                pnl_pips = (trade.entry_price - current_price) / self._get_pip_value(trade.symbol)
                current_r = (trade.entry_price - current_price) / (trade.stop_loss - trade.entry_price) if trade.stop_loss else 0
            
            pnl_amount = pnl_pips * trade.volume * self._get_contract_size(trade.symbol)
            
            # Evaluate position health
            health_score, issues, recommendations = await self._evaluate_position_health(
                trade, market_data, current_r, days_held
            )
            
            # Check for partial profit opportunities
            partial_exit = self._check_partial_profit(trade, current_r, days_held)
            
            # Calculate trailing stop
            trailing_stop = self._calculate_trailing_stop(trade, market_data, current_r)
            
            return PositionHealth(
                symbol=trade.symbol,
                trade_id=trade.id,
                current_pnl=pnl_amount,
                current_r_multiple=current_r,
                days_held=days_held,
                health_score=health_score,
                issues=issues,
                recommendations=recommendations,
                trailing_stop_price=trailing_stop,
                partial_exit_suggestion=partial_exit
            )
            
        except Exception as e:
            logger.error(f"Error monitoring position {trade.id}: {e}")
            raise
    
    async def _get_position_market_data(self, symbol: str) -> Dict[str, MarketData]:
        """Get market data for position monitoring"""
        try:
            # Get daily and weekly data
            d1_data = await self.data_provider.get_market_data(
                symbol, TimeFrame.D1, self.config.timeframes.MIN_BARS_DAILY
            )
            w1_data = await self.data_provider.get_market_data(
                symbol, TimeFrame.W1, self.config.timeframes.MIN_BARS_WEEKLY
            )
            h4_data = await self.data_provider.get_market_data(
                symbol, TimeFrame.H4, 24  # Last 4 days of H4
            )
            
            return {
                'd1': d1_data,
                'w1': w1_data,
                'h4': h4_data
            }
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return None
    
    async def _evaluate_position_health(
        self, 
        trade: Trade, 
        market_data: Dict[str, MarketData],
        current_r: float,
        days_held: int
    ) -> Tuple[float, List[str], List[str]]:
        """Evaluate the health of a position"""
        issues = []
        recommendations = []
        health_score = 100.0
        
        # Check time-based factors
        if days_held > self.config.frequency.MAX_HOLDING_DAYS:
            issues.append(f"Position held for {days_held} days (max: {self.config.frequency.MAX_HOLDING_DAYS})")
            recommendations.append("Consider closing - exceeded maximum holding period")
            health_score -= 20
        
        elif days_held > self.config.frequency.TARGET_HOLDING_DAYS and current_r < 1.0:
            issues.append(f"Position underperforming after {days_held} days")
            recommendations.append("Consider reducing position size or closing")
            health_score -= 15
        
        # Check profit/loss status
        if current_r < -0.5:
            issues.append("Position at 50% of stop loss")
            recommendations.append("Monitor closely for potential exit")
            health_score -= 25
        
        elif current_r > 2.0 and not self._has_trailing_stop(trade):
            issues.append("Profitable position without trailing stop")
            recommendations.append(f"Implement trailing stop at {1.5 * market_data['d1'].atr:.4f} ATR")
            health_score -= 10
        
        # Check trend alignment
        trend_aligned = await self._check_trend_alignment(trade, market_data)
        if not trend_aligned:
            issues.append("Position against current trend")
            recommendations.append("Consider exit on next bounce/pullback")
            health_score -= 30
        
        # Check for upcoming news
        news_risk = await self._check_news_risk(trade.symbol)
        if news_risk:
            issues.append(f"High-impact news upcoming: {news_risk}")
            recommendations.append("Consider reducing position before news")
            health_score -= 15
        
        # Check volatility
        current_volatility = market_data['d1'].atr / market_data['d1'].latest_candle.close
        normal_volatility = 0.01  # 1% daily ATR as baseline
        
        if current_volatility > normal_volatility * 2:
            issues.append("Elevated volatility detected")
            recommendations.append("Consider tightening stops or reducing position")
            health_score -= 10
        
        # Ensure health score stays in valid range
        health_score = max(0, min(100, health_score))
        
        return health_score, issues, recommendations
    
    async def _check_trend_alignment(self, trade: Trade, market_data: Dict[str, MarketData]) -> bool:
        """Check if position is aligned with current trend"""
        d1_data = market_data['d1']
        w1_data = market_data['w1']
        
        # Simple trend check using EMAs
        current_price = d1_data.latest_candle.close
        
        # Calculate EMAs (simplified - in production would use proper indicators)
        ema20 = sum(c.close for c in d1_data.candles[-20:]) / 20
        ema50 = sum(c.close for c in d1_data.candles[-50:]) / 50
        
        if trade.side == SignalType.BUY:
            return current_price > ema20 > ema50
        else:
            return current_price < ema20 < ema50
    
    async def _check_news_risk(self, symbol: str) -> Optional[str]:
        """Check for upcoming high-impact news"""
        try:
            news_context = await self.news_service.get_enhanced_news_context(
                symbol=symbol,
                lookahead_hours=48,
                lookback_hours=0
            )
            
            # Check for high-impact events
            high_impact_events = [
                event for event in news_context.upcoming_events
                if event.impact == "High"
            ]
            
            if high_impact_events:
                return high_impact_events[0].event  # Return first high-impact event
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking news risk: {e}")
            return None
    
    def _check_partial_profit(self, trade: Trade, current_r: float, days_held: int) -> Optional[float]:
        """Check if partial profits should be taken"""
        # Check against partial profit levels
        for portion, r_level in self.config.exit.PARTIAL_PROFIT_LEVELS:
            if current_r >= r_level and not self._has_partial_exit_at_level(trade, r_level):
                return portion
        
        # Time-based partial exit
        if days_held >= self.config.frequency.PARTIAL_PROFIT_DAYS and current_r > 1.0:
            return 0.5  # Take 50% after target days if profitable
        
        return None
    
    def _calculate_trailing_stop(
        self, 
        trade: Trade, 
        market_data: Dict[str, MarketData],
        current_r: float
    ) -> Optional[float]:
        """Calculate trailing stop price"""
        if current_r < self.config.exit.TRAILING_STOP_ACTIVATION:
            return None
        
        d1_data = market_data['d1']
        atr = d1_data.atr
        trailing_distance = atr * self.config.exit.TRAILING_STOP_DISTANCE
        
        current_price = d1_data.latest_candle.close
        
        if trade.side == SignalType.BUY:
            trailing_stop = current_price - trailing_distance
            # Ensure trailing stop is above entry
            return max(trailing_stop, trade.entry_price * 1.001)  # At least breakeven + spread
        else:
            trailing_stop = current_price + trailing_distance
            # Ensure trailing stop is below entry
            return min(trailing_stop, trade.entry_price * 0.999)
    
    async def _calculate_portfolio_heat(self, open_trades: List[Trade]) -> PortfolioHeat:
        """Calculate portfolio-wide risk metrics"""
        total_risk = 0
        recommendations = []
        
        # Calculate total risk
        for trade in open_trades:
            if trade.stop_loss:
                risk_amount = abs(trade.entry_price - trade.stop_loss) * trade.volume * self._get_contract_size(trade.symbol)
                risk_percent = (risk_amount / self.account_balance) * 100
                total_risk += risk_percent
        
        # Check against limits
        if total_risk > self.config.risk.MAX_PORTFOLIO_HEAT:
            recommendations.append(f"Portfolio heat ({total_risk:.1f}%) exceeds maximum ({self.config.risk.MAX_PORTFOLIO_HEAT}%)")
            recommendations.append("Avoid new positions until risk reduces")
        
        # Calculate correlations
        correlations = await self._calculate_position_correlations(open_trades)
        high_correlations = [(s1, s2, corr) for (s1, s2), corr in correlations.items() if abs(corr) > 0.7]
        
        if len(high_correlations) > 0:
            recommendations.append(f"Found {len(high_correlations)} highly correlated position pairs")
            recommendations.append("Consider diversifying or reducing correlated exposure")
        
        # Calculate max drawdown risk
        max_drawdown = sum(
            abs(trade.entry_price - trade.stop_loss) / trade.entry_price * 100
            for trade in open_trades if trade.stop_loss
        )
        
        return PortfolioHeat(
            total_risk_percent=total_risk,
            position_count=len(open_trades),
            correlated_positions=high_correlations,
            max_drawdown_risk=max_drawdown,
            recommendations=recommendations
        )
    
    async def _calculate_position_correlations(
        self, 
        open_trades: List[Trade]
    ) -> Dict[Tuple[str, str], float]:
        """Calculate correlations between open positions"""
        # Cache correlations for 1 hour
        if (self._correlation_cache_time and 
            datetime.now(timezone.utc) - self._correlation_cache_time < timedelta(hours=1)):
            return self.position_correlations
        
        correlations = {}
        symbols = list(set(trade.symbol for trade in open_trades))
        
        # Calculate pairwise correlations
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                correlation = await self._calculate_correlation(symbol1, symbol2)
                correlations[(symbol1, symbol2)] = correlation
        
        self.position_correlations = correlations
        self._correlation_cache_time = datetime.now(timezone.utc)
        
        return correlations
    
    async def _calculate_correlation(self, symbol1: str, symbol2: str) -> float:
        """Calculate correlation between two symbols"""
        try:
            # Get daily data for correlation
            data1 = await self.data_provider.get_market_data(symbol1, TimeFrame.D1, 30)
            data2 = await self.data_provider.get_market_data(symbol2, TimeFrame.D1, 30)
            
            if not data1 or not data2:
                return 0.0
            
            # Calculate returns
            returns1 = [(data1.candles[i].close - data1.candles[i-1].close) / data1.candles[i-1].close 
                       for i in range(1, len(data1.candles))]
            returns2 = [(data2.candles[i].close - data2.candles[i-1].close) / data2.candles[i-1].close 
                       for i in range(1, len(data2.candles))]
            
            # Simple correlation calculation
            if len(returns1) != len(returns2):
                return 0.0
            
            n = len(returns1)
            if n == 0:
                return 0.0
            
            mean1 = sum(returns1) / n
            mean2 = sum(returns2) / n
            
            cov = sum((returns1[i] - mean1) * (returns2[i] - mean2) for i in range(n)) / n
            std1 = (sum((r - mean1) ** 2 for r in returns1) / n) ** 0.5
            std2 = (sum((r - mean2) ** 2 for r in returns2) / n) ** 0.5
            
            if std1 == 0 or std2 == 0:
                return 0.0
            
            return cov / (std1 * std2)
            
        except Exception as e:
            logger.error(f"Error calculating correlation between {symbol1} and {symbol2}: {e}")
            return 0.0
    
    async def execute_position_adjustments(
        self, 
        position_health: Dict[str, PositionHealth]
    ) -> Dict[str, Any]:
        """Execute recommended position adjustments"""
        adjustments = {}
        
        for trade_id, health in position_health.items():
            try:
                trade = self.trade_repository.get_trade_by_id(trade_id)
                if not trade:
                    continue
                
                # Execute trailing stop adjustments
                if health.trailing_stop_price and health.trailing_stop_price != trade.stop_loss:
                    success = await self.order_manager.modify_position(
                        ticket=trade.mt5_ticket,
                        stop_loss=health.trailing_stop_price,
                        take_profit=trade.take_profit
                    )
                    if success:
                        adjustments[trade_id] = {
                            'action': 'trailing_stop_updated',
                            'new_stop': health.trailing_stop_price
                        }
                        logger.info(f"Updated trailing stop for {trade.symbol} to {health.trailing_stop_price}")
                
                # Execute partial profits
                if health.partial_exit_suggestion:
                    partial_volume = trade.volume * health.partial_exit_suggestion
                    success = await self.order_manager.close_position_partial(
                        ticket=trade.mt5_ticket,
                        volume=partial_volume
                    )
                    if success:
                        adjustments[trade_id] = {
                            'action': 'partial_profit_taken',
                            'volume': partial_volume,
                            'percentage': health.partial_exit_suggestion
                        }
                        logger.info(f"Took {health.partial_exit_suggestion*100}% partial profit on {trade.symbol}")
                
            except Exception as e:
                logger.error(f"Error executing adjustments for trade {trade_id}: {e}")
                adjustments[trade_id] = {'error': str(e)}
        
        return adjustments
    
    async def generate_daily_review(self) -> str:
        """Generate a comprehensive daily position review"""
        try:
            # Monitor all positions
            position_health = await self.monitor_all_positions()
            
            # Get portfolio heat
            open_trades = self.trade_repository.get_open_trades()
            portfolio_heat = await self._calculate_portfolio_heat(open_trades)
            
            # Build review report
            report_lines = [
                "📊 DAILY POSITION REVIEW",
                f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
                f"Account Balance: ${self.account_balance:,.2f}",
                "",
                f"📈 OPEN POSITIONS: {len(position_health)}",
                f"Portfolio Heat: {portfolio_heat.total_risk_percent:.1f}%",
                ""
            ]
            
            # Add position details
            total_pnl = 0
            for health in position_health.values():
                total_pnl += health.current_pnl
                
                status_emoji = "🟢" if health.health_score >= 70 else "🟡" if health.health_score >= 40 else "🔴"
                
                report_lines.extend([
                    f"{status_emoji} {health.symbol}",
                    f"  PnL: ${health.current_pnl:,.2f} ({health.current_r_multiple:.2f}R)",
                    f"  Days Held: {health.days_held}",
                    f"  Health Score: {health.health_score:.0f}%"
                ])
                
                if health.issues:
                    report_lines.append(f"  Issues: {', '.join(health.issues[:2])}")
                if health.recommendations:
                    report_lines.append(f"  Action: {health.recommendations[0]}")
                
                report_lines.append("")
            
            # Add summary
            report_lines.extend([
                "📊 SUMMARY",
                f"Total P&L: ${total_pnl:,.2f}",
                f"Average Health: {sum(h.health_score for h in position_health.values())/len(position_health):.0f}%" if position_health else "N/A",
                ""
            ])
            
            # Add portfolio recommendations
            if portfolio_heat.recommendations:
                report_lines.extend([
                    "⚠️ PORTFOLIO ALERTS",
                    *[f"• {rec}" for rec in portfolio_heat.recommendations],
                    ""
                ])
            
            # Add correlation warnings
            if portfolio_heat.correlated_positions:
                report_lines.append("🔗 HIGH CORRELATIONS:")
                for sym1, sym2, corr in portfolio_heat.correlated_positions[:3]:
                    report_lines.append(f"• {sym1}/{sym2}: {corr:.2f}")
            
            report = "\n".join(report_lines)
            
            # Send via Telegram if configured
            if self.telegram:
                await self.telegram.send_message(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating daily review: {e}")
            return f"Error generating daily review: {str(e)}"
    
    async def check_time_based_exits(self) -> List[str]:
        """Check for positions that should be exited based on time"""
        exits_needed = []
        
        open_trades = self.trade_repository.get_open_trades()
        
        for trade in open_trades:
            days_held = (datetime.now(timezone.utc) - trade.entry_time).days
            
            # Check maximum holding period
            if days_held > self.config.frequency.MAX_HOLDING_DAYS:
                exits_needed.append(trade.id)
                logger.warning(f"Trade {trade.id} ({trade.symbol}) exceeded max holding period: {days_held} days")
            
            # Check time stop (no progress)
            elif days_held > self.config.exit.TIME_STOP_DAYS:
                # Check if position is making progress
                market_data = await self.data_provider.get_market_data(
                    trade.symbol, TimeFrame.D1, days_held
                )
                
                if market_data:
                    # Simple progress check - position should be at least breakeven
                    current_price = market_data.latest_candle.close
                    if trade.side == SignalType.BUY:
                        progress = (current_price - trade.entry_price) / trade.entry_price
                    else:
                        progress = (trade.entry_price - current_price) / trade.entry_price
                    
                    if progress < 0.005:  # Less than 0.5% profit
                        exits_needed.append(trade.id)
                        logger.warning(f"Trade {trade.id} ({trade.symbol}) showing no progress after {days_held} days")
        
        return exits_needed
    
    async def _send_position_alert(self, trade: Trade, health: PositionHealth):
        """Send alert for critical position issues"""
        alert_message = f"""
⚠️ POSITION ALERT - {trade.symbol}

Trade ID: {trade.id}
Health Score: {health.health_score:.0f}%
Current P&L: ${health.current_pnl:,.2f} ({health.current_r_multiple:.2f}R)
Days Held: {health.days_held}

Issues:
{chr(10).join(f'• {issue}' for issue in health.issues)}

Recommendations:
{chr(10).join(f'• {rec}' for rec in health.recommendations[:2])}
"""
        
        logger.warning(f"Position alert for {trade.symbol}: {health.issues}")
        
        if self.telegram:
            await self.telegram.send_message(alert_message)
    
    def _get_pip_value(self, symbol: str) -> float:
        """Get pip value for a symbol"""
        # Simplified pip value calculation
        if "JPY" in symbol:
            return 0.01
        else:
            return 0.0001
    
    def _get_contract_size(self, symbol: str) -> float:
        """Get contract size for a symbol"""
        # Simplified - in production would get from broker
        return 100000  # Standard lot for forex
    
    def _has_trailing_stop(self, trade: Trade) -> bool:
        """Check if trade has a trailing stop"""
        # Check if stop loss has been moved from original
        return trade.stop_loss and trade.stop_loss != trade.original_stop_loss
    
    def _has_partial_exit_at_level(self, trade: Trade, r_level: float) -> bool:
        """Check if partial exit has been taken at this R level"""
        # Would check trade history/metadata in production
        return False


# Export the position monitor
__all__ = ['PositionMonitor', 'PositionHealth', 'PortfolioHeat']