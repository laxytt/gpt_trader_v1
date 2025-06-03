"""
Performance analytics and reporting for GPT Trading System.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

from config.settings import get_settings
from core.infrastructure.database.repositories import TradeRepository
from core.domain.models import Trade, TradeStatus

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_percent: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_percent: float
    avg_trade_duration: timedelta
    best_trade: Optional[Trade]
    worst_trade: Optional[Trade]
    current_streak: int
    max_win_streak: int
    max_loss_streak: int


class PerformanceAnalyzer:
    """Comprehensive performance analysis and reporting"""
    
    def __init__(self):
        self.settings = get_settings()
        self.trade_repo = TradeRepository(self.settings.database.db_path)
        self.reports_dir = Path("reports/performance")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for charts
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    async def generate_daily_report(self) -> Dict[str, Any]:
        """Generate comprehensive daily performance report"""
        logger.info("Generating daily performance report")
        
        try:
            # Get today's trades
            end_date = datetime.now(timezone.utc)
            start_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
            
            trades = self.trade_repo.get_trades_by_date_range(start_date, end_date)
            
            # Calculate metrics
            metrics = await self._calculate_metrics(trades)
            
            # Generate report sections
            report = {
                'date': start_date.date().isoformat(),
                'summary': self._generate_summary(metrics),
                'trades': self._analyze_trades(trades),
                'risk_analysis': await self._analyze_risk(trades),
                'symbol_breakdown': self._analyze_by_symbol(trades),
                'recommendations': await self._generate_recommendations(metrics, trades)
            }
            
            # Save report
            report_path = await self._save_report(report, 'daily')
            
            # Generate charts
            charts_path = await self._generate_charts(trades, 'daily')
            
            # Send notifications if configured
            await self._send_daily_summary(report)
            
            return {
                'success': True,
                'report_path': str(report_path),
                'charts_path': str(charts_path),
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Daily report generation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def generate_weekly_report(self) -> Dict[str, Any]:
        """Generate weekly performance report"""
        logger.info("Generating weekly performance report")
        
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=7)
        
        trades = self.trade_repo.get_trades_by_date_range(start_date, end_date)
        
        # Weekly specific analysis
        report = {
            'week_ending': end_date.date().isoformat(),
            'daily_breakdown': await self._analyze_daily_performance(trades),
            'pattern_analysis': self._analyze_patterns(trades),
            'ml_performance': await self._analyze_ml_performance(trades),
            'optimization_suggestions': self._suggest_optimizations(trades)
        }
        
        report_path = await self._save_report(report, 'weekly')
        
        return {
            'success': True,
            'report_path': str(report_path),
            'trade_count': len(trades)
        }
    
    async def compare_backtest_vs_live(
        self, 
        backtest_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare backtest results with live trading performance"""
        logger.info("Comparing backtest vs live performance")
        
        # Get live trades for same period
        live_trades = self.trade_repo.get_trades_by_date_range(
            backtest_results.get('start_date'),
            backtest_results.get('end_date')
        )
        
        live_metrics = await self._calculate_metrics(live_trades)
        
        # Compare key metrics
        comparison = {
            'win_rate_diff': live_metrics.win_rate - backtest_results.get('win_rate', 0),
            'pnl_diff': live_metrics.total_pnl - backtest_results.get('total_pnl', 0),
            'sharpe_diff': live_metrics.sharpe_ratio - backtest_results.get('sharpe_ratio', 0),
            'drawdown_diff': live_metrics.max_drawdown_percent - backtest_results.get('max_drawdown', 0),
            'trade_count_diff': live_metrics.total_trades - backtest_results.get('total_trades', 0)
        }
        
        # Determine if optimization is needed
        optimization_needed = (
            comparison['win_rate_diff'] < -0.1 or  # 10% worse win rate
            comparison['sharpe_diff'] < -0.5 or    # Significantly worse risk-adjusted returns
            comparison['drawdown_diff'] > 10       # 10% higher drawdown
        )
        
        return {
            'live_metrics': live_metrics,
            'backtest_metrics': backtest_results,
            'comparison': comparison,
            'optimization_needed': optimization_needed,
            'recommendations': self._generate_comparison_recommendations(comparison)
        }
    
    async def _calculate_metrics(self, trades: List[Trade]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        if not trades:
            return self._empty_metrics()
        
        # Basic counts
        winning_trades = [t for t in trades if t.result == 'WIN']
        losing_trades = [t for t in trades if t.result == 'LOSS']
        
        # PnL calculations
        pnls = [t.pnl_usd for t in trades if t.pnl_usd is not None]
        total_pnl = sum(pnls)
        
        # Calculate initial balance (approximate)
        initial_balance = 10000  # Could be from settings or calculated
        total_pnl_percent = (total_pnl / initial_balance) * 100
        
        # Win/Loss metrics
        wins = [t.pnl_usd for t in winning_trades if t.pnl_usd is not None]
        losses = [abs(t.pnl_usd) for t in losing_trades if t.pnl_usd is not None]
        
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        profit_factor = sum(wins) / sum(losses) if losses else float('inf')
        
        # Risk-adjusted returns
        sharpe_ratio = self._calculate_sharpe_ratio(pnls)
        sortino_ratio = self._calculate_sortino_ratio(pnls)
        
        # Drawdown
        max_dd, max_dd_percent = self._calculate_max_drawdown(trades)
        
        # Duration
        durations = []
        for trade in trades:
            if trade.exit_timestamp and trade.timestamp:
                durations.append(trade.exit_timestamp - trade.timestamp)
        
        avg_duration = sum(durations, timedelta()) / len(durations) if durations else timedelta()
        
        # Best/Worst trades
        sorted_trades = sorted(trades, key=lambda t: t.pnl_usd or 0)
        worst_trade = sorted_trades[0] if sorted_trades else None
        best_trade = sorted_trades[-1] if sorted_trades else None
        
        # Streaks
        current_streak, max_win_streak, max_loss_streak = self._calculate_streaks(trades)
        
        return PerformanceMetrics(
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=len(winning_trades) / len(trades) if trades else 0,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_dd,
            max_drawdown_percent=max_dd_percent,
            avg_trade_duration=avg_duration,
            best_trade=best_trade,
            worst_trade=worst_trade,
            current_streak=current_streak,
            max_win_streak=max_win_streak,
            max_loss_streak=max_loss_streak
        )
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0
        
        df = pd.DataFrame({'returns': returns})
        # Assume daily returns, annualize with 252 trading days
        return (df['returns'].mean() / df['returns'].std()) * (252 ** 0.5) if df['returns'].std() > 0 else 0
    
    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if len(returns) < 2:
            return 0
        
        df = pd.DataFrame({'returns': returns})
        downside = df[df['returns'] < 0]['returns']
        
        if len(downside) > 0 and downside.std() > 0:
            return (df['returns'].mean() / downside.std()) * (252 ** 0.5)
        return 0
    
    def _calculate_max_drawdown(self, trades: List[Trade]) -> tuple:
        """Calculate maximum drawdown"""
        if not trades:
            return 0, 0
        
        # Sort trades by timestamp
        sorted_trades = sorted(trades, key=lambda t: t.timestamp)
        
        # Calculate cumulative PnL
        cumulative = 0
        peak = 0
        max_dd = 0
        max_dd_percent = 0
        
        for trade in sorted_trades:
            if trade.pnl_usd:
                cumulative += trade.pnl_usd
                if cumulative > peak:
                    peak = cumulative
                
                drawdown = peak - cumulative
                if drawdown > max_dd:
                    max_dd = drawdown
                    if peak > 0:
                        max_dd_percent = (drawdown / peak) * 100
        
        return max_dd, max_dd_percent
    
    def _calculate_streaks(self, trades: List[Trade]) -> tuple:
        """Calculate win/loss streaks"""
        if not trades:
            return 0, 0, 0
        
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for trade in sorted(trades, key=lambda t: t.timestamp):
            if trade.result == 'WIN':
                if current_streak >= 0:
                    current_streak += 1
                    max_win_streak = max(max_win_streak, current_streak)
                else:
                    current_streak = 1
            elif trade.result == 'LOSS':
                if current_streak <= 0:
                    current_streak -= 1
                    max_loss_streak = max(max_loss_streak, abs(current_streak))
                else:
                    current_streak = -1
        
        return current_streak, max_win_streak, max_loss_streak
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics when no trades"""
        return PerformanceMetrics(
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
            total_pnl=0, total_pnl_percent=0, avg_win=0, avg_loss=0,
            profit_factor=0, sharpe_ratio=0, sortino_ratio=0,
            max_drawdown=0, max_drawdown_percent=0,
            avg_trade_duration=timedelta(), best_trade=None, worst_trade=None,
            current_streak=0, max_win_streak=0, max_loss_streak=0
        )
    
    def _generate_summary(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Generate performance summary"""
        return {
            'total_trades': metrics.total_trades,
            'win_rate': f"{metrics.win_rate:.2%}",
            'total_pnl': f"${metrics.total_pnl:.2f}",
            'profit_factor': f"{metrics.profit_factor:.2f}",
            'sharpe_ratio': f"{metrics.sharpe_ratio:.2f}",
            'max_drawdown': f"{metrics.max_drawdown_percent:.2%}",
            'avg_trade_duration': str(metrics.avg_trade_duration),
            'current_streak': metrics.current_streak
        }
    
    def _analyze_trades(self, trades: List[Trade]) -> Dict[str, Any]:
        """Analyze individual trades"""
        if not trades:
            return {'message': 'No trades today'}
        
        return {
            'count': len(trades),
            'symbols_traded': list(set(t.symbol for t in trades)),
            'best_trade': {
                'symbol': trades[0].symbol,
                'pnl': trades[0].pnl_usd,
                'percentage': trades[0].pnl_percent
            } if trades else None,
            'average_risk_reward': sum(t.risk_reward_ratio for t in trades) / len(trades),
            'trade_distribution': {
                'buy': len([t for t in trades if t.side == 'BUY']),
                'sell': len([t for t in trades if t.side == 'SELL'])
            }
        }
    
    async def _analyze_risk(self, trades: List[Trade]) -> Dict[str, Any]:
        """Analyze risk metrics"""
        if not trades:
            return {'risk_level': 'N/A'}
        
        # Calculate various risk metrics
        risk_amounts = [t.risk_amount_usd for t in trades if t.risk_amount_usd]
        
        return {
            'avg_risk_per_trade': sum(risk_amounts) / len(risk_amounts) if risk_amounts else 0,
            'max_risk_taken': max(risk_amounts) if risk_amounts else 0,
            'risk_reward_achieved': sum(t.risk_reward_ratio for t in trades) / len(trades),
            'stop_loss_hit_rate': len([t for t in trades if t.result == 'LOSS' and t.exit_price == t.stop_loss]) / len(trades) if trades else 0
        }
    
    def _analyze_by_symbol(self, trades: List[Trade]) -> Dict[str, Any]:
        """Analyze performance by symbol"""
        symbol_performance = {}
        
        for trade in trades:
            if trade.symbol not in symbol_performance:
                symbol_performance[trade.symbol] = {
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'pnl': 0
                }
            
            symbol_performance[trade.symbol]['trades'] += 1
            if trade.result == 'WIN':
                symbol_performance[trade.symbol]['wins'] += 1
            elif trade.result == 'LOSS':
                symbol_performance[trade.symbol]['losses'] += 1
            
            if trade.pnl_usd:
                symbol_performance[trade.symbol]['pnl'] += trade.pnl_usd
        
        # Calculate win rates
        for symbol, data in symbol_performance.items():
            data['win_rate'] = data['wins'] / data['trades'] if data['trades'] > 0 else 0
        
        return symbol_performance
    
    async def _generate_recommendations(
        self, 
        metrics: PerformanceMetrics,
        trades: List[Trade]
    ) -> List[str]:
        """Generate trading recommendations based on performance"""
        recommendations = []
        
        # Win rate recommendations
        if metrics.win_rate < 0.4:
            recommendations.append("Consider tightening entry criteria - win rate below 40%")
        elif metrics.win_rate > 0.7:
            recommendations.append("Excellent win rate - consider increasing position sizes")
        
        # Risk recommendations
        if metrics.max_drawdown_percent > 20:
            recommendations.append("High drawdown detected - review risk management")
        
        # Streak recommendations
        if metrics.current_streak <= -3:
            recommendations.append("Currently in losing streak - consider reducing risk")
        elif metrics.current_streak >= 5:
            recommendations.append("Strong winning streak - stay disciplined")
        
        # Symbol-specific recommendations
        symbol_perf = self._analyze_by_symbol(trades)
        for symbol, data in symbol_perf.items():
            if data['win_rate'] < 0.3 and data['trades'] > 5:
                recommendations.append(f"Consider removing {symbol} - poor performance")
        
        return recommendations
    
    async def _generate_charts(
        self, 
        trades: List[Trade],
        report_type: str
    ) -> Path:
        """Generate performance charts"""
        if not trades:
            return self.reports_dir / "no_trades.png"
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{report_type.capitalize()} Performance Report', fontsize=16)
        
        # 1. Cumulative PnL
        cumulative_pnl = []
        current = 0
        for trade in sorted(trades, key=lambda t: t.timestamp):
            if trade.pnl_usd:
                current += trade.pnl_usd
                cumulative_pnl.append(current)
        
        ax1.plot(cumulative_pnl, 'g-', linewidth=2)
        ax1.fill_between(range(len(cumulative_pnl)), cumulative_pnl, alpha=0.3)
        ax1.set_title('Cumulative PnL')
        ax1.set_xlabel('Trade Number')
        ax1.set_ylabel('PnL ($)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Win/Loss Distribution
        wins = [t.pnl_usd for t in trades if t.result == 'WIN' and t.pnl_usd]
        losses = [abs(t.pnl_usd) for t in trades if t.result == 'LOSS' and t.pnl_usd]
        
        ax2.hist([wins, losses], label=['Wins', 'Losses'], bins=20, alpha=0.7)
        ax2.set_title('Win/Loss Distribution')
        ax2.set_xlabel('Trade Size ($)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Symbol Performance
        symbol_perf = self._analyze_by_symbol(trades)
        symbols = list(symbol_perf.keys())
        pnls = [data['pnl'] for data in symbol_perf.values()]
        
        ax3.bar(symbols, pnls, color=['g' if p > 0 else 'r' for p in pnls])
        ax3.set_title('Performance by Symbol')
        ax3.set_xlabel('Symbol')
        ax3.set_ylabel('PnL ($)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Daily PnL (if multiple days)
        daily_pnl = {}
        for trade in trades:
            date = trade.timestamp.date()
            if date not in daily_pnl:
                daily_pnl[date] = 0
            if trade.pnl_usd:
                daily_pnl[date] += trade.pnl_usd
        
        if len(daily_pnl) > 1:
            dates = sorted(daily_pnl.keys())
            values = [daily_pnl[d] for d in dates]
            ax4.plot(dates, values, 'bo-')
            ax4.set_title('Daily PnL')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('PnL ($)')
            ax4.tick_params(axis='x', rotation=45)
        else:
            ax4.text(0.5, 0.5, 'Single day data', ha='center', va='center')
            ax4.set_title('Daily PnL')
        
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_path = self.reports_dir / f"{report_type}_charts_{timestamp}.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    async def _save_report(
        self, 
        report: Dict[str, Any],
        report_type: str
    ) -> Path:
        """Save report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"{report_type}_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report_path
    
    async def _send_daily_summary(self, report: Dict[str, Any]):
        """Send daily summary via configured channels"""
        if not self.settings.telegram.enabled:
            return
        
        try:
            from core.infrastructure.notifications.telegram import TelegramNotifier
            
            notifier = TelegramNotifier(
                token=self.settings.telegram.token,
                chat_id=self.settings.telegram.chat_id
            )
            
            # Format message
            summary = report['summary']
            message = f"""
📊 *Daily Trading Report*

📅 Date: {report['date']}
📈 Total Trades: {summary['total_trades']}
🎯 Win Rate: {summary['win_rate']}
💰 Total PnL: {summary['total_pnl']}
📉 Max Drawdown: {summary['max_drawdown']}
⚡ Current Streak: {summary['current_streak']}

🔍 Recommendations:
"""
            for rec in report['recommendations'][:3]:  # Top 3 recommendations
                message += f"• {rec}\n"
            
            await notifier.send_message(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Failed to send daily summary: {e}")
    
    def _generate_comparison_recommendations(
        self, 
        comparison: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations from backtest comparison"""
        recommendations = []
        
        if comparison['win_rate_diff'] < -0.1:
            recommendations.append("Live win rate significantly lower - review entry criteria")
        
        if comparison['sharpe_diff'] < -0.5:
            recommendations.append("Risk-adjusted returns worse in live - check risk management")
        
        if comparison['drawdown_diff'] > 10:
            recommendations.append("Higher drawdown in live trading - consider reducing position sizes")
        
        if comparison['trade_count_diff'] < -10:
            recommendations.append("Fewer trades in live - check if all signals are being executed")
        
        return recommendations
    
    async def _analyze_daily_performance(
        self, 
        trades: List[Trade]
    ) -> Dict[str, Any]:
        """Analyze performance by day"""
        daily_metrics = {}
        
        for trade in trades:
            date = trade.timestamp.date()
            if date not in daily_metrics:
                daily_metrics[date] = {
                    'trades': [],
                    'pnl': 0,
                    'wins': 0,
                    'losses': 0
                }
            
            daily_metrics[date]['trades'].append(trade)
            if trade.pnl_usd:
                daily_metrics[date]['pnl'] += trade.pnl_usd
            
            if trade.result == 'WIN':
                daily_metrics[date]['wins'] += 1
            elif trade.result == 'LOSS':
                daily_metrics[date]['losses'] += 1
        
        # Calculate daily statistics
        for date, data in daily_metrics.items():
            data['trade_count'] = len(data['trades'])
            data['win_rate'] = data['wins'] / data['trade_count'] if data['trade_count'] > 0 else 0
        
        return daily_metrics
    
    def _analyze_patterns(self, trades: List[Trade]) -> Dict[str, Any]:
        """Analyze trading patterns"""
        # Time-based patterns
        hour_distribution = {}
        day_distribution = {}
        
        for trade in trades:
            hour = trade.timestamp.hour
            day = trade.timestamp.strftime('%A')
            
            hour_distribution[hour] = hour_distribution.get(hour, 0) + 1
            day_distribution[day] = day_distribution.get(day, 0) + 1
        
        return {
            'hour_distribution': hour_distribution,
            'day_distribution': day_distribution,
            'most_active_hour': max(hour_distribution.items(), key=lambda x: x[1])[0] if hour_distribution else None,
            'most_active_day': max(day_distribution.items(), key=lambda x: x[1])[0] if day_distribution else None
        }
    
    async def _analyze_ml_performance(self, trades: List[Trade]) -> Dict[str, Any]:
        """Analyze ML model performance if applicable"""
        # This would analyze trades that used ML signals
        ml_trades = [t for t in trades if hasattr(t, 'used_ml') and t.used_ml]
        non_ml_trades = [t for t in trades if not hasattr(t, 'used_ml') or not t.used_ml]
        
        if not ml_trades:
            return {'message': 'No ML-based trades found'}
        
        ml_metrics = await self._calculate_metrics(ml_trades)
        non_ml_metrics = await self._calculate_metrics(non_ml_trades)
        
        return {
            'ml_trades': len(ml_trades),
            'ml_win_rate': ml_metrics.win_rate,
            'ml_pnl': ml_metrics.total_pnl,
            'non_ml_win_rate': non_ml_metrics.win_rate,
            'non_ml_pnl': non_ml_metrics.total_pnl,
            'ml_advantage': ml_metrics.win_rate - non_ml_metrics.win_rate
        }
    
    def _suggest_optimizations(self, trades: List[Trade]) -> List[str]:
        """Suggest strategy optimizations based on trade analysis"""
        suggestions = []
        
        # Analyze losing trades
        losing_trades = [t for t in trades if t.result == 'LOSS']
        if losing_trades:
            # Check if losses are concentrated in specific conditions
            loss_by_hour = {}
            for trade in losing_trades:
                hour = trade.timestamp.hour
                loss_by_hour[hour] = loss_by_hour.get(hour, 0) + 1
            
            worst_hour = max(loss_by_hour.items(), key=lambda x: x[1])[0]
            if loss_by_hour[worst_hour] > len(losing_trades) * 0.3:
                suggestions.append(f"Consider avoiding trades around {worst_hour}:00 UTC - high loss concentration")
        
        # Check trade duration
        durations = []
        for trade in trades:
            if trade.exit_timestamp:
                durations.append((trade.exit_timestamp - trade.timestamp).total_seconds() / 3600)
        
        if durations:
            avg_duration = sum(durations) / len(durations)
            if avg_duration < 1:
                suggestions.append("Very short trade durations - consider longer timeframe analysis")
            elif avg_duration > 24:
                suggestions.append("Long trade durations - consider tighter stop losses")
        
        return suggestions


async def main():
    """Run performance analytics standalone"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Performance Analytics')
    parser.add_argument('--report', choices=['daily', 'weekly', 'custom'],
                      default='daily', help='Report type')
    parser.add_argument('--days', type=int, default=1,
                      help='Number of days for custom report')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    
    analyzer = PerformanceAnalyzer()
    
    if args.report == 'daily':
        result = await analyzer.generate_daily_report()
    elif args.report == 'weekly':
        result = await analyzer.generate_weekly_report()
    else:
        # Custom date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=args.days)
        trades = analyzer.trade_repo.get_trades_by_date_range(start_date, end_date)
        metrics = await analyzer._calculate_metrics(trades)
        
        print(f"Performance for last {args.days} days:")
        print(f"  Total trades: {metrics.total_trades}")
        print(f"  Win rate: {metrics.win_rate:.2%}")
        print(f"  Total PnL: ${metrics.total_pnl:.2f}")
        print(f"  Sharpe ratio: {metrics.sharpe_ratio:.2f}")
        return
    
    if result['success']:
        print(f"✅ Report generated successfully!")
        print(f"   Report: {result['report_path']}")
        if 'charts_path' in result:
            print(f"   Charts: {result['charts_path']}")
    else:
        print(f"❌ Report generation failed: {result['error']}")


if __name__ == "__main__":
    asyncio.run(main())