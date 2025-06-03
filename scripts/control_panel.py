#!/usr/bin/env python3
"""
Control panel for GPT Trading System.
Provides a command-line interface to manage all system components.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import modules individually to avoid circular dependencies


class TradingSystemControl:
    """Central control panel for trading system management"""
    
    def __init__(self):
        # Lazy initialization to avoid import issues
        self._health_checker = None
        self._backup_tool = None
        self._analyzer = None
        self._news_updater = None
        self._ml_updater = None
        self._backtest_runner = None
    
    @property
    def health_checker(self):
        if self._health_checker is None:
            from scripts.automation.health_check import HealthChecker
            self._health_checker = HealthChecker()
        return self._health_checker
    
    @property
    def backup_tool(self):
        if self._backup_tool is None:
            from scripts.automation.database_backup import DatabaseBackup
            self._backup_tool = DatabaseBackup()
        return self._backup_tool
    
    @property
    def analyzer(self):
        if self._analyzer is None:
            from scripts.automation.performance_analytics import PerformanceAnalyzer
            self._analyzer = PerformanceAnalyzer()
        return self._analyzer
    
    @property
    def news_updater(self):
        if self._news_updater is None:
            from scripts.automation.news_updater import NewsUpdater
            self._news_updater = NewsUpdater()
        return self._news_updater
    
    @property
    def ml_updater(self):
        if self._ml_updater is None:
            from scripts.automation.ml_updater import MLUpdater
            self._ml_updater = MLUpdater()
        return self._ml_updater
    
    @property
    def backtest_runner(self):
        if self._backtest_runner is None:
            from scripts.automation.backtest_runner import BacktestRunner
            self._backtest_runner = BacktestRunner()
        return self._backtest_runner
    
    async def status(self):
        """Show system status"""
        print("\n" + "="*60)
        print("GPT TRADING SYSTEM STATUS")
        print("="*60)
        
        # Check health
        health = await self.health_checker.check_all()
        
        print(f"\n{'Status:':<20} {'✅ HEALTHY' if health['healthy'] else '❌ UNHEALTHY'}")
        print(f"{'Timestamp:':<20} {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        print("\nComponent Status:")
        print("-" * 40)
        for component, status in health['checks'].items():
            icon = '✅' if status['healthy'] else '❌'
            print(f"{icon} {component:<20} {status['message']}")
        
        if health['issues']:
            print("\n⚠️  Issues Detected:")
            for issue in health['issues']:
                print(f"   - {issue}")
        
        # Check news data age
        news_age = await self.news_updater.check_news_age()
        print(f"\n{'News Data:':<20} {news_age['age_hours']:.1f} hours old")
        
        # Recent performance
        try:
            end_date = datetime.now(timezone.utc)
            start_date = end_date.replace(hour=0, minute=0, second=0)
            trades = await self.analyzer.trade_repo.get_trades_by_date_range(start_date, end_date)
            metrics = await self.analyzer._calculate_metrics(trades)
            
            print(f"\nToday's Performance:")
            print(f"{'Trades:':<20} {metrics.total_trades}")
            print(f"{'Win Rate:':<20} {metrics.win_rate:.1%}")
            print(f"{'PnL:':<20} ${metrics.total_pnl:.2f}")
        except:
            print(f"\nToday's Performance: No data available")
        
        print("\n" + "="*60)
    
    async def backup(self, backup_type: str = "manual"):
        """Create database backup"""
        print(f"\n🔄 Creating {backup_type} backup...")
        result = await self.backup_tool.create_backup(backup_type)
        
        if result['success']:
            print(f"✅ Backup created successfully!")
            print(f"   Path: {result['backup_path']}")
            print(f"   Size: {result['size_mb']}MB")
        else:
            print(f"❌ Backup failed: {result['error']}")
    
    async def report(self, report_type: str = "daily"):
        """Generate performance report"""
        print(f"\n📊 Generating {report_type} report...")
        
        if report_type == "daily":
            result = await self.analyzer.generate_daily_report()
        elif report_type == "weekly":
            result = await self.analyzer.generate_weekly_report()
        else:
            print("❌ Invalid report type. Use 'daily' or 'weekly'")
            return
        
        if result['success']:
            print(f"✅ Report generated successfully!")
            print(f"   Report: {result['report_path']}")
            if 'charts_path' in result:
                print(f"   Charts: {result['charts_path']}")
        else:
            print(f"❌ Report generation failed: {result['error']}")
    
    async def update_news(self):
        """Update news data"""
        print("\n📰 Updating economic calendar...")
        result = await self.news_updater.update_news_data()
        
        if result['success']:
            print(f"✅ News updated successfully!")
            print(f"   Events: {result['news_count']}")
            
            # Show upcoming high impact
            upcoming = await self.news_updater.get_upcoming_high_impact(24)
            if upcoming:
                print(f"\n🔴 High impact events in next 24h:")
                for event in upcoming[:5]:  # Show top 5
                    print(f"   • {event['event']} ({event['currency']}) - {event['hours_until']:.1f}h")
        else:
            print(f"❌ News update failed: {result['error']}")
    
    async def update_ml(self):
        """Update ML models"""
        print("\n🤖 Updating ML models...")
        result = await self.ml_updater.daily_update()
        
        if result['success']:
            if result.get('model_deployed'):
                print(f"✅ New model deployed!")
                print(f"   Model ID: {result['model_id']}")
                print(f"   Improvement: {result['improvement']:.2f}%")
            else:
                print(f"✅ ML update completed - current model kept")
                print(f"   Reason: {result.get('reason')}")
        else:
            print(f"❌ ML update failed: {result.get('error', result.get('reason'))}")
    
    async def backtest(self, days: int = 90):
        """Run backtest"""
        print(f"\n🔬 Running backtest for last {days} days...")
        
        # Configure backtest
        from datetime import timedelta
        end_date = datetime.now(timezone.utc)
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        from core.services.backtesting_service import BacktestConfig, BacktestMode, BacktestEngine
        from core.infrastructure.mt5.data_provider import MT5DataProvider
        from core.infrastructure.mt5.client import MT5Client
        from config.settings import get_settings
        
        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            symbols=['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],  # Test multiple pairs
            mode=BacktestMode.FULL,
            initial_balance=10000.0,
            risk_per_trade=0.01,  # 1% risk per trade
            max_open_trades=3,
            commission_per_lot=7.0,
            slippage_points=2,
            use_spread_filter=True,
            use_news_filter=False  # Disable for backtest
        )
        
        # Initialize required components
        settings = get_settings()
        mt5_client = MT5Client(settings.mt5)
        data_provider = MT5DataProvider(mt5_client)
        
        # Create BacktestEngine with proper parameters
        engine = BacktestEngine(
            data_provider=data_provider,
            db_path=settings.database.db_path
        )
        
        print("Running backtest...")
        results = await engine.run_backtest(config)
        
        print(f"\n✅ Backtest completed!")
        print(f"   Total trades: {results.total_trades}")
        print(f"   Win rate: {results.win_rate:.1%}")
        print(f"   Total PnL: ${results.total_pnl:.2f}")
        print(f"   Sharpe ratio: {results.sharpe_ratio:.2f}")
        print(f"   Max drawdown: {results.max_drawdown:.1f}%")
        
        # Debug info
        if results.max_drawdown > 100:
            print(f"\n⚠️  WARNING: Drawdown exceeds 100% - this indicates a calculation error!")
            print(f"   Equity curve length: {len(results.equity_curve)}")
            if results.equity_curve:
                print(f"   First value: ${results.equity_curve[0]:.2f}")
                print(f"   Last value: ${results.equity_curve[-1]:.2f}")
                print(f"   Min value: ${min(results.equity_curve):.2f}")
                print(f"   Max value: ${max(results.equity_curve):.2f}")
    
    async def emergency_stop(self):
        """Emergency stop all trading"""
        print("\n🚨 EMERGENCY STOP INITIATED")
        
        # This would send signals to stop trading
        # For now, just show what would happen
        print("   • Closing all open positions")
        print("   • Cancelling all pending orders")
        print("   • Disabling automated trading")
        print("   • Sending notifications")
        
        print("\n⚠️  To actually implement emergency stop:")
        print("   1. Stop the trading_loop.py process")
        print("   2. Close positions manually in MT5")
        print("   3. Review logs for issues")


async def main():
    """Main control panel interface"""
    parser = argparse.ArgumentParser(
        description='GPT Trading System Control Panel',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  status        Show system status and health
  backup        Create database backup
  report        Generate performance report
  news          Update economic calendar
  ml            Update ML models
  backtest      Run quick backtest
  emergency     Emergency stop all trading

Examples:
  python control_panel.py status
  python control_panel.py backup --type weekly
  python control_panel.py report --type daily
  python control_panel.py backtest --days 30
        """
    )
    
    parser.add_argument('command', choices=[
        'status', 'backup', 'report', 'news', 'ml', 'backtest', 'emergency'
    ], help='Command to execute')
    
    parser.add_argument('--type', help='Type for backup/report commands')
    parser.add_argument('--days', type=int, default=90, help='Days for backtest')
    
    args = parser.parse_args()
    
    control = TradingSystemControl()
    
    try:
        if args.command == 'status':
            await control.status()
        elif args.command == 'backup':
            await control.backup(args.type or 'manual')
        elif args.command == 'report':
            await control.report(args.type or 'daily')
        elif args.command == 'news':
            await control.update_news()
        elif args.command == 'ml':
            await control.update_ml()
        elif args.command == 'backtest':
            await control.backtest(args.days)
        elif args.command == 'emergency':
            await control.emergency_stop()
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Configure logging
    import logging
    logging.basicConfig(
        level=logging.WARNING,  # Only show warnings and errors
        format='%(asctime)s | %(levelname)s | %(message)s'
    )
    
    asyncio.run(main())