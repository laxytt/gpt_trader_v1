# run_backtest.py
"""
Script to run backtests on the GPT Trading System with improved data handling
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import logging
from typing import Dict, List, Tuple

# Add project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings
from core.infrastructure.mt5.client import MT5Client
from core.infrastructure.mt5.data_provider import MT5DataProvider
from core.services.backtesting_service import (
    BacktestEngine, BacktestConfig, BacktestMode, BacktestReportGenerator
)
from core.utils.chart_utils import ChartGenerator
from core.utils.data_diagnostics import DataDiagnostics

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def check_data_availability(data_provider: MT5DataProvider, symbols: List[str]):
    """Check and report data availability for all symbols"""
    print("\n" + "="*80)
    print("DATA AVAILABILITY CHECK")
    print("="*80)
    
    diagnostics = DataDiagnostics(data_provider.unified_provider)
    report = await diagnostics.generate_availability_report(symbols)
    
    available_symbols = []
    date_ranges = {}
    
    for symbol, info in report['symbols'].items():
        if info['available']:
            available_symbols.append(symbol)
            start = datetime.fromisoformat(info['start_date'])
            end = datetime.fromisoformat(info['end_date'])
            date_ranges[symbol] = (start, end)
            
            print(f"\n{symbol}:")
            print(f"  Date Range: {start.date()} to {end.date()} ({info['total_days']} days)")
            print(f"  Timeframes:")
            for tf, tf_info in info['timeframes'].items():
                if tf_info['available']:
                    print(f"    {tf}: ✓ {tf_info['bars_available']} bars available")
                else:
                    print(f"    {tf}: ✗ {tf_info.get('error', 'Not available')}")
        else:
            print(f"\n{symbol}: ✗ {info.get('error', 'Not available')}")
    
    print("\n" + "="*80)
    
    return available_symbols, date_ranges


async def determine_backtest_period(date_ranges: Dict[str, Tuple[datetime, datetime]]):
    """Determine optimal backtest period based on available data"""
    if not date_ranges:
        return None, None
    
    # Find common date range
    latest_start = max(start for start, _ in date_ranges.values())
    earliest_end = min(end for _, end in date_ranges.values())
    
    if latest_start >= earliest_end:
        print("\nNo overlapping data period found across symbols!")
        return None, None
    
    # Calculate available period
    available_days = (earliest_end - latest_start).days
    
    print(f"\nCommon data period: {latest_start.date()} to {earliest_end.date()} ({available_days} days)")
    
    # Determine backtest period based on available data
    if available_days < 30:
        print("Warning: Less than 30 days of data available!")
        return latest_start, earliest_end
    
    # Use last 6 months or available data, whichever is smaller
    ideal_days = 180  # 6 months
    actual_days = min(ideal_days, available_days - 10)  # Leave some buffer
    
    backtest_end = earliest_end - timedelta(days=5)  # Small buffer from end
    backtest_start = backtest_end - timedelta(days=actual_days)
    
    # Ensure start is not before available data
    backtest_start = max(backtest_start, latest_start + timedelta(days=5))
    
    return backtest_start, backtest_end


async def main():
    """Run backtest with improved data handling"""
    
    # Load settings
    settings = get_settings()
    
    # Initialize MT5 client
    print("Initializing MT5 connection...")
    mt5_client = MT5Client(settings.mt5)
    if not mt5_client.initialize():
        print("Failed to initialize MT5")
        return
    
    try:
        # Create data provider
        data_provider = MT5DataProvider(mt5_client)
        
        # Check data availability
        symbols_to_test = settings.trading.symbols[:3]  # Test first 3 symbols
        available_symbols, date_ranges = await check_data_availability(
            data_provider, symbols_to_test
        )
        
        if not available_symbols:
            print("\nNo data available for any symbols!")
            return
        
        # Determine backtest period
        backtest_start, backtest_end = await determine_backtest_period(date_ranges)
        
        if not backtest_start or not backtest_end:
            print("\nCannot determine valid backtest period!")
            return
        
        # Create backtest configuration
        config = BacktestConfig(
            start_date=backtest_start,
            end_date=backtest_end,
            symbols=available_symbols,
            mode=BacktestMode.OFFLINE_ONLY,
            initial_balance=10000,
            risk_per_trade=settings.trading.risk_per_trade_percent / 100,
            max_open_trades=settings.trading.max_open_trades,
            save_results=True
        )
        
        print(f"\nBacktest Configuration:")
        print(f"  Period: {config.start_date.date()} to {config.end_date.date()}")
        print(f"  Symbols: {', '.join(config.symbols)}")
        print(f"  Mode: {config.mode.value}")
        print(f"  Initial Balance: ${config.initial_balance:,.2f}")
        print(f"  Risk per Trade: {config.risk_per_trade*100:.1f}%")
        
        # Create backtest engine
        engine = BacktestEngine(
            data_provider=data_provider,
            chart_generator=ChartGenerator() if ChartGenerator else None,
            db_path=settings.database.db_path
        )
        
        print(f"\nStarting backtest...")
        print("-" * 80)
        
        # Run backtest
        results = await engine.run_backtest(config)
        
        # Print results
        print("\n" + "="*80)
        print("BACKTEST RESULTS")
        print("="*80)
        print(f"Total Trades: {results.total_trades}")
        print(f"Winning Trades: {results.winning_trades}")
        print(f"Losing Trades: {results.losing_trades}")
        print(f"Win Rate: {results.win_rate:.1%}")
        print(f"Total P&L: ${results.total_pnl:,.2f}")
        print(f"Total Return: {results.total_return:.2f}%")
        print(f"Max Drawdown: {results.max_drawdown:.2f}%")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {results.sortino_ratio:.2f}")
        print(f"Profit Factor: {results.profit_factor:.2f}")
        print(f"Average Trade Duration: {results.average_bars_held:.1f} hours")
        
        # Performance by symbol
        if 'by_symbol' in results.statistics:
            print("\nPerformance by Symbol:")
            print("-" * 40)
            for symbol, stats in results.statistics['by_symbol'].items():
                print(f"{symbol}:")
                print(f"  Trades: {stats['trades']}")
                print(f"  Win Rate: {stats['win_rate']:.1%}")
                print(f"  P&L: ${stats['pnl']:,.2f}")
        
        # Generate detailed reports
        if results.total_trades > 0:
            print("\nGenerating reports...")
            report_generator = BacktestReportGenerator(ChartGenerator() if ChartGenerator else None)
            await report_generator.generate_report(results, "backtest_results")
            print("Reports saved to backtest_results/")
        else:
            print("\nNo trades executed - skipping report generation")
        
    except Exception as e:
        logger.exception(f"Backtest failed: {e}")
    finally:
        mt5_client.shutdown()
        print("\nMT5 connection closed")


if __name__ == "__main__":
    asyncio.run(main())