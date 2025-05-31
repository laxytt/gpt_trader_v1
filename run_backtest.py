"""
Script to run backtests on the GPT Trading System
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings
from core.infrastructure.mt5.client import MT5Client
from core.infrastructure.mt5.data_provider import MT5DataProvider
from core.services.backtesting_service import (
    BacktestEngine, BacktestConfig, BacktestMode, BacktestReportGenerator, BacktestSignalGenerator
)
from core.utils.chart_utils import ChartGenerator

# In run_backtest.py, update the main function

async def run_full_mode_backtest(settings, mt5_client, data_provider):
    """Run backtest in FULL mode with GPT integration"""
    
    # Initialize GPT components
    from core.infrastructure.gpt.client import GPTClient
    from core.infrastructure.gpt.signal_generator import GPTSignalGenerator
    from core.services.signal_service import SignalService
    from core.services.news_service import NewsService
    from core.services.memory_service import MemoryService
    from core.infrastructure.database.repositories import (
        MemoryCaseRepository, SignalRepository
    )
    
    print("Initializing FULL mode components...")
    
    # Create all required services
    gpt_client = GPTClient(settings.gpt)
    signal_generator = GPTSignalGenerator(gpt_client)
    news_service = NewsService(settings.news)
    memory_case_repo = MemoryCaseRepository(settings.database.db_path)
    memory_service = MemoryService(memory_case_repo, settings.database)
    signal_repository = SignalRepository(settings.database.db_path)
    
    # Create signal service
    signal_service = SignalService(
        data_provider=data_provider,
        signal_generator=signal_generator,
        news_service=news_service,
        memory_service=memory_service,
        signal_repository=signal_repository,
        trading_config=settings.trading,
        enable_offline_validation=True
    )
    
    # Configure backtest for FULL mode
    config = BacktestConfig(
        start_date=datetime(2025, 5, 20, tzinfo=timezone.utc),  # Use recent dates
        end_date=datetime(2025, 5, 27, tzinfo=timezone.utc),    # One week
        symbols=["EURUSD"],
        mode=BacktestMode.FULL,
        initial_balance=10000,
        risk_per_trade=0.015,
        max_open_trades=2
    )
    
    # Create backtest signal generator with data provider
    backtest_signal_gen = BacktestSignalGenerator(
        signal_service=signal_service,
        mode=BacktestMode.FULL,
        data_provider=data_provider  # Pass the data provider
    )
    
    # Create and run engine
    engine = BacktestEngine(
        data_provider=data_provider,
        signal_generator=backtest_signal_gen,
        chart_generator=ChartGenerator() if ChartGenerator else None,
        db_path=settings.database.db_path
    )
    
    print(f"Starting FULL mode backtest (with GPT) from {config.start_date.date()} to {config.end_date.date()}")
    print("WARNING: This will make real GPT API calls and cost money!")
    print("Press Ctrl+C to cancel...")
    
    try:
        await asyncio.sleep(3)  # Give user time to cancel
    except KeyboardInterrupt:
        print("Cancelled by user")
        return None
    
    results = await engine.run_backtest(config)
    return results


async def main():
    """Run backtest with current configuration"""
    
        # Load settings
    settings = get_settings()
    
    # Initialize MT5 client
    mt5_client = MT5Client(settings.mt5)
    if not mt5_client.initialize():
        print("Failed to initialize MT5")
        return
    
    # Add MT5 connection check
    account_info = mt5_client.get_account_info()
    if account_info:
        print(f"MT5 Connected - Account: {account_info.get('login', 'Unknown')}")
    else:
        print("MT5 connected but cannot get account info")
    
    # Check if symbol exists
    symbol_info = mt5_client.get_symbol_info("EURUSD")
    if symbol_info:
        print(f"Symbol EURUSD found: {symbol_info.get('description', 'No description')}")
    else:
        print("Symbol EURUSD not found in MT5")
    
    try:
        # Create data provider
        data_provider = MT5DataProvider(mt5_client)
        
        # Test data retrieval directly
        from core.domain.enums import TimeFrame
        print("Testing direct data retrieval...")
        test_data = await data_provider.get_market_data(
            symbol="EURUSD",
            timeframe=TimeFrame.H1,
            bars=1000  # Get more bars to ensure we have data
        )
        print(f"Retrieved {len(test_data.candles)} candles")
        if test_data.candles:
            print(f"Date range: {test_data.candles[0].timestamp} to {test_data.candles[-1].timestamp}")
        
        
        # Configure backtest - OFFLINE 2024 2025
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 12, 31, tzinfo=timezone.utc),
            symbols=settings.trading.symbols[:2],  # Use first 2 symbols from config
            mode=BacktestMode.OFFLINE_ONLY,
            initial_balance=10000,
            risk_per_trade=settings.trading.risk_per_trade_percent / 100,
            max_open_trades=settings.trading.max_open_trades
        )
        
        # # Test one month with real GPT
        # config = BacktestConfig(
        # start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        # end_date=datetime(2024, 1, 7, tzinfo=timezone.utc),
        # symbols=["EURUSD"],
        # mode=BacktestMode.FULL,
        # initial_balance=10000,
        # risk_per_trade=0.015
        # )

        # Create backtest engine
        engine = BacktestEngine(
            data_provider=data_provider,
            chart_generator=ChartGenerator() if ChartGenerator else None,
            db_path=settings.database.db_path
        )
        
        print(f"Starting backtest for {config.symbols} from {config.start_date.date()} to {config.end_date.date()}")
        
        # Run backtest
        results = await engine.run_backtest(config)
        
        # Print summary
        print(f"\nBacktest Results:")
        print(f"Total Trades: {results.total_trades}")
        print(f"Win Rate: {results.win_rate:.1%}")
        print(f"Total Return: {results.total_return:.2f}%")
        print(f"Max Drawdown: {results.max_drawdown:.2f}%")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"Profit Factor: {results.profit_factor:.2f}")
        
        # Generate reports
        report_generator = BacktestReportGenerator(ChartGenerator() if ChartGenerator else None)
        await report_generator.generate_report(results, "backtest_results")
        
        print("\nReports saved to backtest_results/")
        
    finally:
        mt5_client.shutdown()


if __name__ == "__main__":
    asyncio.run(main())