"""
Test script to verify backtest infrastructure without MT5
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Mock the MT5 module since it's not available in WSL
sys.modules['MetaTrader5'] = type(sys)('MetaTrader5')

print("Starting mock backtest test...")

try:
    from config.settings import get_settings
    settings = get_settings()
    print(f"✓ Settings loaded successfully")
    print(f"  - Trading symbols: {settings.trading.symbols[:3]}")
    print(f"  - Risk per trade: {settings.trading.risk_per_trade_percent}%")
    print(f"  - Max open trades: {settings.trading.max_open_trades}")
    
    from core.domain.models import Trade, TradingSignal, SignalType
    print(f"✓ Domain models imported successfully")
    
    from core.infrastructure.database.repositories import TradeRepository
    print(f"✓ Database repositories imported successfully")
    
    # Test database connection
    db_path = settings.database.db_path
    print(f"  - Database path: {db_path}")
    
    # Test creating a trade object
    test_trade = Trade(
        id="TEST_001",
        symbol="EURUSD",
        signal_type=SignalType.BUY,
        entry_price=1.0900,
        stop_loss=1.0850,
        take_profit=1.1000,
        position_size=0.1,
        entry_time=datetime.now(timezone.utc)
    )
    print(f"✓ Trade object created successfully")
    
    print("\n✅ Basic infrastructure test passed!")
    print("\nNote: Full backtesting requires MetaTrader5 installation on Windows.")
    print("To run on Windows: python run_backtest.py")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete.")