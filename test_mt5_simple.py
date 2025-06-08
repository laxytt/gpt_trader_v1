#!/usr/bin/env python3
"""
Simple MT5 Connection Test
Quick test to verify MT5 is properly connected and configured
"""

import sys
from pathlib import Path
from datetime import datetime
import MetaTrader5 as mt5

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_settings
from core.infrastructure.mt5.client import MT5Client


def test_mt5_connection():
    """Test basic MT5 connection"""
    print("MT5 Connection Test")
    print("=" * 50)
    
    # Get settings
    settings = get_settings()
    print(f"\nSettings loaded from: {settings.model_config.get('env_file', '.env')}")
    print(f"MT5 Files Directory: {settings.mt5.files_dir}")
    
    # Initialize MT5 client
    print("\n1. Initializing MT5 Client...")
    mt5_client = MT5Client(settings.mt5)
    
    if mt5_client.initialize():
        print("✅ MT5 initialized successfully")
        
        # Get terminal info
        terminal_info = mt5.terminal_info()
        if terminal_info:
            print(f"\nTerminal Info:")
            # terminal_info is a named tuple, access by index or name
            print(f"  - Connected: {'Yes' if terminal_info.connected else 'No'}")
            print(f"  - Company: {terminal_info.company}")
            # Version might be 'build' in some MT5 versions
            if hasattr(terminal_info, 'version'):
                print(f"  - Version: {terminal_info.version}")
            elif hasattr(terminal_info, 'build'):
                print(f"  - Build: {terminal_info.build}")
            if hasattr(terminal_info, 'path'):
                print(f"  - Path: {terminal_info.path}")
            
        # Get account info
        account_info = mt5_client.get_account_info()
        if account_info:
            print(f"\n2. Account Information:")
            print(f"  - Login: {account_info.get('login', 'N/A')}")
            print(f"  - Name: {account_info.get('name', 'N/A')}")
            print(f"  - Server: {account_info.get('server', 'N/A')}")
            print(f"  - Currency: {account_info.get('currency', 'N/A')}")
            print(f"  - Balance: ${account_info.get('balance', 0):.2f}")
            print(f"  - Equity: ${account_info.get('equity', 0):.2f}")
            print(f"  - Margin Free: ${account_info.get('margin_free', 0):.2f}")
            print(f"  - Leverage: 1:{account_info.get('leverage', 'N/A')}")
            print(f"  - Trade Allowed: {'Yes' if account_info.get('trade_allowed') else 'No'}")
            print(f"  - Trade Expert: {'Yes' if account_info.get('trade_expert') else 'No'}")
            print(f"  - Account Type: {'DEMO' if account_info.get('trade_mode') == 0 else 'LIVE'}")
            
            # Check if it's a demo account
            is_demo = account_info.get('trade_mode') == 0 or 'demo' in str(account_info.get('server', '')).lower()
            if is_demo:
                print("\n✅ This is a DEMO account - safe for testing")
            else:
                print("\n⚠️  WARNING: This appears to be a LIVE account!")
                
        else:
            print("❌ Failed to get account info")
            
        # Test symbol access with resolver
        print("\n3. Testing Symbol Access:")
        from core.utils.symbol_resolver import SymbolResolver
        resolver = SymbolResolver()
        
        test_symbols = ["EURUSD", "GBPUSD", "USDJPY", "GOLD", "US30"]
        
        for symbol in test_symbols:
            # Try to resolve the symbol name
            actual_symbol = resolver.resolve_symbol(symbol)
            
            if actual_symbol:
                info = mt5.symbol_info(actual_symbol)
                if info and info.visible:
                    tick = mt5.symbol_info_tick(actual_symbol)
                    if tick:
                        spread = (tick.ask - tick.bid) / info.point
                        if symbol != actual_symbol:
                            print(f"  ✅ {symbol} ({actual_symbol}): Bid={tick.bid}, Ask={tick.ask}, Spread={spread:.1f} points")
                        else:
                            print(f"  ✅ {symbol}: Bid={tick.bid}, Ask={tick.ask}, Spread={spread:.1f} points")
                    else:
                        print(f"  ❌ {symbol}: No tick data")
                else:
                    # Try to enable it
                    enabled_symbol = resolver.resolve_and_enable(symbol)
                    if enabled_symbol:
                        print(f"  ✅ {symbol} ({enabled_symbol}): Enabled in Market Watch")
                        # Try to get tick data again
                        tick = mt5.symbol_info_tick(enabled_symbol)
                        if tick:
                            info = mt5.symbol_info(enabled_symbol)
                            spread = (tick.ask - tick.bid) / info.point
                            print(f"     Bid={tick.bid}, Ask={tick.ask}, Spread={spread:.1f} points")
                    else:
                        print(f"  ❌ {symbol}: Not available")
            else:
                print(f"  ❌ {symbol}: Could not resolve symbol name")
                
        # Test data retrieval
        print("\n4. Testing Data Retrieval:")
        symbol = "EURUSD"
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 10)
        
        if rates is not None and len(rates) > 0:
            print(f"  ✅ Retrieved {len(rates)} H1 candles for {symbol}")
            latest = rates[-1]
            print(f"     Latest: Time={datetime.fromtimestamp(latest['time'])}, "
                  f"Close={latest['close']}, Volume={latest['tick_volume']}")
        else:
            print(f"  ❌ Failed to retrieve historical data")
            
        # Check positions
        print("\n5. Current Positions:")
        positions = mt5.positions_get()
        if positions:
            print(f"  Found {len(positions)} open position(s):")
            for pos in positions:
                print(f"    - {pos.symbol}: {pos.type_description}, "
                      f"Volume={pos.volume}, Profit=${pos.profit:.2f}")
        else:
            print("  No open positions")
            
        # Check pending orders
        print("\n6. Pending Orders:")
        orders = mt5.orders_get()
        if orders:
            print(f"  Found {len(orders)} pending order(s):")
            for order in orders:
                print(f"    - {order.symbol}: {order.type_description}, "
                      f"Volume={order.volume_current}")
        else:
            print("  No pending orders")
            
        # Shutdown
        mt5_client.shutdown()
        print("\n✅ MT5 connection test completed successfully")
        
    else:
        error = mt5.last_error()
        print(f"❌ Failed to initialize MT5")
        print(f"   Error code: {error[0]}")
        print(f"   Description: {error[1]}")
        print("\n   Troubleshooting:")
        print("   1. Make sure MT5 terminal is installed")
        print("   2. Check the path in your .env file")
        print("   3. Ensure you're using the correct login credentials")
        print("   4. Try logging into MT5 manually first")
        

def main():
    """Run the test"""
    try:
        test_mt5_connection()
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        

if __name__ == "__main__":
    main()