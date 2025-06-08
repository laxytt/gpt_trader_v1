#!/usr/bin/env python3
"""
Debug MT5 order_check return codes
"""

import sys
from pathlib import Path
import MetaTrader5 as mt5

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def debug_order_check():
    """Debug order_check return codes"""
    print("MT5 Order Check Debug")
    print("=" * 50)
    
    if not mt5.initialize():
        print("Failed to initialize MT5")
        return
        
    try:
        # Get EURUSD info
        symbol = "EURUSD"
        tick = mt5.symbol_info_tick(symbol)
        
        if not tick:
            print(f"No tick data for {symbol}")
            return
            
        print(f"\nCurrent prices for {symbol}:")
        print(f"Bid: {tick.bid}, Ask: {tick.ask}")
        
        # Prepare test order
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": 0.01,
            "type": mt5.ORDER_TYPE_BUY,
            "price": tick.ask,
            "sl": tick.ask - 0.0020,
            "tp": tick.ask + 0.0030,
            "deviation": 20,
            "magic": 123456,
            "comment": "MT5 test order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        print("\nOrder Request:")
        for key, value in request.items():
            print(f"  {key}: {value}")
            
        # Check order
        result = mt5.order_check(request)
        
        print("\nOrder Check Result:")
        if result:
            print(f"  retcode: {result.retcode}")
            print(f"  comment: {result.comment}")
            print(f"  balance: {result.balance}")
            print(f"  equity: {result.equity}")
            print(f"  profit: {result.profit}")
            print(f"  margin: {result.margin}")
            print(f"  margin_free: {result.margin_free}")
            print(f"  margin_level: {result.margin_level}")
            
            # Show what the return codes mean
            print("\nReturn Code Meanings:")
            print(f"  TRADE_RETCODE_DONE = {mt5.TRADE_RETCODE_DONE}")
            print(f"  TRADE_RETCODE_PLACED = {mt5.TRADE_RETCODE_PLACED}")
            print(f"  TRADE_RETCODE_REQUOTE = {mt5.TRADE_RETCODE_REQUOTE}")
            print(f"  TRADE_RETCODE_REJECT = {mt5.TRADE_RETCODE_REJECT}")
            print(f"  TRADE_RETCODE_CANCEL = {mt5.TRADE_RETCODE_CANCEL}")
            
            # Check if order is valid
            if result.retcode == 0:
                print("\n✅ Order is VALID (retcode = 0)")
            else:
                print(f"\n❌ Order check failed with retcode: {result.retcode}")
                
        else:
            print("  No result returned")
            
    finally:
        mt5.shutdown()
        

if __name__ == "__main__":
    debug_order_check()