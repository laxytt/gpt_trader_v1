#!/usr/bin/env python3
"""
Debug script to check MT5 terminal_info attributes
"""

import MetaTrader5 as mt5

# Initialize MT5
if mt5.initialize():
    print("MT5 initialized successfully\n")
    
    # Get terminal info
    terminal_info = mt5.terminal_info()
    if terminal_info:
        print("Terminal Info attributes:")
        print(f"Type: {type(terminal_info)}")
        print(f"Fields: {terminal_info._fields if hasattr(terminal_info, '_fields') else 'No _fields'}")
        print("\nAll attributes:")
        for attr in dir(terminal_info):
            if not attr.startswith('_'):
                try:
                    value = getattr(terminal_info, attr)
                    print(f"  - {attr}: {value}")
                except Exception as e:
                    print(f"  - {attr}: Error accessing ({e})")
    
    # Get account info  
    account_info = mt5.account_info()
    if account_info:
        print("\n\nAccount Info attributes:")
        print(f"Type: {type(account_info)}")
        print(f"Fields: {account_info._fields if hasattr(account_info, '_fields') else 'No _fields'}")
        print("\nSample values:")
        for attr in ['login', 'server', 'balance', 'trade_mode']:
            if hasattr(account_info, attr):
                print(f"  - {attr}: {getattr(account_info, attr)}")
    
    # Shutdown
    mt5.shutdown()
else:
    print("Failed to initialize MT5")
    print(mt5.last_error())