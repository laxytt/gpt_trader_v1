#!/usr/bin/env python3
"""
Test Symbol Finder
Helps find and enable symbols with different naming conventions
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import MetaTrader5 as mt5
from mt5_symbol_manager import MT5SymbolManager
from core.utils.symbol_resolver import SymbolResolver


def interactive_symbol_finder():
    """Interactive tool to find and enable symbols"""
    print("MT5 Symbol Finder")
    print("=" * 50)
    
    if not mt5.initialize():
        print("❌ Failed to initialize MT5")
        return
        
    try:
        manager = MT5SymbolManager()
        resolver = SymbolResolver()
        
        while True:
            print("\nOptions:")
            print("1. Search for a symbol")
            print("2. Enable common symbols (GOLD, US30, OIL, etc.)")
            print("3. Show symbol mappings for this broker")
            print("4. List all symbol categories")
            print("5. Test symbol resolver")
            print("0. Exit")
            
            choice = input("\nEnter choice: ").strip()
            
            if choice == '0':
                break
                
            elif choice == '1':
                search = input("Enter symbol to search for (e.g., GOLD, US30): ").strip()
                if search:
                    print(f"\nSearching for '{search}'...")
                    matches = manager.find_symbol(search)
                    
                    if matches:
                        print(f"Found {len(matches)} matches:")
                        for i, (name, desc) in enumerate(matches[:10]):
                            info = mt5.symbol_info(name)
                            status = "✓ Visible" if info and info.visible else "○ Hidden"
                            print(f"{i+1}. {name} - {desc} [{status}]")
                        
                        if len(matches) > 10:
                            print(f"... and {len(matches) - 10} more")
                            
                        # Offer to enable
                        enable = input("\nEnter number to enable (or press Enter to skip): ").strip()
                        if enable.isdigit() and 1 <= int(enable) <= len(matches):
                            symbol_to_enable = matches[int(enable) - 1][0]
                            if manager.enable_symbol(symbol_to_enable):
                                print(f"✅ Enabled {symbol_to_enable}")
                                
                                # Show tick data
                                tick = mt5.symbol_info_tick(symbol_to_enable)
                                if tick:
                                    print(f"   Current price: Bid={tick.bid}, Ask={tick.ask}")
                            else:
                                print(f"❌ Failed to enable {symbol_to_enable}")
                    else:
                        print("No symbols found")
                        
            elif choice == '2':
                print("\nEnabling common symbols...")
                enabled = manager.enable_common_symbols()
                print(f"\nEnabled {len(enabled)} symbols")
                
            elif choice == '3':
                print("\nSymbol mappings for this broker:")
                mappings = manager.get_symbol_mapping()
                if mappings:
                    for common, actual in sorted(mappings.items()):
                        print(f"  {common:10} -> {actual}")
                else:
                    print("No mappings found")
                    
            elif choice == '4':
                print("\nSymbol categories:")
                categories = manager.list_all_symbols_by_category()
                for category, symbols in sorted(categories.items()):
                    print(f"\n{category}: {len(symbols)} symbols")
                    
            elif choice == '5':
                print("\nTesting symbol resolver...")
                test_symbols = ['GOLD', 'US30', 'OIL', 'DAX', 'BITCOIN']
                
                for symbol in test_symbols:
                    resolved = resolver.resolve_symbol(symbol)
                    if resolved:
                        print(f"  {symbol:10} -> {resolved}")
                    else:
                        print(f"  {symbol:10} -> Not found")
                        
    finally:
        mt5.shutdown()
        print("\nMT5 connection closed")


def quick_enable_symbols():
    """Quick function to enable common trading symbols"""
    print("Quick Symbol Enabler")
    print("=" * 50)
    
    if not mt5.initialize():
        print("❌ Failed to initialize MT5")
        return
        
    try:
        resolver = SymbolResolver()
        
        # List of symbols to enable
        symbols_to_enable = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD',  # Major pairs
            'GOLD', 'SILVER',  # Metals
            'US30', 'US500', 'NASDAQ',  # US Indices
            'DAX', 'FTSE',  # European Indices
            'OIL', 'BRENT',  # Energy
        ]
        
        print(f"\nAttempting to enable {len(symbols_to_enable)} symbols...")
        
        enabled_count = 0
        for symbol in symbols_to_enable:
            actual = resolver.resolve_and_enable(symbol)
            if actual:
                if symbol != actual:
                    print(f"✅ {symbol} -> {actual}")
                else:
                    print(f"✅ {symbol}")
                enabled_count += 1
            else:
                print(f"❌ {symbol} - not found")
                
        print(f"\nSuccessfully enabled {enabled_count}/{len(symbols_to_enable)} symbols")
        
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        quick_enable_symbols()
    else:
        interactive_symbol_finder()