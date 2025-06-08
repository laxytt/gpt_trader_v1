#!/usr/bin/env python3
"""
MT5 Symbol Manager
Helps find, enable, and manage trading symbols with different naming conventions
"""

import MetaTrader5 as mt5
from typing import List, Dict, Optional, Tuple
import re


class MT5SymbolManager:
    """Manages MT5 symbols - finding, enabling, and mapping different names"""
    
    # Common symbol aliases used by different brokers
    SYMBOL_ALIASES = {
        'GOLD': ['XAUUSD', 'Gold', 'GOLD', 'XAU/USD', 'XAUUSDm'],
        'SILVER': ['XAGUSD', 'Silver', 'SILVER', 'XAG/USD', 'XAGUSDm'],
        'US30': ['US30', 'DJ30', 'US30Cash', 'US30.cash', 'USA30', 'USA30.cash', 'Wall Street 30', 'WS30', 'US30Index'],
        'US500': ['US500', 'SP500', 'US500Cash', 'US500.cash', 'USA500', 'S&P500', 'SPX500'],
        'NASDAQ': ['USTEC', 'NAS100', 'US100', 'NASDAQ', 'US100Cash', 'US100.cash', 'USA100'],
        'DAX': ['GER30', 'GER40', 'DAX30', 'DAX40', 'Germany30', 'Germany40', 'DE30', 'DE40'],
        'FTSE': ['UK100', 'FTSE100', 'UK100Cash', 'UK100.cash'],
        'OIL': ['XTIUSD', 'USOil', 'CrudeOil', 'WTI', 'USOIL', 'OIL', 'WTICrude'],
        'BRENT': ['XBRUSD', 'UKOil', 'BrentOil', 'BRENT', 'UKOIL', 'BRENTCrude'],
        'BITCOIN': ['BTCUSD', 'Bitcoin', 'BTC/USD', 'BTCUSD.', 'BTCUSD!'],
        'ETHEREUM': ['ETHUSD', 'Ethereum', 'ETH/USD', 'ETHUSD.', 'ETHUSD!'],
    }
    
    def __init__(self):
        """Initialize the symbol manager"""
        if not mt5.initialize():
            raise Exception("Failed to initialize MT5")
            
    def find_symbol(self, search_term: str) -> List[Tuple[str, str]]:
        """
        Find symbols matching the search term
        
        Args:
            search_term: Symbol name or partial name to search for
            
        Returns:
            List of tuples (symbol_name, description)
        """
        search_term = search_term.upper()
        matches = []
        
        # First check aliases
        if search_term in self.SYMBOL_ALIASES:
            for alias in self.SYMBOL_ALIASES[search_term]:
                symbol_info = mt5.symbol_info(alias)
                if symbol_info:
                    matches.append((symbol_info.name, symbol_info.description))
        
        # Then search all symbols
        all_symbols = mt5.symbols_get()
        if all_symbols:
            for symbol in all_symbols:
                # Check if search term is in symbol name or description
                if (search_term in symbol.name.upper() or 
                    search_term in symbol.description.upper()):
                    matches.append((symbol.name, symbol.description))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_matches = []
        for match in matches:
            if match[0] not in seen:
                seen.add(match[0])
                unique_matches.append(match)
                
        return unique_matches
    
    def enable_symbol(self, symbol_name: str) -> bool:
        """
        Enable a symbol for trading (make it visible in Market Watch)
        
        Args:
            symbol_name: Exact symbol name to enable
            
        Returns:
            True if successful, False otherwise
        """
        return mt5.symbol_select(symbol_name, True)
    
    def disable_symbol(self, symbol_name: str) -> bool:
        """
        Disable a symbol (remove from Market Watch)
        
        Args:
            symbol_name: Exact symbol name to disable
            
        Returns:
            True if successful, False otherwise
        """
        return mt5.symbol_select(symbol_name, False)
    
    def get_symbol_info(self, symbol_name: str) -> Optional[Dict]:
        """
        Get detailed information about a symbol
        
        Args:
            symbol_name: Symbol name
            
        Returns:
            Dictionary with symbol information or None
        """
        info = mt5.symbol_info(symbol_name)
        if not info:
            return None
            
        return {
            'name': info.name,
            'description': info.description,
            'path': info.path,
            'visible': info.visible,
            'selected': info.select,
            'spread': info.spread,
            'digits': info.digits,
            'trade_mode': info.trade_mode,
            'min_volume': info.volume_min,
            'max_volume': info.volume_max,
            'volume_step': info.volume_step,
            'contract_size': info.trade_contract_size,
            'tick_value': info.trade_tick_value,
            'tick_size': info.trade_tick_size,
            'swap_long': info.swap_long,
            'swap_short': info.swap_short,
            'margin_initial': info.margin_initial,
            'margin_maintenance': info.margin_maintenance,
        }
    
    def find_and_enable_symbol(self, search_term: str) -> Optional[str]:
        """
        Find a symbol and enable it for trading
        
        Args:
            search_term: Symbol name or alias to search for
            
        Returns:
            The enabled symbol name or None if not found
        """
        matches = self.find_symbol(search_term)
        
        if not matches:
            print(f"No symbols found matching '{search_term}'")
            return None
            
        if len(matches) == 1:
            # Only one match, enable it
            symbol_name = matches[0][0]
            if self.enable_symbol(symbol_name):
                print(f"Enabled symbol: {symbol_name} ({matches[0][1]})")
                return symbol_name
            else:
                print(f"Failed to enable symbol: {symbol_name}")
                return None
        else:
            # Multiple matches, let user choose
            print(f"\nMultiple symbols found matching '{search_term}':")
            for i, (name, desc) in enumerate(matches):
                print(f"{i+1}. {name} - {desc}")
            
            try:
                choice = int(input("\nEnter number to enable (0 to cancel): "))
                if choice == 0:
                    return None
                if 1 <= choice <= len(matches):
                    symbol_name = matches[choice-1][0]
                    if self.enable_symbol(symbol_name):
                        print(f"Enabled symbol: {symbol_name}")
                        return symbol_name
                    else:
                        print(f"Failed to enable symbol: {symbol_name}")
                        return None
            except ValueError:
                print("Invalid choice")
                return None
    
    def enable_common_symbols(self) -> Dict[str, str]:
        """
        Enable commonly used symbols
        
        Returns:
            Dictionary mapping common names to actual symbol names
        """
        common_symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'GOLD', 'US30', 'OIL']
        enabled_symbols = {}
        
        for symbol in common_symbols:
            # First try direct name
            if mt5.symbol_info(symbol):
                if self.enable_symbol(symbol):
                    enabled_symbols[symbol] = symbol
                    print(f"✓ Enabled {symbol}")
            else:
                # Try aliases
                matches = self.find_symbol(symbol)
                if matches:
                    actual_symbol = matches[0][0]
                    if self.enable_symbol(actual_symbol):
                        enabled_symbols[symbol] = actual_symbol
                        print(f"✓ Enabled {symbol} as {actual_symbol}")
                else:
                    print(f"✗ Could not find {symbol}")
                    
        return enabled_symbols
    
    def list_all_symbols_by_category(self) -> Dict[str, List[str]]:
        """
        List all available symbols organized by category
        
        Returns:
            Dictionary with categories as keys and symbol lists as values
        """
        categories = {}
        all_symbols = mt5.symbols_get()
        
        if all_symbols:
            for symbol in all_symbols:
                # Extract category from path (e.g., "Forex\Majors\EURUSD")
                path_parts = symbol.path.split('\\')
                if len(path_parts) > 1:
                    category = path_parts[0]
                    subcategory = path_parts[1] if len(path_parts) > 2 else ""
                    full_category = f"{category}/{subcategory}" if subcategory else category
                else:
                    full_category = "Other"
                
                if full_category not in categories:
                    categories[full_category] = []
                    
                categories[full_category].append(symbol.name)
        
        # Sort symbols within each category
        for category in categories:
            categories[category].sort()
            
        return categories
    
    def get_symbol_mapping(self) -> Dict[str, str]:
        """
        Get mapping of common names to actual broker symbols
        
        Returns:
            Dictionary mapping common names to broker-specific names
        """
        mapping = {}
        
        for common_name, aliases in self.SYMBOL_ALIASES.items():
            for alias in aliases:
                if mt5.symbol_info(alias):
                    mapping[common_name] = alias
                    break
                    
        return mapping


def test_symbol_manager():
    """Test the symbol manager functionality"""
    print("MT5 Symbol Manager Test")
    print("=" * 50)
    
    manager = MT5SymbolManager()
    
    # Test 1: Find symbols
    print("\n1. Finding symbols:")
    test_searches = ['GOLD', 'US30', 'EUR', 'OIL']
    
    for search in test_searches:
        print(f"\nSearching for '{search}':")
        matches = manager.find_symbol(search)
        for name, desc in matches[:5]:  # Show first 5 matches
            print(f"  - {name}: {desc}")
        if len(matches) > 5:
            print(f"  ... and {len(matches) - 5} more")
    
    # Test 2: Get symbol mapping
    print("\n2. Symbol mapping for this broker:")
    mapping = manager.get_symbol_mapping()
    for common, actual in mapping.items():
        print(f"  {common} -> {actual}")
    
    # Test 3: Enable common symbols
    print("\n3. Enabling common symbols:")
    enabled = manager.enable_common_symbols()
    
    # Test 4: List categories
    print("\n4. Symbol categories:")
    categories = manager.list_all_symbols_by_category()
    for category, symbols in sorted(categories.items()):
        print(f"\n{category}: {len(symbols)} symbols")
        # Show first 3 symbols in each category
        for symbol in symbols[:3]:
            print(f"  - {symbol}")
        if len(symbols) > 3:
            print(f"  ... and {len(symbols) - 3} more")


if __name__ == "__main__":
    if mt5.initialize():
        try:
            test_symbol_manager()
        finally:
            mt5.shutdown()
    else:
        print("Failed to initialize MT5")