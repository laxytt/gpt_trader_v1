"""
Symbol Resolver Utility
Handles different symbol naming conventions across brokers
"""

import logging
from typing import Optional, Dict, List
import MetaTrader5 as mt5

logger = logging.getLogger(__name__)


class SymbolResolver:
    """Resolves symbol names across different broker conventions"""
    
    # Common symbol mappings
    SYMBOL_MAPPINGS = {
        # Metals
        'GOLD': ['XAUUSD', 'Gold', 'GOLD', 'XAU/USD', 'XAUUSDm', 'XAUUSD.', 'XAUUSD!'],
        'SILVER': ['XAGUSD', 'Silver', 'SILVER', 'XAG/USD', 'XAGUSDm', 'XAGUSD.'],
        
        # Indices
        'US30': ['US30', 'DJ30', 'US30Cash', 'US30.cash', 'USA30', 'USA30.cash', 
                 'Wall Street 30', 'WS30', 'US30Index', 'US30.', 'DJI30', 'DJIA'],
        'US500': ['US500', 'SP500', 'US500Cash', 'US500.cash', 'USA500', 
                  'S&P500', 'SPX500', 'US500.', 'SPX', 'USA.500'],
        'NASDAQ': ['USTEC', 'NAS100', 'US100', 'NASDAQ', 'US100Cash', 
                   'US100.cash', 'USA100', 'NDX100', 'TECH100'],
        'DAX': ['GER30', 'GER40', 'DAX30', 'DAX40', 'Germany30', 
                'Germany40', 'DE30', 'DE40', 'DAX', 'GER.30', 'GER.40'],
        'FTSE': ['UK100', 'FTSE100', 'UK100Cash', 'UK100.cash', 'FTSE', 'UK.100'],
        
        # Energy
        'OIL': ['XTIUSD', 'USOil', 'CrudeOil', 'WTI', 'USOIL', 'OIL', 
                'WTICrude', 'XTIUSD.', 'OIL.WTI', 'CL'],
        'BRENT': ['XBRUSD', 'UKOil', 'BrentOil', 'BRENT', 'UKOIL', 
                  'BRENTCrude', 'XBRUSD.', 'OIL.BRENT'],
        
        # Crypto
        'BITCOIN': ['BTCUSD', 'Bitcoin', 'BTC/USD', 'BTCUSD.', 'BTCUSD!', 'BTC'],
        'ETHEREUM': ['ETHUSD', 'Ethereum', 'ETH/USD', 'ETHUSD.', 'ETHUSD!', 'ETH'],
    }
    
    def __init__(self):
        """Initialize the symbol resolver"""
        self._symbol_cache: Dict[str, str] = {}
        self._populate_cache()
        
    def _populate_cache(self):
        """Populate the symbol cache with available symbols"""
        try:
            all_symbols = mt5.symbols_get()
            if all_symbols:
                # Create a mapping of uppercase names for quick lookup
                for symbol in all_symbols:
                    self._symbol_cache[symbol.name.upper()] = symbol.name
                    
                logger.info(f"Symbol cache populated with {len(self._symbol_cache)} symbols")
        except Exception as e:
            logger.error(f"Failed to populate symbol cache: {e}")
            
    def resolve_symbol(self, symbol: str) -> Optional[str]:
        """
        Resolve a symbol name to the broker's actual symbol
        
        Args:
            symbol: The symbol to resolve (e.g., 'GOLD', 'US30')
            
        Returns:
            The actual symbol name used by the broker, or None if not found
        """
        symbol_upper = symbol.upper()
        
        # First, check if it's already a valid symbol
        if symbol_upper in self._symbol_cache:
            return self._symbol_cache[symbol_upper]
            
        # Check if it's a known alias
        if symbol_upper in self.SYMBOL_MAPPINGS:
            for variant in self.SYMBOL_MAPPINGS[symbol_upper]:
                variant_upper = variant.upper()
                if variant_upper in self._symbol_cache:
                    actual_symbol = self._symbol_cache[variant_upper]
                    logger.info(f"Resolved {symbol} -> {actual_symbol}")
                    return actual_symbol
                    
        # Try partial matching as last resort
        matches = self._find_partial_matches(symbol_upper)
        if matches:
            if len(matches) == 1:
                actual_symbol = matches[0]
                logger.info(f"Resolved {symbol} -> {actual_symbol} (partial match)")
                return actual_symbol
            else:
                logger.warning(f"Multiple matches for {symbol}: {matches[:5]}")
                # Return the shortest match (usually the most common)
                actual_symbol = min(matches, key=len)
                logger.info(f"Resolved {symbol} -> {actual_symbol} (best match)")
                return actual_symbol
                
        logger.warning(f"Could not resolve symbol: {symbol}")
        return None
        
    def _find_partial_matches(self, symbol: str) -> List[str]:
        """Find symbols that contain the search term"""
        matches = []
        for cached_symbol in self._symbol_cache.values():
            if symbol in cached_symbol.upper():
                matches.append(cached_symbol)
        return sorted(matches, key=len)  # Sort by length (shorter first)
        
    def resolve_and_enable(self, symbol: str) -> Optional[str]:
        """
        Resolve a symbol and enable it in Market Watch
        
        Args:
            symbol: The symbol to resolve and enable
            
        Returns:
            The actual symbol name if successful, None otherwise
        """
        actual_symbol = self.resolve_symbol(symbol)
        
        if actual_symbol:
            # Enable the symbol in Market Watch
            if mt5.symbol_select(actual_symbol, True):
                logger.info(f"Enabled symbol in Market Watch: {actual_symbol}")
                return actual_symbol
            else:
                logger.error(f"Failed to enable symbol: {actual_symbol}")
                
        return None
        
    def get_available_symbols(self, category: Optional[str] = None) -> List[str]:
        """
        Get list of available symbols, optionally filtered by category
        
        Args:
            category: Category to filter by (e.g., 'Forex', 'Metals', 'Indices')
            
        Returns:
            List of symbol names
        """
        symbols = []
        
        try:
            all_symbols = mt5.symbols_get()
            if all_symbols:
                for symbol in all_symbols:
                    if category:
                        # Check if symbol path contains the category
                        if category.lower() in symbol.path.lower():
                            symbols.append(symbol.name)
                    else:
                        symbols.append(symbol.name)
                        
        except Exception as e:
            logger.error(f"Failed to get available symbols: {e}")
            
        return sorted(symbols)
        
    def ensure_symbols_enabled(self, symbols: List[str]) -> Dict[str, str]:
        """
        Ensure a list of symbols are enabled in Market Watch
        
        Args:
            symbols: List of symbols to enable
            
        Returns:
            Dictionary mapping requested symbols to actual symbols
        """
        resolved_symbols = {}
        
        for symbol in symbols:
            actual = self.resolve_and_enable(symbol)
            if actual:
                resolved_symbols[symbol] = actual
            else:
                logger.warning(f"Failed to enable symbol: {symbol}")
                
        return resolved_symbols


# Global instance
_symbol_resolver: Optional[SymbolResolver] = None


def get_symbol_resolver() -> SymbolResolver:
    """Get the global symbol resolver instance"""
    global _symbol_resolver
    if _symbol_resolver is None:
        _symbol_resolver = SymbolResolver()
    return _symbol_resolver


def resolve_symbol(symbol: str) -> Optional[str]:
    """
    Convenience function to resolve a symbol
    
    Args:
        symbol: Symbol to resolve
        
    Returns:
        Actual symbol name or None
    """
    resolver = get_symbol_resolver()
    return resolver.resolve_symbol(symbol)


def ensure_symbol_enabled(symbol: str) -> Optional[str]:
    """
    Convenience function to resolve and enable a symbol
    
    Args:
        symbol: Symbol to enable
        
    Returns:
        Actual symbol name or None
    """
    resolver = get_symbol_resolver()
    return resolver.resolve_and_enable(symbol)