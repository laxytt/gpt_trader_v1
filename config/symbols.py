"""
Symbol-specific configurations for the GPT Trading System.
Contains trading parameters and specifications for each symbol.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class SymbolSpec:
    """Specification for a trading symbol"""
    pip_value: float
    min_atr: float
    min_volume: int
    typical_spread: float
    commission_per_lot: float
    currencies: Tuple[str, ...]
    point_multiplier: int = 10000  # For converting to pips
    
    @property
    def pip_size(self) -> float:
        """Get pip size for this symbol"""
        return self.pip_value
    
    def price_to_pips(self, price_diff: float) -> float:
        """Convert price difference to pips"""
        return price_diff * self.point_multiplier


# Symbol specifications - these define trading characteristics for each symbol
SYMBOL_SPECIFICATIONS: Dict[str, SymbolSpec] = {
    # Major Forex Pairs
    "EURUSD": SymbolSpec(
        pip_value=0.0001,
        min_atr=0.0008,
        min_volume=60,
        typical_spread=1.0,
        commission_per_lot=3.5,
        currencies=("EUR", "USD")
    ),
    
    "GBPUSD": SymbolSpec(
        pip_value=0.0001,
        min_atr=0.0010,
        min_volume=60,
        typical_spread=1.5,
        commission_per_lot=3.5,
        currencies=("GBP", "USD")
    ),
    
    "USDJPY": SymbolSpec(
        pip_value=0.01,
        min_atr=0.08,
        min_volume=50,
        typical_spread=1.0,
        commission_per_lot=3.5,
        currencies=("USD", "JPY"),
        point_multiplier=100  # JPY pairs have different pip calculation
    ),
    
    "USDCAD": SymbolSpec(
        pip_value=0.0001,
        min_atr=0.0008,
        min_volume=45,
        typical_spread=1.2,
        commission_per_lot=3.5,
        currencies=("USD", "CAD")
    ),
    
    "AUDUSD": SymbolSpec(
        pip_value=0.0001,
        min_atr=0.0007,
        min_volume=50,
        typical_spread=1.0,
        commission_per_lot=3.5,
        currencies=("AUD", "USD")
    ),
    
    "NZDUSD": SymbolSpec(
        pip_value=0.0001,
        min_atr=0.0006,
        min_volume=40,
        typical_spread=1.5,
        commission_per_lot=3.5,
        currencies=("NZD", "USD")
    ),
    
    "USDCHF": SymbolSpec(
        pip_value=0.0001,
        min_atr=0.0007,
        min_volume=45,
        typical_spread=1.2,
        commission_per_lot=3.5,
        currencies=("USD", "CHF")
    ),
    
    # Cross Pairs
    "EURJPY": SymbolSpec(
        pip_value=0.01,
        min_atr=0.10,
        min_volume=40,
        typical_spread=1.5,
        commission_per_lot=4.0,
        currencies=("EUR", "JPY"),
        point_multiplier=100
    ),
    
    "GBPJPY": SymbolSpec(
        pip_value=0.01,
        min_atr=0.12,
        min_volume=35,
        typical_spread=2.0,
        commission_per_lot=4.0,
        currencies=("GBP", "JPY"),
        point_multiplier=100
    ),
    
    # Commodities - Precious Metals
    "XAUUSD": SymbolSpec(
        pip_value=0.01,
        min_atr=2.0,
        min_volume=200,
        typical_spread=3.0,
        commission_per_lot=5.0,
        currencies=("USD",),
        point_multiplier=100
    ),
    
    "XAGUSD": SymbolSpec(
        pip_value=0.001,
        min_atr=0.15,
        min_volume=150,
        typical_spread=2.5,
        commission_per_lot=4.5,
        currencies=("USD",),
        point_multiplier=1000
    ),
    
    # Commodities - Energy
    "WTIUSD": SymbolSpec(
        pip_value=0.01,
        min_atr=0.50,
        min_volume=300,
        typical_spread=3.0,
        commission_per_lot=6.0,
        currencies=("USD",),
        point_multiplier=100
    ),
    
    "UKOUSD": SymbolSpec(
        pip_value=0.01,
        min_atr=0.55,
        min_volume=280,
        typical_spread=3.5,
        commission_per_lot=6.0,
        currencies=("USD",),
        point_multiplier=100
    ),
    
    "NATGAS": SymbolSpec(
        pip_value=0.001,
        min_atr=0.050,
        min_volume=250,
        typical_spread=4.0,
        commission_per_lot=6.5,
        currencies=("USD",),
        point_multiplier=1000
    ),
    
    # Indices
    "US30.cash": SymbolSpec(
        pip_value=0.1,
        min_atr=30.0,
        min_volume=200,
        typical_spread=2.0,
        commission_per_lot=5.0,
        currencies=("USD",),
        point_multiplier=10
    ),
    
    "US100.cash": SymbolSpec(
        pip_value=0.1,
        min_atr=15.0,
        min_volume=200,
        typical_spread=1.5,
        commission_per_lot=5.0,
        currencies=("USD",),
        point_multiplier=10
    ),
    
    "US500.cash": SymbolSpec(
        pip_value=0.1,
        min_atr=8.0,
        min_volume=180,
        typical_spread=1.0,
        commission_per_lot=4.5,
        currencies=("USD",),
        point_multiplier=10
    ),
    
    "GER40.cash": SymbolSpec(
        pip_value=0.1,
        min_atr=12.0,
        min_volume=150,
        typical_spread=1.0,
        commission_per_lot=4.0,
        currencies=("EUR",),
        point_multiplier=10
    ),
    
    "UK100.cash": SymbolSpec(
        pip_value=0.1,
        min_atr=10.0,
        min_volume=120,
        typical_spread=1.2,
        commission_per_lot=4.0,
        currencies=("GBP",),
        point_multiplier=10
    ),
    
    "FR40.cash": SymbolSpec(
        pip_value=0.1,
        min_atr=8.0,
        min_volume=100,
        typical_spread=1.5,
        commission_per_lot=4.0,
        currencies=("EUR",),
        point_multiplier=10
    )
}


# Predefined symbol groups for easy configuration
SYMBOL_GROUPS = {
    "major_forex": [
        "EURUSD", "GBPUSD", "USDJPY", "USDCAD", 
        "AUDUSD", "NZDUSD", "USDCHF"
    ],
    
    "cross_pairs": [
        "EURJPY", "GBPJPY", "EURGBP", "EURAUD",
        "EURCAD", "GBPAUD", "GBPCAD"
    ],
    
    "commodities": [
        "XAUUSD", "XAGUSD", "WTIUSD", "UKOUSD", "NATGAS"
    ],
    
    "us_indices": [
        "US30.cash", "US100.cash", "US500.cash"
    ],
    
    "european_indices": [
        "GER40.cash", "UK100.cash", "FR40.cash"
    ],
    
    "conservative": [
        "EURUSD", "GBPUSD", "USDCAD", "AUDUSD"
    ],
    
    "moderate": [
        "EURJPY", "XAUUSD", "US500.cash", "GER40.cash"
    ],
    
    "aggressive": [
        "GBPJPY", "NATGAS", "US30.cash", "WTIUSD"
    ],
    
    "ftmo_recommended": [
        "EURUSD", "GBPUSD", "XAUUSD", "US500.cash", 
        "GER40.cash", "WTIUSD"
    ],
    
    "beginner_friendly": [
        "EURUSD", "GBPUSD", "USDCAD"
    ]
}


# News event blacklists by symbol
NEWS_BLACKLISTS = {
    "EURUSD": [
        "Federal Funds Rate", "FOMC Statement", "Non-Farm Employment Change",
        "Unemployment Rate", "Average Hourly Earnings", "Advance GDP q/q",
        "FOMC Meeting Minutes", "CPI y/y", "Main Refinancing Rate",
        "ECB Press Conference", "ECB Interest Rate Decision"
    ],
    
    "GBPUSD": [
        "Federal Funds Rate", "FOMC Statement", "Non-Farm Employment Change",
        "Bank of England Interest Rate Decision", "MPC Meeting Minutes",
        "GDP q/q", "CPI y/y", "Unemployment Rate"
    ],
    
    "USDJPY": [
        "Federal Funds Rate", "FOMC Statement", "Non-Farm Employment Change",
        "Bank of Japan Interest Rate Decision", "BOJ Press Conference",
        "Tankan Large Manufacturers Index"
    ],
    
    "XAUUSD": [
        "Federal Funds Rate", "FOMC Statement", "CPI y/y",
        "Non-Farm Employment Change", "FOMC Meeting Minutes"
    ],
    
    "US30.cash": [
        "Federal Funds Rate", "FOMC Statement", "Non-Farm Employment Change",
        "GDP q/q", "CPI y/y", "FOMC Meeting Minutes"
    ],
    
    "US100.cash": [
        "Federal Funds Rate", "FOMC Statement", "Non-Farm Employment Change",
        "GDP q/q", "CPI y/y", "FOMC Meeting Minutes"
    ],
    
    "GER40.cash": [
        "ECB Interest Rate Decision", "ECB Press Conference",
        "Main Refinancing Rate", "German GDP q/q", "German CPI y/y"
    ]
}


# Trading session preferences by symbol
SYMBOL_SESSIONS = {
    "EURUSD": ["Europe", "New York"],
    "GBPUSD": ["Europe", "New York"], 
    "USDJPY": ["Asia", "New York"],
    "AUDUSD": ["Asia", "Europe"],
    "XAUUSD": ["Europe", "New York"],
    "US30.cash": ["New York"],
    "GER40.cash": ["Europe"]
}


# Utility functions
def get_symbol_spec(symbol: str) -> SymbolSpec:
    """Get symbol specification, with fallback to default"""
    return SYMBOL_SPECIFICATIONS.get(symbol.upper(), 
        SymbolSpec(
            pip_value=0.0001,
            min_atr=0.0005,
            min_volume=50,
            typical_spread=2.0,
            commission_per_lot=5.0,
            currencies=("USD", "EUR")
        )
    )


def get_symbol_currencies(symbol: str) -> Tuple[str, ...]:
    """Get currencies relevant to a symbol"""
    spec = get_symbol_spec(symbol)
    return spec.currencies


def get_symbols_by_group(group_name: str) -> List[str]:
    """Get symbols by predefined group"""
    return SYMBOL_GROUPS.get(group_name, [])


def is_symbol_supported(symbol: str) -> bool:
    """Check if symbol is in our specifications"""
    return symbol.upper() in SYMBOL_SPECIFICATIONS


def get_all_supported_symbols() -> List[str]:
    """Get list of all supported symbols"""
    return list(SYMBOL_SPECIFICATIONS.keys())


def get_symbol_news_blacklist(symbol: str) -> List[str]:
    """Get news blacklist for symbol"""
    return NEWS_BLACKLISTS.get(symbol.upper(), [])


# Export main components
__all__ = [
    'SymbolSpec',
    'SYMBOL_SPECIFICATIONS', 
    'SYMBOL_GROUPS',
    'NEWS_BLACKLISTS',
    'SYMBOL_SESSIONS',
    'get_symbol_spec',
    'get_symbol_currencies', 
    'get_symbols_by_group',
    'is_symbol_supported',
    'get_all_supported_symbols',
    'get_symbol_news_blacklist'
]