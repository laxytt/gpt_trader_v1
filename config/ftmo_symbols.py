"""
FTMO-compliant symbol configurations for commodities and indices.
Based on FTMO's allowed trading instruments (2025).
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class FTMOSymbolSpec:
    """Extended specification for FTMO trading symbols"""
    symbol: str
    market_type: str  # 'commodity', 'index', 'forex'
    pip_value: float
    min_atr: float
    min_volume: int
    typical_spread: float
    commission_per_lot: float
    point_multiplier: int
    trading_hours: str
    optimal_sessions: List[str]
    volatility_profile: str  # 'low', 'medium', 'high', 'extreme'
    risk_factor: float  # 1.0 = normal, >1.0 = higher risk
    description: str


# Additional FTMO-allowed commodities
FTMO_COMMODITIES: Dict[str, FTMOSymbolSpec] = {
    # Energy Commodities
    "WTIUSD": FTMOSymbolSpec(
        symbol="WTIUSD",
        market_type="commodity",
        pip_value=0.01,
        min_atr=0.50,
        min_volume=300,
        typical_spread=3.0,
        commission_per_lot=6.0,
        point_multiplier=100,
        trading_hours="00:00-23:00",
        optimal_sessions=["Europe", "New York"],
        volatility_profile="high",
        risk_factor=1.5,
        description="WTI Crude Oil"
    ),
    
    "UKOUSD": FTMOSymbolSpec(
        symbol="UKOUSD",
        market_type="commodity",
        pip_value=0.01,
        min_atr=0.55,
        min_volume=280,
        typical_spread=3.5,
        commission_per_lot=6.0,
        point_multiplier=100,
        trading_hours="02:00-23:00",
        optimal_sessions=["Europe", "New York"],
        volatility_profile="high",
        risk_factor=1.5,
        description="Brent Crude Oil"
    ),
    
    "NATGAS": FTMOSymbolSpec(
        symbol="NATGAS",
        market_type="commodity",
        pip_value=0.001,
        min_atr=0.050,
        min_volume=250,
        typical_spread=4.0,
        commission_per_lot=6.5,
        point_multiplier=1000,
        trading_hours="00:00-23:00",
        optimal_sessions=["New York"],
        volatility_profile="extreme",
        risk_factor=2.0,
        description="Natural Gas"
    ),
    
    # Precious Metals (additional to existing XAUUSD/XAGUSD)
    "XPDUSD": FTMOSymbolSpec(
        symbol="XPDUSD",
        market_type="commodity",
        pip_value=0.01,
        min_atr=15.0,
        min_volume=100,
        typical_spread=10.0,
        commission_per_lot=7.0,
        point_multiplier=100,
        trading_hours="00:00-23:00",
        optimal_sessions=["Europe", "New York"],
        volatility_profile="high",
        risk_factor=1.8,
        description="Palladium"
    ),
    
    "XPTUSD": FTMOSymbolSpec(
        symbol="XPTUSD",
        market_type="commodity",
        pip_value=0.01,
        min_atr=8.0,
        min_volume=120,
        typical_spread=5.0,
        commission_per_lot=6.0,
        point_multiplier=100,
        trading_hours="00:00-23:00",
        optimal_sessions=["Europe", "New York"],
        volatility_profile="medium",
        risk_factor=1.3,
        description="Platinum"
    ),
    
    # Agricultural Commodities
    "COFFEE": FTMOSymbolSpec(
        symbol="COFFEE",
        market_type="commodity",
        pip_value=0.01,
        min_atr=2.0,
        min_volume=150,
        typical_spread=8.0,
        commission_per_lot=5.0,
        point_multiplier=100,
        trading_hours="09:00-18:30",
        optimal_sessions=["New York"],
        volatility_profile="medium",
        risk_factor=1.4,
        description="Coffee"
    ),
    
    "COCOA": FTMOSymbolSpec(
        symbol="COCOA",
        market_type="commodity",
        pip_value=1.0,
        min_atr=20.0,
        min_volume=100,
        typical_spread=10.0,
        commission_per_lot=5.0,
        point_multiplier=1,
        trading_hours="09:00-18:30",
        optimal_sessions=["New York"],
        volatility_profile="medium",
        risk_factor=1.3,
        description="Cocoa"
    ),
    
    "SOYBEAN": FTMOSymbolSpec(
        symbol="SOYBEAN",
        market_type="commodity",
        pip_value=0.25,
        min_atr=10.0,
        min_volume=120,
        typical_spread=2.0,
        commission_per_lot=5.0,
        point_multiplier=4,
        trading_hours="01:00-19:00",
        optimal_sessions=["New York"],
        volatility_profile="medium",
        risk_factor=1.2,
        description="Soybeans"
    ),
    
    "CORN": FTMOSymbolSpec(
        symbol="CORN",
        market_type="commodity",
        pip_value=0.25,
        min_atr=5.0,
        min_volume=140,
        typical_spread=1.5,
        commission_per_lot=4.5,
        point_multiplier=4,
        trading_hours="01:00-19:00",
        optimal_sessions=["New York"],
        volatility_profile="low",
        risk_factor=1.1,
        description="Corn"
    ),
    
    "WHEAT": FTMOSymbolSpec(
        symbol="WHEAT",
        market_type="commodity",
        pip_value=0.25,
        min_atr=6.0,
        min_volume=130,
        typical_spread=2.0,
        commission_per_lot=4.5,
        point_multiplier=4,
        trading_hours="01:00-19:00",
        optimal_sessions=["New York"],
        volatility_profile="medium",
        risk_factor=1.2,
        description="Wheat"
    )
}


# Additional FTMO-allowed indices
FTMO_INDICES: Dict[str, FTMOSymbolSpec] = {
    # Asian Indices
    "JP225.cash": FTMOSymbolSpec(
        symbol="JP225.cash",
        market_type="index",
        pip_value=1.0,
        min_atr=200.0,
        min_volume=180,
        typical_spread=10.0,
        commission_per_lot=5.0,
        point_multiplier=1,
        trading_hours="00:00-23:00",
        optimal_sessions=["Asia"],
        volatility_profile="medium",
        risk_factor=1.2,
        description="Nikkei 225"
    ),
    
    "HK50.cash": FTMOSymbolSpec(
        symbol="HK50.cash",
        market_type="index",
        pip_value=1.0,
        min_atr=150.0,
        min_volume=150,
        typical_spread=8.0,
        commission_per_lot=5.0,
        point_multiplier=1,
        trading_hours="02:00-17:00",
        optimal_sessions=["Asia"],
        volatility_profile="high",
        risk_factor=1.4,
        description="Hang Seng 50"
    ),
    
    "CN50.cash": FTMOSymbolSpec(
        symbol="CN50.cash",
        market_type="index",
        pip_value=1.0,
        min_atr=120.0,
        min_volume=140,
        typical_spread=15.0,
        commission_per_lot=5.5,
        point_multiplier=1,
        trading_hours="02:00-17:00",
        optimal_sessions=["Asia"],
        volatility_profile="high",
        risk_factor=1.5,
        description="China A50"
    ),
    
    # European Indices (additional)
    "ESP35.cash": FTMOSymbolSpec(
        symbol="ESP35.cash",
        market_type="index",
        pip_value=0.1,
        min_atr=15.0,
        min_volume=100,
        typical_spread=2.0,
        commission_per_lot=4.0,
        point_multiplier=10,
        trading_hours="08:00-20:00",
        optimal_sessions=["Europe"],
        volatility_profile="medium",
        risk_factor=1.2,
        description="Spain 35"
    ),
    
    "EU50.cash": FTMOSymbolSpec(
        symbol="EU50.cash",
        market_type="index",
        pip_value=0.1,
        min_atr=12.0,
        min_volume=120,
        typical_spread=1.5,
        commission_per_lot=4.0,
        point_multiplier=10,
        trading_hours="08:00-22:00",
        optimal_sessions=["Europe"],
        volatility_profile="medium",
        risk_factor=1.1,
        description="Euro Stoxx 50"
    ),
    
    "NL25.cash": FTMOSymbolSpec(
        symbol="NL25.cash",
        market_type="index",
        pip_value=0.01,
        min_atr=2.0,
        min_volume=100,
        typical_spread=0.5,
        commission_per_lot=3.5,
        point_multiplier=100,
        trading_hours="08:00-22:00",
        optimal_sessions=["Europe"],
        volatility_profile="low",
        risk_factor=1.0,
        description="Netherlands 25"
    ),
    
    # Other Global Indices
    "AUS200.cash": FTMOSymbolSpec(
        symbol="AUS200.cash",
        market_type="index",
        pip_value=0.1,
        min_atr=20.0,
        min_volume=130,
        typical_spread=2.0,
        commission_per_lot=4.5,
        point_multiplier=10,
        trading_hours="00:00-22:00",
        optimal_sessions=["Asia", "Europe"],
        volatility_profile="medium",
        risk_factor=1.2,
        description="Australia 200"
    ),
    
    "CAN60.cash": FTMOSymbolSpec(
        symbol="CAN60.cash",
        market_type="index",
        pip_value=0.1,
        min_atr=15.0,
        min_volume=100,
        typical_spread=2.5,
        commission_per_lot=4.5,
        point_multiplier=10,
        trading_hours="14:30-21:00",
        optimal_sessions=["New York"],
        volatility_profile="medium",
        risk_factor=1.1,
        description="Canada 60"
    )
}


# Exotic forex pairs that FTMO allows (higher spreads, more volatility)
FTMO_EXOTIC_FOREX: Dict[str, FTMOSymbolSpec] = {
    "USDMXN": FTMOSymbolSpec(
        symbol="USDMXN",
        market_type="forex",
        pip_value=0.0001,
        min_atr=0.015,
        min_volume=80,
        typical_spread=15.0,
        commission_per_lot=5.0,
        point_multiplier=10000,
        trading_hours="00:00-23:00",
        optimal_sessions=["New York"],
        volatility_profile="high",
        risk_factor=1.6,
        description="USD/Mexican Peso"
    ),
    
    "USDTRY": FTMOSymbolSpec(
        symbol="USDTRY",
        market_type="forex",
        pip_value=0.0001,
        min_atr=0.025,
        min_volume=70,
        typical_spread=50.0,
        commission_per_lot=6.0,
        point_multiplier=10000,
        trading_hours="00:00-23:00",
        optimal_sessions=["Europe"],
        volatility_profile="extreme",
        risk_factor=2.0,
        description="USD/Turkish Lira"
    ),
    
    "USDZAR": FTMOSymbolSpec(
        symbol="USDZAR",
        market_type="forex",
        pip_value=0.0001,
        min_atr=0.020,
        min_volume=75,
        typical_spread=30.0,
        commission_per_lot=5.5,
        point_multiplier=10000,
        trading_hours="00:00-23:00",
        optimal_sessions=["Europe"],
        volatility_profile="high",
        risk_factor=1.7,
        description="USD/South African Rand"
    ),
    
    "USDSGD": FTMOSymbolSpec(
        symbol="USDSGD",
        market_type="forex",
        pip_value=0.0001,
        min_atr=0.0008,
        min_volume=60,
        typical_spread=3.0,
        commission_per_lot=4.0,
        point_multiplier=10000,
        trading_hours="00:00-23:00",
        optimal_sessions=["Asia"],
        volatility_profile="low",
        risk_factor=1.1,
        description="USD/Singapore Dollar"
    ),
    
    "USDHKD": FTMOSymbolSpec(
        symbol="USDHKD",
        market_type="forex",
        pip_value=0.0001,
        min_atr=0.0005,
        min_volume=50,
        typical_spread=5.0,
        commission_per_lot=4.0,
        point_multiplier=10000,
        trading_hours="00:00-23:00",
        optimal_sessions=["Asia"],
        volatility_profile="low",
        risk_factor=1.0,
        description="USD/Hong Kong Dollar"
    )
}


# Updated symbol groups for FTMO trading
FTMO_SYMBOL_GROUPS = {
    # Conservative groups - lower risk, stable instruments
    "conservative_commodities": [
        "XAUUSD",  # Gold - most stable commodity
        "XAGUSD",  # Silver
        "CORN",    # Low volatility agricultural
    ],
    
    "conservative_indices": [
        "US500.cash",  # S&P 500 - most stable
        "EU50.cash",   # Euro Stoxx 50
        "NL25.cash",   # Netherlands 25
    ],
    
    # Moderate risk groups
    "moderate_commodities": [
        "WTIUSD",   # Crude oil
        "UKOUSD",   # Brent oil
        "XPTUSD",   # Platinum
        "COFFEE",   # Coffee
        "WHEAT",    # Wheat
    ],
    
    "moderate_indices": [
        "US30.cash",    # Dow Jones
        "US100.cash",   # Nasdaq
        "GER40.cash",   # DAX
        "UK100.cash",   # FTSE 100
        "JP225.cash",   # Nikkei
        "AUS200.cash",  # Australia 200
    ],
    
    # Aggressive groups - higher risk/reward
    "aggressive_commodities": [
        "NATGAS",   # Natural gas - very volatile
        "XPDUSD",   # Palladium
        "COCOA",    # Cocoa
    ],
    
    "aggressive_indices": [
        "HK50.cash",   # Hang Seng - volatile
        "CN50.cash",   # China A50 - volatile
        "ESP35.cash",  # Spain 35
    ],
    
    "aggressive_forex": [
        "USDMXN",   # Mexican Peso
        "USDTRY",   # Turkish Lira - extreme volatility
        "USDZAR",   # South African Rand
    ],
    
    # Session-specific groups
    "asian_session": [
        "JP225.cash", "HK50.cash", "CN50.cash", "AUS200.cash",
        "USDSGD", "USDHKD"
    ],
    
    "european_session": [
        "GER40.cash", "UK100.cash", "FR40.cash", "EU50.cash",
        "ESP35.cash", "NL25.cash", "XAUUSD", "XAGUSD"
    ],
    
    "american_session": [
        "US30.cash", "US100.cash", "US500.cash", "CAN60.cash",
        "WTIUSD", "CORN", "WHEAT", "SOYBEAN", "COFFEE"
    ],
    
    # FTMO recommended diversification
    "ftmo_diversified_portfolio": [
        "EURUSD",      # Major forex
        "GBPUSD",      # Major forex
        "XAUUSD",      # Precious metal
        "US500.cash",  # US index
        "GER40.cash",  # European index
        "WTIUSD",      # Energy
    ]
}


def get_ftmo_symbol_spec(symbol: str) -> FTMOSymbolSpec:
    """Get FTMO symbol specification"""
    # Check all dictionaries
    for specs_dict in [FTMO_COMMODITIES, FTMO_INDICES, FTMO_EXOTIC_FOREX]:
        if symbol in specs_dict:
            return specs_dict[symbol]
    return None


def get_symbols_by_volatility(volatility_level: str) -> List[str]:
    """Get symbols by volatility profile"""
    symbols = []
    for specs_dict in [FTMO_COMMODITIES, FTMO_INDICES, FTMO_EXOTIC_FOREX]:
        for symbol, spec in specs_dict.items():
            if spec.volatility_profile == volatility_level:
                symbols.append(symbol)
    return symbols


def get_symbols_by_session(session: str) -> List[str]:
    """Get symbols optimal for a specific trading session"""
    symbols = []
    for specs_dict in [FTMO_COMMODITIES, FTMO_INDICES, FTMO_EXOTIC_FOREX]:
        for symbol, spec in specs_dict.items():
            if session in spec.optimal_sessions:
                symbols.append(symbol)
    return symbols


def get_ftmo_risk_adjusted_symbols(max_risk_factor: float = 1.5) -> List[str]:
    """Get symbols below a certain risk threshold"""
    symbols = []
    for specs_dict in [FTMO_COMMODITIES, FTMO_INDICES, FTMO_EXOTIC_FOREX]:
        for symbol, spec in specs_dict.items():
            if spec.risk_factor <= max_risk_factor:
                symbols.append(symbol)
    return symbols


# Export main components
__all__ = [
    'FTMOSymbolSpec',
    'FTMO_COMMODITIES',
    'FTMO_INDICES',
    'FTMO_EXOTIC_FOREX',
    'FTMO_SYMBOL_GROUPS',
    'get_ftmo_symbol_spec',
    'get_symbols_by_volatility',
    'get_symbols_by_session',
    'get_ftmo_risk_adjusted_symbols'
]