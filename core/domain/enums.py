"""
Enums and constants for the GPT Trading System.
Centralizes all constant values and enumerated types.
"""

from enum import Enum, IntEnum
from typing import Dict, List, Tuple


class TimeFrame(Enum):
    """MetaTrader 5 timeframe constants"""
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"
    MN1 = "MN1"


class MT5TimeFrame(IntEnum):
    """MT5 native timeframe values"""
    M1 = 1
    M2 = 2
    M3 = 3
    M4 = 4
    M5 = 5
    M6 = 6
    M10 = 10
    M12 = 12
    M15 = 15
    M20 = 20
    M30 = 30
    H1 = 16385
    H2 = 16386
    H3 = 16387
    H4 = 16388
    H6 = 16390
    H8 = 16392
    H12 = 16396
    D1 = 16408
    W1 = 32769
    MN1 = 49153


class OrderType(IntEnum):
    """MT5 order types"""
    BUY = 0
    SELL = 1
    BUY_LIMIT = 2
    SELL_LIMIT = 3
    BUY_STOP = 4
    SELL_STOP = 5
    BUY_STOP_LIMIT = 6
    SELL_STOP_LIMIT = 7


class TradeAction(IntEnum):
    """MT5 trade actions"""
    DEAL = 1
    PENDING = 5
    SLTP = 6
    MODIFY = 7
    REMOVE = 8
    CLOSE_BY = 10


class ReturnCode(IntEnum):
    """MT5 return codes"""
    DONE = 10009
    COMMON_ERROR = 10004
    INVALID_REQUEST = 10013
    INVALID_VOLUME = 10014
    INVALID_PRICE = 10015
    INVALID_STOPS = 10016
    TRADE_DISABLED = 10017
    MARKET_CLOSED = 10018
    NO_MONEY = 10019
    PRICE_CHANGED = 10020
    OFF_QUOTES = 10021
    EXPIRY_DENIED = 10022
    ORDER_CHANGED = 10023
    TOO_MANY_REQUESTS = 10024
    NO_CHANGES = 10025
    SERVER_DISABLES_AT = 10026
    CLIENT_DISABLES_AT = 10027
    LOCKED = 10028
    FROZEN = 10029
    INVALID_EXPIRATION = 10030


class NewsImpact(Enum):
    """News impact levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Currency(Enum):
    """Major currencies"""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    AUD = "AUD"
    CAD = "CAD"
    CHF = "CHF"
    NZD = "NZD"


class VSAPattern(Enum):
    """Volume Spread Analysis patterns"""
    # Strength signals
    STOPPING_VOLUME = "stopping_volume"
    SHAKE_OUT = "shake_out"
    TWO_BAR_REVERSAL = "two_bar_reversal"
    NO_SUPPLY = "no_supply"
    TEST = "test"
    SELLING_CLIMAX = "selling_climax"
    
    # Weakness signals
    UPTHRUST = "upthrust"
    NO_DEMAND = "no_demand"
    SUPPLY_COMING_IN = "supply_coming_in"
    TRAP_UP_MOVE = "trap_up_move"
    BUYING_CLIMAX = "buying_climax"
    END_OF_RISING_MARKET = "end_of_rising_market"
    
    # Neutral/Test patterns
    INSIDE_BAR = "inside_bar"
    OUTSIDE_BAR = "outside_bar"


class TrendDirection(Enum):
    """Market trend directions"""
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"
    UNDEFINED = "undefined"


class SupportResistanceLevel(Enum):
    """Support and resistance level types"""
    MAJOR = "major"
    MINOR = "minor"
    PSYCHOLOGICAL = "psychological"
    FIBONACCI = "fibonacci"
    PIVOT = "pivot"


# Trading Constants
class TradingConstants:
    """Trading system constants"""
    
    # Risk management
    DEFAULT_RISK_PERCENT = 1.5
    MAX_RISK_PERCENT = 5.0
    MIN_RISK_REWARD_RATIO = 2.0
    MAX_DRAWDOWN_PERCENT = 10.0
    
    # Position sizing
    MIN_LOT_SIZE = 0.01
    MAX_LOT_SIZE = 10.0
    LOT_STEP = 0.01
    
    # Time limits
    MAX_TRADE_DURATION_HOURS = 24
    MAX_TRADE_DURATION_CANDLES = 15
    
    # Technical analysis
    DEFAULT_ATR_PERIOD = 14
    DEFAULT_RSI_PERIOD = 14
    DEFAULT_EMA_FAST = 50
    DEFAULT_EMA_SLOW = 200
    
    # Volume analysis
    MIN_VOLUME_THRESHOLD = 50
    HIGH_VOLUME_MULTIPLIER = 2.0
    LOW_VOLUME_MULTIPLIER = 0.5
    
    # Spread thresholds
    MAX_SPREAD_MULTIPLIER = 3.0
    
    # Magic numbers for MT5
    MAGIC_NUMBER = 10032024


# Symbol configurations
class SymbolConfig:
    """Symbol-specific configuration - now imported from config.symbols"""
    
    @staticmethod
    def get_symbol_specifications():
        """Get symbol specifications from config module"""
        try:
            from config.symbols import SYMBOL_SPECIFICATIONS
            # Convert to old format for backward compatibility
            old_format = {}
            for symbol, spec in SYMBOL_SPECIFICATIONS.items():
                old_format[symbol] = {
                    "pip_value": spec.pip_value,
                    "min_atr": spec.min_atr,
                    "min_volume": spec.min_volume,
                    "typical_spread": spec.typical_spread,
                    "commission_per_lot": spec.commission_per_lot,
                    "currencies": spec.currencies
                }
            return old_format
        except ImportError:
            # Fallback to old hardcoded values
            return {
                "EURUSD": {
                    "pip_value": 0.0001,
                    "min_atr": 0.0008,
                    "min_volume": 60,
                    "typical_spread": 1.0,
                    "commission_per_lot": 3.5,
                    "currencies": ("EUR", "USD")
                }
            }
    
    SYMBOL_SPECIFICATIONS = get_symbol_specifications()


# News event blacklists by symbol
class NewsBlacklists:
    """News events that restrict trading for specific symbols"""
    
    EVENT_BLACKLIST_BY_SYMBOL = {
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


# Session timing
class TradingSessions:
    """Trading session definitions (UTC hours)"""
    
    SESSIONS = {
        "ASIA": {"start": 0, "end": 7, "major_pairs": ["USDJPY", "AUDUSD"]},
        "EUROPE": {"start": 7, "end": 15, "major_pairs": ["EURUSD", "GBPUSD"]},
        "NEW_YORK": {"start": 13, "end": 22, "major_pairs": ["EURUSD", "GBPUSD", "USDJPY"]},
        "OVERLAP_EU_NY": {"start": 13, "end": 15, "major_pairs": ["EURUSD", "GBPUSD"]}
    }


# Risk tier mappings
class RiskTiers:
    """Risk tier configurations"""
    
    TIER_RISK_MAPPING = {
        "A": 2.0,  # High confidence
        "B": 1.5,  # Medium confidence
        "C": 1.0   # Low confidence
    }
    
    TIER_DESCRIPTIONS = {
        "A": "High confidence signal with strong VSA confirmation",
        "B": "Medium confidence signal with partial confirmation", 
        "C": "Low confidence signal or uncertain market conditions"
    }


# GPT Model configurations
class GPTModels:
    """GPT model specifications"""
    
    MODELS = {
        "gpt-4.1-2025-04-14": {
            "max_tokens": 4096,
            "supports_vision": True,
            "cost_per_1k_input": 0.01,
            "cost_per_1k_output": 0.03
        },
        "gpt-4-turbo": {
            "max_tokens": 4096,
            "supports_vision": True,
            "cost_per_1k_input": 0.01,
            "cost_per_1k_output": 0.03
        }
    }


# File paths and extensions
class FilePaths:
    """Common file paths and extensions"""
    
    CHART_EXTENSIONS = [".png", ".jpg", ".jpeg"]
    DATA_EXTENSIONS = [".json", ".csv", ".xlsx"]
    LOG_EXTENSIONS = [".log", ".txt"]
    
    DEFAULT_FILENAMES = {
        "trades_db": "trades.db",
        "news_data": "forexfactory_week.json",
        "completed_trades": "completed_trades.jsonl",
        "open_trades": "open_trades.json",
        "last_signal": "last_signal.json",
        "trigger_file": "trigger.txt"
    }


# Utility mappings
TIMEFRAME_TO_MT5 = {
    TimeFrame.M1: MT5TimeFrame.M1,
    TimeFrame.M5: MT5TimeFrame.M5,
    TimeFrame.M15: MT5TimeFrame.M15,
    TimeFrame.M30: MT5TimeFrame.M30,
    TimeFrame.H1: MT5TimeFrame.H1,
    TimeFrame.H4: MT5TimeFrame.H4,
    TimeFrame.D1: MT5TimeFrame.D1,
    TimeFrame.W1: MT5TimeFrame.W1,
    TimeFrame.MN1: MT5TimeFrame.MN1
}

SIGNAL_TO_ORDER_TYPE = {
    "BUY": OrderType.BUY,
    "SELL": OrderType.SELL
}

# Export all enums and constants
__all__ = [
    # Enums
    'TimeFrame', 'MT5TimeFrame', 'OrderType', 'TradeAction', 'ReturnCode',
    'NewsImpact', 'Currency', 'VSAPattern', 'TrendDirection', 'SupportResistanceLevel',
    
    # Constants classes
    'TradingConstants', 'SymbolConfig', 'NewsBlacklists', 'TradingSessions',
    'RiskTiers', 'GPTModels', 'FilePaths',
    
    # Utility mappings
    'TIMEFRAME_TO_MT5', 'SIGNAL_TO_ORDER_TYPE'
]