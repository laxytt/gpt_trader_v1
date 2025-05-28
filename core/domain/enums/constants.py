# core/domain/enums/constants.py
"""Trading system constants"""

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

class SymbolConfig:
    """Symbol-specific configuration"""
    
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
            return {}
    
    SYMBOL_SPECIFICATIONS = get_symbol_specifications()

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

class TradingSessions:
    """Trading session definitions (UTC hours)"""
    
    SESSIONS = {
        "ASIA": {"start": 0, "end": 7, "major_pairs": ["USDJPY", "AUDUSD"]},
        "EUROPE": {"start": 7, "end": 15, "major_pairs": ["EURUSD", "GBPUSD"]},
        "NEW_YORK": {"start": 13, "end": 22, "major_pairs": ["EURUSD", "GBPUSD", "USDJPY"]},
        "OVERLAP_EU_NY": {"start": 13, "end": 15, "major_pairs": ["EURUSD", "GBPUSD"]}
    }

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