"""
Position Trading Configuration for Phase 2.2: Shift to Longer Timeframes

This configuration implements a position trading strategy focusing on:
- Daily and Weekly timeframes for reduced noise
- Wider stops with better risk-reward ratios
- Fewer, higher quality trades
- Longer holding periods (5-20 days average)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import timedelta

@dataclass
class PositionTradingTimeframes:
    """Timeframe configurations for position trading"""
    
    # Primary analysis timeframes
    ENTRY_TIMEFRAME = "D1"      # Daily for entry signals
    BACKGROUND_TIMEFRAME = "W1"  # Weekly for major trend
    CONFIRMATION_TIMEFRAME = "H4" # 4-hour for fine-tuning entry
    
    # Minimum bars required for proper analysis
    MIN_BARS_DAILY = 100        # 100 days of history
    MIN_BARS_WEEKLY = 52        # 52 weeks (1 year)
    MIN_BARS_H4 = 120          # 20 days of H4 data
    
    # Data quality requirements
    MAX_ALLOWED_GAPS = 5        # Maximum missing candles
    MIN_LIQUIDITY_THRESHOLD = 0.7  # Minimum volume consistency


@dataclass
class PositionTradingRiskParameters:
    """Risk management for position trading"""
    
    # Stop loss configuration (ATR-based)
    STOP_LOSS_ATR_MULTIPLIER_MIN = 3.0    # Minimum 3x daily ATR
    STOP_LOSS_ATR_MULTIPLIER_MAX = 5.0    # Maximum 5x daily ATR
    STOP_LOSS_ATR_PERIOD = 20             # 20-day ATR for stability
    
    # Take profit configuration
    TAKE_PROFIT_ATR_MULTIPLIER_MIN = 9.0   # Minimum 9x ATR (3:1 R:R)
    TAKE_PROFIT_ATR_MULTIPLIER_MAX = 15.0  # Maximum 15x ATR (5:1 R:R)
    
    # Risk-reward requirements
    MIN_RISK_REWARD_RATIO = 3.0    # Minimum 3:1 R:R
    TARGET_RISK_REWARD_RATIO = 4.0  # Target 4:1 R:R
    MAX_RISK_REWARD_RATIO = 5.0     # Maximum 5:1 R:R
    
    # Position sizing
    RISK_PER_TRADE_PERCENT = 1.0    # Conservative 1% per trade
    MAX_RISK_PER_DAY_PERCENT = 2.0  # Maximum 2% daily risk
    MAX_PORTFOLIO_HEAT = 6.0        # Maximum 6% total portfolio risk
    
    # Volatility adjustments
    HIGH_VOLATILITY_ADJUSTMENT = 0.7   # Reduce position size by 30% in high volatility
    LOW_VOLATILITY_ADJUSTMENT = 1.2    # Increase position size by 20% in low volatility
    VOLATILITY_LOOKBACK_DAYS = 20      # 20-day volatility calculation


@dataclass
class PositionTradingFrequency:
    """Trading frequency controls for position trading"""
    
    # Maximum positions
    MAX_CONCURRENT_POSITIONS = 4       # Fewer positions for focus
    MAX_POSITIONS_PER_SYMBOL = 1       # One position per symbol
    MAX_CORRELATED_POSITIONS = 2       # Max 2 highly correlated positions
    
    # Time-based restrictions
    MIN_HOURS_BETWEEN_TRADES = 24      # Minimum 24 hours between new trades
    MIN_HOURS_BETWEEN_SAME_SYMBOL = 48 # 48 hours before re-entering same symbol
    
    # Quality over quantity filters
    MIN_CONFIDENCE_SCORE = 70.0        # Higher confidence requirement
    MIN_EDGE_PERCENT = 2.0             # Minimum 2% expected edge
    
    # Holding period targets
    MIN_HOLDING_DAYS = 3               # Minimum 3 days hold
    TARGET_HOLDING_DAYS = 10           # Target 10 days hold
    MAX_HOLDING_DAYS = 30              # Maximum 30 days hold
    
    # Exit patience
    PARTIAL_PROFIT_DAYS = 5            # Consider partial profits after 5 days
    BREAKEVEN_STOP_DAYS = 7            # Move to breakeven after 7 days


@dataclass
class MarketSpecificAdjustments:
    """Market-specific configurations for different asset classes"""
    
    # Forex majors (EURUSD, GBPUSD, USDJPY)
    FOREX_MAJORS = {
        "stop_loss_multiplier": 3.0,      # Tighter stops for liquid pairs
        "take_profit_multiplier": 9.0,    # 3:1 baseline
        "min_atr_filter": 0.0005,         # Minimum daily ATR (50 pips)
        "session_preference": ["london", "newyork"],  # Best sessions
        "avoid_hours": [22, 23, 0, 1, 2], # Avoid Asian session gaps
        "max_spread_pips": 2.0,           # Maximum acceptable spread
    }
    
    # Forex crosses (EURJPY, GBPJPY, EURGBP)
    FOREX_CROSSES = {
        "stop_loss_multiplier": 3.5,      # Slightly wider for crosses
        "take_profit_multiplier": 10.5,   # Maintain 3:1 ratio
        "min_atr_filter": 0.0007,         # Higher ATR requirement
        "session_preference": ["london", "newyork"],
        "avoid_hours": [22, 23, 0, 1, 2],
        "max_spread_pips": 3.0,
    }
    
    # Commodities (GOLD, OIL, SILVER)
    COMMODITIES = {
        "stop_loss_multiplier": 4.0,      # Wider stops for commodities
        "take_profit_multiplier": 12.0,   # Higher targets
        "min_atr_filter": 1.0,            # Commodity-specific ATR
        "session_preference": ["london", "newyork"],
        "avoid_hours": [20, 21, 22, 23],  # Avoid overnight gaps
        "max_spread_percent": 0.15,        # Percentage-based spread limit
        "volatility_adjustment": 0.8,      # Reduce size for volatility
    }
    
    # Indices (US30, NAS100, SPX500)
    INDICES = {
        "stop_loss_multiplier": 5.0,      # Widest stops for indices
        "take_profit_multiplier": 15.0,   # Highest targets
        "min_atr_filter": 50.0,           # Index-specific ATR
        "session_preference": ["newyork"], # Focus on US session
        "avoid_hours": list(range(0, 14)), # Only trade US hours
        "max_spread_points": 5.0,          # Points-based spread
        "gap_risk_filter": True,           # Check for gap risk
        "earnings_season_adjustment": 0.5, # Reduce during earnings
    }


@dataclass
class PositionTradingIndicators:
    """Technical indicators optimized for position trading"""
    
    # Trend indicators (Daily timeframe)
    TREND_INDICATORS = [
        ("EMA", [20, 50, 200]),          # Key moving averages
        ("SMA", [50, 200]),              # Simple MAs for major levels
        ("KAMA", [30]),                  # Adaptive moving average
        ("Ichimoku", [9, 26, 52, 26]),  # Full Ichimoku system
        ("ADX", [14]),                   # Trend strength
        ("Aroon", [25]),                 # Trend changes
    ]
    
    # Momentum indicators (Daily timeframe)
    MOMENTUM_INDICATORS = [
        ("RSI", [14, 21]),               # Multiple RSI periods
        ("MACD", [12, 26, 9]),          # Standard MACD
        ("Stochastic", [14, 3, 3]),     # Slow stochastic
        ("CCI", [20]),                   # Commodity Channel Index
        ("ROC", [10, 20]),              # Rate of Change
        ("TSI", [25, 13]),              # True Strength Index
    ]
    
    # Volatility indicators
    VOLATILITY_INDICATORS = [
        ("ATR", [14, 20]),              # Multiple ATR periods
        ("Bollinger Bands", [20, 2]),   # Standard BB
        ("Keltner Channels", [20, 2]),  # KC for volatility
        ("Donchian Channels", [20]),    # Price channels
        ("Volatility Ratio", [14]),     # Volatility assessment
    ]
    
    # Volume indicators (where applicable)
    VOLUME_INDICATORS = [
        ("OBV", []),                    # On Balance Volume
        ("Chaikin Money Flow", [20]),   # Money flow
        ("Volume Profile", []),         # Volume at price levels
        ("VWAP", []),                  # Volume Weighted Average Price
    ]


@dataclass
class PositionEntryFilters:
    """Strict entry filters for position trades"""
    
    # Market structure requirements
    STRUCTURE_FILTERS = {
        "clear_trend": True,             # Must have clear D1 trend
        "trend_alignment": True,         # D1 and W1 must align
        "pullback_required": True,       # Enter on pullbacks only
        "support_resistance": True,      # Must be near key levels
        "no_major_news_24h": True,      # Avoid major news events
    }
    
    # Technical confirmations required
    CONFIRMATIONS_REQUIRED = 3          # Minimum confirmations
    CONFLUENCE_SCORE_MIN = 70          # Minimum confluence score
    
    # Fundamental filters
    FUNDAMENTAL_ALIGNMENT = True        # Fundamentals must support
    SENTIMENT_CONSIDERATION = True      # Consider market sentiment
    INTERMARKET_CHECK = True           # Check correlated markets


@dataclass
class PositionExitRules:
    """Exit management for position trades"""
    
    # Profit taking rules
    PARTIAL_PROFIT_LEVELS = [
        (0.5, 1.5),    # Take 50% at 1.5R
        (0.3, 3.0),    # Take 30% at 3R
        (0.2, 4.0),    # Final 20% at 4R
    ]
    
    # Trailing stop rules
    TRAILING_STOP_ACTIVATION = 2.0     # Activate at 2R profit
    TRAILING_STOP_DISTANCE = 1.5       # Trail by 1.5x ATR
    
    # Time-based exits
    TIME_STOP_DAYS = 21               # Exit if no progress in 21 days
    WEEKEND_PROTECTION = True         # Reduce before weekends
    MONTH_END_REDUCTION = True        # Reduce at month end
    
    # Invalidation rules
    INVALIDATION_SIGNALS = [
        "trend_change",               # Major trend reversal
        "support_break",              # Key support broken
        "volume_divergence",          # Volume not confirming
        "momentum_loss",              # Momentum indicators failing
        "correlation_break",          # Correlations breaking down
    ]


class PositionTradingConfig:
    """Main configuration class for position trading"""
    
    # Timeframe settings
    timeframes = PositionTradingTimeframes()
    
    # Risk parameters
    risk = PositionTradingRiskParameters()
    
    # Frequency controls
    frequency = PositionTradingFrequency()
    
    # Market-specific settings
    markets = MarketSpecificAdjustments()
    
    # Technical indicators
    indicators = PositionTradingIndicators()
    
    # Entry filters
    entry = PositionEntryFilters()
    
    # Exit rules
    exit = PositionExitRules()
    
    @classmethod
    def get_market_config(cls, symbol: str) -> Dict:
        """Get market-specific configuration for a symbol"""
        symbol_upper = symbol.upper()
        
        # Forex pairs
        forex_majors = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
        forex_crosses = ["EURJPY", "GBPJPY", "EURGBP", "EURAUD", "GBPAUD", "AUDNZD"]
        
        # Commodities
        commodities = ["GOLD", "XAUUSD", "SILVER", "XAGUSD", "OIL", "WTI", "BRENT"]
        
        # Indices
        indices = ["US30", "DJI", "NAS100", "NDX", "SPX500", "SP500", "DAX", "FTSE"]
        
        if symbol_upper in forex_majors:
            return cls.markets.FOREX_MAJORS
        elif symbol_upper in forex_crosses:
            return cls.markets.FOREX_CROSSES
        elif symbol_upper in commodities:
            return cls.markets.COMMODITIES
        elif symbol_upper in indices:
            return cls.markets.INDICES
        else:
            # Default to forex majors config
            return cls.markets.FOREX_MAJORS
    
    @classmethod
    def calculate_position_size(cls, account_balance: float, risk_percent: float, 
                              stop_loss_pips: float, symbol: str) -> float:
        """Calculate position size based on risk parameters"""
        risk_amount = account_balance * (risk_percent / 100)
        
        # Get market-specific adjustments
        market_config = cls.get_market_config(symbol)
        
        # Apply volatility adjustment if available
        if "volatility_adjustment" in market_config:
            risk_amount *= market_config["volatility_adjustment"]
        
        # Basic position size calculation (simplified)
        # In production, this would consider pip values, contract sizes, etc.
        position_size = risk_amount / stop_loss_pips
        
        return round(position_size, 2)
    
    @classmethod
    def validate_trade_timing(cls, last_trade_time: Optional[float], 
                            symbol: str, existing_positions: List[str]) -> Tuple[bool, str]:
        """Validate if enough time has passed for a new trade"""
        import time
        
        current_time = time.time()
        
        # Check minimum time between any trades
        if last_trade_time:
            hours_passed = (current_time - last_trade_time) / 3600
            if hours_passed < cls.frequency.MIN_HOURS_BETWEEN_TRADES:
                return False, f"Only {hours_passed:.1f} hours since last trade (need {cls.frequency.MIN_HOURS_BETWEEN_TRADES})"
        
        # Check if we already have a position in this symbol
        if symbol in existing_positions:
            return False, f"Already have a position in {symbol}"
        
        # Check maximum concurrent positions
        if len(existing_positions) >= cls.frequency.MAX_CONCURRENT_POSITIONS:
            return False, f"Maximum {cls.frequency.MAX_CONCURRENT_POSITIONS} positions reached"
        
        return True, "Trade timing validated"
    
    @classmethod
    def get_session_filter(cls, symbol: str, current_hour: int) -> bool:
        """Check if current hour is suitable for trading this symbol"""
        market_config = cls.get_market_config(symbol)
        
        # Check if we should avoid this hour
        if "avoid_hours" in market_config:
            if current_hour in market_config["avoid_hours"]:
                return False
        
        return True


# Integration with existing system
def integrate_position_trading_config(settings):
    """
    Integrate position trading configuration with existing settings
    
    Usage in trading_loop.py or settings.py:
    
    from config.position_trading_config import integrate_position_trading_config
    settings = integrate_position_trading_config(settings)
    """
    
    config = PositionTradingConfig()
    
    # Update timeframe settings
    settings.trading.entry_timeframe = config.timeframes.ENTRY_TIMEFRAME
    settings.trading.background_timeframe = config.timeframes.BACKGROUND_TIMEFRAME
    
    # Update risk settings
    settings.trading.stop_loss_atr_multiplier = config.risk.STOP_LOSS_ATR_MULTIPLIER_MIN
    settings.trading.take_profit_atr_multiplier = config.risk.TAKE_PROFIT_ATR_MULTIPLIER_MIN
    settings.trading.min_risk_reward_ratio = config.risk.MIN_RISK_REWARD_RATIO
    
    # Update frequency settings
    settings.trading.max_open_trades = config.frequency.MAX_CONCURRENT_POSITIONS
    settings.trading.min_confidence = config.frequency.MIN_CONFIDENCE_SCORE
    
    # Update council settings for position trading
    settings.trading.council_debate_rounds = 3  # More thorough analysis
    settings.trading.council_quick_mode = False  # Full analysis for position trades
    
    return settings


# Environment variable configuration for easy deployment
POSITION_TRADING_ENV_SETTINGS = """
# Position Trading Configuration (Phase 2.2)
# Add these to your .env file

# Timeframes
TRADING_ENTRY_TIMEFRAME=D1
TRADING_BACKGROUND_TIMEFRAME=W1
TRADING_CONFIRMATION_TIMEFRAME=H4

# Risk Management
TRADING_STOP_LOSS_ATR_MULTIPLIER=3.5
TRADING_TAKE_PROFIT_ATR_MULTIPLIER=12.0
TRADING_MIN_RISK_REWARD_RATIO=3.0
TRADING_RISK_PER_TRADE_PERCENT=1.0

# Trading Frequency
TRADING_MAX_OPEN_TRADES=4
TRADING_MIN_CONFIDENCE=70.0
TRADING_MIN_HOURS_BETWEEN_TRADES=24
TRADING_COUNCIL_DEBATE_ROUNDS=3
TRADING_COUNCIL_QUICK_MODE=false

# Position Management
TRADING_MIN_HOLDING_DAYS=3
TRADING_TARGET_HOLDING_DAYS=10
TRADING_PARTIAL_PROFITS_ENABLED=true
TRADING_TRAILING_STOP_ENABLED=true

# Market Specific
TRADING_VOLATILITY_FILTER_ENABLED=true
TRADING_SESSION_FILTER_ENABLED=true
TRADING_NEWS_FILTER_ENABLED=true
"""