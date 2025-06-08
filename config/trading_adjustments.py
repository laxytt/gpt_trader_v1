"""
Recommended adjustments to increase trading signal frequency

Copy these settings to your .env file or update config/settings.py
"""

# 1. LOWER THE CONFIDENCE THRESHOLD (most important)
# Current: 75% - This is very high!
# Recommended for testing:
TRADING_COUNCIL_MIN_CONFIDENCE=65.0  # Was 75.0

# 2. REDUCE OFFLINE VALIDATION THRESHOLD
# This pre-filters opportunities before they reach the council
TRADING_OFFLINE_VALIDATION_THRESHOLD=0.3  # Was 0.5

# 3. INCREASE TRADING HOURS
# Remove some of the hour restrictions
TRADING_START_HOUR=6   # Was 7
TRADING_END_HOUR=23    # Keep same

# 4. DISABLE QUICK MODE FOR BETTER SIGNALS
# Allow council debates for more nuanced decisions
TRADING_USE_COUNCIL=true
TRADING_COUNCIL_QUICK_MODE=false  # Was true
TRADING_COUNCIL_DEBATE_ROUNDS=2   # Was 1

# 5. REDUCE CYCLE INTERVAL FOR MORE FREQUENT CHECKS
TRADING_CYCLE_INTERVAL_MINUTES=30  # Was 60

# 6. ADJUST RISK PARAMETERS
# Slightly more aggressive to catch more opportunities
TRADING_RISK_PER_TRADE_PERCENT=2.0  # Was 1.5
TRADING_MAX_OPEN_TRADES=4           # Was 3

# 7. MARKETAUX SETTINGS
# Reduce sentiment influence if it's too restrictive
MARKETAUX_SENTIMENT_WEIGHT=0.2      # Was 0.3
MARKETAUX_MIN_RELEVANCE_SCORE=0.3   # Lower to get more news

# Alternative: Create a test configuration
# In your trading_loop.py or a test script:
"""
test_settings = {
    "council_min_confidence": 60.0,      # Much lower for testing
    "offline_validation_threshold": 0.2,  # Very permissive
    "council_quick_mode": False,         # Full debates
    "cycle_interval_minutes": 15,        # Check every 15 minutes
}
"""