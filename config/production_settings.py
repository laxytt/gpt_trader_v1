"""
Production-ready settings for the trading system
Includes comprehensive rate limiting and monitoring
"""

# OpenAI Rate Limiting Configuration
OPENAI_RATE_LIMITS = {
    # Tier 1 ($5/month) limits
    "tier_1": {
        "gpt-4": {"rpm": 500, "tpm": 10000, "rpd": 10000},
        "gpt-4-turbo": {"rpm": 500, "tpm": 30000, "rpd": 10000},
        "gpt-3.5-turbo": {"rpm": 3500, "tpm": 60000, "rpd": 10000},
        "gpt-4o-mini": {"rpm": 500, "tpm": 30000, "rpd": 10000},
    },
    # Tier 2 ($50/month) limits
    "tier_2": {
        "gpt-4": {"rpm": 5000, "tpm": 40000, "rpd": None},
        "gpt-4-turbo": {"rpm": 5000, "tpm": 80000, "rpd": None},
        "gpt-3.5-turbo": {"rpm": 10000, "tpm": 200000, "rpd": None},
        "gpt-4o-mini": {"rpm": 5000, "tpm": 80000, "rpd": None},
    },
    # Tier 3 ($100/month) limits
    "tier_3": {
        "gpt-4": {"rpm": 10000, "tpm": 150000, "rpd": None},
        "gpt-4-turbo": {"rpm": 10000, "tpm": 300000, "rpd": None},
        "gpt-3.5-turbo": {"rpm": 10000, "tpm": 1000000, "rpd": None},
        "gpt-4o-mini": {"rpm": 10000, "tpm": 300000, "rpd": None},
    }
}

# Production Trading Council Configuration
PRODUCTION_COUNCIL_CONFIG = {
    # Rate limiting
    "agent_delay_seconds": 0.75,  # 750ms between agent calls
    "symbol_delay_seconds": 2.0,   # 2s between symbols
    "retry_delay_seconds": 3.0,    # 3s before retry on rate limit
    
    # API optimization
    "max_concurrent_agents": 2,    # Max agents calling API at once
    "batch_symbols": False,        # Process symbols one at a time
    
    # Council settings
    "quick_mode": True,            # Skip debates to reduce API calls
    "debate_rounds": 1,            # Minimal debates if quick_mode is False
    "min_confidence": 50.0,        # Lower threshold for production
    
    # Monitoring
    "log_gpt_payloads": True,      # Log all GPT requests for dashboard
    "track_token_usage": True,     # Track token costs
    "alert_on_rate_limit": True,   # Alert when hitting rate limits
}

# Rate Limiter Configuration
RATE_LIMITER_CONFIG = {
    # Based on your OpenAI tier
    "openai_tier": "tier_1",  # Change to your tier: tier_1, tier_2, tier_3
    
    # Safety margins (use % of actual limits)
    "safety_margin": 0.8,  # Use only 80% of rate limits
    
    # Burst control
    "max_burst_requests": 10,  # Max requests in a burst
    "burst_window_seconds": 10,  # Time window for burst
    
    # Backoff strategy
    "initial_backoff": 1.0,
    "max_backoff": 60.0,
    "backoff_multiplier": 2.0,
}

# GPT Request Logging for Dashboard
GPT_LOGGING_CONFIG = {
    "log_requests": True,
    "log_responses": True,
    "log_tokens": True,
    "log_costs": True,
    "max_payload_length": 5000,  # Truncate long payloads
    "store_in_database": True,    # Store in SQLite for dashboard
}

# Production Environment Settings
PRODUCTION_ENV = """
# Add these to your .env file for production

# OpenAI Configuration
OPENAI_TIER=tier_1  # Your OpenAI tier
GPT_MODEL=gpt-4o-mini  # Most cost-effective model
GPT_TEMPERATURE=0.1  # Low temperature for consistency

# Trading Council Production Settings
TRADING_COUNCIL_AGENT_DELAY=0.75  # Delay between agents
TRADING_COUNCIL_QUICK_MODE=true  # Reduce API calls
TRADING_COUNCIL_DEBATE_ROUNDS=1  # Minimal debates
TRADING_COUNCIL_MIN_CONFIDENCE=50.0  # Production threshold

# Rate Limiting
ENABLE_RATE_LIMITING=true
RATE_LIMIT_SAFETY_MARGIN=0.8
MAX_RETRIES=3

# Monitoring
LOG_GPT_REQUESTS=true
TRACK_TOKEN_COSTS=true
ALERT_ON_ERRORS=true

# Symbol Management
TRADING_SYMBOLS=EURUSD,GBPUSD  # Start with 2 symbols
SYMBOL_PROCESSING_DELAY=2.0  # Delay between symbols

# Error Handling
CIRCUIT_BREAKER_ENABLED=true
CIRCUIT_BREAKER_THRESHOLD=5
CIRCUIT_BREAKER_TIMEOUT=300  # 5 minutes
"""