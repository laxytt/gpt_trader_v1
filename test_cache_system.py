"""
Test the intelligent caching system
"""

import asyncio
import logging
from datetime import datetime
from core.infrastructure.cache.market_state_cache import MarketStateCache
from core.domain.models import MarketData, TradingSignal, SignalType, RiskClass
from core.domain.models import Candle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_candle(close: float, volume: float = 1000, rsi: float = 50) -> Candle:
    """Create a test candle"""
    return Candle(
        timestamp=datetime.now(),
        open=close - 0.0001,
        high=close + 0.0002,
        low=close - 0.0002,
        close=close,
        volume=volume,
        spread=0.00002,
        ema_50=close - 0.0005,
        ema_200=close - 0.001,
        rsi=rsi,
        atr=0.0005,
        rsi_slope=0.0
    )


def create_test_market_data(symbol: str, price: float = 1.1000) -> MarketData:
    """Create test market data"""
    # Create 20 candles with slight variations
    h1_candles = []
    for i in range(20):
        candle_price = price + (i - 10) * 0.0001  # Small price variations
        h1_candles.append(create_test_candle(candle_price))
    
    return MarketData(
        symbol=symbol,
        h1_candles=h1_candles,
        h4_candles=[],
        h1_screenshot_path="",
        h4_screenshot_path=""
    )


def create_test_signal(symbol: str, signal_type: SignalType = SignalType.BUY) -> TradingSignal:
    """Create a test trading signal"""
    return TradingSignal(
        symbol=symbol,
        signal=signal_type,
        entry=1.1005,
        sl=1.0995,
        tp=1.1025,
        rr=2.0,
        risk_class=RiskClass.A,
        reason="Test signal",
        timestamp=datetime.now()
    )


async def test_cache_operations():
    """Test cache basic operations"""
    logger.info("Starting cache system tests...")
    
    # Initialize cache
    cache = MarketStateCache(
        similarity_threshold=0.85,
        ttl_minutes=60,
        max_size_mb=100
    )
    
    # Test 1: Cache and retrieve exact match
    logger.info("\n=== Test 1: Exact Match ===")
    market_data = create_test_market_data("EURUSD", 1.1000)
    signal = create_test_signal("EURUSD", SignalType.BUY)
    
    # Cache the decision
    cache.cache_decision(
        market_data=market_data,
        signal=signal,
        council_decision={"test": "data"},
        confidence_score=85.5
    )
    
    # Try to retrieve it
    result = await cache.get_cached_decision(market_data)
    if result:
        cached_signal, cached_council = result
        logger.info(f"✅ Cache hit! Retrieved signal: {cached_signal.signal}")
    else:
        logger.error("❌ Cache miss on exact match")
    
    # Test 2: Similar market conditions
    logger.info("\n=== Test 2: Similar Market Conditions ===")
    similar_market = create_test_market_data("EURUSD", 1.1001)  # Slightly different price
    result = await cache.get_cached_decision(similar_market)
    if result:
        logger.info(f"✅ Found similar cached decision")
    else:
        logger.info("❌ No similar match found (expected behavior)")
    
    # Test 3: Should skip analysis
    logger.info("\n=== Test 3: Skip Analysis Conditions ===")
    
    # Create market with high spread
    high_spread_market = create_test_market_data("EURUSD", 1.1000)
    high_spread_market.h1_candles[-1].spread = 0.01  # Very high spread
    
    should_skip, reason = cache.should_skip_analysis(high_spread_market)
    if should_skip:
        logger.info(f"✅ Correctly identified skip condition: {reason}")
    else:
        logger.error("❌ Failed to identify high spread")
    
    # Test 4: Cache statistics
    logger.info("\n=== Test 4: Cache Statistics ===")
    stats = cache.get_stats()
    logger.info(f"Cache stats: {stats}")
    logger.info(f"Hit rate: {stats['hit_rate']:.1%}")
    logger.info(f"Total requests: {stats['total_requests']}")
    logger.info(f"Cache size: {stats['cache_size']}")
    
    # Test 5: Multiple symbols
    logger.info("\n=== Test 5: Multiple Symbols ===")
    
    # Cache decisions for different symbols
    for symbol in ["GBPUSD", "USDJPY", "EURJPY"]:
        market = create_test_market_data(symbol, 1.3000)
        signal = create_test_signal(symbol, SignalType.SELL)
        cache.cache_decision(market, signal, {}, 75.0)
    
    # Check cache size
    stats = cache.get_stats()
    logger.info(f"Cache now contains {stats['cache_size']} entries")
    
    # Try to retrieve GBPUSD
    gbp_market = create_test_market_data("GBPUSD", 1.3000)
    result = await cache.get_cached_decision(gbp_market)
    if result:
        logger.info("✅ Retrieved GBPUSD from cache")
    else:
        logger.error("❌ Failed to retrieve GBPUSD")
    
    # Test 6: Cache expiration (simulated)
    logger.info("\n=== Test 6: Cache Features ===")
    
    # Generate cache key
    key = cache.get_cache_key(market_data)
    logger.info(f"Cache key generated: {key[:8]}...")
    
    # Test different market conditions
    conditions = [
        ("Trending market", create_test_market_data("EURUSD", 1.1050)),
        ("Ranging market", create_test_market_data("EURUSD", 1.1000)),
        ("High volatility", create_test_market_data("EURUSD", 1.1000))
    ]
    
    for name, market in conditions:
        market.h1_candles[-1].atr = 0.002 if "volatility" in name else 0.0005
        features = cache._extract_key_features(market)
        logger.info(f"{name} features: trend={features.get('trend')}, "
                   f"volatility={features.get('volatility')}")
    
    logger.info("\n✅ All cache tests completed!")


if __name__ == "__main__":
    asyncio.run(test_cache_operations())