"""
Test the pre-trade filtering system
"""

import asyncio
import logging
from datetime import datetime, timedelta
from core.services.pre_trade_filter import PreTradeFilter
from core.domain.models import MarketData, Candle
from config.settings import TradingSettings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_candle(
    close: float, 
    volume: float = 1000, 
    spread: float = 0.00002,
    atr: float = 0.0005,
    rsi: float = 50
) -> Candle:
    """Create a test candle"""
    return Candle(
        timestamp=datetime.now(),
        open=close - 0.0001,
        high=close + 0.0002,
        low=close - 0.0002,
        close=close,
        volume=volume,
        spread=spread,
        ema_50=close - 0.0005,
        ema_200=close - 0.001,
        rsi=rsi,
        atr=atr,
        rsi_slope=0.0
    )


def create_market_data(
    symbol: str = "EURUSD",
    price: float = 1.1000,
    spread: float = 0.00002,
    atr: float = 0.0005,
    volume: float = 1000,
    trending: bool = True
) -> MarketData:
    """Create test market data with various conditions"""
    h1_candles = []
    
    for i in range(20):
        if trending:
            # Create trending market
            candle_price = price + (i - 10) * 0.0002
            candle_volume = volume + (i * 50)
            candle_rsi = 50 + (i - 10)
        else:
            # Create ranging market
            candle_price = price + (0.0001 if i % 2 == 0 else -0.0001)
            candle_volume = volume
            candle_rsi = 50 + (5 if i % 2 == 0 else -5)
        
        candle = create_test_candle(
            close=candle_price,
            volume=candle_volume,
            spread=spread,
            atr=atr,
            rsi=candle_rsi
        )
        
        # Adjust EMAs for trending market
        if trending:
            candle.ema_50 = candle_price - 0.0003
            candle.ema_200 = candle_price - 0.0006
        
        h1_candles.append(candle)
    
    return MarketData(
        symbol=symbol,
        h1_candles=h1_candles,
        h4_candles=[],
        h1_screenshot_path="",
        h4_screenshot_path=""
    )


async def test_pre_filter():
    """Test pre-trade filter functionality"""
    logger.info("Starting pre-trade filter tests...")
    
    # Initialize filter
    trading_config = TradingSettings()
    pre_filter = PreTradeFilter(trading_config)
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Good trading conditions',
            'data': create_market_data(spread=0.00002, atr=0.0005, volume=1500, trending=True),
            'expected': True
        },
        {
            'name': 'Wide spread',
            'data': create_market_data(spread=0.0003, atr=0.0005),  # 60% of ATR
            'expected': False
        },
        {
            'name': 'Low volatility',
            'data': create_market_data(atr=0.0001),  # Below threshold
            'expected': False
        },
        {
            'name': 'Low volume',
            'data': create_market_data(volume=200),  # 20% of normal
            'expected': False
        },
        {
            'name': 'Choppy market',
            'data': create_market_data(trending=False),
            'expected': True  # May pass but with lower score
        }
    ]
    
    # Run tests
    for scenario in scenarios:
        logger.info(f"\n=== Testing: {scenario['name']} ===")
        
        result = await pre_filter.should_analyze(scenario['data'])
        
        logger.info(f"Result: {'PASS' if result.should_analyze else 'FAIL'}")
        logger.info(f"Reason: {result.reason}")
        logger.info(f"Quality score: {result.details.get('quality_score', 0):.2f}")
        
        if result.details.get('filter_scores'):
            logger.info("Filter scores:")
            for filter_name, score in result.details['filter_scores'].items():
                logger.info(f"  {filter_name}: {score:.2f}")
        
        if result.details.get('failed_checks'):
            logger.info(f"Failed checks: {result.details['failed_checks']}")
        
        # Check expectation
        if result.should_analyze == scenario['expected']:
            logger.info("✅ Test passed")
        else:
            logger.error(f"❌ Test failed - expected {scenario['expected']}")
    
    # Test with news events
    logger.info("\n=== Testing: News blackout ===")
    
    news_events = [
        {
            'timestamp': datetime.now() + timedelta(minutes=15),
            'impact': 'high',
            'title': 'NFP Release'
        }
    ]
    
    good_market = create_market_data()
    result = await pre_filter.should_analyze(good_market, news_events)
    
    logger.info(f"Result with imminent news: {'PASS' if result.should_analyze else 'FAIL'}")
    logger.info(f"Reason: {result.reason}")
    
    # Get statistics
    logger.info("\n=== Filter Statistics ===")
    stats = pre_filter.get_stats()
    logger.info(f"Total analyzed: {stats['total_analyzed']}")
    logger.info(f"Pass rate: {stats['pass_rate']}")
    logger.info(f"Filter reasons: {stats['filter_reasons']}")
    
    logger.info("\n✅ All pre-filter tests completed!")


if __name__ == "__main__":
    asyncio.run(test_pre_filter())