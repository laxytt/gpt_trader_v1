#!/usr/bin/env python3
"""
Test script to verify daily and weekly timeframe support for position trading.
"""

import asyncio
import logging
from datetime import datetime, timezone

from core.infrastructure.mt5.client import MT5Client
from core.infrastructure.mt5.data_provider import MT5DataProvider
from core.domain.enums import TimeFrame
from core.utils.chart_utils import ChartGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_position_trading_data():
    """Test fetching daily and weekly data for position trading"""
    
    # Initialize MT5 client
    mt5_client = MT5Client()
    if not mt5_client.initialize():
        logger.error("Failed to initialize MT5")
        return
    
    try:
        # Initialize data provider
        chart_generator = ChartGenerator(mt5_client)
        data_provider = MT5DataProvider(mt5_client, chart_generator)
        
        # Test symbols
        test_symbols = ["EURUSD", "GOLD", "US30"]
        
        for symbol in test_symbols:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing position trading data for {symbol}")
            logger.info(f"{'='*60}")
            
            # Test daily timeframe
            logger.info("\n--- Daily Timeframe (D1) ---")
            try:
                daily_data = await data_provider.get_market_data(
                    symbol,
                    TimeFrame.D1,
                    bars=100  # Will be adjusted to minimum for position trading
                )
                
                logger.info(f"Retrieved {len(daily_data.candles)} daily bars")
                
                if daily_data.candles:
                    latest = daily_data.candles[-1]
                    logger.info(f"Latest daily candle:")
                    logger.info(f"  - Timestamp: {latest.timestamp}")
                    logger.info(f"  - OHLC: {latest.open:.5f} / {latest.high:.5f} / {latest.low:.5f} / {latest.close:.5f}")
                    logger.info(f"  - ATR14: {latest.atr14:.5f}" if latest.atr14 else "  - ATR14: N/A")
                    logger.info(f"  - ATR %: {latest.atr_percentage:.2f}%" if latest.atr_percentage else "  - ATR %: N/A")
                    logger.info(f"  - RSI14: {latest.rsi14:.2f}" if latest.rsi14 else "  - RSI14: N/A")
                    logger.info(f"  - EMA50: {latest.ema50:.5f}" if latest.ema50 else "  - EMA50: N/A")
                    logger.info(f"  - SMA200: {latest.sma200:.5f}" if latest.sma200 else "  - SMA200: N/A")
                    
            except Exception as e:
                logger.error(f"Failed to get daily data: {e}")
            
            # Test weekly timeframe
            logger.info("\n--- Weekly Timeframe (W1) ---")
            try:
                weekly_data = await data_provider.get_market_data(
                    symbol,
                    TimeFrame.W1,
                    bars=52  # Will be adjusted to minimum for position trading
                )
                
                logger.info(f"Retrieved {len(weekly_data.candles)} weekly bars")
                
                if weekly_data.candles:
                    latest = weekly_data.candles[-1]
                    logger.info(f"Latest weekly candle:")
                    logger.info(f"  - Timestamp: {latest.timestamp}")
                    logger.info(f"  - OHLC: {latest.open:.5f} / {latest.high:.5f} / {latest.low:.5f} / {latest.close:.5f}")
                    logger.info(f"  - ATR14: {latest.atr14:.5f}" if latest.atr14 else "  - ATR14: N/A")
                    logger.info(f"  - ATR %: {latest.atr_percentage:.2f}%" if latest.atr_percentage else "  - ATR %: N/A")
                    logger.info(f"  - RSI14: {latest.rsi14:.2f}" if latest.rsi14 else "  - RSI14: N/A")
                    
            except Exception as e:
                logger.error(f"Failed to get weekly data: {e}")
            
            # Test position trading data method
            logger.info("\n--- Position Trading Data (Combined) ---")
            try:
                position_data = await data_provider.get_position_trading_data(
                    symbol,
                    include_weekly=True
                )
                
                logger.info(f"Retrieved position trading data:")
                logger.info(f"  - Daily bars: {len(position_data['daily'].candles)}")
                if 'weekly' in position_data:
                    logger.info(f"  - Weekly bars: {len(position_data['weekly'].candles)}")
                
                # Test position sizing calculation
                if 'daily' in position_data:
                    sizing = data_provider.calculate_position_size_metrics(
                        position_data['daily'],
                        account_balance=10000,
                        risk_percentage=1.0
                    )
                    
                    if sizing:
                        logger.info("\nPosition Sizing Metrics:")
                        logger.info(f"  - Current Price: {sizing['current_price']:.5f}")
                        logger.info(f"  - Current ATR: {sizing['current_atr']:.5f}")
                        logger.info(f"  - ATR %: {sizing['atr_percentage']:.2f}%")
                        logger.info(f"  - Suggested Stop Distance: {sizing['suggested_stop_distance']:.5f}")
                        logger.info(f"  - Suggested Lots: {sizing['lots']}")
                        logger.info(f"  - Risk Amount: ${sizing['risk_amount']:.2f}")
                        
            except Exception as e:
                logger.error(f"Failed to get position trading data: {e}")
            
            # Small delay between symbols
            await asyncio.sleep(1)
            
    finally:
        mt5_client.shutdown()
        logger.info("\nMT5 client shutdown complete")


if __name__ == "__main__":
    asyncio.run(test_position_trading_data())