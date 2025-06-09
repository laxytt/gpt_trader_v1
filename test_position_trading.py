#!/usr/bin/env python3
"""
Test script for position trading mode
Tests the updated agents with daily/weekly timeframes
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict

from core.domain.models import MarketData, Candle, SignalType
from core.agents.technical_analyst import TechnicalAnalyst
from core.agents.momentum_trader import MomentumTrader
from core.agents.risk_manager import RiskManager
from core.infrastructure.gpt.client import GPTClient
from config.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_mock_daily_data() -> MarketData:
    """Create mock daily timeframe data for testing"""
    candles = []
    base_price = 1.2000
    
    # Create 200 daily candles with an uptrend
    for i in range(200):
        # Add some trend and volatility
        trend = i * 0.0001  # Gradual uptrend
        noise = (i % 10 - 5) * 0.0005  # Some volatility
        
        open_price = base_price + trend + noise
        close_price = open_price + (0.0010 if i % 3 == 0 else -0.0005)  # More up days
        high_price = max(open_price, close_price) + 0.0005
        low_price = min(open_price, close_price) - 0.0005
        
        candle = Candle(
            time=datetime.now(),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=1000000 + i * 1000,
            spread=0.00010,
            ema50=base_price + (i - 25) * 0.0001 if i > 50 else base_price,
            ema200=base_price + (i - 100) * 0.00008 if i > 200 else base_price,
            rsi14=50 + (i % 20),  # Oscillating RSI
            atr14=0.0015  # ~15 pips daily ATR
        )
        candles.append(candle)
        base_price = close_price
    
    return MarketData(
        symbol="EURUSD",
        timeframe="D1",
        candles=candles,
        latest_candle=candles[-1],
        current_time=datetime.now()
    )


def create_mock_weekly_data() -> MarketData:
    """Create mock weekly timeframe data for testing"""
    candles = []
    base_price = 1.1800
    
    # Create 52 weekly candles (1 year)
    for i in range(52):
        trend = i * 0.0020  # Stronger weekly trend
        noise = (i % 4 - 2) * 0.0020
        
        open_price = base_price + trend + noise
        close_price = open_price + (0.0050 if i % 3 == 0 else -0.0020)
        high_price = max(open_price, close_price) + 0.0030
        low_price = min(open_price, close_price) - 0.0030
        
        candle = Candle(
            time=datetime.now(),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=5000000 + i * 10000,
            spread=0.00010,
            ema50=base_price + (i - 25) * 0.0015 if i > 50 else base_price,
            ema200=base_price,  # Not enough data for 200-week EMA
            rsi14=45 + (i % 30),
            atr14=0.0080  # ~80 pips weekly ATR
        )
        candles.append(candle)
        base_price = close_price
    
    return MarketData(
        symbol="EURUSD",
        timeframe="W1",
        candles=candles,
        latest_candle=candles[-1],
        current_time=datetime.now()
    )


def create_mock_h4_data() -> MarketData:
    """Create mock H4 data for refined entry"""
    candles = []
    base_price = 1.2180  # Near daily close
    
    # Create 100 H4 candles
    for i in range(100):
        noise = (i % 6 - 3) * 0.0002
        
        open_price = base_price + noise
        close_price = open_price + (0.0003 if i % 2 == 0 else -0.0002)
        high_price = max(open_price, close_price) + 0.0002
        low_price = min(open_price, close_price) - 0.0002
        
        candle = Candle(
            time=datetime.now(),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=500000,
            spread=0.00010,
            ema50=base_price,
            ema200=base_price - 0.0050,
            rsi14=55,
            atr14=0.0008  # ~8 pips H4 ATR
        )
        candles.append(candle)
        base_price = close_price
    
    return MarketData(
        symbol="EURUSD",
        timeframe="H4",
        candles=candles,
        latest_candle=candles[-1],
        current_time=datetime.now()
    )


async def test_position_trading_agents():
    """Test the agents in position trading mode"""
    logger.info("Testing position trading mode with daily/weekly timeframes")
    
    # Initialize settings and GPT client
    settings = get_settings()
    gpt_client = GPTClient(api_key=settings.openai_api_key)
    
    # Create test data
    daily_data = create_mock_daily_data()
    weekly_data = create_mock_weekly_data()
    h4_data = create_mock_h4_data()
    
    # Position trading market data (D1 + W1)
    position_market_data = {
        'd1': daily_data,
        'w1': weekly_data,
        'h4': h4_data  # For refined entry
    }
    
    # Scalping/day trading market data (H1 + H4)
    scalping_market_data = {
        'h1': h4_data,  # Using H4 as mock H1
        'h4': h4_data
    }
    
    # Initialize agents
    technical_analyst = TechnicalAnalyst(gpt_client)
    momentum_trader = MomentumTrader(gpt_client)
    risk_manager = RiskManager(gpt_client, account_balance=10000)
    
    agents = [
        ("Technical Analyst", technical_analyst),
        ("Momentum Trader", momentum_trader),
        ("Risk Manager", risk_manager)
    ]
    
    # Test position trading mode
    logger.info("\n" + "="*80)
    logger.info("TESTING POSITION TRADING MODE (D1/W1)")
    logger.info("="*80)
    
    for agent_name, agent in agents:
        logger.info(f"\n--- {agent_name} Position Analysis ---")
        try:
            # Check if agent detects position mode
            is_position_mode = agent._is_position_trading_mode(position_market_data)
            logger.info(f"Position mode detected: {is_position_mode}")
            
            # Analyze with position data
            analysis = agent.analyze(position_market_data)
            
            logger.info(f"Recommendation: {analysis.recommendation.value}")
            logger.info(f"Confidence: {analysis.confidence}%")
            logger.info(f"Entry: {analysis.entry_price:.5f}")
            logger.info(f"Stop Loss: {analysis.stop_loss:.5f}")
            logger.info(f"Take Profit: {analysis.take_profit:.5f}")
            logger.info(f"Risk/Reward: {analysis.risk_reward_ratio:.2f}" if analysis.risk_reward_ratio else "Risk/Reward: N/A")
            logger.info(f"Reasoning: {analysis.reasoning}")
            
            # Check position-specific metadata
            if analysis.metadata.get('position_mode'):
                logger.info("Position-specific metadata:")
                for key, value in analysis.metadata.items():
                    if key != 'raw_parsed':
                        logger.info(f"  {key}: {value}")
            
        except Exception as e:
            logger.error(f"Error analyzing with {agent_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Test standard mode for comparison
    logger.info("\n" + "="*80)
    logger.info("TESTING STANDARD MODE (H1/H4) FOR COMPARISON")
    logger.info("="*80)
    
    for agent_name, agent in agents[:1]:  # Just test technical analyst
        logger.info(f"\n--- {agent_name} Standard Analysis ---")
        try:
            # Check if agent detects position mode
            is_position_mode = agent._is_position_trading_mode(scalping_market_data)
            logger.info(f"Position mode detected: {is_position_mode}")
            
            # Analyze with standard data
            analysis = agent.analyze(scalping_market_data)
            
            logger.info(f"Recommendation: {analysis.recommendation.value}")
            logger.info(f"Confidence: {analysis.confidence}%")
            logger.info(f"Stop Loss Distance: {abs(analysis.stop_loss - analysis.entry_price) * 10000:.1f} pips" if analysis.stop_loss else "No stop loss")
            
        except Exception as e:
            logger.error(f"Error analyzing with {agent_name}: {e}")


async def test_mock_analysis():
    """Test with mock responses (no API calls)"""
    logger.info("\n" + "="*80)
    logger.info("TESTING WITH MOCK RESPONSES")
    logger.info("="*80)
    
    # Create mock data
    daily_data = create_mock_daily_data()
    weekly_data = create_mock_weekly_data()
    
    # Test position mode detection
    from core.agents.technical_analyst import TechnicalAnalyst
    
    # Create a mock GPT client
    class MockGPTClient:
        def analyze_with_response(self, prompt, **kwargs):
            # Return mock position trading response
            return """
TREND: Strong bullish on daily, confirmed by weekly
PATTERN: Ascending triangle forming on daily
KEY_LEVELS: [Critical support: 1.2100, resistance: 1.2250]
SETUP_QUALITY: Excellent
RECOMMENDATION: BUY
CONFIDENCE: 85
ENTRY_STRATEGY: Buy on pullback to 1.2150-1.2160 zone
POSITION_STOP: 1.2050 (3.5x daily ATR)
TARGETS: T1: 1.2250 (5 days), T2: 1.2350 (10 days), T3: 1.2500 (20 days)
HOLD_DURATION: 10-20 days expected
EXIT_TRIGGER: Close below 20-day EMA or 15 days time stop
"""
    
    mock_client = MockGPTClient()
    analyst = TechnicalAnalyst(mock_client)
    
    # Test position mode detection
    position_data = {'d1': daily_data, 'w1': weekly_data}
    standard_data = {'h1': daily_data, 'h4': daily_data}  # Using daily as mock
    
    logger.info(f"Position mode with D1/W1: {analyst._is_position_trading_mode(position_data)}")
    logger.info(f"Position mode with H1/H4: {analyst._is_position_trading_mode(standard_data)}")
    
    # Test analysis
    try:
        analysis = analyst.analyze(position_data)
        logger.info(f"\nPosition Trading Analysis:")
        logger.info(f"Recommendation: {analysis.recommendation}")
        logger.info(f"Confidence: {analysis.confidence}%")
        logger.info(f"Metadata: {analysis.metadata}")
    except Exception as e:
        logger.error(f"Error in mock analysis: {e}")


if __name__ == "__main__":
    # Choose which test to run
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--mock":
        # Run mock test (no API calls)
        asyncio.run(test_mock_analysis())
    else:
        # Run full test (requires API key)
        logger.warning("This will make real API calls. Use --mock for testing without API calls.")
        response = input("Continue? (y/n): ")
        if response.lower() == 'y':
            asyncio.run(test_position_trading_agents())
        else:
            logger.info("Running mock test instead...")
            asyncio.run(test_mock_analysis())