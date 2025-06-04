"""
Test script for the Trading Council multi-agent system
"""

import asyncio
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Disable some verbose loggers
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)


async def test_council():
    """Test the Trading Council signal generation"""
    
    logger.info("Starting Trading Council test...")
    
    try:
        # Import dependencies
        from config.settings import get_settings
        from trading_loop import DependencyContainer
        from core.domain.enums import TimeFrame
        
        # Initialize settings
        settings = get_settings()
        
        # Create dependency container
        container = DependencyContainer(settings)
        
        # Get services
        mt5_client = container.mt5_client()
        signal_service = container.signal_service()
        
        # Connect to MT5
        if not await mt5_client.connect():
            logger.error("Failed to connect to MT5")
            return
        
        logger.info("Connected to MT5 successfully")
        
        # Test symbol
        test_symbol = "EURUSD"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing Trading Council for {test_symbol}")
        logger.info(f"{'='*60}\n")
        
        # Generate signal using Trading Council
        logger.info("Convening the Trading Council...")
        signal = await signal_service.generate_signal(test_symbol)
        
        # Display results
        logger.info(f"\n{'='*50}")
        logger.info("COUNCIL DECISION")
        logger.info(f"{'='*50}")
        logger.info(f"Symbol: {signal.symbol}")
        logger.info(f"Decision: {signal.signal.value}")
        logger.info(f"Confidence: {signal.confidence}%")
        logger.info(f"Risk Class: {signal.risk_class.value}")
        
        if signal.entry_price:
            logger.info(f"Entry Price: {signal.entry_price}")
        if signal.stop_loss:
            logger.info(f"Stop Loss: {signal.stop_loss}")
        if signal.take_profit:
            logger.info(f"Take Profit: {signal.take_profit}")
        
        if signal.rationale:
            logger.info(f"\nRationale: {signal.rationale}")
        elif signal.reason:
            logger.info(f"\nReason: {signal.reason}")
        
        # Display council metadata if available
        if signal.metadata:
            logger.info(f"\n{'='*50}")
            logger.info("COUNCIL METADATA")
            logger.info(f"{'='*50}")
            
            if 'llm_confidence' in signal.metadata:
                logger.info(f"LLM Confidence: {signal.metadata['llm_confidence']:.1f}%")
            if 'ml_confidence' in signal.metadata:
                logger.info(f"ML Confidence: {signal.metadata['ml_confidence']:.1f}%")
            if 'consensus_level' in signal.metadata:
                logger.info(f"Consensus Level: {signal.metadata['consensus_level']:.1f}%")
            if 'dissent_count' in signal.metadata:
                logger.info(f"Dissenting Views: {signal.metadata['dissent_count']}")
            
            if 'agent_votes' in signal.metadata:
                logger.info("\nAgent Votes:")
                for agent, vote in signal.metadata['agent_votes'].items():
                    logger.info(f"  {agent}: {vote}")
            
            if 'validation_score' in signal.metadata:
                logger.info(f"\nValidation Score: {signal.metadata['validation_score']:.2f}")
        
        # Get council history
        logger.info(f"\n{'='*50}")
        logger.info("COUNCIL ANALYSIS SUMMARY")
        logger.info(f"{'='*50}")
        
        history = await signal_service.get_council_history(limit=1)
        if history:
            latest = history[0]
            logger.info(f"Decision made at: {latest['timestamp']}")
            logger.info(f"Agent consensus breakdown:")
            for agent, vote in latest['agent_votes'].items():
                logger.info(f"  {agent}: {vote}")
        
        logger.info(f"\n{'='*60}")
        logger.info("Trading Council test completed successfully!")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Error during council test: {str(e)}", exc_info=True)
    
    finally:
        # Disconnect from MT5
        try:
            await mt5_client.disconnect()
            logger.info("Disconnected from MT5")
        except:
            pass


if __name__ == "__main__":
    asyncio.run(test_council())