"""
GPT signal generator for creating trading signals from market data.
Handles data preparation, prompt construction, and signal validation.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import json
from pathlib import Path

from core.infrastructure.gpt.client import GPTClient
from core.domain.models import (
    TradingSignal, MarketData, NewsEvent, MarketContext, 
    SignalType, RiskClass
)
from core.domain.exceptions import (
    SignalGenerationError, GPTResponseError, ValidationError,
    ErrorContext
)
from core.utils.validation import SignalValidator
from core.utils.chart_utils import encode_image_as_b64


logger = logging.getLogger(__name__)


class SignalDataPreparator:
    """Prepares market data and context for GPT signal generation"""
    
    def __init__(self):
        self.max_candles_for_gpt = 20  # Limit candles sent to GPT for cost control
    
    def prepare_signal_data(
        self,
        h1_data: MarketData,
        h4_data: MarketData,
        news_events: List[NewsEvent],
        market_context: MarketContext,
        chart_paths: Optional[Dict[str, str]] = None,
        historical_cases: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Prepare all data needed for signal generation.
        
        Args:
            h1_data: H1 timeframe market data
            h4_data: H4 timeframe market data  
            news_events: Upcoming news events
            market_context: Current market context
            chart_paths: Optional paths to chart images
            historical_cases: Optional historical trade cases
            
        Returns:
            Dictionary with prepared data for GPT
        """
        # Limit candles for cost efficiency
        h1_candles = h1_data.candles[-self.max_candles_for_gpt:] if h1_data.candles else []
        h4_candles = h4_data.candles[-self.max_candles_for_gpt:] if h4_data.candles else []
        
        prepared_data = {
            'symbol': h1_data.symbol,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            
            # Market data
            'history_h1': [candle.to_dict() for candle in h1_candles],
            'history_h4': [candle.to_dict() for candle in h4_candles],
            
            # Market context
            'market_context': market_context.to_dict(),
            
            # News events (next 2 hours)
            'upcoming_news': [
                {
                    'timestamp': event.timestamp.isoformat(),
                    'country': event.country,
                    'title': event.title,
                    'impact': event.impact
                }
                for event in news_events[:10]  # Limit to 10 most relevant
            ],
            
            # Historical context
            'historical_cases': historical_cases or []
        }
        
        return prepared_data
    
    def prepare_chart_images(self, chart_paths: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Prepare chart images for GPT vision analysis.
        
        Args:
            chart_paths: Dictionary mapping timeframe to chart path
            
        Returns:
            List of image content dictionaries
        """
        images = []
        
        for timeframe, path in chart_paths.items():
            if path and Path(path).exists():
                try:
                    encoded_image = encode_image_as_b64(path)
                    if encoded_image:
                        images.append(encoded_image)
                        logger.debug(f"Added {timeframe} chart image to analysis")
                except Exception as e:
                    logger.warning(f"Failed to encode chart {path}: {e}")
        
        return images


class GPTSignalGenerator:
    """
    Generates trading signals using GPT analysis of market data and charts.
    """
    
    def __init__(self, gpt_client: GPTClient, prompts_dir: str = "config/prompts"):
        self.gpt_client = gpt_client
        self.data_preparator = SignalDataPreparator()
        self.validator = SignalValidator()
        
        # Load system prompt
        self.system_prompt = self._load_system_prompt(prompts_dir)
        
        # Required keys in GPT response
        self.required_signal_keys = [
            'symbol', 'signal', 'entry', 'sl', 'tp', 'rr', 'risk_class', 'reason'
        ]
    
    def _load_system_prompt(self, prompts_dir: str) -> str:
        """Load the system prompt template from file"""
        try:
            prompt_path = Path(prompts_dir) / "system_prompt.txt"
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"System prompt file not found: {prompt_path}")
            raise SignalGenerationError("System prompt file not found")
        except Exception as e:
            logger.error(f"Error loading system prompt: {e}")
            raise SignalGenerationError(f"Failed to load system prompt: {str(e)}")
    
    async def generate_signal(
        self,
        h1_data: MarketData,
        h4_data: MarketData,
        news_events: List[NewsEvent],
        market_context: MarketContext,
        chart_paths: Optional[Dict[str, str]] = None,
        historical_cases: Optional[List[Dict[str, Any]]] = None
    ) -> TradingSignal:
        """
        Generate a trading signal using GPT analysis.
        
        Args:
            h1_data: H1 timeframe market data
            h4_data: H4 timeframe market data
            news_events: Upcoming news events
            market_context: Current market context
            chart_paths: Optional chart image paths
            historical_cases: Optional historical cases for context
            
        Returns:
            TradingSignal object
            
        Raises:
            SignalGenerationError: If signal generation fails
        """
        symbol = h1_data.symbol
        
        with ErrorContext("Signal generation", symbol=symbol) as ctx:
            ctx.add_detail("h1_candles", len(h1_data.candles))
            ctx.add_detail("h4_candles", len(h4_data.candles))
            ctx.add_detail("news_events", len(news_events))
            
            # Prepare data for GPT
            signal_data = self.data_preparator.prepare_signal_data(
                h1_data=h1_data,
                h4_data=h4_data,
                news_events=news_events,
                market_context=market_context,
                chart_paths=chart_paths,
                historical_cases=historical_cases
            )
            
            # Prepare chart images if available
            chart_images = []
            if chart_paths:
                chart_images = self.data_preparator.prepare_chart_images(chart_paths)
            
            # Build GPT messages
            messages = self._build_gpt_messages(signal_data, chart_images, symbol)
            
            # Call GPT API
            gpt_response = await self.gpt_client.chat_completion(
                messages=messages,
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=1000   # Sufficient for signal response
            )
            
            # Parse and validate response
            signal_dict = self.gpt_client.parse_json_response(gpt_response['content'])
            self.gpt_client.validate_response_schema(signal_dict, self.required_signal_keys)
            
            # Convert to TradingSignal object
            trading_signal = self._create_trading_signal(
                signal_dict, 
                symbol, 
                news_events,
                market_context
            )
            
            # Validate signal
            self.validator.validate_signal(trading_signal)
            
            logger.info(f"Signal generated for {symbol}: {trading_signal.signal.value} "
                       f"({trading_signal.risk_class.value})")
            
            return trading_signal
    
    def _build_gpt_messages(
        self, 
        signal_data: Dict[str, Any], 
        chart_images: List[Dict[str, Any]],
        symbol: str
    ) -> List[Dict[str, Any]]:
        """Build messages for GPT API call"""
        
        # System message with symbol-specific prompt
        system_message = {
            "role": "system",
            "content": self.system_prompt.format(symbol=symbol)
        }
        
        # User message content (multimodal with text and images)
        user_content = []
        
        # Add explanatory text
        user_content.append({
            "type": "text",
            "text": f"Analyze {symbol} for trading signals using the provided H1/H4 data, charts, and context:"
        })
        
        # Add chart images
        user_content.extend(chart_images)
        
        # Add market context text
        market_context = signal_data['market_context']
        context_text = (
            f"Market Context:\n"
            f"- Session: {market_context['session']}\n"
            f"- Volatility: {market_context['volatility']}\n"
            f"- Win streak: {market_context['win_streak_type']} ({market_context['win_streak_length']})\n"
            f"- Win rate: {market_context['win_rate']:.1%}\n\n"
        )
        
        user_content.append({
            "type": "text", 
            "text": context_text
        })
        
        # Add JSON data
        json_data = json.dumps(signal_data, indent=2, default=str)
        user_content.append({
            "type": "text",
            "text": f"Market Data:\n```json\n{json_data}\n```"
        })
        
        user_message = {
            "role": "user",
            "content": user_content
        }
        
        return [system_message, user_message]
    
    def _create_trading_signal(
        self,
        signal_dict: Dict[str, Any],
        symbol: str,
        news_events: List[NewsEvent],
        market_context: MarketContext
    ) -> TradingSignal:
        """Convert GPT response dictionary to TradingSignal object"""
        
        try:
            # Parse signal type
            signal_str = signal_dict['signal'].upper()
            signal_type = SignalType(signal_str)
            
            # Parse risk class
            risk_class_str = signal_dict['risk_class'].upper()
            risk_class = RiskClass(risk_class_str)
            
            # Create signal object
            trading_signal = TradingSignal(
                symbol=symbol,
                signal=signal_type,
                reason=signal_dict['reason'],
                risk_class=risk_class,
                timestamp=datetime.now(timezone.utc),
                
                # Price levels (None for WAIT signals)
                entry=float(signal_dict['entry']) if signal_dict['entry'] else None,
                stop_loss=float(signal_dict['sl']) if signal_dict['sl'] else None,
                take_profit=float(signal_dict['tp']) if signal_dict['tp'] else None,
                risk_reward=float(signal_dict['rr']) if signal_dict['rr'] else None,
                
                # Context
                market_context=market_context.to_dict(),
                news_events=news_events
            )
            
            return trading_signal
            
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Error creating TradingSignal from GPT response: {e}")
            logger.debug(f"Signal dict: {signal_dict}")
            raise SignalGenerationError(f"Invalid GPT signal response: {str(e)}")
    
    def create_fallback_signal(
        self, 
        symbol: str, 
        reason: str,
        news_events: Optional[List[NewsEvent]] = None,
        market_context: Optional[MarketContext] = None
    ) -> TradingSignal:
        """
        Create a fallback WAIT signal when generation fails.
        
        Args:
            symbol: Trading symbol
            reason: Reason for fallback signal
            news_events: Optional news events
            market_context: Optional market context
            
        Returns:
            WAIT signal
        """
        return TradingSignal(
            symbol=symbol,
            signal=SignalType.WAIT,
            reason=f"Fallback signal: {reason}",
            risk_class=RiskClass.C,
            timestamp=datetime.now(timezone.utc),
            news_events=news_events or [],
            market_context=market_context.to_dict() if market_context else {}
        )


# Export main class
__all__ = ['GPTSignalGenerator', 'SignalDataPreparator']