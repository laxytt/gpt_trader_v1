import json
import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from core.debug_utils import pretty_print_json_box, print_section
from core.statistics import get_win_loss_streak
import openai
import pandas as pd
import tiktoken
from core.mt5_data import get_recent_candle_history_and_chart
from core.news_utils import get_upcoming_news
from core.utils import (
    encode_image_as_b64, get_market_session,
    get_volatility_context,
    np_encoder
)
from core.database import save_gpt_signal_decision, memory

logger = logging.getLogger(__name__)

# Configuration constants
GPT_MODEL = "gpt-4.1-2025-04-14"
ENCODING_MODEL = "gpt-4"  # For token estimation
DEFAULT_NEWS_WINDOW_MINUTES = 2880  # 48 hours

class GPTConfig:
    """Configuration for GPT API interactions"""
    MODEL = GPT_MODEL
    ENCODING_FALLBACK = "cl100k_base"
    MAX_RETRIES = 3
    REQUIRED_SIGNAL_KEYS = ["signal", "entry", "sl", "tp", "rr", "risk_class", "reason"]
    VALID_SIGNALS = ("BUY", "SELL", "WAIT")

class DataPreparator:
    """Prepares market data and context for GPT analysis"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.client = openai.OpenAI()
    
    def prepare_signal_data(self, bars_json: int = 20, bars_chart: int = 80) -> Dict[str, Any]:
        """Prepare all data needed for signal generation"""
        logger.info(f"Preparing signal data for {self.symbol}")
        
        # Get market data
        prompt_data = get_recent_candle_history_and_chart(self.symbol, bars_json, bars_chart)
        
        if prompt_data.get("error"):
            return {"error": prompt_data["error"]}
        
        # Add historical context
        self._add_previous_signal(prompt_data)
        
        # Prepare chart images
        images = self._prepare_chart_images(prompt_data)
        
        # Prepare market context
        market_context = self._prepare_market_context(prompt_data)
        
        # Add news and historical cases
        self._add_contextual_data(prompt_data)
        
        return {
            "prompt_data": prompt_data,
            "images": images,
            "market_context": market_context
        }
    
    def _add_previous_signal(self, prompt_data: Dict[str, Any]) -> None:
        """Add previous GPT decision to context"""
        last_signal = memory.get_previous_signal(self.symbol)
        if last_signal:
            prompt_data["previous_gpt_decision"] = last_signal
    
    def _prepare_chart_images(self, prompt_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Prepare chart images for GPT analysis"""
        image_paths = [
            prompt_data.pop("screenshot_h1", None),
            prompt_data.pop("screenshot_h4", None)
        ]
        
        images = []
        for path in image_paths:
            if path:
                try:
                    encoded_image = encode_image_as_b64(path)
                    images.append(encoded_image)
                except Exception as e:
                    logger.warning(f"Failed to encode image {path}: {e}")
        
        return images
    
    def _prepare_market_context(self, prompt_data: Dict[str, Any]) -> str:
        """Prepare market session and volatility context"""
        df_h1 = pd.DataFrame(prompt_data.get("history_h1", []))
        
        now = datetime.now(timezone.utc)
        session = get_market_session(now)
        volatility = get_volatility_context(df_h1)
        streak_info = get_win_loss_streak(self.symbol)
        
        return (
            f"Market session: {session}\n"
            f"Volatility: {volatility}\n"
            f"Win/loss streak: {streak_info['streak_type']} ({streak_info['streak_length']}), "
            f"Win rate: {streak_info['win_rate']:.1%}\n"
        )
    
    def _add_contextual_data(self, prompt_data: Dict[str, Any]) -> None:
        """Add news and historical cases to prompt data"""
        # Add historical similar cases
        df_h1 = pd.DataFrame(prompt_data.get("history_h1", []))
        if not df_h1.empty:
            context_text = df_h1.iloc[-1].to_dict()
            retrieved_cases = memory.query(context_text, self.symbol)
            prompt_data["historical_cases"] = retrieved_cases or []
        else:
            prompt_data["historical_cases"] = []
        
        # Add upcoming news
        prompt_data["upcoming_news"] = get_upcoming_news(
            symbol=self.symbol, 
            within_minutes=DEFAULT_NEWS_WINDOW_MINUTES
        )

class GPTResponseProcessor:
    """Processes and validates GPT responses"""
    
    @staticmethod
    def parse_json_response(text: str) -> Dict[str, Any]:
        """Safely parse JSON from GPT response"""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                raise e
        else:
            logger.error("No JSON found in GPT response!")
            raise ValueError("No JSON found in GPT response")
    
    @staticmethod
    def validate_signal_decision(signal_decision: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Validate and sanitize signal decision"""
        signal_decision["symbol"] = symbol
        
        # Ensure all required keys are present
        for key in GPTConfig.REQUIRED_SIGNAL_KEYS:
            signal_decision.setdefault(key, None)
        
        # Validate signal type
        if signal_decision["signal"] not in GPTConfig.VALID_SIGNALS:
            signal_decision["signal"] = "WAIT"
            signal_decision["reason"] = "Invalid GPT signal - defaulted to WAIT"
        
        return signal_decision
    
    @staticmethod
    def estimate_tokens_and_cost(prompt: str, completion: str, model: str = ENCODING_MODEL) -> Dict[str, int]:
        """Estimate token usage and cost"""
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning(f"Model {model} not found, using fallback encoding")
            enc = tiktoken.get_encoding(GPTConfig.ENCODING_FALLBACK)
        
        prompt_tokens = len(enc.encode(prompt))
        completion_tokens = len(enc.encode(completion))
        
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

class GPTSignalGenerator:
    """Generates trading signals using GPT"""
    
    def __init__(self):
        self.client = openai.OpenAI()
        self.processor = GPTResponseProcessor()
        
        # Load system prompt
        try:
            with open('core/prompts/system_prompt.txt', 'r') as f:
                self.system_prompt_template = f.read()
        except FileNotFoundError:
            logger.error("System prompt file not found!")
            raise
    
    def generate_signal(self, symbol: str = "EURUSD", bars_json: int = 20, bars_chart: int = 80) -> Dict[str, Any]:
        """Generate trading signal for given symbol"""
        logger.info(f"GPT signal request initiated for {symbol}")
        
        # Prepare data
        preparator = DataPreparator(symbol)
        prepared_data = preparator.prepare_signal_data(bars_json, bars_chart)
        
        if prepared_data.get("error"):
            return self._create_error_response(symbol, prepared_data["error"])
        
        # Generate signal
        try:
            signal_decision = self._call_gpt_api(symbol, prepared_data)
            
            # Process and validate response
            signal_decision = self.processor.validate_signal_decision(signal_decision, symbol)
            
            # Log and save
            pretty_print_json_box(signal_decision, title=f"GPT Signal Output — {symbol}")
            save_gpt_signal_decision(signal_decision)
            
            return signal_decision
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return self._create_error_response(symbol, f"GPT API error: {str(e)}")
    
    def _call_gpt_api(self, symbol: str, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make the actual GPT API call"""
        prompt_data = prepared_data["prompt_data"]
        images = prepared_data["images"]
        market_context = prepared_data["market_context"]
        
        # Prepare JSON data
        prompt_json = json.dumps(prompt_data, default=np_encoder)
        
        # Build message content
        message_content = [
            {"type": "text", "text": f"{symbol} H1/H4 data and context:"},
            *images,
            {"type": "text", "text": market_context},
            {"type": "text", "text": prompt_json}
        ]
        
        # Make API call
        response = self.client.chat.completions.create(
            model=GPTConfig.MODEL,
            messages=[
                {"role": "system", "content": self.system_prompt_template.format(symbol=symbol)},
                {"role": "user", "content": message_content}
            ]
        )
        
        reply = response.choices[0].message.content
        
        # Log token usage
        token_info = self.processor.estimate_tokens_and_cost(prompt_json, reply)
        logger.info(f"Token usage for {symbol}: {token_info}")
        
        # Parse response
        return self.processor.parse_json_response(reply)
    
    def _create_error_response(self, symbol: str, reason: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "signal": "WAIT",
            "symbol": symbol,
            "reason": reason,
            "entry": None,
            "sl": None,
            "tp": None,
            "rr": None,
            "risk_class": None
        }

class GPTReflectionGenerator:
    """Generates trade reflections using GPT"""
    
    def __init__(self):
        self.client = openai.OpenAI()
    
    def generate_reflection(self, trade: Dict[str, Any], bars_json: int = 20, bars_chart: int = 80) -> Dict[str, Any]:
        """Generate reflection analysis for completed trade"""
        symbol = trade.get("symbol", "EURUSD")
        print_section(f"GPT REFLECTION — {symbol}")
        
        try:
            # Get recent market data
            recent_data = get_recent_candle_history_and_chart(
                symbol=symbol, 
                bars_json=bars_json, 
                bars_chart=bars_chart
            )
            
            # Prepare reflection prompt
            reflection_prompt = self._build_reflection_prompt(trade, recent_data)
            
            # Call GPT API
            response = self.client.chat.completions.create(
                model=GPTConfig.MODEL,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert trading coach evaluating closed trades using VSA, multi-timeframe, and strict rules."
                    },
                    {"role": "user", "content": reflection_prompt}
                ]
            )
            
            reflection_text = response.choices[0].message.content.strip()
            
            # Log output
            print_section("GPT REFLECTION OUTPUT")
            print(reflection_text)
            
            # Add reflection to trade data
            trade["reflection"] = reflection_text
            
            return trade
            
        except Exception as e:
            logger.error(f"Error generating reflection for {symbol}: {e}")
            trade["reflection"] = f"Reflection generation failed: {str(e)}"
            return trade
    
    def _build_reflection_prompt(self, trade: Dict[str, Any], recent_data: Dict[str, Any]) -> str:
        """Build the reflection prompt with trade and market data"""
        history_h1 = recent_data.get("history_h1", [])
        history_h4 = recent_data.get("history_h4", [])
        
        return f"""
You are a trading assistant reviewing a completed trade. You are given:
- Trade summary (side, entry, exit, SL, TP, RR, result).
- Recent H1 and H4 candles with indicators.

Trade Details:
- Side: {trade.get('side', 'N/A')}
- Entry: {trade.get('entry', 'N/A')}
- Exit: {trade.get('exit_price', 'N/A')}
- Stop-loss: {trade.get('sl', 'N/A')}
- Take-profit: {trade.get('tp', 'N/A')}
- Risk/Reward: {trade.get('rr', 'N/A')}
- Outcome: {trade.get('result', 'N/A')}

Market Context:
- H1 Last Candle: {history_h1[-1] if history_h1 else 'N/A'}
- H4 Last Candle: {history_h4[-1] if history_h4 else 'N/A'}

Analysis Request:
Evaluate this trade using strict VSA and multi-timeframe analysis principles. 
Was the signal valid? What could be improved in future similar setups?
Provide actionable insights in one focused paragraph.
        """.strip()

# ========== PUBLIC API ==========

def ask_gpt_for_signal(symbol: str = "EURUSD", bars_json: int = 20, bars_chart: int = 80) -> Dict[str, Any]:
    """
    Generate trading signal using GPT analysis
    
    Args:
        symbol: Trading symbol to analyze
        bars_json: Number of bars for JSON data
        bars_chart: Number of bars for chart analysis
        
    Returns:
        Dict containing signal decision and metadata
    """
    generator = GPTSignalGenerator()
    return generator.generate_signal(symbol, bars_json, bars_chart)

def ask_gpt_for_reflection(trade: Dict[str, Any], bars_json: int = 20, bars_chart: int = 80) -> Dict[str, Any]:
    """
    Generate reflection analysis for completed trade
    
    Args:
        trade: Trade data dictionary
        bars_json: Number of bars for analysis
        bars_chart: Number of bars for chart context
        
    Returns:
        Trade dictionary with added reflection
    """
    generator = GPTReflectionGenerator()
    return generator.generate_reflection(trade, bars_json, bars_chart)

# ========== BACKWARD COMPATIBILITY ==========
# These functions are maintained for backward compatibility with existing code

def safe_parse_json(text: str) -> dict:
    """
    Legacy function for backward compatibility.
    Use GPTResponseProcessor.parse_json_response() for new code.
    """
    return GPTResponseProcessor.parse_json_response(text)

def estimate_tokens_and_cost(prompt: str, completion: str, model: str = "gpt-4-1106-preview") -> Dict[str, int]:
    """
    Legacy function for backward compatibility.
    Use GPTResponseProcessor.estimate_tokens_and_cost() for new code.
    """
    return GPTResponseProcessor.estimate_tokens_and_cost(prompt, completion, model)