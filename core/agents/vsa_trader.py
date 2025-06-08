"""
VSA (Volume Spread Analysis) Trader Agent
Specializes in volume-based market analysis following VSA principles
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from core.agents.base_agent import BaseAgent, AgentAnalysis
from core.domain.models import MarketData, SignalType
from core.domain.enums import AgentType

logger = logging.getLogger(__name__)


class VSATrader(BaseAgent):
    """
    VSA specialist that analyzes volume, spread, and price action
    to identify smart money movements and market manipulation
    """
    
    def __init__(self, gpt_client, **kwargs):
        super().__init__(
            agent_type=AgentType.TECHNICAL,  # Or create new AgentType.VSA
            personality="methodical VSA specialist focused on volume analysis and smart money movements",
            gpt_client=gpt_client,
            **kwargs
        )
    
    async def analyze_market(
        self,
        market_data: Dict[str, MarketData],
        news_context: List[str],
        ml_context: Optional[Dict[str, Any]] = None
    ) -> AgentAnalysis:
        """Analyze market using VSA principles"""
        
        h1_data = market_data.get('h1')
        h4_data = market_data.get('h4')
        
        if not h1_data or not h4_data:
            return self._create_error_analysis("Missing market data")
        
        # Prepare VSA-specific analysis
        vsa_context = self._prepare_vsa_context(h1_data, h4_data)
        
        # Get latest price for reference
        current_price = h1_data.latest_candle.close if h1_data.latest_candle else 0
        
        # Create VSA-focused prompt
        prompt = self._create_vsa_prompt(h1_data, h4_data, vsa_context, news_context)
        
        # Get GPT analysis
        response = await self._get_gpt_analysis(prompt, market_data)
        
        # Parse response
        return self._parse_analysis(response, current_price)
    
    def _prepare_vsa_context(self, h1_data: MarketData, h4_data: MarketData) -> Dict[str, Any]:
        """Prepare VSA-specific context from market data"""
        
        # Analyze last 5 candles for VSA patterns
        recent_candles = h1_data.candles[-5:] if len(h1_data.candles) >= 5 else h1_data.candles
        
        vsa_patterns = []
        for i, candle in enumerate(recent_candles):
            # Calculate spread
            spread = candle.high - candle.low
            body = abs(candle.close - candle.open)
            
            # Analyze volume (using tick_volume as proxy)
            volume = candle.tick_volume if hasattr(candle, 'tick_volume') else 0
            
            # Identify VSA patterns
            if i > 0:
                prev_candle = recent_candles[i-1]
                prev_volume = prev_candle.tick_volume if hasattr(prev_candle, 'tick_volume') else 0
                
                # Stopping Volume
                if candle.close < candle.open and volume > prev_volume * 1.5:
                    vsa_patterns.append("Stopping Volume detected")
                
                # No Supply (Test)
                if candle.close < candle.open and volume < prev_volume * 0.5:
                    vsa_patterns.append("No Supply signal")
                
                # Shake Out
                if candle.low < prev_candle.low and candle.close > (candle.low + spread * 0.66):
                    vsa_patterns.append("Potential Shake Out")
        
        return {
            'patterns': vsa_patterns,
            'last_spread': spread if 'spread' in locals() else 0,
            'last_volume': volume if 'volume' in locals() else 0
        }
    
    def _create_vsa_prompt(
        self,
        h1_data: MarketData,
        h4_data: MarketData,
        vsa_context: Dict[str, Any],
        news_context: List[str]
    ) -> str:
        """Create VSA-specific analysis prompt"""
        
        h1_latest = h1_data.latest_candle
        h4_latest = h4_data.latest_candle
        
        # Get last 5 candles for VSA analysis
        candle_descriptions = []
        for i, candle in enumerate(h1_data.candles[-5:]):
            volume = candle.tick_volume if hasattr(candle, 'tick_volume') else "N/A"
            spread = candle.high - candle.low
            close_position = (candle.close - candle.low) / spread if spread > 0 else 0.5
            
            candle_descriptions.append(
                f"Candle {i+1}: O:{candle.open:.5f} H:{candle.high:.5f} "
                f"L:{candle.low:.5f} C:{candle.close:.5f} "
                f"Volume:{volume} Spread:{spread:.5f} ClosePos:{close_position:.2f}"
            )
        
        return f"""Analyze {h1_data.symbol} using Volume Spread Analysis (VSA) principles.

Last 5 H1 Candles:
{chr(10).join(candle_descriptions)}

VSA Patterns Detected: {', '.join(vsa_context['patterns']) if vsa_context['patterns'] else 'None'}

Background (H4):
- Trend: {"Bullish" if h4_latest.close > h4_latest.ema50 else "Bearish"}
- Position in range: {((h4_latest.close - min(c.low for c in h4_data.candles[-20:])) / 
                      (max(c.high for c in h4_data.candles[-20:]) - min(c.low for c in h4_data.candles[-20:])) * 100):.1f}%

Apply VSA principles:
1. Analyze volume in relation to price spread
2. Identify smart money accumulation/distribution
3. Look for tests of supply/demand
4. Check for climactic action
5. Assess market strength/weakness

Consider these VSA signals:
- Stopping Volume, Shake Out, No Supply/Demand
- Buying/Selling Climax, Upthrust, Spring
- Two Bar Reversal, Test patterns

Provide:
VSA_SIGNAL: [Main VSA pattern or "No clear signal"]
VOLUME_ANALYSIS: [Volume behavior assessment]
SMART_MONEY: [What smart money appears to be doing]
MARKET_PHASE: [Accumulation/Distribution/Trend/Test]
RECOMMENDATION: [BUY/SELL/WAIT]
CONFIDENCE: [0-100]
ENTRY: [price if BUY/SELL]
STOP_LOSS: [price if BUY/SELL]
REASONING: [VSA-based explanation]
"""
    
    def _parse_analysis(self, response: str, current_price: float) -> AgentAnalysis:
        """Parse VSA analysis response"""
        lines = response.strip().split('\n')
        parsed = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                parsed[key.strip()] = value.strip()
        
        # Extract recommendation
        rec_text = parsed.get('RECOMMENDATION', 'WAIT').upper()
        if 'BUY' in rec_text:
            recommendation = SignalType.BUY
        elif 'SELL' in rec_text:
            recommendation = SignalType.SELL
        else:
            recommendation = SignalType.WAIT
        
        # Extract confidence
        try:
            confidence = float(parsed.get('CONFIDENCE', '50'))
        except:
            confidence = 50.0
        
        # Build reasoning based on VSA
        reasoning = []
        if parsed.get('VSA_SIGNAL') and parsed['VSA_SIGNAL'] != "No clear signal":
            reasoning.append(f"VSA Signal: {parsed['VSA_SIGNAL']}")
        if parsed.get('VOLUME_ANALYSIS'):
            reasoning.append(f"Volume: {parsed['VOLUME_ANALYSIS']}")
        if parsed.get('SMART_MONEY'):
            reasoning.append(f"Smart Money: {parsed['SMART_MONEY']}")
        if parsed.get('MARKET_PHASE'):
            reasoning.append(f"Market Phase: {parsed['MARKET_PHASE']}")
        
        # Extract levels
        try:
            entry = float(parsed.get('ENTRY', str(current_price)))
            stop_loss = float(parsed.get('STOP_LOSS', '0'))
        except:
            entry = current_price
            stop_loss = 0
        
        return AgentAnalysis(
            agent_type=self.agent_type,
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning,
            concerns=[parsed.get('REASONING', '')],
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=0,  # VSA typically doesn't use fixed TP
            market_assessment={
                'vsa_signal': parsed.get('VSA_SIGNAL', ''),
                'smart_money': parsed.get('SMART_MONEY', ''),
                'phase': parsed.get('MARKET_PHASE', '')
            }
        )