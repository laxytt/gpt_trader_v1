"""
Contrarian Trader Agent
Specializes in fading extremes and mean reversion
"""

import logging
from typing import Dict, List, Optional, Any

from core.agents.base_agent import TradingAgent, AgentType, AgentAnalysis, DebateResponse
from core.domain.models import MarketData, SignalType
from core.infrastructure.gpt.client import GPTClient

logger = logging.getLogger(__name__)


class ContrarianTrader(TradingAgent):
    """The Skeptic - fades extremes and crowds"""
    
    def __init__(self, gpt_client: GPTClient):
        super().__init__(
            agent_type=AgentType.CONTRARIAN_TRADER,
            personality="skeptical and contrarian fade trader",
            specialty="reversals, extremes, mean reversion, failed moves, and divergences"
        )
        self.gpt_client = gpt_client
    
    def analyze(
        self,
        market_data: Dict[str, MarketData],
        news_context: Optional[List[str]] = None,
        ml_context: Optional[Dict[str, Any]] = None
    ) -> AgentAnalysis:
        """Look for contrarian opportunities"""
        
        h1_data = market_data.get('h1')
        h4_data = market_data.get('h4')
        
        if not h1_data:
            raise ValueError("H1 data required for contrarian analysis")
        
        # Look for extremes
        recent_candles = h1_data.candles[-50:] if len(h1_data.candles) >= 50 else h1_data.candles
        
        # Price extremes
        highest = max(c.high for c in recent_candles)
        lowest = min(c.low for c in recent_candles)
        current_price = h1_data.latest_candle.close
        
        # Position in range
        range_size = highest - lowest
        position_in_range = (current_price - lowest) / range_size if range_size > 0 else 0.5
        
        # RSI divergence check
        rsi_values = [c.rsi14 for c in recent_candles[-10:] if c.rsi14]
        price_values = [c.close for c in recent_candles[-10:]]
        
        # Failed breakout check
        recent_high_test = any(c.high >= highest * 0.999 for c in recent_candles[-5:])
        recent_low_test = any(c.low <= lowest * 1.001 for c in recent_candles[-5:])
        
        analysis_prompt = self._build_prompt("""
Analyze {symbol} for contrarian trading opportunities.

Current Price: {price}
Position in 50-bar range: {position:.1%}
RSI: {rsi}
Recent High: {highest}, Recent Low: {lowest}
High tested recently: {high_test}
Low tested recently: {low_test}

Recent Price Action:
- 10-bar momentum: {momentum_10:.2f}%
- Current vs 20-bar average: {vs_average:.2f}%

Look for:
1. Are we at an extreme (overbought/oversold)?
2. Any failed breakouts or false moves?
3. Divergences (price vs indicators)?
4. Is everyone on the same side of the trade?
5. Mean reversion opportunities?

Provide:
1. Contrarian opportunity (Strong/Moderate/None)
2. Type of setup (Fade extreme/Failed move/Divergence/Mean reversion)
3. Why the crowd is wrong
4. Your recommendation (BUY/SELL/WAIT)
5. Confidence level (0-100)
6. Reversal triggers
7. What would invalidate the contrarian view

Format your response as:
OPPORTUNITY: [Strong/Moderate/None]
SETUP_TYPE: [type of contrarian setup]
CROWD_ERROR: [why the crowd is wrong]
RECOMMENDATION: [BUY/SELL/WAIT]
CONFIDENCE: [0-100]
REVERSAL_TRIGGERS: [what to watch for reversal]
INVALIDATION: [what kills the contrarian thesis]
""",
            symbol=h1_data.symbol,
            price=current_price,
            position=position_in_range,
            rsi=h1_data.latest_candle.rsi14,
            highest=highest,
            lowest=lowest,
            high_test="Yes" if recent_high_test else "No",
            low_test="Yes" if recent_low_test else "No",
            momentum_10=((current_price - recent_candles[-10].close) / recent_candles[-10].close * 100) if len(recent_candles) >= 10 else 0,
            vs_average=((current_price - sum(c.close for c in recent_candles[-20:]) / 20) / current_price * 100) if len(recent_candles) >= 20 else 0
        )
        
        response = self.gpt_client.analyze_with_response(
            analysis_prompt,
            temperature=0.6  # Higher for creative contrarian thinking
        )
        
        return self._parse_analysis(response, current_price)
    
    def debate(
        self,
        other_analyses: List[AgentAnalysis],
        round_number: int,
        previous_responses: Optional[List[DebateResponse]] = None
    ) -> DebateResponse:
        """Participate in debate with contrarian perspective"""
        
        my_position = self.analysis_history[-1]
        
        # Count how many agree - contrarian loves to fade consensus
        consensus_count = sum(1 for a in other_analyses if a.recommendation == my_position.recommendation)
        
        if round_number == 1:
            statement = self._generate_opening_statement(my_position, len(other_analyses) - consensus_count)
        elif round_number == 2:
            # Challenge the momentum trader specifically
            momentum = next((a for a in other_analyses if a.agent_type == AgentType.MOMENTUM_TRADER), None)
            if momentum and momentum.recommendation != SignalType.WAIT and momentum.recommendation != my_position.recommendation:
                statement = "The momentum trader sees strength, but that's exactly when reversals happen. The rubber band is stretched too far."
            else:
                statement = self._challenge_consensus(other_analyses)
        else:
            statement = self._generate_closing_statement(other_analyses)
        
        # Contrarian gets more confident when everyone disagrees
        updated_confidence = my_position.confidence
        
        if consensus_count <= 1:  # Only contrarian sees this
            updated_confidence = min(100, my_position.confidence + 10)
        elif consensus_count >= 5:  # Everyone agrees (suspicious to contrarian)
            updated_confidence = max(30, my_position.confidence - 20)
        
        return DebateResponse(
            agent_type=self.agent_type,
            round=round_number,
            statement=statement,
            maintains_position=True,  # Contrarians don't follow the crowd
            updated_confidence=updated_confidence
        )
    
    def _parse_analysis(self, response: str, current_price: float) -> AgentAnalysis:
        """Parse GPT response into AgentAnalysis"""
        lines = response.strip().split('\n')
        parsed = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                parsed[key.strip()] = value.strip()
        
        # Contrarian logic
        opportunity = parsed.get('OPPORTUNITY', 'None')
        setup_type = parsed.get('SETUP_TYPE', '')
        
        if 'Strong' in opportunity:
            # Strong contrarian signal
            if 'extreme' in setup_type.lower() and 'over' in setup_type.lower():
                recommendation = SignalType.SELL  # Fade overbought
            elif 'extreme' in setup_type.lower():
                recommendation = SignalType.BUY  # Fade oversold
            else:
                rec_text = parsed.get('RECOMMENDATION', 'WAIT').upper()
                if 'BUY' in rec_text:
                    recommendation = SignalType.BUY
                elif 'SELL' in rec_text:
                    recommendation = SignalType.SELL
                else:
                    recommendation = SignalType.WAIT
            base_confidence = 70
        elif 'Moderate' in opportunity:
            rec_text = parsed.get('RECOMMENDATION', 'WAIT').upper()
            if 'BUY' in rec_text:
                recommendation = SignalType.BUY
            elif 'SELL' in rec_text:
                recommendation = SignalType.SELL
            else:
                recommendation = SignalType.WAIT
            base_confidence = 50
        else:
            recommendation = SignalType.WAIT
            base_confidence = 30
        
        try:
            confidence = float(parsed.get('CONFIDENCE', str(base_confidence)))
        except:
            confidence = base_confidence
        
        # Build reasoning
        reasoning = []
        if parsed.get('SETUP_TYPE'):
            reasoning.append(f"Setup: {parsed['SETUP_TYPE']}")
        if parsed.get('CROWD_ERROR'):
            reasoning.append(f"Crowd error: {parsed['CROWD_ERROR']}")
        if parsed.get('REVERSAL_TRIGGERS'):
            reasoning.append(f"Watch for: {parsed['REVERSAL_TRIGGERS']}")
        
        concerns = []
        if parsed.get('INVALIDATION'):
            concerns.append(parsed['INVALIDATION'])
        
        analysis = AgentAnalysis(
            agent_type=self.agent_type,
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning[:3],
            concerns=concerns,
            entry_price=current_price,
            metadata={
                'opportunity_strength': opportunity,
                'setup_type': setup_type,
                'crowd_error': parsed.get('CROWD_ERROR'),
                'reversal_triggers': parsed.get('REVERSAL_TRIGGERS'),
                'invalidation': parsed.get('INVALIDATION')
            }
        )
        
        self.analysis_history.append(analysis)
        return analysis
    
    def _generate_opening_statement(self, analysis: AgentAnalysis, disagreement_count: int) -> str:
        """Generate opening debate statement"""
        setup = analysis.metadata.get('setup_type', 'contrarian setup')
        
        if analysis.recommendation != SignalType.WAIT:
            intro = f"While others see strength, I see exhaustion. {setup} suggests {analysis.recommendation.value}."
            if disagreement_count >= 4:
                intro += " The fact that most disagree only strengthens my conviction."
            return intro
        else:
            return "No contrarian opportunity here. Sometimes the trend is genuine, not ready to fade yet."
    
    def _challenge_consensus(self, analyses: List[AgentAnalysis]) -> str:
        """Challenge the consensus view"""
        buy_count = sum(1 for a in analyses if a.recommendation == SignalType.BUY)
        sell_count = sum(1 for a in analyses if a.recommendation == SignalType.SELL)
        
        if buy_count >= 4:
            return "Everyone's bullish? That's precisely when tops form. The last buyer is in."
        elif sell_count >= 4:
            return "Universal bearishness is a contrarian's dream. Bottoms are made in despair."
        else:
            return "The market lacks conviction here. Perhaps that's the real signal - confusion before a move."
    
    def _generate_closing_statement(self, all_analyses: List[AgentAnalysis]) -> str:
        """Generate closing statement"""
        my_position = self.analysis_history[-1]
        crowd_error = my_position.metadata.get('crowd_error', 'following the herd')
        
        if my_position.recommendation != SignalType.WAIT:
            return f"The crowd is {crowd_error}. Contrarian trades are uncomfortable by design - that's why they work."
        else:
            return "No clear extreme to fade. A good contrarian knows when NOT to be contrarian."