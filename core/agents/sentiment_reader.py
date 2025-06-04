"""
Sentiment Reader Agent
Specializes in market psychology and sentiment analysis
"""

import logging
from typing import Dict, List, Optional, Any

from core.agents.base_agent import TradingAgent, AgentType, AgentAnalysis, DebateResponse
from core.domain.models import MarketData, SignalType
from core.infrastructure.gpt.client import GPTClient

logger = logging.getLogger(__name__)


class SentimentReader(TradingAgent):
    """The Psychologist - reads market sentiment and crowd behavior"""
    
    def __init__(self, gpt_client: GPTClient):
        super().__init__(
            agent_type=AgentType.SENTIMENT_READER,
            personality="intuitive and mood-sensitive market psychologist",
            specialty="market psychology, fear/greed dynamics, positioning extremes, and crowd behavior"
        )
        self.gpt_client = gpt_client
    
    def analyze(
        self,
        market_data: Dict[str, MarketData],
        news_context: Optional[List[str]] = None,
        ml_context: Optional[Dict[str, Any]] = None
    ) -> AgentAnalysis:
        """Analyze market sentiment and psychology"""
        
        h1_data = market_data.get('h1')
        h4_data = market_data.get('h4')
        
        # Analyze recent price action for sentiment
        recent_candles = h1_data.candles[-20:] if h1_data else []
        
        # Calculate sentiment indicators
        bullish_candles = sum(1 for c in recent_candles if c.close > c.open)
        bearish_candles = len(recent_candles) - bullish_candles
        
        # Volume analysis
        avg_volume = sum(c.volume for c in recent_candles[:-5]) / max(1, len(recent_candles) - 5)
        recent_volume = sum(c.volume for c in recent_candles[-5:]) / min(5, len(recent_candles))
        volume_surge = recent_volume / max(1, avg_volume)
        
        # Price momentum
        if len(recent_candles) >= 10:
            momentum = (recent_candles[-1].close - recent_candles[-10].close) / recent_candles[-10].close * 100
        else:
            momentum = 0
        
        analysis_prompt = self._build_prompt("""
Read the market sentiment for {symbol} like a trading psychologist.

Current Price: {price}
RSI: {rsi}
Recent Momentum: {momentum:.2f}% over 10 bars
Bullish/Bearish Candles: {bullish}/{bearish}
Volume Surge: {volume_surge:.1f}x average

Recent Price Action:
- High: {recent_high}
- Low: {recent_low}
- Range: {range_pips} pips

Consider:
1. Is the market showing fear or greed?
2. Are traders trapped (failed breakouts, false moves)?
3. Is there exhaustion or fresh momentum?
4. What's the crowd likely thinking/doing?
5. Any signs of capitulation or euphoria?

Provide:
1. Overall market mood (fear/greed/uncertainty)
2. Crowd positioning assessment
3. Key psychological levels
4. Sentiment-based recommendation (BUY/SELL/WAIT)
5. Confidence level (0-100)
6. Psychological trigger points
7. Main sentiment concern

Format your response as:
MOOD: [fear/greed/neutral/uncertainty]
CROWD: [positioning assessment]
PSYCH_LEVELS: [key psychological levels]
SENTIMENT_SIGNS: [what the market is telling us]
RECOMMENDATION: [BUY/SELL/WAIT]
CONFIDENCE: [0-100]
TRIGGERS: [levels that would change sentiment]
CONCERN: [main psychological risk]
""",
            symbol=h1_data.symbol,
            price=h1_data.latest_candle.close,
            rsi=h1_data.latest_candle.rsi14,
            momentum=momentum,
            bullish=bullish_candles,
            bearish=bearish_candles,
            volume_surge=volume_surge,
            recent_high=max(c.high for c in recent_candles),
            recent_low=min(c.low for c in recent_candles),
            range_pips=(max(c.high for c in recent_candles) - min(c.low for c in recent_candles)) * 10000
        )
        
        response = self.gpt_client.analyze_with_response(
            analysis_prompt,
            temperature=0.6  # Higher temperature for intuitive reading
        )
        
        return self._parse_analysis(response, h1_data.latest_candle.close if h1_data else 0)
    
    def debate(
        self,
        other_analyses: List[AgentAnalysis],
        round_number: int,
        previous_responses: Optional[List[DebateResponse]] = None
    ) -> DebateResponse:
        """Participate in debate with sentiment insights"""
        
        my_position = self.analysis_history[-1]
        
        if round_number == 1:
            statement = self._generate_opening_statement(my_position)
        elif round_number == 2:
            # Find who to address
            momentum_trader = next((a for a in other_analyses if a.agent_type == AgentType.MOMENTUM_TRADER), None)
            if momentum_trader and momentum_trader.recommendation != my_position.recommendation:
                statement = f"The {momentum_trader.agent_type.value} may see momentum, but I sense exhaustion. The crowd is already all-in."
            else:
                statement = self._provide_psychological_insight(other_analyses)
        else:
            statement = self._generate_closing_statement(other_analyses)
        
        # Sentiment reader may change view if overwhelming technical evidence
        maintains_position = True
        updated_confidence = my_position.confidence
        
        tech_analysis = next((a for a in other_analyses if a.agent_type == AgentType.TECHNICAL_ANALYST), None)
        if tech_analysis and tech_analysis.confidence > 80:
            if tech_analysis.recommendation != my_position.recommendation:
                updated_confidence = max(30, my_position.confidence - 20)
                if updated_confidence < 50:
                    maintains_position = False
        
        return DebateResponse(
            agent_type=self.agent_type,
            round=round_number,
            statement=statement,
            maintains_position=maintains_position,
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
        
        # Determine recommendation based on mood and crowd
        mood = parsed.get('MOOD', 'neutral').lower()
        crowd = parsed.get('CROWD', '').lower()
        
        # Contrarian logic
        if 'fear' in mood and 'oversold' in crowd:
            recommendation = SignalType.BUY
        elif 'greed' in mood and 'overbought' in crowd:
            recommendation = SignalType.SELL
        else:
            rec_text = parsed.get('RECOMMENDATION', 'WAIT').upper()
            if 'BUY' in rec_text:
                recommendation = SignalType.BUY
            elif 'SELL' in rec_text:
                recommendation = SignalType.SELL
            else:
                recommendation = SignalType.WAIT
        
        try:
            confidence = float(parsed.get('CONFIDENCE', '50'))
        except:
            confidence = 50.0
        
        # Build reasoning
        reasoning = []
        if parsed.get('MOOD'):
            reasoning.append(f"Market mood: {parsed['MOOD']}")
        if parsed.get('CROWD'):
            reasoning.append(f"Crowd positioning: {parsed['CROWD']}")
        if parsed.get('SENTIMENT_SIGNS'):
            reasoning.append(f"Sentiment signs: {parsed['SENTIMENT_SIGNS']}")
        
        concerns = []
        if parsed.get('CONCERN'):
            concerns.append(parsed['CONCERN'])
        
        analysis = AgentAnalysis(
            agent_type=self.agent_type,
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning[:3],
            concerns=concerns,
            entry_price=current_price,
            metadata={
                'mood': parsed.get('MOOD'),
                'crowd_positioning': parsed.get('CROWD'),
                'psychological_levels': parsed.get('PSYCH_LEVELS'),
                'triggers': parsed.get('TRIGGERS')
            }
        )
        
        self.analysis_history.append(analysis)
        return analysis
    
    def _generate_opening_statement(self, analysis: AgentAnalysis) -> str:
        """Generate opening debate statement"""
        mood = analysis.metadata.get('mood', 'uncertain')
        return f"I sense {mood} in the market. {analysis.reasoning[0]}. The psychological setup suggests {analysis.recommendation.value}."
    
    def _provide_psychological_insight(self, analyses: List[AgentAnalysis]) -> str:
        """Provide psychological insight to the debate"""
        agreements = sum(1 for a in analyses if a.recommendation == self.analysis_history[-1].recommendation)
        
        if agreements >= 4:
            return "The council's agreement itself is telling - when everyone sees the same thing, the market often does the opposite."
        else:
            return "This disagreement reflects market uncertainty. Perhaps that's the real signal - stay cautious."
    
    def _generate_closing_statement(self, all_analyses: List[AgentAnalysis]) -> str:
        """Generate closing statement"""
        my_mood = self.analysis_history[-1].metadata.get('mood', 'uncertain')
        
        if my_mood in ['fear', 'greed']:
            return f"Extreme {my_mood} creates opportunities. The crowd's emotion is our edge."
        else:
            return "The market psychology isn't clear. Without a strong sentiment read, perhaps we should be patient."