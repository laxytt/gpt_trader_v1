"""
Momentum Trader Agent
Specializes in trend following and momentum strategies
"""

import logging
from typing import Dict, List, Optional, Any

from core.agents.base_agent import TradingAgent, AgentType, AgentAnalysis, DebateResponse
from core.domain.models import MarketData, SignalType
from core.infrastructure.gpt.client import GPTClient

logger = logging.getLogger(__name__)


class MomentumTrader(TradingAgent):
    """The Trend Rider - follows strong momentum and trends"""
    
    def __init__(self, gpt_client: GPTClient):
        super().__init__(
            agent_type=AgentType.MOMENTUM_TRADER,
            personality="aggressive and trend-following momentum trader",
            specialty="breakouts, strong trends, continuation patterns, and momentum acceleration"
        )
        self.gpt_client = gpt_client
    
    def analyze(
        self,
        market_data: Dict[str, MarketData],
        news_context: Optional[List[str]] = None,
        ml_context: Optional[Dict[str, Any]] = None
    ) -> AgentAnalysis:
        """Analyze momentum and trend strength"""
        
        h1_data = market_data.get('h1')
        h4_data = market_data.get('h4')
        
        if not h1_data or not h4_data:
            raise ValueError("Both H1 and H4 data required for momentum analysis")
        
        # Calculate momentum indicators
        recent_candles = h1_data.candles[-20:]
        h4_candles = h4_data.candles[-10:]
        
        # Price momentum
        momentum_10 = ((recent_candles[-1].close - recent_candles[-10].close) / recent_candles[-10].close * 100) if len(recent_candles) >= 10 else 0
        momentum_20 = ((recent_candles[-1].close - recent_candles[0].close) / recent_candles[0].close * 100) if len(recent_candles) >= 20 else 0
        
        # Trend strength on H4
        h4_trend_bars = self._count_trend_bars(h4_candles)
        
        # Volume momentum
        recent_volume = sum(c.volume for c in recent_candles[-5:]) / 5
        older_volume = sum(c.volume for c in recent_candles[-10:-5]) / 5
        volume_momentum = (recent_volume - older_volume) / older_volume * 100 if older_volume > 0 else 0
        
        analysis_prompt = self._build_prompt("""
Analyze {symbol} for momentum trading opportunities.

Current Price: {price}
10-bar Momentum: {momentum_10:.2f}%
20-bar Momentum: {momentum_20:.2f}%
Volume Momentum: {volume_momentum:.1f}%

H1 Trend:
- Price vs EMA20: {price_vs_ema20}
- Price vs EMA50: {price_vs_ema50}
- RSI: {rsi}

H4 Trend:
- Consecutive trend bars: {h4_trend_bars}
- H4 RSI: {h4_rsi}

Assess:
1. Is there strong directional momentum?
2. Is the trend accelerating or decelerating?
3. Any signs of breakout or continuation?
4. Volume confirming the move?
5. Where to join the momentum (pullback levels)?

Provide:
1. Momentum state (Strong Buy/Strong Sell/Neutral/Weakening)
2. Trend quality (Excellent/Good/Poor)
3. Entry strategy for momentum
4. Your recommendation (BUY/SELL/WAIT)
5. Confidence level (0-100)
6. Momentum targets
7. When momentum would fail

Format your response as:
MOMENTUM_STATE: [Strong Buy/Strong Sell/Neutral/Weakening]
TREND_QUALITY: [Excellent/Good/Poor]
ENTRY_STRATEGY: [how to enter the momentum]
RECOMMENDATION: [BUY/SELL/WAIT]
CONFIDENCE: [0-100]
TARGETS: [momentum-based targets]
INVALIDATION: [what would kill the momentum]
""",
            symbol=h1_data.symbol,
            price=h1_data.latest_candle.close,
            momentum_10=momentum_10,
            momentum_20=momentum_20,
            volume_momentum=volume_momentum,
            price_vs_ema20="Above" if h1_data.latest_candle.close > h1_data.latest_candle.ema50 * 0.996 else "Below",  # Using EMA50 as proxy
            price_vs_ema50="Above" if h1_data.latest_candle.close > h1_data.latest_candle.ema50 else "Below",
            rsi=h1_data.latest_candle.rsi14,
            h4_trend_bars=h4_trend_bars,
            h4_rsi=h4_data.latest_candle.rsi14
        )
        
        response = self.gpt_client.analyze_with_response(
            analysis_prompt,
            temperature=0.5
        )
        
        return self._parse_analysis(response, h1_data.latest_candle.close)
    
    def debate(
        self,
        other_analyses: List[AgentAnalysis],
        round_number: int,
        previous_responses: Optional[List[DebateResponse]] = None
    ) -> DebateResponse:
        """Participate in debate with momentum arguments"""
        
        my_position = self.analysis_history[-1]
        
        if round_number == 1:
            statement = self._generate_opening_statement(my_position)
        elif round_number == 2:
            # Address contrarian if they oppose
            contrarian = next((a for a in other_analyses if a.agent_type == AgentType.CONTRARIAN_TRADER), None)
            risk_manager = next((a for a in other_analyses if a.agent_type == AgentType.RISK_MANAGER), None)
            
            if contrarian and contrarian.recommendation != my_position.recommendation:
                statement = "The contrarian may fade this move, but the trend is your friend until it ends. Momentum is clearly present."
            elif risk_manager and risk_manager.recommendation == SignalType.WAIT:
                statement = "Risk concerns are valid, but strong trends offer the best risk/reward when properly managed. We can use tight stops."
            else:
                statement = self._reinforce_momentum_case(other_analyses)
        else:
            statement = self._generate_closing_statement(other_analyses)
        
        # Momentum trader may reduce confidence if technical structure is poor
        updated_confidence = my_position.confidence
        tech_analysis = next((a for a in other_analyses if a.agent_type == AgentType.TECHNICAL_ANALYST), None)
        
        if tech_analysis and tech_analysis.recommendation != my_position.recommendation:
            updated_confidence = max(40, my_position.confidence - 15)
        
        return DebateResponse(
            agent_type=self.agent_type,
            round=round_number,
            statement=statement,
            maintains_position=True,  # Momentum traders stick to trends
            updated_confidence=updated_confidence
        )
    
    def _count_trend_bars(self, candles: List) -> int:
        """Count consecutive bars in same direction"""
        if len(candles) < 2:
            return 0
        
        count = 1
        direction = 1 if candles[-1].close > candles[-1].open else -1
        
        for i in range(len(candles) - 2, -1, -1):
            candle_dir = 1 if candles[i].close > candles[i].open else -1
            if candle_dir == direction:
                count += 1
            else:
                break
        
        return count * direction  # Positive for bullish, negative for bearish
    
    def _parse_analysis(self, response: str, current_price: float) -> AgentAnalysis:
        """Parse GPT response into AgentAnalysis"""
        lines = response.strip().split('\n')
        parsed = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                parsed[key.strip()] = value.strip()
        
        # Momentum trader is directional
        momentum_state = parsed.get('MOMENTUM_STATE', 'Neutral')
        
        if 'Strong Buy' in momentum_state:
            recommendation = SignalType.BUY
            base_confidence = 80
        elif 'Strong Sell' in momentum_state:
            recommendation = SignalType.SELL
            base_confidence = 80
        else:
            rec_text = parsed.get('RECOMMENDATION', 'WAIT').upper()
            if 'BUY' in rec_text:
                recommendation = SignalType.BUY
                base_confidence = 60
            elif 'SELL' in rec_text:
                recommendation = SignalType.SELL
                base_confidence = 60
            else:
                recommendation = SignalType.WAIT
                base_confidence = 40
        
        # Adjust confidence based on trend quality
        trend_quality = parsed.get('TREND_QUALITY', 'Poor')
        if 'Excellent' in trend_quality:
            confidence = min(100, base_confidence + 15)
        elif 'Good' in trend_quality:
            confidence = base_confidence
        else:
            confidence = max(30, base_confidence - 20)
        
        # Build reasoning
        reasoning = []
        reasoning.append(f"Momentum: {momentum_state}")
        reasoning.append(f"Trend quality: {trend_quality}")
        if parsed.get('ENTRY_STRATEGY'):
            reasoning.append(f"Entry: {parsed['ENTRY_STRATEGY']}")
        
        concerns = []
        if parsed.get('INVALIDATION'):
            concerns.append(f"Invalidation: {parsed['INVALIDATION']}")
        
        analysis = AgentAnalysis(
            agent_type=self.agent_type,
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning[:3],
            concerns=concerns,
            entry_price=current_price,
            metadata={
                'momentum_state': momentum_state,
                'trend_quality': trend_quality,
                'entry_strategy': parsed.get('ENTRY_STRATEGY'),
                'targets': parsed.get('TARGETS'),
                'invalidation': parsed.get('INVALIDATION')
            }
        )
        
        self.analysis_history.append(analysis)
        return analysis
    
    def _generate_opening_statement(self, analysis: AgentAnalysis) -> str:
        """Generate opening debate statement"""
        momentum = analysis.metadata.get('momentum_state', 'Neutral')
        quality = analysis.metadata.get('trend_quality', 'Poor')
        
        if analysis.recommendation != SignalType.WAIT:
            return f"{momentum} momentum detected with {quality} trend quality. This is a clear opportunity to ride the trend. {analysis.reasoning[2]}"
        else:
            return "No clear momentum present. As a momentum trader, I prefer to wait for strong directional moves."
    
    def _reinforce_momentum_case(self, analyses: List[AgentAnalysis]) -> str:
        """Reinforce the momentum argument"""
        my_rec = self.analysis_history[-1].recommendation
        if my_rec != SignalType.WAIT:
            return "The momentum is undeniable here. Missing this move would be leaving money on the table. We can manage risk with trailing stops."
        else:
            return "Without clear momentum, there's no edge. Patience until a strong trend emerges."
    
    def _generate_closing_statement(self, all_analyses: List[AgentAnalysis]) -> str:
        """Generate closing statement"""
        my_position = self.analysis_history[-1]
        
        if my_position.recommendation != SignalType.WAIT:
            targets = my_position.metadata.get('targets', 'momentum targets')
            return f"Strong trends don't come often. When momentum aligns, we must act. Target: {targets}."
        else:
            return "No momentum, no trade. Waiting for the next strong directional move is the disciplined approach."