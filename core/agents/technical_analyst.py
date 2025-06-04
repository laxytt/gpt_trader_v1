"""
Technical Analyst Agent
Specializes in chart patterns, indicators, and price action
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from core.agents.base_agent import TradingAgent, AgentType, AgentAnalysis, DebateResponse
from core.domain.models import MarketData, SignalType
from core.infrastructure.gpt.client import GPTClient

logger = logging.getLogger(__name__)


class TechnicalAnalyst(TradingAgent):
    """The Chartist - focuses on technical analysis"""
    
    def __init__(self, gpt_client: GPTClient):
        super().__init__(
            agent_type=AgentType.TECHNICAL_ANALYST,
            personality="methodical and pattern-focused chartist",
            specialty="price action, support/resistance, technical indicators, and chart patterns"
        )
        self.gpt_client = gpt_client
    
    def analyze(
        self,
        market_data: Dict[str, MarketData],
        news_context: Optional[List[str]] = None,
        ml_context: Optional[Dict[str, Any]] = None
    ) -> AgentAnalysis:
        """Perform technical analysis"""
        
        h1_data = market_data.get('h1')
        h4_data = market_data.get('h4')
        
        if not h1_data or not h4_data:
            raise ValueError("Both H1 and H4 data required for technical analysis")
        
        # Extract key technical data
        h1_latest = h1_data.latest_candle
        h4_latest = h4_data.latest_candle
        
        analysis_prompt = self._build_prompt("""
Analyze the {symbol} chart from a pure technical perspective.

H1 Timeframe:
- Current Price: {h1_price}
- EMA50: {h1_ema50}, EMA200: {h1_ema200}
- RSI: {h1_rsi}
- ATR: {h1_atr}
- Recent High: {h1_high}, Recent Low: {h1_low}

H4 Timeframe:
- Current Price: {h4_price}
- EMA50: {h4_ema50}, EMA200: {h4_ema200}
- RSI: {h4_rsi}
- Trend: {h4_trend}

Provide:
1. Chart pattern identification (if any)
2. Key support and resistance levels
3. Indicator confluence analysis
4. Trend structure assessment
5. Your trading recommendation (BUY/SELL/WAIT)
6. Confidence level (0-100)
7. Specific entry, stop loss, and take profit levels
8. Main technical concern

Format your response as:
PATTERN: [identified pattern or "None clear"]
KEY_LEVELS: [support: X.XXXX, resistance: X.XXXX]
INDICATORS: [summary of indicator signals]
TREND: [trend assessment]
RECOMMENDATION: [BUY/SELL/WAIT]
CONFIDENCE: [0-100]
ENTRY: [price]
STOP_LOSS: [price]
TAKE_PROFIT: [price]
CONCERN: [main technical concern]
""",
            symbol=h1_data.symbol,
            h1_price=h1_latest.close,
            h1_ema50=h1_latest.ema50,
            h1_ema200=h1_latest.ema200,
            h1_rsi=h1_latest.rsi14,
            h1_atr=h1_latest.atr14,
            h1_high=max(c.high for c in h1_data.candles[-20:]),
            h1_low=min(c.low for c in h1_data.candles[-20:]),
            h4_price=h4_latest.close,
            h4_ema50=h4_latest.ema50,
            h4_ema200=h4_latest.ema200,
            h4_rsi=h4_latest.rsi14,
            h4_trend="Bullish" if h4_latest.close > h4_latest.ema50 > h4_latest.ema200 else "Bearish" if h4_latest.close < h4_latest.ema50 < h4_latest.ema200 else "Neutral"
        )
        
        # Get GPT analysis
        response = self.gpt_client.analyze_with_response(
            analysis_prompt,
            temperature=0.3  # Lower temperature for technical analysis
        )
        
        # Parse response
        return self._parse_analysis(response, h1_latest.close)
    
    def debate(
        self,
        other_analyses: List[AgentAnalysis],
        round_number: int,
        previous_responses: Optional[List[DebateResponse]] = None
    ) -> DebateResponse:
        """Participate in debate with technical arguments"""
        
        # Find opposing views
        opposing_views = [
            a for a in other_analyses 
            if a.recommendation != self.analysis_history[-1].recommendation
            and a.agent_type != self.agent_type
        ]
        
        if round_number == 1:
            # Initial position
            statement = self._generate_opening_statement(self.analysis_history[-1])
        elif round_number == 2 and opposing_views:
            # Counter-argument
            statement = self._generate_counter_argument(opposing_views[0])
        else:
            # Final position
            statement = self._generate_closing_statement(other_analyses)
        
        maintains_position = True  # Technical analyst rarely changes based on non-technical factors
        
        return DebateResponse(
            agent_type=self.agent_type,
            round=round_number,
            statement=statement,
            maintains_position=maintains_position,
            updated_confidence=self.analysis_history[-1].confidence
        )
    
    def _parse_analysis(self, response: str, current_price: float) -> AgentAnalysis:
        """Parse GPT response into AgentAnalysis"""
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
        
        # Extract levels
        try:
            entry = float(parsed.get('ENTRY', str(current_price)))
            stop_loss = float(parsed.get('STOP_LOSS', '0'))
            take_profit = float(parsed.get('TAKE_PROFIT', '0'))
        except:
            entry = current_price
            stop_loss = 0
            take_profit = 0
        
        # Build reasoning
        reasoning = []
        if parsed.get('PATTERN') and parsed['PATTERN'] != "None clear":
            reasoning.append(f"Pattern identified: {parsed['PATTERN']}")
        if parsed.get('INDICATORS'):
            reasoning.append(f"Indicators show: {parsed['INDICATORS']}")
        if parsed.get('TREND'):
            reasoning.append(f"Trend assessment: {parsed['TREND']}")
        
        # Concerns
        concerns = []
        if parsed.get('CONCERN'):
            concerns.append(parsed['CONCERN'])
        
        analysis = AgentAnalysis(
            agent_type=self.agent_type,
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning,
            concerns=concerns,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=self._calculate_risk_reward(entry, stop_loss, take_profit) if stop_loss and take_profit else None,
            metadata={
                'pattern': parsed.get('PATTERN'),
                'key_levels': parsed.get('KEY_LEVELS'),
                'indicators': parsed.get('INDICATORS')
            }
        )
        
        self.analysis_history.append(analysis)
        return analysis
    
    def _generate_opening_statement(self, analysis: AgentAnalysis) -> str:
        """Generate opening debate statement"""
        key_point = analysis.reasoning[0] if analysis.reasoning else "Technical setup present"
        return f"The charts clearly show a {analysis.recommendation.value} opportunity. {key_point}. My confidence is {analysis.confidence}% based on pure technical factors."
    
    def _generate_counter_argument(self, opposing: AgentAnalysis) -> str:
        """Generate counter-argument"""
        if opposing.agent_type == AgentType.FUNDAMENTAL_ANALYST:
            return "While fundamentals matter long-term, the technical setup is clear right now. Price action doesn't lie."
        elif opposing.agent_type == AgentType.RISK_MANAGER:
            return f"The risk is well-defined with stops at {self.analysis_history[-1].stop_loss}. Technical levels provide clear invalidation."
        else:
            return "The technical picture overrides other concerns in this timeframe. Charts reflect all known information."
    
    def _generate_closing_statement(self, all_analyses: List[AgentAnalysis]) -> str:
        """Generate closing statement"""
        agreement_count = sum(1 for a in all_analyses if a.recommendation == self.analysis_history[-1].recommendation)
        if agreement_count >= 4:
            return "The technical analysis aligns with the majority view. This strengthens the setup."
        else:
            return "Despite differing opinions, the technical setup remains valid. I maintain my position."