"""
Fundamental Analyst Agent
Specializes in economic data, news, and macro trends
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from core.agents.base_agent import TradingAgent, AgentType, AgentAnalysis, DebateResponse
from core.domain.models import MarketData, SignalType
from core.infrastructure.gpt.client import GPTClient

logger = logging.getLogger(__name__)


class FundamentalAnalyst(TradingAgent):
    """The Economist - focuses on fundamental analysis"""
    
    def __init__(self, gpt_client: GPTClient):
        super().__init__(
            agent_type=AgentType.FUNDAMENTAL_ANALYST,
            personality="news-driven and macro-focused economist",
            specialty="economic data, central bank policy, news impact, and cross-market correlations"
        )
        self.gpt_client = gpt_client
    
    def analyze(
        self,
        market_data: Dict[str, MarketData],
        news_context: Optional[List[str]] = None,
        ml_context: Optional[Dict[str, Any]] = None
    ) -> AgentAnalysis:
        """Perform fundamental analysis"""
        
        h1_data = market_data.get('h1')
        h4_data = market_data.get('h4')
        symbol = h1_data.symbol if h1_data else "UNKNOWN"
        
        # Extract currency pair components
        base_currency = symbol[:3] if len(symbol) >= 6 else "XXX"
        quote_currency = symbol[3:6] if len(symbol) >= 6 else "XXX"
        
        # Format news context
        news_summary = "\n".join(news_context[:5]) if news_context else "No recent high-impact news"
        
        analysis_prompt = self._build_prompt("""
Analyze {symbol} from a fundamental perspective.

Currency Pair: {base}/{quote}
Current Price: {current_price}

Recent Economic Context:
{news_summary}

Consider:
1. Current monetary policy stance for both currencies
2. Recent economic data trends
3. Risk sentiment (risk-on vs risk-off)
4. Cross-market correlations (bonds, commodities, equities)
5. Geopolitical factors

Provide:
1. Fundamental bias for each currency
2. Key macro drivers currently
3. Impact of recent/upcoming news
4. Your trading recommendation (BUY/SELL/WAIT)
5. Confidence level (0-100)
6. Time horizon for fundamental view
7. Main fundamental risk

Format your response as:
BASE_BIAS: [Bullish/Bearish/Neutral] - [reason]
QUOTE_BIAS: [Bullish/Bearish/Neutral] - [reason]
MACRO_DRIVERS: [key drivers affecting pair]
NEWS_IMPACT: [how news affects outlook]
RECOMMENDATION: [BUY/SELL/WAIT]
CONFIDENCE: [0-100]
TIME_HORIZON: [hours/days/weeks]
RISK: [main fundamental risk]
""",
            symbol=symbol,
            base=base_currency,
            quote=quote_currency,
            current_price=h1_data.latest_candle.close if h1_data else 0,
            news_summary=news_summary
        )
        
        # Get GPT analysis
        response = self.gpt_client.analyze_with_response(
            analysis_prompt,
            temperature=0.4
        )
        
        return self._parse_analysis(response, h1_data.latest_candle.close if h1_data else 0)
    
    def debate(
        self,
        other_analyses: List[AgentAnalysis],
        round_number: int,
        previous_responses: Optional[List[DebateResponse]] = None
    ) -> DebateResponse:
        """Participate in debate with fundamental arguments"""
        
        my_position = self.analysis_history[-1]
        
        if round_number == 1:
            statement = self._generate_opening_statement(my_position)
        elif round_number == 2:
            # Address technical analyst if they disagree
            tech_analysis = next((a for a in other_analyses if a.agent_type == AgentType.TECHNICAL_ANALYST), None)
            if tech_analysis and tech_analysis.recommendation != my_position.recommendation:
                statement = f"The technical picture may suggest {tech_analysis.recommendation.value}, but fundamentals drive the bigger moves. {my_position.reasoning[0]}"
            else:
                statement = self._support_allied_view(other_analyses)
        else:
            statement = self._generate_closing_statement(other_analyses)
        
        # Fundamental analyst may adjust confidence based on technical alignment
        updated_confidence = my_position.confidence
        tech_analysis = next((a for a in other_analyses if a.agent_type == AgentType.TECHNICAL_ANALYST), None)
        if tech_analysis:
            if tech_analysis.recommendation == my_position.recommendation:
                updated_confidence = min(100, my_position.confidence + 10)
            else:
                updated_confidence = max(0, my_position.confidence - 5)
        
        return DebateResponse(
            agent_type=self.agent_type,
            round=round_number,
            statement=statement,
            maintains_position=True,
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
        
        # Determine recommendation based on biases
        base_bias = parsed.get('BASE_BIAS', '')
        quote_bias = parsed.get('QUOTE_BIAS', '')
        
        if 'Bullish' in base_bias and 'Bearish' in quote_bias:
            recommendation = SignalType.BUY
        elif 'Bearish' in base_bias and 'Bullish' in quote_bias:
            recommendation = SignalType.SELL
        else:
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
        
        # Build reasoning
        reasoning = []
        if parsed.get('BASE_BIAS'):
            reasoning.append(f"Base currency: {parsed['BASE_BIAS']}")
        if parsed.get('MACRO_DRIVERS'):
            reasoning.append(f"Macro drivers: {parsed['MACRO_DRIVERS']}")
        if parsed.get('NEWS_IMPACT'):
            reasoning.append(f"News impact: {parsed['NEWS_IMPACT']}")
        
        # Concerns
        concerns = []
        if parsed.get('RISK'):
            concerns.append(parsed['RISK'])
        
        analysis = AgentAnalysis(
            agent_type=self.agent_type,
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning[:3],  # Limit to 3 points
            concerns=concerns,
            entry_price=current_price,  # Fundamental doesn't specify exact entry
            metadata={
                'base_bias': parsed.get('BASE_BIAS'),
                'quote_bias': parsed.get('QUOTE_BIAS'),
                'time_horizon': parsed.get('TIME_HORIZON', 'medium-term'),
                'macro_drivers': parsed.get('MACRO_DRIVERS')
            }
        )
        
        self.analysis_history.append(analysis)
        return analysis
    
    def _generate_opening_statement(self, analysis: AgentAnalysis) -> str:
        """Generate opening debate statement"""
        horizon = analysis.metadata.get('time_horizon', 'medium-term')
        return f"From a fundamental perspective, {analysis.recommendation.value} is the clear choice. {analysis.reasoning[0]}. This view is based on {horizon} fundamentals."
    
    def _support_allied_view(self, analyses: List[AgentAnalysis]) -> str:
        """Support agents who agree"""
        my_rec = self.analysis_history[-1].recommendation
        allies = [a for a in analyses if a.recommendation == my_rec and a.agent_type != self.agent_type]
        
        if allies:
            ally = allies[0]
            return f"I agree with the {ally.agent_type.value}. The fundamentals support this view, adding conviction to the setup."
        else:
            return "The fundamental picture is clear despite mixed technical signals. Major trends are driven by fundamentals."
    
    def _generate_closing_statement(self, all_analyses: List[AgentAnalysis]) -> str:
        """Generate closing statement"""
        my_analysis = self.analysis_history[-1]
        time_horizon = my_analysis.metadata.get('time_horizon', 'medium-term')
        
        if time_horizon in ['hours', 'short-term']:
            return "While fundamentals favor this direction, be aware this is a shorter-term fundamental view. Manage risk accordingly."
        else:
            return f"The fundamental thesis is strong for the {time_horizon}. This provides a solid foundation for the trade."