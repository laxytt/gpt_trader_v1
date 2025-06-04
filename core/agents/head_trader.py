"""
Head Trader Agent
The decision maker who synthesizes all views and makes final calls
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter

from core.agents.base_agent import TradingAgent, AgentType, AgentAnalysis, DebateResponse
from core.domain.models import MarketData, SignalType, RiskClass
from core.infrastructure.gpt.client import GPTClient

logger = logging.getLogger(__name__)


class HeadTrader(TradingAgent):
    """The Decider - synthesizes all views and makes final decisions"""
    
    def __init__(self, gpt_client: GPTClient):
        super().__init__(
            agent_type=AgentType.HEAD_TRADER,
            personality="balanced and decisive head trader",
            specialty="synthesizing opinions, final decisions, and strategic thinking"
        )
        self.gpt_client = gpt_client
        self.decision_history = []
    
    def analyze(
        self,
        market_data: Dict[str, MarketData],
        news_context: Optional[List[str]] = None,
        ml_context: Optional[Dict[str, Any]] = None
    ) -> AgentAnalysis:
        """Head trader doesn't do independent analysis - synthesizes others"""
        # This method won't be used in normal flow
        # Head trader only acts after hearing from others
        raise NotImplementedError("Head trader synthesizes other analyses, doesn't analyze independently")
    
    def synthesize_decision(
        self,
        agent_analyses: List[AgentAnalysis],
        debate_log: List[DebateResponse],
        market_data: Dict[str, MarketData],
        ml_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[AgentAnalysis, Dict[str, Any]]:
        """
        Synthesize all inputs into final trading decision
        
        Returns:
            Tuple of (final_decision, council_summary)
        """
        
        h1_data = market_data.get('h1')
        current_price = h1_data.latest_candle.close if h1_data else 0
        symbol = h1_data.symbol if h1_data else "UNKNOWN"
        
        # Analyze votes and confidence
        vote_summary = self._analyze_votes(agent_analyses)
        debate_insights = self._extract_debate_insights(debate_log)
        
        # Build synthesis prompt
        synthesis_prompt = self._build_prompt("""
As Head Trader, synthesize the council's analysis for {symbol} at {price}.

Council Votes:
{vote_details}

Vote Summary:
- BUY votes: {buy_votes} (avg confidence: {buy_conf:.1f})
- SELL votes: {sell_votes} (avg confidence: {sell_conf:.1f})
- WAIT votes: {wait_votes} (avg confidence: {wait_conf:.1f})

Key Debate Points:
{debate_summary}

ML Context:
{ml_summary}

Your task:
1. Weigh all perspectives considering their expertise
2. Identify the strongest arguments
3. Make the final decision
4. Set specific entry, stop loss, and take profit
5. Determine position sizing based on conviction
6. Assign risk class (A/B/C)

Consider:
- Technical and fundamentals alignment
- Risk manager's concerns
- Market sentiment
- Momentum vs contrarian views

Provide:
DECISION: [BUY/SELL/WAIT]
CONFIDENCE: [0-100]
ENTRY: [price]
STOP_LOSS: [price]
TAKE_PROFIT: [price]
RISK_CLASS: [A/B/C]
POSITION_SIZE: [0.1-1.0 lots]
KEY_RATIONALE: [main reason for decision]
DISSENT_ADDRESSED: [how you addressed opposing views]
EXECUTION_NOTES: [specific execution instructions]
""",
            symbol=symbol,
            price=current_price,
            vote_details=self._format_vote_details(agent_analyses),
            buy_votes=vote_summary['buy_count'],
            buy_conf=vote_summary['buy_avg_confidence'],
            sell_votes=vote_summary['sell_count'],
            sell_conf=vote_summary['sell_avg_confidence'],
            wait_votes=vote_summary['wait_count'],
            wait_conf=vote_summary['wait_avg_confidence'],
            debate_summary=self._format_debate_summary(debate_insights),
            ml_summary=self._format_ml_context(ml_context)
        )
        
        response = self.gpt_client.analyze_with_response(
            synthesis_prompt,
            temperature=0.3  # Low temperature for decisive action
        )
        
        # Parse decision
        final_decision = self._parse_synthesis(response, current_price, agent_analyses)
        
        # Create council summary
        council_summary = {
            'vote_summary': vote_summary,
            'debate_insights': debate_insights,
            'dissenting_views': self._identify_dissent(agent_analyses, final_decision),
            'consensus_level': self._calculate_consensus(agent_analyses),
            'decision_rationale': final_decision.reasoning
        }
        
        self.decision_history.append({
            'decision': final_decision,
            'council_summary': council_summary,
            'timestamp': h1_data.latest_candle.timestamp if h1_data else None
        })
        
        return final_decision, council_summary
    
    def debate(
        self,
        other_analyses: List[AgentAnalysis],
        round_number: int,
        previous_responses: Optional[List[DebateResponse]] = None
    ) -> DebateResponse:
        """Head trader moderates but doesn't debate until synthesis"""
        
        if round_number < 3:
            # Moderate the debate
            statement = self._moderate_debate(round_number, other_analyses, previous_responses)
        else:
            # Prepare for synthesis
            statement = "I've heard all perspectives. Let me now synthesize our decision."
        
        return DebateResponse(
            agent_type=self.agent_type,
            round=round_number,
            statement=statement,
            maintains_position=True,
            updated_confidence=None  # No position yet
        )
    
    def _analyze_votes(self, analyses: List[AgentAnalysis]) -> Dict[str, Any]:
        """Analyze voting patterns"""
        votes = Counter(a.recommendation for a in analyses)
        
        # Calculate average confidence by recommendation
        buy_confidences = [a.confidence for a in analyses if a.recommendation == SignalType.BUY]
        sell_confidences = [a.confidence for a in analyses if a.recommendation == SignalType.SELL]
        wait_confidences = [a.confidence for a in analyses if a.recommendation == SignalType.WAIT]
        
        return {
            'buy_count': votes[SignalType.BUY],
            'sell_count': votes[SignalType.SELL],
            'wait_count': votes[SignalType.WAIT],
            'buy_avg_confidence': sum(buy_confidences) / len(buy_confidences) if buy_confidences else 0,
            'sell_avg_confidence': sum(sell_confidences) / len(sell_confidences) if sell_confidences else 0,
            'wait_avg_confidence': sum(wait_confidences) / len(wait_confidences) if wait_confidences else 0,
            'total_votes': len(analyses)
        }
    
    def _extract_debate_insights(self, debate_log: List[DebateResponse]) -> Dict[str, List[str]]:
        """Extract key insights from debate"""
        insights = {
            'agreements': [],
            'disagreements': [],
            'key_concerns': [],
            'position_changes': []
        }
        
        # Track position changes
        for response in debate_log:
            if not response.maintains_position:
                insights['position_changes'].append(
                    f"{response.agent_type.value} changed position in round {response.round}"
                )
        
        # Extract key statements (simplified for now)
        for response in debate_log:
            if response.round == 2:  # Counter-arguments round
                insights['disagreements'].append(response.statement[:100] + "...")
        
        return insights
    
    def _format_vote_details(self, analyses: List[AgentAnalysis]) -> str:
        """Format individual votes for prompt"""
        lines = []
        for analysis in analyses:
            lines.append(
                f"- {analysis.agent_type.value}: {analysis.recommendation.value} "
                f"(confidence: {analysis.confidence}%) - {analysis.reasoning[0] if analysis.reasoning else 'No reason'}"
            )
        return "\n".join(lines)
    
    def _format_debate_summary(self, insights: Dict[str, List[str]]) -> str:
        """Format debate insights for prompt"""
        summary = []
        
        if insights['position_changes']:
            summary.append(f"Position changes: {', '.join(insights['position_changes'])}")
        
        if insights['disagreements']:
            summary.append(f"Key disagreements: {len(insights['disagreements'])} points of contention")
        
        return "\n".join(summary) if summary else "No significant debate developments"
    
    def _format_ml_context(self, ml_context: Optional[Dict[str, Any]]) -> str:
        """Format ML context for prompt"""
        if not ml_context:
            return "No ML input available"
        
        return f"ML Signal: {ml_context.get('signal', 'None')}, Confidence: {ml_context.get('confidence', 0)}%"
    
    def _parse_synthesis(
        self,
        response: str,
        current_price: float,
        agent_analyses: List[AgentAnalysis]
    ) -> AgentAnalysis:
        """Parse head trader's synthesis into final decision"""
        lines = response.strip().split('\n')
        parsed = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                parsed[key.strip()] = value.strip()
        
        # Extract decision
        decision_text = parsed.get('DECISION', 'WAIT').upper()
        if 'BUY' in decision_text:
            recommendation = SignalType.BUY
        elif 'SELL' in decision_text:
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
            # Use best levels from agents
            entry, stop_loss, take_profit = self._extract_best_levels(agent_analyses, recommendation, current_price)
        
        # Risk class
        risk_class_text = parsed.get('RISK_CLASS', 'C')
        risk_class = RiskClass.A if 'A' in risk_class_text else RiskClass.B if 'B' in risk_class_text else RiskClass.C
        
        # Position size
        try:
            position_size = float(parsed.get('POSITION_SIZE', '0.1'))
        except:
            position_size = 0.1
        
        # Build reasoning
        reasoning = []
        if parsed.get('KEY_RATIONALE'):
            reasoning.append(parsed['KEY_RATIONALE'])
        if parsed.get('DISSENT_ADDRESSED'):
            reasoning.append(f"Dissent addressed: {parsed['DISSENT_ADDRESSED']}")
        if parsed.get('EXECUTION_NOTES'):
            reasoning.append(f"Execution: {parsed['EXECUTION_NOTES']}")
        
        return AgentAnalysis(
            agent_type=self.agent_type,
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning[:3],
            concerns=[],  # Head trader has already addressed concerns
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=self._calculate_risk_reward(entry, stop_loss, take_profit) if stop_loss and take_profit else None,
            metadata={
                'risk_class': risk_class.value,
                'position_size': position_size,
                'execution_notes': parsed.get('EXECUTION_NOTES', '')
            }
        )
    
    def _extract_best_levels(
        self,
        analyses: List[AgentAnalysis],
        recommendation: SignalType,
        current_price: float
    ) -> Tuple[float, float, float]:
        """Extract best levels from agent analyses"""
        # Prefer technical analyst's levels
        tech_analysis = next((a for a in analyses if a.agent_type == AgentType.TECHNICAL_ANALYST), None)
        
        if tech_analysis and tech_analysis.stop_loss and tech_analysis.take_profit:
            return tech_analysis.entry_price or current_price, tech_analysis.stop_loss, tech_analysis.take_profit
        
        # Fallback to risk manager's levels
        risk_analysis = next((a for a in analyses if a.agent_type == AgentType.RISK_MANAGER), None)
        
        if risk_analysis and risk_analysis.stop_loss and risk_analysis.take_profit:
            return risk_analysis.entry_price or current_price, risk_analysis.stop_loss, risk_analysis.take_profit
        
        # Default levels
        atr = 0.0010  # 10 pips default
        if recommendation == SignalType.BUY:
            return current_price, current_price - 2*atr, current_price + 3*atr
        elif recommendation == SignalType.SELL:
            return current_price, current_price + 2*atr, current_price - 3*atr
        else:
            return current_price, 0, 0
    
    def _moderate_debate(
        self,
        round_number: int,
        analyses: List[AgentAnalysis],
        previous_responses: Optional[List[DebateResponse]]
    ) -> str:
        """Moderate the debate rounds"""
        if round_number == 1:
            vote_summary = self._analyze_votes(analyses)
            if vote_summary['buy_count'] == vote_summary['total_votes']:
                return "Unanimous buy recommendation. Let's ensure we're not missing any risks."
            elif vote_summary['sell_count'] == vote_summary['total_votes']:
                return "Unanimous sell recommendation. Let's confirm this isn't groupthink."
            else:
                return "We have diverse views. Let's explore the key disagreements."
        else:
            return "Good points raised. Any final considerations before I synthesize?"
    
    def _identify_dissent(
        self,
        analyses: List[AgentAnalysis],
        final_decision: AgentAnalysis
    ) -> List[Dict[str, Any]]:
        """Identify dissenting views from final decision"""
        dissenters = []
        
        for analysis in analyses:
            if analysis.recommendation != final_decision.recommendation and analysis.confidence > 60:
                dissenters.append({
                    'agent': analysis.agent_type.value,
                    'wanted': analysis.recommendation.value,
                    'confidence': analysis.confidence,
                    'main_reason': analysis.reasoning[0] if analysis.reasoning else "No reason given"
                })
        
        return dissenters
    
    def _calculate_consensus(self, analyses: List[AgentAnalysis]) -> float:
        """Calculate consensus level (0-100)"""
        if not analyses:
            return 0
        
        # Count most common recommendation
        votes = Counter(a.recommendation for a in analyses)
        most_common_count = votes.most_common(1)[0][1]
        
        # Consensus percentage
        consensus = (most_common_count / len(analyses)) * 100
        
        # Adjust for confidence alignment
        if consensus > 50:
            avg_confidence = sum(a.confidence for a in analyses) / len(analyses)
            confidence_variance = sum((a.confidence - avg_confidence) ** 2 for a in analyses) / len(analyses)
            
            # Lower consensus if confidence varies widely
            if confidence_variance > 400:  # Standard deviation > 20
                consensus *= 0.8
        
        return consensus