"""
Base Trading Agent
Foundation for all specialized trading council members
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from core.domain.models import MarketData, TradingSignal, SignalType, RiskClass
from core.domain.exceptions import AgentError


class AgentType(Enum):
    """Types of trading agents in the council"""
    TECHNICAL_ANALYST = "technical_analyst"
    FUNDAMENTAL_ANALYST = "fundamental_analyst"
    SENTIMENT_READER = "sentiment_reader"
    RISK_MANAGER = "risk_manager"
    MOMENTUM_TRADER = "momentum_trader"
    CONTRARIAN_TRADER = "contrarian_trader"
    HEAD_TRADER = "head_trader"


@dataclass
class AgentAnalysis:
    """Result of an agent's analysis"""
    agent_type: AgentType
    recommendation: SignalType  # BUY, SELL, WAIT
    confidence: float  # 0-100
    reasoning: List[str]  # Key points
    concerns: List[str]  # Risk factors
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    metadata: Dict[str, Any] = None


@dataclass
class DebateResponse:
    """Agent's response during debate"""
    agent_type: AgentType
    round: int
    statement: str
    maintains_position: bool
    updated_confidence: Optional[float] = None
    addressing_agent: Optional[AgentType] = None


class TradingAgent(ABC):
    """Base class for all trading council agents"""
    
    def __init__(self, agent_type: AgentType, personality: str, specialty: str):
        self.agent_type = agent_type
        self.personality = personality
        self.specialty = specialty
        self.speaking_style = self._get_speaking_style()
        self.analysis_history = []
        self.performance_stats = {
            'total_analyses': 0,
            'correct_calls': 0,
            'accuracy': 0.0
        }
    
    @abstractmethod
    def analyze(
        self,
        market_data: Dict[str, MarketData],
        news_context: Optional[List[str]] = None,
        ml_context: Optional[Dict[str, Any]] = None
    ) -> AgentAnalysis:
        """
        Perform analysis and provide recommendation
        
        Args:
            market_data: Dict with 'h1' and 'h4' MarketData
            news_context: Relevant news items
            ml_context: ML model predictions and confidence
            
        Returns:
            AgentAnalysis with recommendation and reasoning
        """
        pass
    
    @abstractmethod
    def debate(
        self,
        other_analyses: List[AgentAnalysis],
        round_number: int,
        previous_responses: Optional[List[DebateResponse]] = None
    ) -> DebateResponse:
        """
        Participate in council debate
        
        Args:
            other_analyses: Analyses from other agents
            round_number: Current debate round (1-3)
            previous_responses: Previous debate responses
            
        Returns:
            DebateResponse with agent's position
        """
        pass
    
    def _get_speaking_style(self) -> str:
        """Get agent-specific speaking style"""
        styles = {
            AgentType.TECHNICAL_ANALYST: "precise and methodical",
            AgentType.FUNDAMENTAL_ANALYST: "macro-focused and analytical",
            AgentType.SENTIMENT_READER: "intuitive and psychological",
            AgentType.RISK_MANAGER: "cautious and protective",
            AgentType.MOMENTUM_TRADER: "aggressive and trend-following",
            AgentType.CONTRARIAN_TRADER: "skeptical and contrarian",
            AgentType.HEAD_TRADER: "balanced and decisive"
        }
        return styles.get(self.agent_type, "professional")
    
    def _calculate_risk_reward(
        self,
        entry: float,
        stop_loss: float,
        take_profit: float
    ) -> float:
        """Calculate risk/reward ratio"""
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        return reward / risk if risk > 0 else 0
    
    def _format_reasoning(self, points: List[str]) -> List[str]:
        """Format reasoning points in agent's style"""
        formatted = []
        for point in points[:3]:  # Limit to 3 key points
            formatted.append(f"{point}")
        return formatted
    
    def update_performance(self, was_correct: bool):
        """Update agent's performance statistics"""
        self.performance_stats['total_analyses'] += 1
        if was_correct:
            self.performance_stats['correct_calls'] += 1
        
        self.performance_stats['accuracy'] = (
            self.performance_stats['correct_calls'] / 
            self.performance_stats['total_analyses']
        )
    
    def get_confidence_modifiers(self) -> Dict[str, float]:
        """Get factors that modify confidence based on agent type"""
        # Override in subclasses for specific modifiers
        return {
            'base_confidence': 50.0,
            'agreement_bonus': 10.0,
            'disagreement_penalty': -10.0
        }
    
    def _build_prompt(self, template: str, **kwargs) -> str:
        """Build prompt with agent personality"""
        personality_prefix = f"""You are a {self.personality} {self.agent_type.value} in a professional trading council.
Your specialty is {self.specialty}.
Your speaking style is {self.speaking_style}.

"""
        return personality_prefix + template.format(**kwargs)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.agent_type.value}, accuracy={self.performance_stats['accuracy']:.1%})"