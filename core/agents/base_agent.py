"""
Base Trading Agent
Foundation for all specialized trading council members
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

from core.domain.models import MarketData, TradingSignal, SignalType, RiskClass
from core.domain.exceptions import AgentError

logger = logging.getLogger(__name__)


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
    
    def _safe_parse_response(self, response: str, current_price: float) -> Dict[str, Any]:
        """
        Safely parse agent response with comprehensive error handling
        
        Args:
            response: Raw GPT response
            current_price: Current market price for fallback
            
        Returns:
            Dict with parsed values and safe defaults
        """
        parsed = {}
        
        try:
            # Parse key-value pairs
            lines = response.strip().split('\n')
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    parsed[key.strip().upper()] = value.strip()
        except Exception as e:
            # Log parsing error but continue with defaults
            parsed = {}
        
        # Safe extraction with defaults
        result = {
            'recommendation': SignalType.WAIT,
            'confidence': 50.0,
            'entry': current_price,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'pattern': None,
            'indicators': None,
            'trend': None,
            'concern': None,
            'reasoning': [],
            'metadata': {}
        }
        
        # Extract recommendation safely
        try:
            rec_text = parsed.get('RECOMMENDATION', 'WAIT').upper()
            if 'BUY' in rec_text:
                result['recommendation'] = SignalType.BUY
            elif 'SELL' in rec_text:
                result['recommendation'] = SignalType.SELL
        except (AttributeError, TypeError) as e:
            logger.debug(f"Failed to parse recommendation: {e}")
        
        # Extract confidence safely
        try:
            conf_str = parsed.get('CONFIDENCE', '50')
            # Remove % sign if present
            conf_str = conf_str.replace('%', '').strip()
            result['confidence'] = max(0.0, min(100.0, float(conf_str)))
        except (ValueError, AttributeError, TypeError) as e:
            logger.debug(f"Failed to parse confidence: {e}")
        
        # Extract price levels safely
        try:
            if 'ENTRY' in parsed:
                result['entry'] = float(parsed['ENTRY'])
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to parse entry price: {e}")
            
        try:
            if 'STOP_LOSS' in parsed:
                result['stop_loss'] = float(parsed['STOP_LOSS'])
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to parse stop loss: {e}")
            
        try:
            if 'TAKE_PROFIT' in parsed:
                result['take_profit'] = float(parsed['TAKE_PROFIT'])
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to parse take profit: {e}")
        
        # Extract other fields safely - comprehensive list from all agents
        for field in ['PATTERN', 'INDICATORS', 'TREND', 'CONCERN', 'KEY_LEVELS',
                     'MOMENTUM_STATE', 'TREND_QUALITY', 'ENTRY_STRATEGY', 'TARGETS', 'INVALIDATION',
                     'MOOD', 'CROWD', 'PSYCH_LEVELS', 'SENTIMENT_SIGNS', 'TRIGGERS',
                     'RISK_LEVEL', 'POSITION_SIZE', 'STOP_DISTANCE', 'RISK_REWARD', 'MAX_LOSS', 'KEY_RISKS',
                     'OPPORTUNITY', 'SETUP_TYPE', 'CROWD_ERROR', 'REVERSAL_TRIGGERS', 'ENTRY_ZONE',
                     'BASE_BIAS', 'QUOTE_BIAS', 'MACRO_DRIVERS', 'NEWS_IMPACT', 'SENTIMENT_IMPACT', 
                     'TIME_HORIZON', 'RISK', 'CORRELATIONS', 'VAR_BREACH']:
            if field in parsed:
                result['metadata'][field] = parsed[field]
        
        # Build reasoning from available data
        if result['pattern'] and result['pattern'] != "None clear":
            result['reasoning'].append(f"Pattern identified: {result['pattern']}")
        if result['indicators']:
            result['reasoning'].append(f"Indicators show: {result['indicators']}")
        if result['trend']:
            result['reasoning'].append(f"Trend assessment: {result['trend']}")
        
        # Store all parsed data in metadata for reference
        result['metadata'] = {
            'raw_parsed': parsed,
            'pattern': result.get('pattern'),
            'indicators': result.get('indicators'),
            'key_levels': result.get('key_levels')
        }
        
        return result
    
    def _calculate_risk_reward(self, entry: float, stop_loss: float, take_profit: float) -> Optional[float]:
        """Calculate risk/reward ratio"""
        if not all([entry, stop_loss, take_profit]) or entry == 0 or stop_loss == 0:
            return None
            
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        
        if risk == 0:
            return None
            
        return reward / risk