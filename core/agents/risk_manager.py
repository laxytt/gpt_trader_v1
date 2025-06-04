"""
Risk Manager Agent
Specializes in position sizing, risk assessment, and capital preservation
"""

import logging
from typing import Dict, List, Optional, Any

from core.agents.base_agent import TradingAgent, AgentType, AgentAnalysis, DebateResponse
from core.domain.models import MarketData, SignalType
from core.infrastructure.gpt.client import GPTClient

logger = logging.getLogger(__name__)


class RiskManager(TradingAgent):
    """The Guardian - protects capital and manages risk"""
    
    def __init__(self, gpt_client: GPTClient, account_balance: float = 10000, risk_per_trade: float = 0.02):
        super().__init__(
            agent_type=AgentType.RISK_MANAGER,
            personality="conservative and protective risk guardian",
            specialty="position sizing, stop placement, risk/reward ratios, and capital preservation"
        )
        self.gpt_client = gpt_client
        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade  # 2% default
        self.max_risk_per_trade = risk_per_trade * account_balance
    
    def analyze(
        self,
        market_data: Dict[str, MarketData],
        news_context: Optional[List[str]] = None,
        ml_context: Optional[Dict[str, Any]] = None
    ) -> AgentAnalysis:
        """Analyze from risk management perspective"""
        
        h1_data = market_data.get('h1')
        h4_data = market_data.get('h4')
        
        if not h1_data:
            raise ValueError("H1 data required for risk analysis")
        
        current_price = h1_data.latest_candle.close
        atr = h1_data.latest_candle.atr14
        spread = h1_data.latest_candle.spread or 0.0001
        
        # Calculate key risk metrics
        atr_pips = atr * 10000  # Convert to pips
        spread_pips = spread * 10000
        
        # Check recent volatility
        recent_candles = h1_data.candles[-24:]  # Last 24 hours
        max_move = max(c.high for c in recent_candles) - min(c.low for c in recent_candles)
        max_move_pips = max_move * 10000
        
        analysis_prompt = self._build_prompt("""
Analyze {symbol} from a risk management perspective.

Current Price: {price}
ATR (14): {atr_pips:.1f} pips
Current Spread: {spread_pips:.1f} pips
24h Range: {max_move_pips:.1f} pips
Account Risk per Trade: ${max_risk:.2f} ({risk_percent}%)

Market Conditions:
- Volatility: {volatility_state}
- Spread vs ATR: {spread_ratio:.1%}

Assess:
1. Is current volatility manageable?
2. Are spreads acceptable for trading?
3. Can we place stops at safe technical levels?
4. What's the maximum position size?
5. Risk/reward potential
6. Correlation risk with open positions

Provide:
1. Risk assessment (Low/Medium/High)
2. Recommended position size (in lots)
3. Suggested stop distance (in pips)
4. Risk/reward evaluation
5. Trading recommendation (BUY/SELL/WAIT)
6. Confidence level (0-100)
7. Key risk factors

Format your response as:
RISK_LEVEL: [Low/Medium/High]
POSITION_SIZE: [lots]
STOP_DISTANCE: [pips]
RISK_REWARD: [ratio]
MAX_LOSS: [dollar amount]
RECOMMENDATION: [BUY/SELL/WAIT]
CONFIDENCE: [0-100]
KEY_RISKS: [main risk factors]
""",
            symbol=h1_data.symbol,
            price=current_price,
            atr_pips=atr_pips,
            spread_pips=spread_pips,
            max_move_pips=max_move_pips,
            max_risk=self.max_risk_per_trade,
            risk_percent=self.risk_per_trade * 100,
            volatility_state="High" if atr_pips > 15 else "Normal" if atr_pips > 8 else "Low",
            spread_ratio=spread/atr if atr > 0 else 0
        )
        
        # Risk Manager uses GPT-4 for its veto power
        from config.settings import get_settings
        settings = get_settings()
        
        response = self.gpt_client.analyze_with_response(
            analysis_prompt,
            temperature=0.2,  # Very low temperature for conservative risk assessment
            model_override=settings.gpt.risk_manager_model
        )
        
        return self._parse_analysis(response, current_price)
    
    def debate(
        self,
        other_analyses: List[AgentAnalysis],
        round_number: int,
        previous_responses: Optional[List[DebateResponse]] = None
    ) -> DebateResponse:
        """Participate in debate focusing on risk"""
        
        my_position = self.analysis_history[-1]
        
        # Risk manager has veto power on high-risk trades
        high_risk_agents = [
            a for a in other_analyses 
            if a.agent_type in [AgentType.MOMENTUM_TRADER, AgentType.SENTIMENT_READER]
            and a.recommendation != SignalType.WAIT
            and a.confidence > 80
        ]
        
        if round_number == 1:
            statement = self._generate_opening_statement(my_position)
        elif round_number == 2 and high_risk_agents:
            statement = self._address_aggressive_traders(high_risk_agents[0])
        else:
            statement = self._generate_closing_statement(other_analyses, my_position)
        
        # Risk manager may strengthen position if others are too aggressive
        updated_confidence = my_position.confidence
        
        aggressive_count = sum(1 for a in other_analyses if a.recommendation != SignalType.WAIT and a.confidence > 70)
        if aggressive_count >= 4 and my_position.recommendation == SignalType.WAIT:
            updated_confidence = min(100, my_position.confidence + 20)  # Stronger conviction to wait
        
        return DebateResponse(
            agent_type=self.agent_type,
            round=round_number,
            statement=statement,
            maintains_position=True,  # Risk manager rarely changes position
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
        
        # Risk manager is conservative - high bar for trades
        risk_level = parsed.get('RISK_LEVEL', 'High').upper()
        
        if 'HIGH' in risk_level:
            recommendation = SignalType.WAIT
            confidence = 80.0  # High confidence in waiting
        else:
            rec_text = parsed.get('RECOMMENDATION', 'WAIT').upper()
            if 'BUY' in rec_text and 'LOW' in risk_level:
                recommendation = SignalType.BUY
            elif 'SELL' in rec_text and 'LOW' in risk_level:
                recommendation = SignalType.SELL
            else:
                recommendation = SignalType.WAIT
            
            try:
                confidence = float(parsed.get('CONFIDENCE', '50'))
            except:
                confidence = 50.0
        
        # Extract risk metrics
        try:
            stop_distance = float(parsed.get('STOP_DISTANCE', '20').replace('pips', '').strip())
            risk_reward = float(parsed.get('RISK_REWARD', '1').replace(':1', '').strip())
            position_size = float(parsed.get('POSITION_SIZE', '0.1'))
        except:
            stop_distance = 20
            risk_reward = 1
            position_size = 0.1
        
        # Calculate stop loss price
        if recommendation == SignalType.BUY:
            stop_loss = current_price - (stop_distance / 10000)
            take_profit = current_price + (stop_distance * risk_reward / 10000)
        elif recommendation == SignalType.SELL:
            stop_loss = current_price + (stop_distance / 10000)
            take_profit = current_price - (stop_distance * risk_reward / 10000)
        else:
            stop_loss = 0
            take_profit = 0
        
        # Build reasoning
        reasoning = []
        reasoning.append(f"Risk level: {risk_level}")
        if parsed.get('RISK_REWARD'):
            reasoning.append(f"Risk/Reward: {parsed['RISK_REWARD']}")
        reasoning.append(f"Position size: {position_size} lots")
        
        concerns = []
        if parsed.get('KEY_RISKS'):
            concerns.append(parsed['KEY_RISKS'])
        
        analysis = AgentAnalysis(
            agent_type=self.agent_type,
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning[:3],
            concerns=concerns,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward,
            metadata={
                'risk_level': risk_level,
                'position_size': position_size,
                'stop_distance': stop_distance,
                'max_loss': parsed.get('MAX_LOSS', 'Unknown')
            }
        )
        
        self.analysis_history.append(analysis)
        return analysis
    
    def _generate_opening_statement(self, analysis: AgentAnalysis) -> str:
        """Generate opening debate statement"""
        risk_level = analysis.metadata.get('risk_level', 'Unknown')
        if analysis.recommendation == SignalType.WAIT:
            return f"Risk assessment shows {risk_level} risk. We should wait for better conditions. Capital preservation is paramount."
        else:
            return f"Risk is {risk_level} and manageable. {analysis.reasoning[1]}. We can proceed with proper position sizing."
    
    def _address_aggressive_traders(self, aggressive_agent: AgentAnalysis) -> str:
        """Address aggressive trading suggestions"""
        return f"The {aggressive_agent.agent_type.value} suggests aggressive entry, but consider the downside. One bad trade can erase multiple winners. Let's wait for optimal risk/reward."
    
    def _generate_closing_statement(self, all_analyses: List[AgentAnalysis], my_position: AgentAnalysis) -> str:
        """Generate closing statement"""
        risk_level = my_position.metadata.get('risk_level', 'Unknown')
        
        if my_position.recommendation == SignalType.WAIT:
            return f"With {risk_level} risk identified, protecting capital takes priority. No trade is better than a bad trade."
        else:
            position_size = my_position.metadata.get('position_size', 0.1)
            return f"If we proceed, strict risk management is essential. Maximum {position_size} lots with stops at technical levels."