"""
Trading Council
Orchestrates the multi-agent trading decision process
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
import numpy as np

from core.agents.base_agent import AgentType, AgentAnalysis, DebateResponse
from core.agents.technical_analyst import TechnicalAnalyst
from core.agents.fundamental_analyst import FundamentalAnalyst
from core.agents.sentiment_reader import SentimentReader
from core.agents.risk_manager import RiskManager
from core.agents.momentum_trader import MomentumTrader
from core.agents.contrarian_trader import ContrarianTrader
from core.agents.head_trader import HeadTrader
from core.domain.models import MarketData, TradingSignal, SignalType, RiskClass
from core.infrastructure.gpt.client import GPTClient
from core.services.council_optimizer import CouncilOptimizer

logger = logging.getLogger(__name__)


@dataclass
class CouncilDecision:
    """Complete council decision package"""
    signal: TradingSignal
    agent_analyses: List[AgentAnalysis]
    debate_log: List[DebateResponse]
    llm_confidence: float
    ml_confidence: float
    final_confidence: float
    consensus_level: float
    dissenting_views: List[Dict[str, Any]]
    decision_rationale: str
    timestamp: datetime


class TradingCouncil:
    """
    Orchestrates multi-agent trading decisions
    Implements the debate process and confidence fusion
    """
    
    def __init__(
        self,
        gpt_client: GPTClient,
        account_balance: float = 10000,
        risk_per_trade: float = 0.02,
        min_confidence_threshold: float = 75.0,
        llm_weight: float = 0.7,
        ml_weight: float = 0.3,
        agent_delay: float = 0.5,
        trading_config: Optional[Any] = None
    ):
        """
        Initialize the trading council
        
        Args:
            gpt_client: GPT client for agent communication
            account_balance: Account balance for risk calculations
            risk_per_trade: Risk percentage per trade
            min_confidence_threshold: Minimum confidence to trade
            llm_weight: Weight for LLM confidence (0-1)
            ml_weight: Weight for ML confidence (0-1)
            trading_config: Trading configuration for optimization
        """
        self.gpt_client = gpt_client
        self.min_confidence = min_confidence_threshold
        self.llm_weight = llm_weight
        self.ml_weight = ml_weight
        self.agent_delay = agent_delay
        
        # Initialize all agents
        self.agents = {
            AgentType.TECHNICAL_ANALYST: TechnicalAnalyst(gpt_client),
            AgentType.FUNDAMENTAL_ANALYST: FundamentalAnalyst(gpt_client),
            AgentType.SENTIMENT_READER: SentimentReader(gpt_client),
            AgentType.RISK_MANAGER: RiskManager(gpt_client, account_balance, risk_per_trade),
            AgentType.MOMENTUM_TRADER: MomentumTrader(gpt_client),
            AgentType.CONTRARIAN_TRADER: ContrarianTrader(gpt_client),
            AgentType.HEAD_TRADER: HeadTrader(gpt_client)
        }
        
        self.council_history = []
        
        # Initialize optimizer if config provided
        self.optimizer = CouncilOptimizer(trading_config) if trading_config else None
    
    def _get_majority_decision(self, agent_analyses: List[AgentAnalysis]) -> AgentAnalysis:
        """Get decision based on majority vote WITH risk manager veto power"""
        
        # CRITICAL: Check for Risk Manager veto first
        risk_manager_analysis = next(
            (a for a in agent_analyses if a.agent_type == AgentType.RISK_MANAGER), 
            None
        )
        
        if risk_manager_analysis:
            # Risk Manager has veto power on high-risk trades
            if risk_manager_analysis.recommendation == SignalType.WAIT:
                # Check if it's a strong veto (confidence > 80%)
                if risk_manager_analysis.confidence >= 80:
                    return AgentAnalysis(
                        agent_type=AgentType.HEAD_TRADER,
                        recommendation=SignalType.WAIT,
                        confidence=risk_manager_analysis.confidence,
                        reasoning=[
                            "Risk Manager VETO: Trade rejected due to unacceptable risk",
                            *risk_manager_analysis.reasoning
                        ],
                        concerns=risk_manager_analysis.concerns,
                        metadata={'veto': True, 'veto_by': 'risk_manager'}
                    )
            
            # Also check for explicit high risk in metadata
            if risk_manager_analysis.metadata and risk_manager_analysis.metadata.get('risk_level') == 'High':
                return AgentAnalysis(
                    agent_type=AgentType.HEAD_TRADER,
                    recommendation=SignalType.WAIT,
                    confidence=90.0,  # High confidence in avoiding high risk
                    reasoning=[
                        "Risk Manager VETO: High risk level detected",
                        *risk_manager_analysis.concerns
                    ],
                    concerns=risk_manager_analysis.concerns,
                    metadata={'veto': True, 'veto_by': 'risk_manager', 'risk_level': 'High'}
                )
        
        # If no veto, proceed with normal voting
        votes = {'BUY': [], 'SELL': [], 'WAIT': []}
        
        for analysis in agent_analyses:
            if analysis.recommendation == SignalType.BUY:
                votes['BUY'].append(analysis)
            elif analysis.recommendation == SignalType.SELL:
                votes['SELL'].append(analysis)
            else:
                votes['WAIT'].append(analysis)
        
        # Find majority
        max_votes = max(len(votes['BUY']), len(votes['SELL']), len(votes['WAIT']))
        
        if len(votes['BUY']) == max_votes and len(votes['BUY']) > len(votes['WAIT']):
            # Average the buy analyses
            avg_confidence = sum(a.confidence for a in votes['BUY']) / len(votes['BUY'])
            return AgentAnalysis(
                agent_type=AgentType.HEAD_TRADER,
                recommendation=SignalType.BUY,
                confidence=avg_confidence,
                reasoning=["Majority vote: BUY (Risk Manager approved)"],
                concerns=[],
                entry_price=votes['BUY'][0].entry_price,
                stop_loss=votes['BUY'][0].stop_loss,
                take_profit=votes['BUY'][0].take_profit
            )
        elif len(votes['SELL']) == max_votes and len(votes['SELL']) > len(votes['WAIT']):
            avg_confidence = sum(a.confidence for a in votes['SELL']) / len(votes['SELL'])
            return AgentAnalysis(
                agent_type=AgentType.HEAD_TRADER,
                recommendation=SignalType.SELL,
                confidence=avg_confidence,
                reasoning=["Majority vote: SELL (Risk Manager approved)"],
                concerns=[],
                entry_price=votes['SELL'][0].entry_price,
                stop_loss=votes['SELL'][0].stop_loss,
                take_profit=votes['SELL'][0].take_profit
            )
        else:
            return AgentAnalysis(
                agent_type=AgentType.HEAD_TRADER,
                recommendation=SignalType.WAIT,
                confidence=50.0,
                reasoning=["Majority vote: WAIT or no consensus"],
                concerns=[]
            )
    
    def _count_votes(self, agent_analyses: List[AgentAnalysis]) -> Dict[str, Any]:
        """Count votes from analyses"""
        votes = {'buy_count': 0, 'sell_count': 0, 'wait_count': 0,
                'buy_avg_confidence': 0, 'sell_avg_confidence': 0, 'wait_avg_confidence': 0}
        
        buy_confidences = []
        sell_confidences = []
        wait_confidences = []
        
        for analysis in agent_analyses:
            if analysis.recommendation == SignalType.BUY:
                votes['buy_count'] += 1
                buy_confidences.append(analysis.confidence)
            elif analysis.recommendation == SignalType.SELL:
                votes['sell_count'] += 1
                sell_confidences.append(analysis.confidence)
            else:
                votes['wait_count'] += 1
                wait_confidences.append(analysis.confidence)
        
        if buy_confidences:
            votes['buy_avg_confidence'] = sum(buy_confidences) / len(buy_confidences)
        if sell_confidences:
            votes['sell_avg_confidence'] = sum(sell_confidences) / len(sell_confidences)
        if wait_confidences:
            votes['wait_avg_confidence'] = sum(wait_confidences) / len(wait_confidences)
            
        return votes
    
    def _calculate_consensus_level(self, agent_analyses: List[AgentAnalysis]) -> float:
        """Calculate consensus level (0-1)"""
        if not agent_analyses:
            return 0.0
            
        recommendations = [a.recommendation for a in agent_analyses]
        most_common = max(set(recommendations), key=recommendations.count)
        consensus_count = recommendations.count(most_common)
        
        return consensus_count / len(agent_analyses)
    
    async def convene_council(
        self,
        market_data: Dict[str, MarketData],
        news_context: Optional[List[str]] = None,
        ml_context: Optional[Dict[str, Any]] = None
    ) -> CouncilDecision:
        """
        Convene the full trading council for a decision
        
        Args:
            market_data: Market data dict with 'h1' and 'h4' keys
            news_context: Recent news items
            ml_context: ML predictions and confidence
            
        Returns:
            Complete council decision
        """
        logger.info(f"Convening trading council for {market_data.get('h1', {}).symbol}")
        
        try:
            # Phase 1: Individual Analysis (parallel)
            agent_analyses = await self._conduct_individual_analysis(
                market_data, news_context, ml_context
            )
            
            # Check if quick mode is enabled
            from config.settings import get_settings
            settings = get_settings()
            
            if settings.trading.council_quick_mode:
                # Skip debates and use simple majority vote
                debate_log = []
                final_decision = self._get_majority_decision(agent_analyses)
                council_summary = {
                    'vote_summary': self._count_votes(agent_analyses),
                    'debate_insights': ["Quick mode - no debate conducted"],
                    'dissenting_views': [],
                    'consensus_level': self._calculate_consensus_level(agent_analyses),
                    'decision_rationale': final_decision.reasoning if hasattr(final_decision, 'reasoning') else []
                }
            else:
                # Phase 2: Three-round debate
                debate_log = await self._conduct_debate(agent_analyses, market_data)
                
                # Phase 3: Head trader synthesis
                final_decision, council_summary = await self._synthesize_decision(
                    agent_analyses, debate_log, market_data, ml_context
                )
            
            # Phase 4: Calculate confidence scores
            llm_confidence = self._calculate_llm_confidence(
                agent_analyses, debate_log, council_summary
            )
            
            ml_confidence = ml_context.get('confidence', 50.0) if ml_context else 50.0
            
            # Final confidence fusion
            final_confidence = (self.llm_weight * llm_confidence) + (self.ml_weight * ml_confidence)
            
            # Create trading signal
            symbol = market_data.get('h1', {}).symbol or "UNKNOWN"
            
            if final_decision.recommendation == SignalType.WAIT or final_confidence < self.min_confidence:
                signal = self._create_wait_signal(
                    symbol, final_decision, final_confidence
                )
            else:
                signal = self._create_trading_signal(
                    symbol, final_decision, final_confidence, council_summary
                )
            
            # Package complete decision
            decision = CouncilDecision(
                signal=signal,
                agent_analyses=agent_analyses,
                debate_log=debate_log,
                llm_confidence=llm_confidence,
                ml_confidence=ml_confidence,
                final_confidence=final_confidence,
                consensus_level=council_summary['consensus_level'],
                dissenting_views=council_summary['dissenting_views'],
                decision_rationale=council_summary['decision_rationale'][0] if council_summary['decision_rationale'] else "Council decision",
                timestamp=datetime.now()
            )
            
            # Store in history
            self.council_history.append(decision)
            
            logger.info(
                f"Council decision: {signal.signal.value} with {final_confidence:.1f}% confidence "
                f"(LLM: {llm_confidence:.1f}%, ML: {ml_confidence:.1f}%)"
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Council error: {e}", exc_info=True)
            # Return safe WAIT signal on error
            return self._create_error_decision(market_data, str(e))
    
    async def _conduct_individual_analysis(
        self,
        market_data: Dict[str, MarketData],
        news_context: Optional[List[str]],
        ml_context: Optional[Dict[str, Any]]
    ) -> List[AgentAnalysis]:
        """Phase 1: Get individual analysis from each agent (except head trader)"""
        
        # Since analyze methods are synchronous, run them in executor
        loop = asyncio.get_event_loop()
        analysis_tasks = []
        
        for agent_type, agent in self.agents.items():
            if agent_type != AgentType.HEAD_TRADER:  # Head trader doesn't analyze independently
                # Run synchronous analyze method in thread pool
                analysis_tasks.append(
                    loop.run_in_executor(
                        None,
                        agent.analyze,
                        market_data,
                        news_context,
                        ml_context
                    )
                )
        
        # Run analyses sequentially with delays to avoid rate limits
        analyses = []
        agent_delay = self.agent_delay  # Use configurable delay
        
        for i, task in enumerate(analysis_tasks):
            if i > 0:  # Add delay after first agent
                await asyncio.sleep(agent_delay)
            
            try:
                analysis = await task
                analyses.append(analysis)
            except Exception as e:
                logger.error(f"Agent analysis failed: {e}")
                analyses.append(e)
        
        # Filter out any errors
        valid_analyses = []
        for i, analysis in enumerate(analyses):
            if isinstance(analysis, Exception):
                logger.error(f"Agent analysis error: {analysis}")
            else:
                valid_analyses.append(analysis)
        
        return valid_analyses
    
    async def _conduct_debate(
        self,
        agent_analyses: List[AgentAnalysis],
        market_data: Dict[str, MarketData]
    ) -> List[DebateResponse]:
        """Phase 2: Conduct debate with optimization"""
        
        debate_log = []
        
        # Check if we need full debate using optimizer
        if self.optimizer:
            # Extract initial signals for optimization check
            initial_signals = {
                str(a.agent_type.value): {
                    'signal': a.recommendation,
                    'confidence': a.confidence
                }
                for a in agent_analyses
            }
            
            needs_debate, reason = self.optimizer.needs_full_debate(
                initial_signals, 
                list(market_data.values())[0] if market_data else None
            )
            
            if not needs_debate:
                logger.info(f"Skipping full debate: {reason}")
                self.optimizer.update_stats('immediate' if reason in ['risk_veto', 'obvious_pattern'] else 'early_stop', 1)
                
                # Create minimal debate log
                for analysis in agent_analyses:
                    response = DebateResponse(
                        agent_type=analysis.agent_type,
                        round=1,
                        maintains_position=True,
                        points=[f"Initial position: {analysis.recommendation.value}"],
                        confidence_change=0
                    )
                    debate_log.append(response)
                
                return debate_log
        
        # Conduct full debate with potential early stopping
        max_rounds = 3 if not self.optimizer else self.optimizer.trading_config.council_debate_rounds
        round_results = []
        
        for round_number in range(1, max_rounds + 1):
            logger.debug(f"Starting debate round {round_number}")
            
            round_responses = []
            
            # Each agent responds to the current state
            for analysis in agent_analyses:
                agent = self.agents[analysis.agent_type]
                
                # Get previous responses for this round
                previous_responses = [r for r in debate_log if r.round < round_number]
                
                response = agent.debate(
                    agent_analyses,
                    round_number,
                    previous_responses
                )
                
                round_responses.append(response)
            
            # Head trader moderates
            head_trader = self.agents[AgentType.HEAD_TRADER]
            moderation = head_trader.debate(
                agent_analyses,
                round_number,
                round_responses
            )
            round_responses.append(moderation)
            
            # Add all responses to debate log
            debate_log.extend(round_responses)
            
            # Check if we should continue debate
            if self.optimizer and round_number < max_rounds:
                # Calculate current consensus
                current_consensus = self._calculate_consensus(round_responses)
                round_results.append({
                    'confidence': current_consensus * 100,
                    'consensus_level': current_consensus
                })
                
                continue_debate, stop_reason = self.optimizer.optimize_debate_rounds(
                    round_results,
                    self.min_confidence
                )
                
                if not continue_debate:
                    logger.info(f"Early stopping debate at round {round_number}: {stop_reason}")
                    self.optimizer.update_stats('early_stop', round_number)
                    break
        
        if self.optimizer and round_number == max_rounds:
            self.optimizer.update_stats('full_debate', max_rounds)
        
        return debate_log
    
    async def _synthesize_decision(
        self,
        agent_analyses: List[AgentAnalysis],
        debate_log: List[DebateResponse],
        market_data: Dict[str, MarketData],
        ml_context: Optional[Dict[str, Any]]
    ) -> Tuple[AgentAnalysis, Dict[str, Any]]:
        """Phase 3: Head trader synthesizes final decision"""
        
        head_trader = self.agents[AgentType.HEAD_TRADER]
        
        final_decision, council_summary = head_trader.synthesize_decision(
            agent_analyses,
            debate_log,
            market_data,
            ml_context
        )
        
        return final_decision, council_summary
    
    def _calculate_llm_confidence(
        self,
        agent_analyses: List[AgentAnalysis],
        debate_log: List[DebateResponse],
        council_summary: Dict[str, Any]
    ) -> float:
        """Calculate LLM confidence based on council dynamics"""
        
        # 1. Agreement level (0-40 points)
        consensus = council_summary['consensus_level']
        agreement_score = (consensus / 100) * 40
        
        # 2. Debate quality (0-30 points)
        position_changes = sum(1 for r in debate_log if not r.maintains_position)
        debate_score = 30  # Start with full score
        
        # Penalize too many position changes (uncertainty)
        if position_changes > 3:
            debate_score -= 10
        elif position_changes == 0:
            debate_score -= 5  # No evolution of thinking
        
        # 3. Confidence alignment (0-30 points)
        confidences = [a.confidence for a in agent_analyses]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 50
        confidence_std = (
            sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
        ) ** 0.5 if confidences else 0
        
        # Better if confidences are aligned (low std dev)
        if confidence_std < 10:
            context_score = 30
        elif confidence_std < 20:
            context_score = 20
        else:
            context_score = 10
        
        # Total LLM confidence
        llm_confidence = agreement_score + debate_score + context_score
        
        # Boost if risk manager agrees with majority
        risk_analysis = next((a for a in agent_analyses if a.agent_type == AgentType.RISK_MANAGER), None)
        if risk_analysis:
            vote_summary = council_summary['vote_summary']
            majority_rec = max(
                SignalType.BUY if vote_summary['buy_count'] >= vote_summary['sell_count'] else SignalType.SELL,
                SignalType.WAIT,
                key=lambda x: vote_summary[f'{x.value.lower()}_count']
            )
            
            if risk_analysis.recommendation == majority_rec:
                llm_confidence = min(100, llm_confidence + 10)
        
        return llm_confidence
    
    def _create_trading_signal(
        self,
        symbol: str,
        decision: AgentAnalysis,
        confidence: float,
        council_summary: Dict[str, Any]
    ) -> TradingSignal:
        """Create trading signal from council decision"""
        
        # Extract metadata
        risk_class_str = decision.metadata.get('risk_class', 'C')
        risk_class = RiskClass.A if risk_class_str == 'A' else RiskClass.B if risk_class_str == 'B' else RiskClass.C
        
        # Build comprehensive rationale
        rationale_parts = [
            f"Council decision with {confidence:.0f}% confidence.",
            f"Consensus level: {council_summary['consensus_level']:.0f}%."
        ]
        
        if decision.reasoning:
            rationale_parts.append(decision.reasoning[0])
        
        if council_summary['dissenting_views']:
            dissent_count = len(council_summary['dissenting_views'])
            rationale_parts.append(f"{dissent_count} dissenting view(s) considered.")
        
        return TradingSignal(
            symbol=symbol,
            signal=decision.recommendation,
            entry=decision.entry_price,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            risk_class=risk_class,
            reason=" ".join(rationale_parts),
            timestamp=datetime.now(),
            market_context={
                'council_decision': True,
                'consensus_level': council_summary['consensus_level'],
                'position_size': decision.metadata.get('position_size', 0.1),
                'execution_notes': decision.metadata.get('execution_notes', ''),
                'dissent_count': len(council_summary['dissenting_views']),
                'confidence': int(confidence)
            }
        )
    
    def _create_wait_signal(
        self,
        symbol: str,
        decision: AgentAnalysis,
        confidence: float
    ) -> TradingSignal:
        """Create WAIT signal when confidence is too low or decision is WAIT"""
        
        if decision.recommendation == SignalType.WAIT:
            reason = f"Council recommends waiting. {decision.reasoning[0] if decision.reasoning else 'No clear opportunity.'}"
        else:
            reason = f"Confidence too low ({confidence:.0f}% < {self.min_confidence}% threshold)"
        
        return TradingSignal(
            symbol=symbol,
            signal=SignalType.WAIT,
            reason=reason,
            risk_class=RiskClass.C,
            timestamp=datetime.now(),
            market_context={
                'council_decision': True,
                'below_threshold': confidence < self.min_confidence,
                'confidence': int(confidence)
            }
        )
    
    def _create_error_decision(
        self,
        market_data: Dict[str, MarketData],
        error: str
    ) -> CouncilDecision:
        """Create error decision when council fails"""
        
        symbol = market_data.get('h1', {}).symbol or "UNKNOWN"
        
        signal = TradingSignal(
            symbol=symbol,
            signal=SignalType.WAIT,
            reason=f"Council error: {error}",
            risk_class=RiskClass.C,
            timestamp=datetime.now(),
            market_context={
                'confidence': 0,
                'error': True
            }
        )
        
        return CouncilDecision(
            signal=signal,
            agent_analyses=[],
            debate_log=[],
            llm_confidence=0,
            ml_confidence=0,
            final_confidence=0,
            consensus_level=0,
            dissenting_views=[],
            decision_rationale=f"Error: {error}",
            timestamp=datetime.now()
        )
    
    def get_agent_performance(self) -> Dict[AgentType, Dict[str, Any]]:
        """Get performance statistics for each agent"""
        
        performance = {}
        
        for agent_type, agent in self.agents.items():
            if hasattr(agent, 'performance_stats'):
                performance[agent_type] = agent.performance_stats
        
        return performance
    
    def get_recent_decisions(self, limit: int = 10) -> List[CouncilDecision]:
        """Get recent council decisions"""
        return self.council_history[-limit:]
    
    def _calculate_consensus(self, responses: List[DebateResponse]) -> float:
        """Calculate consensus level from debate responses"""
        if not responses:
            return 0.0
        
        # Count agents maintaining positions
        maintaining = sum(1 for r in responses if r.maintains_position)
        total = len(responses)
        
        # Base consensus on position stability
        consensus = maintaining / total
        
        # Adjust for confidence changes
        confidence_changes = [abs(r.confidence_change) for r in responses]
        avg_change = np.mean(confidence_changes) if confidence_changes else 0
        
        # High confidence changes indicate uncertainty
        if avg_change > 10:
            consensus *= 0.8
        
        return consensus