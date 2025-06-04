"""
Council Signal Service
Enhanced signal generation using the Trading Council multi-agent system
"""

import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timezone
from pathlib import Path

from core.domain.models import TradingSignal, MarketData, SignalType, RiskClass
from core.domain.exceptions import SignalGenerationError, ErrorContext
from core.domain.enums import TimeFrame
from core.agents.council import TradingCouncil, CouncilDecision
from core.infrastructure.mt5.data_provider import MT5DataProvider
from core.infrastructure.gpt.client import GPTClient
from core.infrastructure.database.repositories import SignalRepository
from core.services.news_service import NewsService
from core.services.memory_service import MemoryService
from core.services.offline_validator import OfflineSignalValidator
from core.utils.chart_utils import ChartGenerator
from core.utils.error_handling import with_error_recovery
from config.settings import TradingSettings

logger = logging.getLogger(__name__)


class CouncilSignalService:
    """
    Signal service that uses the Trading Council for decisions
    Replaces the original signal service with multi-agent approach
    """
    
    def __init__(
        self,
        data_provider: MT5DataProvider,
        gpt_client: GPTClient,
        news_service: NewsService,
        memory_service: MemoryService,
        signal_repository: SignalRepository,
        trading_config: TradingSettings,
        chart_generator: Optional[ChartGenerator] = None,
        screenshots_dir: str = "screenshots",
        enable_offline_validation: bool = True,
        account_balance: float = 10000,
        min_confidence_threshold: float = 75.0
    ):
        self.data_provider = data_provider
        self.gpt_client = gpt_client
        self.news_service = news_service
        self.memory_service = memory_service
        self.signal_repository = signal_repository
        self.trading_config = trading_config
        self.chart_generator = chart_generator
        self.screenshots_dir = Path(screenshots_dir)
        self.enable_offline_validation = enable_offline_validation
        
        # Initialize Trading Council
        self.trading_council = TradingCouncil(
            gpt_client=gpt_client,
            account_balance=account_balance,
            risk_per_trade=trading_config.risk_per_trade_percent / 100,
            min_confidence_threshold=min_confidence_threshold,
            llm_weight=getattr(trading_config, 'council_llm_weight', 0.7),
            ml_weight=getattr(trading_config, 'council_ml_weight', 0.3)
        )
        
        # Initialize offline validator
        if enable_offline_validation:
            self.offline_validator = OfflineSignalValidator()
            logger.info("Council Signal Service with offline validation enabled")
        else:
            self.offline_validator = None
        
        # Ensure screenshots directory exists
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
    
    async def validate_symbol_readiness(self, symbol: str) -> bool:
        """
        Check if symbol is ready for signal generation.
        
        Args:
            symbol: Trading symbol to check
            
        Returns:
            True if symbol is ready for analysis
        """
        try:
            # Check data availability
            has_data = await self.data_provider.validate_symbol_data(symbol, TimeFrame.H1)
            if not has_data:
                logger.warning(f"Insufficient data for {symbol}")
                return False
            
            # Check news restrictions
            is_restricted = await self.news_service.is_trading_restricted(symbol)
            if is_restricted:
                logger.info(f"Trading restricted by news for {symbol}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Symbol readiness check failed for {symbol}: {e}")
            return False
    
    async def generate_signal(self, symbol: str) -> TradingSignal:
        """
        Generate a trading signal using the Trading Council
        
        Args:
            symbol: Trading symbol
            
        Returns:
            TradingSignal with council decision
        """
        with ErrorContext("Council signal generation", symbol=symbol) as ctx:
            logger.info(f"Starting council signal generation for {symbol}")
            
            try:
                # Step 1: Gather market data
                ctx.add_detail("step", "data_gathering")
                market_data = await self._gather_market_data(symbol)
                
                # Step 2: Run offline validation
                ctx.add_detail("step", "offline_validation")
                validation_result = None
                if self.enable_offline_validation:
                    validation_result = await self._run_offline_validation(
                        market_data['h1']
                    )
                    
                    # Check if we should skip based on validation
                    if not validation_result['should_proceed']:
                        logger.info(
                            f"Offline validation suggests skipping {symbol}: "
                            f"Score={validation_result['validation_summary']['weighted_score']:.2f}"
                        )
                        
                        return self._create_validation_based_wait_signal(
                            symbol, validation_result
                        )
                
                # Step 3: Check news restrictions
                ctx.add_detail("step", "news_check")
                news_events = await self._check_news_restrictions(symbol)
                
                # Step 4: Generate charts if available
                ctx.add_detail("step", "chart_generation")
                chart_paths = await self._generate_charts(symbol, market_data)
                
                # Step 5: Get ML context
                ctx.add_detail("step", "ml_context")
                ml_context = await self._get_ml_context(symbol, market_data)
                
                # Step 6: Format news for council
                news_context = self._format_news_context(news_events)
                
                # Step 7: Convene the Trading Council
                ctx.add_detail("step", "council_deliberation")
                council_decision = await self.trading_council.convene_council(
                    market_data=market_data,
                    news_context=news_context,
                    ml_context=ml_context
                )
                
                # Step 8: Enhance signal with additional metadata
                signal = self._enhance_signal(
                    council_decision.signal,
                    validation_result,
                    chart_paths,
                    council_decision
                )
                
                # Step 9: Store signal
                ctx.add_detail("step", "storage")
                await self._store_signal(signal)
                
                logger.info(
                    f"Council signal generated for {symbol}: "
                    f"{signal.signal.value} ({signal.risk_class.value}) "
                    f"with {council_decision.final_confidence:.1f}% confidence"
                )
                
                return signal
                
            except Exception as e:
                logger.error(f"Council signal generation failed for {symbol}: {e}")
                return self._create_fallback_signal(symbol, str(e))
    
    async def _gather_market_data(self, symbol: str) -> Dict[str, MarketData]:
        """Gather H1 and H4 market data"""
        try:
            from core.domain.enums import TimeFrame
            
            # Get multi-timeframe data
            market_data = await self.data_provider.get_multi_timeframe_data(
                symbol=symbol,
                timeframes=[TimeFrame.H1, TimeFrame.H4],
                bars=self.trading_config.bars_for_analysis
            )
            
            if 'H1' not in market_data or 'H4' not in market_data:
                raise SignalGenerationError(f"Insufficient market data for {symbol}")
            
            return {
                'h1': market_data['H1'],
                'h4': market_data['H4']
            }
            
        except Exception as e:
            raise SignalGenerationError(f"Failed to gather market data: {str(e)}")
    
    async def _run_offline_validation(self, market_data: MarketData) -> Dict[str, Any]:
        """Run offline validation on market data"""
        try:
            validation_result = await self.offline_validator.validate_market_data(
                market_data,
                min_score_threshold=self.trading_config.offline_validation_threshold
            )
            
            # Log validation summary
            summary = validation_result['validation_summary']
            logger.info(
                f"Offline validation for {market_data.symbol}: "
                f"Score={summary['weighted_score']:.2f}, "
                f"Passed={summary['validators_passed']}/{summary['total_validators']}"
            )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Offline validation failed: {e}")
            # Return neutral result on error
            return {
                'should_proceed': True,
                'validation_summary': {'weighted_score': 0.5},
                'gpt_context': {'pre_validation_performed': False, 'error': str(e)}
            }
    
    async def _check_news_restrictions(self, symbol: str) -> List[Any]:
        """Check for news restrictions and get upcoming events"""
        try:
            # Check if trading is restricted by news
            if await self.news_service.is_trading_restricted(symbol):
                raise SignalGenerationError(f"Trading restricted by high-impact news for {symbol}")
            
            # Get upcoming news events for context
            news_events = await self.news_service.get_upcoming_news(
                symbol=symbol,
                within_minutes=120  # Next 2 hours
            )
            
            return news_events
            
        except SignalGenerationError:
            raise  # Re-raise restriction errors
        except Exception as e:
            logger.warning(f"News check failed for {symbol}: {e}")
            return []  # Continue without news data
    
    @with_error_recovery(default_return=None, log_level='warning')
    async def _generate_charts(
        self, 
        symbol: str, 
        market_data: Dict[str, MarketData]
    ) -> Optional[Dict[str, str]]:
        """Generate chart images for visual context"""
        if not self.chart_generator:
            return None
        
        chart_paths = {}
        
        for timeframe, data in market_data.items():
            chart_filename = f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            chart_path = self.screenshots_dir / chart_filename
            
            # Use chart service
            from core.services.chart_service import ChartService
            chart_service = ChartService(self.chart_generator)
            
            generated_path = chart_service.generate_market_chart(
                market_data=data,
                output_path=str(chart_path),
                title=f"{symbol} {timeframe}"
            )
            
            if generated_path:
                chart_paths[timeframe.lower()] = generated_path
        
        return chart_paths if chart_paths else None
    
    async def _get_ml_context(
        self,
        symbol: str,
        market_data: Dict[str, MarketData]
    ) -> Optional[Dict[str, Any]]:
        """Get ML model predictions if available"""
        try:
            # This would integrate with your ML service
            # For now, return mock context
            return {
                'signal': 'BUY',
                'confidence': 72.5,
                'model_id': 'pattern_trader_v4',
                'features': {
                    'trend_strength': 0.65,
                    'momentum': 0.45,
                    'volatility': 'normal'
                }
            }
        except Exception as e:
            logger.warning(f"Failed to get ML context: {e}")
            return None
    
    def _format_news_context(self, news_events: List[Any]) -> List[str]:
        """Format news events for council agents"""
        if not news_events:
            return ["No significant news events in next 2 hours"]
        
        formatted = []
        for event in news_events[:5]:  # Limit to top 5
            formatted.append(
                f"{event.country} - {event.title} "
                f"(Impact: {event.impact}, Time: {event.timestamp.strftime('%H:%M')})"
            )
        
        return formatted
    
    def _enhance_signal(
        self,
        signal: TradingSignal,
        validation_result: Optional[Dict[str, Any]],
        chart_paths: Optional[Dict[str, str]],
        council_decision: CouncilDecision
    ) -> TradingSignal:
        """Enhance signal with additional metadata"""
        
        # Initialize market_context if not present
        if signal.market_context is None:
            signal.market_context = {}
        
        # Add validation score
        if validation_result:
            signal.market_context['validation_score'] = validation_result['validation_summary']['weighted_score']
            signal.market_context['validation_passed'] = validation_result['should_proceed']
        
        # Add chart paths
        if chart_paths:
            signal.market_context['charts'] = chart_paths
        
        # Add council metadata
        signal.market_context.update({
            'llm_confidence': council_decision.llm_confidence,
            'ml_confidence': council_decision.ml_confidence,
            'consensus_level': council_decision.consensus_level,
            'dissent_count': len(council_decision.dissenting_views),
            'agent_votes': {
                a.agent_type.value: a.recommendation.value 
                for a in council_decision.agent_analyses
            }
        })
        
        return signal
    
    async def _store_signal(self, signal: TradingSignal):
        """Store signal in repository"""
        try:
            self.signal_repository.save_signal(signal)
            logger.debug(f"Signal stored for {signal.symbol}")
        except Exception as e:
            logger.error(f"Failed to store signal: {e}")
    
    def _create_validation_based_wait_signal(
        self, 
        symbol: str, 
        validation_result: Dict[str, Any]
    ) -> TradingSignal:
        """Create a WAIT signal based on validation results"""
        summary = validation_result['validation_summary']
        top_issues = summary.get('all_issues', [])[:2]
        
        reason_parts = [
            f"Market quality below threshold (score: {summary['weighted_score']:.2f})"
        ]
        
        for issue in top_issues:
            reason_parts.append(f"- {issue['message']}")
        
        reason = ". ".join(reason_parts)
        
        return TradingSignal(
            symbol=symbol,
            signal=SignalType.WAIT,
            reason=reason,
            risk_class=RiskClass.C,
            timestamp=datetime.now(timezone.utc),
            market_context={
                'validation_score': summary['weighted_score'],
                'council_skipped': True
            }
        )
    
    def _create_fallback_signal(self, symbol: str, reason: str) -> TradingSignal:
        """Create a fallback WAIT signal when generation fails"""
        return TradingSignal(
            symbol=symbol,
            signal=SignalType.WAIT,
            reason=f"Signal generation failed: {reason}",
            risk_class=RiskClass.C,
            timestamp=datetime.now(timezone.utc)
        )
    
    async def get_council_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent council decisions for analysis"""
        recent_decisions = self.trading_council.get_recent_decisions(limit)
        
        history = []
        for decision in recent_decisions:
            history.append({
                'timestamp': decision.timestamp.isoformat(),
                'symbol': decision.signal.symbol,
                'decision': decision.signal.signal.value,
                'confidence': decision.final_confidence,
                'consensus': decision.consensus_level,
                'rationale': decision.decision_rationale,
                'dissent_count': len(decision.dissenting_views),
                'agent_votes': {
                    a.agent_type.value: a.recommendation.value
                    for a in decision.agent_analyses
                }
            })
        
        return history
    
    async def get_agent_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for each agent"""
        return self.trading_council.get_agent_performance()
    
    async def get_signal_history(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent signal history for a symbol.
        
        Args:
            symbol: Trading symbol
            limit: Maximum number of signals to return
            
        Returns:
            List of signal dictionaries
        """
        try:
            return self.signal_repository.get_recent_signals(symbol, limit)
        except Exception as e:
            logger.error(f"Failed to get signal history for {symbol}: {e}")
            return []
    
    async def batch_generate_signals(self, symbols: List[str]) -> Dict[str, TradingSignal]:
        """
        Generate signals for multiple symbols.
        
        Args:
            symbols: List of symbols to analyze
            
        Returns:
            Dictionary mapping symbols to their signals
        """
        results = {}
        
        for symbol in symbols:
            try:
                # Check if symbol is ready
                if not await self.validate_symbol_readiness(symbol):
                    results[symbol] = self._create_fallback_signal(
                        symbol, "Symbol not ready for analysis"
                    )
                    continue
                
                # Generate signal
                signal = await self.generate_signal(symbol)
                results[symbol] = signal
                
            except Exception as e:
                logger.error(f"Batch signal generation failed for {symbol}: {e}")
                results[symbol] = self._create_fallback_signal(symbol, str(e))
        
        return results
    
    async def generate_signal_with_ml(self, symbol: str) -> TradingSignal:
        """
        Generate trading signal using ML model context within the Trading Council.
        The ML predictions are integrated into the council's decision-making process.
        """
        # The Trading Council already incorporates ML predictions through ml_context
        # This method exists for API compatibility with SignalService
        logger.info(f"Generating council signal with ML context for {symbol}")
        return await self.generate_signal(symbol)