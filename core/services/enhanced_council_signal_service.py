"""
Enhanced Council Signal Service with MarketAux integration
Uses enhanced news service for sentiment-aware trading decisions
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
from core.services.enhanced_news_service import EnhancedNewsService, EnhancedNewsContext
from core.services.memory_service import MemoryService
from core.services.offline_validator import OfflineSignalValidator
from core.ml.ml_predictor import MLPredictor
from core.utils.chart_utils import ChartGenerator
from core.utils.error_handling import with_error_recovery
from config.settings import TradingSettings, MarketAuxSettings
from core.infrastructure.cache.market_state_cache import MarketStateCache
from core.services.pre_trade_filter import PreTradeFilter

logger = logging.getLogger(__name__)


class EnhancedCouncilSignalService:
    """
    Enhanced signal service that uses Trading Council with MarketAux sentiment
    """
    
    def __init__(
        self,
        data_provider: MT5DataProvider,
        gpt_client: GPTClient,
        enhanced_news_service: EnhancedNewsService,
        memory_service: MemoryService,
        signal_repository: SignalRepository,
        trading_config: TradingSettings,
        marketaux_config: MarketAuxSettings,
        chart_generator: Optional[ChartGenerator] = None,
        screenshots_dir: str = "screenshots",
        enable_offline_validation: bool = True,
        account_balance: float = 10000,
        min_confidence_threshold: float = 75.0,
        ml_predictor: Optional[MLPredictor] = None
    ):
        self.data_provider = data_provider
        self.gpt_client = gpt_client
        self.enhanced_news_service = enhanced_news_service
        self.memory_service = memory_service
        self.signal_repository = signal_repository
        self.trading_config = trading_config
        self.marketaux_config = marketaux_config
        self.chart_generator = chart_generator
        self.screenshots_dir = Path(screenshots_dir)
        self.enable_offline_validation = enable_offline_validation
        self.account_balance = account_balance
        self.min_confidence_threshold = min_confidence_threshold
        
        # Initialize ML predictor
        self.ml_predictor = ml_predictor or MLPredictor()
        logger.info(f"ML Predictor initialized with models for: {list(self.ml_predictor.loaded_models.keys())}")
        
        # Initialize Trading Council
        self.trading_council = TradingCouncil(
            gpt_client=gpt_client,
            account_balance=account_balance,
            risk_per_trade=trading_config.risk_per_trade_percent / 100,
            min_confidence_threshold=min_confidence_threshold,
            llm_weight=getattr(trading_config, 'council_llm_weight', 0.7),
            ml_weight=getattr(trading_config, 'council_ml_weight', 0.3),
            agent_delay=getattr(trading_config, 'council_agent_delay', 0.5),
            trading_config=trading_config
        )
        
        # Initialize offline validator if enabled
        if enable_offline_validation:
            self.validator = OfflineSignalValidator()
        else:
            self.validator = None
        
        # Initialize market state cache if enabled
        if getattr(trading_config, 'cache_enabled', True):
            self.cache = MarketStateCache(
                similarity_threshold=getattr(trading_config, 'cache_similarity_threshold', 0.85),
                ttl_minutes=getattr(trading_config, 'cache_ttl_minutes', 60),
                max_size_mb=getattr(trading_config, 'cache_size_mb', 500)
            )
            logger.info("Market state cache initialized")
        else:
            self.cache = None
        
        # Initialize pre-trade filter
        self.pre_filter = PreTradeFilter(trading_config)
        logger.info("Pre-trade filter initialized")
    
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
            
            # Check news restrictions including sentiment
            is_restricted, reason = await self.enhanced_news_service.is_trading_restricted(
                symbol=symbol,
                consider_sentiment=self.marketaux_config.enabled
            )
            if is_restricted:
                logger.info(f"Trading restricted for {symbol}: {reason}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Symbol readiness check failed for {symbol}: {e}")
            return False
    
    async def generate_signal(
        self,
        symbol: str,
        force_generation: bool = False
    ) -> Optional[TradingSignal]:
        """
        Generate trading signal with enhanced news sentiment
        """
        with ErrorContext("Enhanced signal generation", symbol=symbol) as ctx:
            try:
                # Step 1: Get market data
                market_data = await self._get_market_data(symbol)
                if not market_data:
                    logger.warning(f"No market data available for {symbol}")
                    return None
                
                # Step 1.5: Pre-trade filtering
                if not force_generation:
                    # Get news events for filter
                    news_events = []
                    try:
                        news_context = await self.enhanced_news_service.get_enhanced_news_context(
                            symbol=symbol,
                            lookahead_hours=1,  # Only need near-term news for filter
                            lookback_hours=0
                        )
                        news_events = news_context.upcoming_events
                    except Exception as e:
                        logger.debug(f"Could not get news for pre-filter: {e}")
                    
                    # Run pre-trade filter
                    filter_result = await self.pre_filter.should_analyze(market_data, news_events)
                    
                    if not filter_result.should_analyze:
                        logger.info(f"Pre-filter rejected {symbol}: {filter_result.reason}")
                        logger.debug(f"Filter details: {filter_result.details}")
                        return self._create_filter_signal(symbol, filter_result)
                
                # Step 1.6: Check cache for similar market conditions
                if self.cache and not force_generation:
                    # Check if we should skip analysis entirely
                    should_skip, skip_reason = self.cache.should_skip_analysis(market_data)
                    if should_skip:
                        logger.info(f"Skipping analysis for {symbol}: {skip_reason}")
                        return self._create_skip_signal(symbol, skip_reason)
                    
                    # Check for cached decision
                    cached_result = await self.cache.get_cached_decision(market_data)
                    if cached_result:
                        cached_signal, cached_council = cached_result
                        logger.info(f"Using cached decision for {symbol}: {cached_signal.signal}")
                        
                        # Update signal timestamp and return
                        cached_signal.timestamp = datetime.now(timezone.utc)
                        return cached_signal
                
                # Step 2: Offline validation
                if self.enable_offline_validation and not force_generation:
                    # Use validate_market_data method with h1 data
                    validation_result = await self.validator.validate_market_data(
                        market_data['h1'],
                        min_score_threshold=self.trading_config.offline_validation_threshold
                    )
                    
                    validation_score = validation_result['validation_summary']['weighted_score']
                    ctx.add_detail("validation_score", validation_score)
                    
                    if validation_score < self.trading_config.offline_validation_threshold:
                        logger.debug(
                            f"Signal opportunity below threshold for {symbol}: "
                            f"{validation_score:.2f} < {self.trading_config.offline_validation_threshold}"
                        )
                        return self._create_validation_based_wait_signal(
                            symbol, validation_result
                        )
                
                # Step 3: Get enhanced news context with sentiment
                ctx.add_detail("step", "enhanced_news_check")
                enhanced_news_context = await self._get_enhanced_news_context(symbol)
                
                # Check for restrictions based on news and sentiment
                is_restricted, restriction_reason = await self.enhanced_news_service.is_trading_restricted(
                    symbol=symbol,
                    consider_sentiment=self.marketaux_config.enabled
                )
                
                if is_restricted:
                    raise SignalGenerationError(f"Trading restricted: {restriction_reason}")
                
                # Step 4: Generate charts
                ctx.add_detail("step", "chart_generation")
                chart_paths = await self._generate_charts(symbol, market_data)
                
                # Step 5: Get ML context
                ctx.add_detail("step", "ml_context")
                ml_context = await self._get_ml_context(symbol, market_data)
                
                # Add sentiment to ML context
                if enhanced_news_context.sentiment_summary:
                    ml_context['sentiment_score'] = enhanced_news_context.sentiment_summary.get('average_sentiment', 0)
                    ml_context['sentiment_label'] = enhanced_news_context.sentiment_summary.get('sentiment_label', 'neutral')
                
                # Step 6: Format enhanced news for council
                news_context = enhanced_news_context.to_text_context()
                
                # Step 7: Convene the Trading Council
                ctx.add_detail("step", "council_deliberation")
                council_decision = await self.trading_council.convene_council(
                    market_data=market_data,
                    news_context=news_context,
                    ml_context=ml_context
                )
                
                # Step 8: Adjust confidence based on sentiment alignment
                adjusted_decision = self._adjust_for_sentiment(
                    council_decision,
                    enhanced_news_context
                )
                
                # Step 9: Enhance signal with metadata
                signal = self._enhance_signal(
                    adjusted_decision.signal,
                    validation_result if 'validation_result' in locals() else None,
                    chart_paths,
                    adjusted_decision,
                    enhanced_news_context
                )
                
                # Step 10: Store signal
                ctx.add_detail("step", "storage")
                await self._store_signal(signal)
                
                # Step 11: Cache the decision
                if self.cache and signal.signal != SignalType.WAIT:
                    confidence = signal.market_context.get('council_confidence', 0) if signal.market_context else 0
                    self.cache.cache_decision(
                        market_data=market_data,
                        signal=signal,
                        council_decision=council_decision.__dict__ if hasattr(council_decision, '__dict__') else {},
                        confidence_score=confidence
                    )
                
                logger.info(
                    f"Enhanced signal generated for {symbol}: {signal.signal.value} "
                    f"with confidence {signal.market_context.get('council_confidence', 0) if signal.market_context else 0:.1f}% "
                    f"(sentiment: {enhanced_news_context.sentiment_summary.get('sentiment_label', 'N/A')})"
                )
                
                return signal
                
            except SignalGenerationError:
                raise
            except Exception as e:
                logger.error(f"Failed to generate enhanced signal for {symbol}: {e}")
                raise SignalGenerationError(f"Signal generation failed: {str(e)}")
    
    async def _get_enhanced_news_context(self, symbol: str) -> EnhancedNewsContext:
        """Get enhanced news context with sentiment"""
        try:
            return await self.enhanced_news_service.get_enhanced_news_context(
                symbol=symbol,
                lookahead_hours=48,
                lookback_hours=24,
                use_marketaux=self.marketaux_config.enabled
            )
        except Exception as e:
            logger.error(f"Failed to get enhanced news context: {e}")
            # Return empty context on error
            return EnhancedNewsContext([], [], {})
    
    def _adjust_for_sentiment(
        self,
        council_decision: CouncilDecision,
        enhanced_news_context: EnhancedNewsContext
    ) -> CouncilDecision:
        """Adjust council decision based on sentiment analysis"""
        if not self.marketaux_config.enabled:
            return council_decision
        
        sentiment_score = enhanced_news_context.sentiment_summary.get('average_sentiment', 0)
        sentiment_weight = self.marketaux_config.sentiment_weight
        
        # Calculate sentiment adjustment
        if council_decision.signal.signal == SignalType.BUY:
            # Positive sentiment supports BUY, negative opposes
            sentiment_adjustment = sentiment_score * sentiment_weight * 20  # Max ±20% adjustment
        elif council_decision.signal.signal == SignalType.SELL:
            # Negative sentiment supports SELL, positive opposes
            sentiment_adjustment = -sentiment_score * sentiment_weight * 20
        else:
            sentiment_adjustment = 0
        
        # Adjust confidence
        original_confidence = council_decision.final_confidence
        adjusted_confidence = max(0, min(100, original_confidence + sentiment_adjustment))
        
        logger.info(
            f"Sentiment adjustment for {council_decision.signal.symbol}: "
            f"{original_confidence:.1f}% -> {adjusted_confidence:.1f}% "
            f"(sentiment: {sentiment_score:.2f})"
        )
        
        # Update decision
        council_decision.final_confidence = adjusted_confidence
        # Initialize metadata if not present
        if not hasattr(council_decision, 'metadata'):
            council_decision.metadata = {}
        council_decision.metadata['sentiment_adjustment'] = sentiment_adjustment
        council_decision.metadata['sentiment_score'] = sentiment_score
        
        # Check if adjustment changes the signal
        if adjusted_confidence < self.min_confidence_threshold and council_decision.signal.signal != SignalType.WAIT:
            logger.warning(
                f"Sentiment adjustment reduced confidence below threshold for {council_decision.signal.symbol}"
            )
            # Convert to WAIT signal
            council_decision.signal.signal = SignalType.WAIT
            council_decision.signal.market_context = council_decision.signal.market_context or {}
            council_decision.signal.market_context['original_signal'] = council_decision.signal.signal.value
            council_decision.signal.market_context['sentiment_override'] = True
        
        return council_decision
    
    def _enhance_signal(
        self,
        signal: TradingSignal,
        validation_result: Optional[Any],
        chart_paths: Dict[str, str],
        council_decision: CouncilDecision,
        enhanced_news_context: EnhancedNewsContext
    ) -> TradingSignal:
        """Enhance signal with additional metadata"""
        # Add council metadata
        # Initialize market_context if not present
        if signal.market_context is None:
            signal.market_context = {}
        
        # Add council metadata
        signal.market_context.update({
            'council_confidence': council_decision.final_confidence,
            'council_consensus': getattr(council_decision, 'consensus_type', 'unknown').value if hasattr(getattr(council_decision, 'consensus_type', None), 'value') else 'unknown',
            'agent_votes': council_decision.get_vote_summary() if hasattr(council_decision, 'get_vote_summary') else {},
            'debate_rounds': getattr(council_decision, 'debate_rounds', 1),
            'chart_paths': chart_paths,
            'generated_at': datetime.now(timezone.utc).isoformat()
        })
        
        # Add validation metadata
        if validation_result:
            signal.market_context['validation_score'] = validation_result['validation_summary']['weighted_score']
            signal.market_context['validation_issues'] = len(validation_result['validation_summary'].get('all_issues', []))
        
        # Add sentiment metadata
        if enhanced_news_context.sentiment_summary:
            signal.market_context['sentiment'] = {
                'score': enhanced_news_context.sentiment_summary.get('average_sentiment', 0),
                'label': enhanced_news_context.sentiment_summary.get('sentiment_label', 'neutral'),
                'article_count': enhanced_news_context.sentiment_summary.get('article_count', 0),
                'high_impact_count': enhanced_news_context.sentiment_summary.get('high_impact_count', 0)
            }
        
        # Add news bias
        news_bias = enhanced_news_context.get_trading_bias()
        signal.market_context['news_bias'] = news_bias
        
        return signal
    
    async def _get_market_data(self, symbol: str) -> Optional[Dict[str, MarketData]]:
        """Get market data for multiple timeframes"""
        try:
            # Get data for entry and background timeframes
            h1_data = await self.data_provider.get_market_data(
                symbol, TimeFrame.H1, self.trading_config.bars_for_analysis
            )
            h4_data = await self.data_provider.get_market_data(
                symbol, TimeFrame.H4, self.trading_config.bars_for_analysis
            )
            
            if not h1_data or not h4_data:
                return None
            
            return {
                'h1': h1_data,
                'h4': h4_data
            }
            
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return None
    
    async def _generate_charts(self, symbol: str, market_data: Dict[str, MarketData]) -> Dict[str, str]:
        """Generate charts for the symbol"""
        if not self.chart_generator:
            return {}
        
        try:
            chart_paths = {}
            # Chart generation is sync, not async
            for timeframe, data in market_data.items():
                # Skip chart generation for now - not critical for signals
                # TODO: Implement proper chart generation
                pass
            
            return chart_paths
            
        except Exception as e:
            logger.error(f"Failed to generate charts for {symbol}: {e}")
            return {}
    
    async def _get_ml_context(self, symbol: str, market_data: Dict[str, MarketData]) -> Dict[str, Any]:
        """Get ML model context if available"""
        try:
            # Get ML settings
            from config.settings import get_settings
            settings = get_settings()
            ml_enabled = settings.ml.enabled
            
            if not ml_enabled:
                logger.debug("ML is disabled in configuration")
                return {
                    'ml_enabled': False,
                    'ml_confidence': None,
                    'ml_signal': None,
                    'confidence': 50.0  # Default confidence for council
                }
                
            # Get ML prediction
            ml_result = await self.ml_predictor.get_ml_prediction(symbol, market_data)
            
            # Extract results
            ml_enabled = ml_result.get('ml_enabled', False)
            ml_signal = ml_result.get('ml_signal')
            ml_confidence = ml_result.get('ml_confidence')
            ml_metadata = ml_result.get('ml_metadata', {})
            
            # Log ML prediction
            if ml_enabled and ml_signal:
                logger.info(
                    f"ML prediction for {symbol}: {ml_signal.value} "
                    f"with confidence {ml_confidence:.1f}%"
                )
                
                # Log top features if available
                top_features = ml_metadata.get('feature_importances', [])
                if top_features:
                    features_str = ", ".join([f"{f['name']}: {f['importance']:.3f}" for f in top_features[:3]])
                    logger.debug(f"Top ML features: {features_str}")
            else:
                logger.debug(f"ML prediction not available for {symbol}: {ml_metadata.get('reason', 'Unknown')}")
                
            return {
                'ml_enabled': ml_enabled,
                'ml_confidence': ml_confidence,
                'ml_signal': ml_signal.value if ml_signal else None,
                'confidence': ml_confidence if ml_confidence else 50.0,  # For council to use
                'ml_metadata': ml_metadata
            }
            
        except Exception as e:
            logger.error(f"Error getting ML context for {symbol}: {e}", exc_info=True)
            return {
                'ml_enabled': False,
                'ml_confidence': None,
                'ml_signal': None,
                'confidence': 50.0,  # Default confidence
                'ml_metadata': {'error': str(e)}
            }
    
    async def _store_signal(self, signal: TradingSignal):
        """Store signal in repository"""
        try:
            # save_signal is synchronous, don't await it
            self.signal_repository.save_signal(signal)
        except Exception as e:
            logger.error(f"Failed to store signal: {e}")
    
    def _create_validation_based_wait_signal(
        self,
        symbol: str,
        validation_result: Dict[str, Any]
    ) -> TradingSignal:
        """Create a WAIT signal based on validation results"""
        validation_score = validation_result['validation_summary']['weighted_score']
        issues = validation_result['validation_summary'].get('all_issues', [])
        
        # Build reason from top issues
        reason_parts = [f"Low validation score: {validation_score:.2f}"]
        for issue in issues[:2]:  # Top 2 issues
            reason_parts.append(f"- {issue['message']}")
        
        return TradingSignal(
            symbol=symbol,
            signal=SignalType.WAIT,
            risk_class=RiskClass.C,
            reason=". ".join(reason_parts),
            timestamp=datetime.now(timezone.utc),
            market_context={
                'validation_score': validation_score,
                'validation_issues': len(issues),
                'source': 'offline_validation'
            }
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
                },
                'sentiment': getattr(decision, 'metadata', {}).get('sentiment_score', 0)
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
        logger.info(f"Generating enhanced council signal with ML context for {symbol}")
        return await self.generate_signal(symbol)
    
    def _create_fallback_signal(self, symbol: str, reason: str) -> TradingSignal:
        """Create a fallback WAIT signal when generation fails"""
        return TradingSignal(
            id=TradingSignal.generate_id(),
            symbol=symbol,
            signal=SignalType.WAIT,
            risk_class=RiskClass.CONSERVATIVE,
            reason=f"Signal generation failed: {reason}",
            timestamp=datetime.now(timezone.utc),
            metadata={
                'error': reason,
                'source': 'enhanced_council_fallback'
            }
        )
    
    def _create_skip_signal(self, symbol: str, reason: str) -> TradingSignal:
        """Create a WAIT signal for skipped analysis"""
        return TradingSignal(
            symbol=symbol,
            signal=SignalType.WAIT,
            reason=f"Analysis skipped: {reason}",
            risk_class=RiskClass.C,
            timestamp=datetime.now(timezone.utc),
            market_context={
                'source': 'cache_skip',
                'skip_reason': reason
            }
        )
    
    def _create_validation_based_wait_signal(
        self,
        symbol: str,
        validation_result: Dict[str, Any]
    ) -> TradingSignal:
        """Create a WAIT signal based on validation results"""
        issues = []
        for check, result in validation_result['quality_checks'].items():
            if not result['passed']:
                issues.append(f"{check}: {result['reason']}")
        
        validation_score = validation_result['validation_summary']['weighted_score']
        
        return TradingSignal(
            symbol=symbol,
            signal=SignalType.WAIT,
            reason=f"Low validation score ({validation_score:.2f}): {'; '.join(issues[:2])}",
            risk_class=RiskClass.C,
            timestamp=datetime.now(timezone.utc),
            market_context={
                'validation_score': validation_score,
                'validation_issues': len(issues),
                'source': 'offline_validation'
            }
        )
    
    def _create_filter_signal(self, symbol: str, filter_result: Any) -> TradingSignal:
        """Create a WAIT signal based on pre-filter results"""
        return TradingSignal(
            symbol=symbol,
            signal=SignalType.WAIT,
            reason=f"Pre-filter: {filter_result.reason}",
            risk_class=RiskClass.C,
            timestamp=datetime.now(timezone.utc),
            market_context={
                'source': 'pre_filter',
                'quality_score': filter_result.details.get('quality_score', 0),
                'failed_checks': filter_result.details.get('failed_checks', []),
                'filter_scores': filter_result.details.get('filter_scores', {})
            }
        )


# Export the enhanced service
__all__ = ['EnhancedCouncilSignalService']