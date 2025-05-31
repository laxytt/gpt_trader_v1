"""
Signal service for orchestrating trading signal generation.
Coordinates data gathering, analysis, and signal validation.
"""

import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.domain.models import (
    TradingSignal, MarketData, NewsEvent, MarketContext, 
    SignalType, RiskClass, MarketSession, VolatilityLevel
)
from core.domain.exceptions import (
    SignalGenerationError, ErrorContext, ServiceError
)
from core.domain.enums import TimeFrame
from core.infrastructure.data.unified_data_provider import DataRequest
from core.infrastructure.mt5.data_provider import MT5DataProvider  
from core.infrastructure.gpt.signal_generator import GPTSignalGenerator
from core.infrastructure.database.repositories import SignalRepository
from core.services.news_service import NewsService
from core.services.memory_service import MemoryService
from core.services.offline_validator import OfflineSignalValidator
from core.utils.chart_utils import ChartGenerator, create_market_data_chart
from core.utils.error_handling import with_error_recovery
from core.utils.validation import SignalValidator
from config.settings import TradingSettings


logger = logging.getLogger(__name__)


class MarketContextAnalyzer:
    """Analyzes current market context for signal generation"""
    
    def __init__(self, memory_service: MemoryService):
        self.memory_service = memory_service
    
    def analyze_context(self, symbol: str, h1_data: MarketData) -> MarketContext:
        """
        Analyze current market context.
        
        Args:
            symbol: Trading symbol
            h1_data: H1 market data for analysis
            
        Returns:
            MarketContext object
        """
        # Determine market session
        now = datetime.now(timezone.utc)
        session = self._get_market_session(now)
        
        # Analyze volatility
        volatility = self._analyze_volatility(h1_data)
        
        # Get win/loss statistics
        stats = self.memory_service.get_performance_stats(symbol)
        
        return MarketContext(
            session=session,
            volatility=volatility,
            win_streak_type=stats.get('streak_type', 'unknown'),
            win_streak_length=stats.get('streak_length', 0),
            win_rate=stats.get('win_rate', 0.0),
            sample_size=stats.get('sample_size', 0),
            timestamp=now
        )
    
    def _get_market_session(self, timestamp: datetime) -> MarketSession:
        """Determine current market session"""
        hour = timestamp.hour
        
        if 0 <= hour < 7:
            return MarketSession.ASIA
        elif 7 <= hour < 13:
            return MarketSession.EUROPE
        elif 13 <= hour < 15:
            return MarketSession.OVERLAP  # EU-NY overlap
        else:
            return MarketSession.NEW_YORK
    
    def _analyze_volatility(self, market_data: MarketData) -> VolatilityLevel:
        """Analyze market volatility from ATR data"""
        if not market_data.candles:
            return VolatilityLevel.MEDIUM
        
        # Get recent ATR values
        atr_values = [
            candle.atr14 for candle in market_data.candles[-20:] 
            if candle.atr14 is not None
        ]
        
        if len(atr_values) < 5:
            return VolatilityLevel.MEDIUM
        
        current_atr = atr_values[-1]
        avg_atr = sum(atr_values) / len(atr_values)
        
        if avg_atr == 0:
            return VolatilityLevel.MEDIUM
        
        ratio = current_atr / avg_atr
        
        if ratio > 1.3:
            return VolatilityLevel.HIGH
        elif ratio < 0.7:
            return VolatilityLevel.LOW
        else:
            return VolatilityLevel.MEDIUM


class SignalService:
    """Update to use date-based requests for backtesting"""
    
    async def generate_signal(
        self, 
        symbol: str,
        as_of_date: Optional[datetime] = None  # Add this parameter
    ) -> TradingSignal:
        """
        Generate signal as of specific date (for backtesting) or current (for live).
        """
        if as_of_date:
            # Backtesting mode - get historical data up to as_of_date
            h1_request = DataRequest(
                symbol=symbol,
                timeframe=TimeFrame.H1,
                end_date=as_of_date,
                start_date=as_of_date - timedelta(days=10)  # 10 days of H1 data
            )
            h4_request = DataRequest(
                symbol=symbol,
                timeframe=TimeFrame.H4,
                end_date=as_of_date,
                start_date=as_of_date - timedelta(days=30)  # 30 days of H4 data
            )
        else:
            # Live mode - get recent bars
            h1_request = DataRequest(
                symbol=symbol,
                timeframe=TimeFrame.H1,
                num_bars=self.trading_config.bars_for_analysis
            )
            h4_request = DataRequest(
                symbol=symbol,
                timeframe=TimeFrame.H4,
                num_bars=self.trading_config.bars_for_analysis
            )
        
        # Use unified provider through data_provider
        h1_data = await self.data_provider.unified_provider.get_data(h1_request)
        h4_data = await self.data_provider.unified_provider.get_data(h4_request)
    
    def __init__(
        self,
        data_provider: MT5DataProvider,
        signal_generator: GPTSignalGenerator,
        news_service: NewsService,
        memory_service: MemoryService,
        signal_repository: SignalRepository,
        trading_config: TradingSettings,
        chart_generator: Optional[ChartGenerator] = None,
        screenshots_dir: str = "screenshots",
        enable_offline_validation: bool = True
    ):
        self.data_provider = data_provider
        self.signal_generator = signal_generator
        self.news_service = news_service
        self.memory_service = memory_service
        self.signal_repository = signal_repository
        self.trading_config = trading_config
        self.chart_generator = chart_generator
        self.screenshots_dir = Path(screenshots_dir)
        
        # Initialize components
        self.context_analyzer = MarketContextAnalyzer(memory_service)
        self.validator = SignalValidator()
        
        # Ensure screenshots directory exists
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

        # Initialize offline validator
        self.enable_offline_validation = enable_offline_validation
        if enable_offline_validation:
            self.offline_validator = OfflineSignalValidator()
            logger.info("Offline signal validation enabled")
        else:
            self.offline_validator = None
            logger.info("Offline signal validation disabled")
    
    async def generate_signal(self, symbol: str) -> TradingSignal:
        """
        Generate a comprehensive trading signal for the given symbol.
        Now includes offline validation step before GPT analysis.
        """
        with ErrorContext("Signal generation", symbol=symbol) as ctx:
            logger.info(f"Starting signal generation for {symbol}")
            
            try:
                # Step 1: Gather market data
                ctx.add_detail("step", "data_gathering")
                market_data = await self._gather_market_data(symbol)
                
                # Step 2: Run offline validation (NEW)
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
                        
                        # Return a WAIT signal with validation context
                        return self._create_validation_based_wait_signal(
                            symbol, validation_result
                        )
                
                # Step 3: Check news restrictions (existing)
                ctx.add_detail("step", "news_check")
                news_events = await self._check_news_restrictions(symbol)
                
                # Step 4: Analyze market context (existing)
                ctx.add_detail("step", "context_analysis")
                market_context = self.context_analyzer.analyze_context(
                    symbol, market_data['h1']
                )
                
                # Step 5: Generate charts if available (existing)
                ctx.add_detail("step", "chart_generation")
                chart_paths = await self._generate_charts(symbol, market_data)
                
                # Step 6: Get historical context (existing)
                ctx.add_detail("step", "historical_context")
                historical_cases = await self._get_historical_context(
                    symbol, market_data['h1']
                )
                
                # Step 7: Generate signal using GPT with validation context (MODIFIED)
                ctx.add_detail("step", "gpt_analysis")
                signal = await self._generate_signal_with_validation_context(
                    h1_data=market_data['h1'],
                    h4_data=market_data['h4'],
                    news_events=news_events,
                    market_context=market_context,
                    chart_paths=chart_paths,
                    historical_cases=historical_cases,
                    validation_result=validation_result  # NEW: Pass validation context
                )
                
                # Step 8: Validate signal (existing)
                ctx.add_detail("step", "validation")
                self.validator.validate_signal(signal)
                
                # Step 9: Store signal with validation metadata (MODIFIED)
                ctx.add_detail("step", "storage")
                await self._store_signal_with_metadata(signal, validation_result)
                
                logger.info(
                    f"Signal generated successfully for {symbol}: "
                    f"{signal.signal.value} ({signal.risk_class.value})"
                )
                
                return signal
                
            except Exception as e:
                logger.error(f"Signal generation failed for {symbol}: {e}")
                return self._create_fallback_signal(symbol, str(e))
    
    async def _gather_market_data(self, symbol: str) -> Dict[str, MarketData]:
        """Gather H1 and H4 market data"""
        try:
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
    
    async def _check_news_restrictions(self, symbol: str) -> List[NewsEvent]:
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
        """Generate chart images for GPT analysis"""
        if not self.chart_generator:
            return None
        
        chart_paths = {}
        
        for timeframe, data in market_data.items():
            chart_filename = f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            chart_path = self.screenshots_dir / chart_filename
            
            # Use the new chart service
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
    
    @with_error_recovery(default_return=[], log_level='warning')
    async def _get_historical_context(
        self, 
        symbol: str, 
        h1_data: MarketData
    ) -> List[Dict[str, Any]]:
        """Get historical trade cases for context"""
        if not h1_data.candles:
            return []
        
        # Create context from current market conditions
        latest_candle = h1_data.latest_candle
        if not latest_candle:
            return []
        
        context_text = (
            f"EMA50={latest_candle.ema50:.5f}, EMA200={latest_candle.ema200:.5f}, "
            f"RSI={latest_candle.rsi14:.1f}, Volume={latest_candle.volume}, "
            f"ATR={latest_candle.atr14:.5f}"
        )
        
        # Query similar historical cases
        similar_cases = await self.memory_service.find_similar_cases(
            context=context_text,
            symbol=symbol,
            limit=3
        )
        
        return similar_cases
    
    async def _store_signal(self, signal: TradingSignal):
        """Store signal in repository"""
        try:
            self.signal_repository.save_signal(signal)
            logger.debug(f"Signal stored for {signal.symbol}")
        except Exception as e:
            logger.error(f"Failed to store signal: {e}")
            # Don't fail signal generation if storage fails
    
    async def _run_offline_validation(
        self, 
        market_data: MarketData
    ) -> Dict[str, Any]:
        """Run offline validation on market data"""
        try:
            validation_result = await self.offline_validator.validate_market_data(
                market_data,
                min_score_threshold=0.5  # Configurable threshold
            )
            
            # Log validation summary
            summary = validation_result['validation_summary']
            logger.info(
                f"Offline validation for {market_data.symbol}: "
                f"Score={summary['weighted_score']:.2f}, "
                f"Passed={summary['validators_passed']}/{summary['total_validators']}"
            )
            
            # Log any critical issues
            if summary['critical_issues']:
                logger.warning(
                    f"Critical issues for {market_data.symbol}: "
                    f"{[issue['message'] for issue in summary['critical_issues']]}"
                )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Offline validation failed: {e}")
            # Return a neutral result on error (don't block trading)
            return {
                'should_proceed': True,
                'validation_summary': {'weighted_score': 0.5},
                'gpt_context': {'pre_validation_performed': False, 'error': str(e)}
            }

    def _create_validation_based_wait_signal(
        self, 
        symbol: str, 
        validation_result: Dict[str, Any]
    ) -> TradingSignal:
        """Create a WAIT signal based on validation results"""
        summary = validation_result['validation_summary']
        top_issues = summary.get('all_issues', [])[:2]  # Top 2 issues
        
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
            market_context={'validation_score': summary['weighted_score']}
        )
    
    async def _generate_signal_with_validation_context(
        self,
        h1_data: MarketData,
        h4_data: MarketData,
        news_events: List[NewsEvent],
        market_context: MarketContext,
        chart_paths: Optional[Dict[str, str]],
        historical_cases: Optional[List[Dict[str, Any]]],
        validation_result: Optional[Dict[str, Any]]
    ) -> TradingSignal:
        """Generate signal with offline validation context"""
        
        # If no validation result, use standard generation
        if not validation_result:
            return await self.signal_generator.generate_signal(
                h1_data=h1_data,
                h4_data=h4_data,
                news_events=news_events,
                market_context=market_context,
                chart_paths=chart_paths,
                historical_cases=historical_cases
            )
        
        # Enhance market context with validation data
        enhanced_context = market_context.to_dict()
        enhanced_context['offline_validation'] = validation_result['gpt_context']
        
        # Create enhanced market context object
        enhanced_market_context = MarketContext(
            session=market_context.session,
            volatility=market_context.volatility,
            win_streak_type=market_context.win_streak_type,
            win_streak_length=market_context.win_streak_length,
            win_rate=market_context.win_rate,
            sample_size=market_context.sample_size,
            timestamp=market_context.timestamp
        )
        
        # Add validation data to the market context dictionary representation
        enhanced_market_context_dict = enhanced_market_context.to_dict()
        enhanced_market_context_dict.update({'offline_validation': validation_result['gpt_context']})
        
        # We need to modify the signal generator to accept this enhanced context
        # For now, we'll pass it through the historical_cases parameter
        enhanced_historical = historical_cases or []
        enhanced_historical.append({
            'type': 'current_validation',
            'validation_data': validation_result['gpt_context']
        })
        
        return await self.signal_generator.generate_signal(
            h1_data=h1_data,
            h4_data=h4_data,
            news_events=news_events,
            market_context=enhanced_market_context,
            chart_paths=chart_paths,
            historical_cases=enhanced_historical
        )
    
    async def _store_signal_with_metadata(
        self, 
        signal: TradingSignal,
        validation_result: Optional[Dict[str, Any]]
    ):
        """Store signal with validation metadata"""
        try:
            # Add validation score to signal metadata if available
            if validation_result and hasattr(signal, 'metadata'):
                signal.metadata = signal.metadata or {}
                signal.metadata['validation_score'] = validation_result['validation_summary']['weighted_score']
                signal.metadata['validation_passed'] = validation_result['should_proceed']
            
            self.signal_repository.save_signal(signal)
            logger.debug(f"Signal stored for {signal.symbol}")
            
        except Exception as e:
            logger.error(f"Failed to store signal: {e}")
            # Don't fail signal generation if storage fails

    def _create_fallback_signal(self, symbol: str, reason: str) -> TradingSignal:
        """Create a fallback WAIT signal when generation fails"""
        return TradingSignal(
            symbol=symbol,
            signal=SignalType.WAIT,
            reason=f"Signal generation failed: {reason}",
            risk_class=RiskClass.C,
            timestamp=datetime.now(timezone.utc)
        )
    
    async def validate_symbol_readiness(self, symbol: str) -> bool:
        """
        Check if symbol is ready for signal generation.
        
        Args:
            symbol: Trading symbol to check
            
        Returns:
            True if symbol is ready for analysis
        """
        try:
            # Check data availability - properly await the async call
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


# Export main service
__all__ = ['SignalService', 'MarketContextAnalyzer']