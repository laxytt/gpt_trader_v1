"""
Signal service for orchestrating trading signal generation.
Coordinates data gathering, analysis, and signal validation.
"""

import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timezone
from pathlib import Path

from core.domain.models import (
    TradingSignal, MarketData, NewsEvent, MarketContext, 
    SignalType, RiskClass, MarketSession, VolatilityLevel
)
from core.domain.exceptions import (
    SignalGenerationError, ErrorContext, ServiceError
)
from core.domain.enums import TimeFrame
from core.infrastructure.mt5.data_provider import MT5DataProvider  
from core.infrastructure.gpt.signal_generator import GPTSignalGenerator
from core.infrastructure.database.repositories import SignalRepository
from core.services.news_service import NewsService
from core.services.memory_service import MemoryService
from core.utils.chart_utils import ChartGenerator, create_market_data_chart
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
    """
    Orchestrates trading signal generation by coordinating all required components.
    """
    
    def __init__(
        self,
        data_provider: MT5DataProvider,
        signal_generator: GPTSignalGenerator,
        news_service: NewsService,
        memory_service: MemoryService,
        signal_repository: SignalRepository,
        trading_config: TradingSettings,
        chart_generator: Optional[ChartGenerator] = None,
        screenshots_dir: str = "screenshots"
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
    
    async def generate_signal(self, symbol: str) -> TradingSignal:
        """
        Generate a comprehensive trading signal for the given symbol.
        
        Args:
            symbol: Trading symbol to analyze
            
        Returns:
            TradingSignal object
            
        Raises:
            SignalGenerationError: If signal generation fails
        """
        with ErrorContext("Signal generation", symbol=symbol) as ctx:
            logger.info(f"Starting signal generation for {symbol}")
            
            try:
                # Step 1: Gather market data
                ctx.add_detail("step", "data_gathering")
                market_data = await self._gather_market_data(symbol)
                
                # Step 2: Check news restrictions
                ctx.add_detail("step", "news_check")
                news_events = await self._check_news_restrictions(symbol)
                
                # Step 3: Analyze market context
                ctx.add_detail("step", "context_analysis")
                market_context = self.context_analyzer.analyze_context(
                    symbol, market_data['h1']
                )
                
                # Step 4: Generate charts if available
                ctx.add_detail("step", "chart_generation")
                chart_paths = await self._generate_charts(symbol, market_data)
                
                # Step 5: Get historical context
                ctx.add_detail("step", "historical_context")
                historical_cases = await self._get_historical_context(symbol, market_data['h1'])
                
                # Step 6: Generate signal using GPT
                ctx.add_detail("step", "gpt_analysis")
                signal = await self.signal_generator.generate_signal(
                    h1_data=market_data['h1'],
                    h4_data=market_data['h4'],
                    news_events=news_events,
                    market_context=market_context,
                    chart_paths=chart_paths,
                    historical_cases=historical_cases
                )
                
                # Step 7: Validate signal
                ctx.add_detail("step", "validation")
                self.validator.validate_signal(signal)
                
                # Step 8: Store signal
                ctx.add_detail("step", "storage")
                await self._store_signal(signal)
                
                logger.info(f"Signal generated successfully for {symbol}: "
                           f"{signal.signal.value} ({signal.risk_class.value})")
                
                return signal
                
            except Exception as e:
                logger.error(f"Signal generation failed for {symbol}: {e}")
                
                # Return fallback signal
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
    
    async def _generate_charts(
        self, 
        symbol: str, 
        market_data: Dict[str, MarketData]
    ) -> Optional[Dict[str, str]]:
        """Generate chart images for GPT analysis"""
        if not self.chart_generator:
            return None
        
        try:
            chart_paths = {}
            
            for timeframe, data in market_data.items():
                chart_filename = f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                chart_path = self.screenshots_dir / chart_filename
                
                generated_path = create_market_data_chart(
                    market_data=data,
                    output_path=str(chart_path),
                    title=f"{symbol} {timeframe}"
                )
                
                if generated_path:
                    chart_paths[timeframe.lower()] = generated_path
            
            return chart_paths if chart_paths else None
            
        except Exception as e:
            logger.warning(f"Chart generation failed for {symbol}: {e}")
            return None
    
    async def _get_historical_context(
        self, 
        symbol: str, 
        h1_data: MarketData
    ) -> List[Dict[str, Any]]:
        """Get historical trade cases for context"""
        try:
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
            
        except Exception as e:
            logger.warning(f"Historical context retrieval failed for {symbol}: {e}")
            return []
    
    async def _store_signal(self, signal: TradingSignal):
        """Store signal in repository"""
        try:
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
            # Check data availability
            has_data = self.data_provider.validate_symbol_data(symbol, TimeFrame.H1)
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