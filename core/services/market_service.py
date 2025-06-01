"""
Market service for comprehensive market analysis and data coordination.
Provides high-level market intelligence and conditions assessment.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum

from core.domain.models import MarketData, MarketSession, VolatilityLevel
from core.domain.exceptions import ServiceError, ErrorContext
from core.domain.enums import TimeFrame
from core.infrastructure.mt5.data_provider import MT5DataProvider
from core.infrastructure.mt5.client import MT5Client
from config.symbols import get_symbol_spec, SYMBOL_SESSIONS


logger = logging.getLogger(__name__)


class MarketCondition(Enum):
    """Overall market condition assessment"""
    EXCELLENT = "excellent"     # High volatility, clear trends, good volume
    GOOD = "good"              # Moderate conditions, some opportunities
    FAIR = "fair"              # Mixed conditions, selective opportunities
    POOR = "poor"              # Low volatility, choppy, avoid trading
    CLOSED = "closed"          # Market closed


class MarketIntelligence:
    """Market intelligence data structure"""
    
    def __init__(
        self,
        symbol: str,
        condition: MarketCondition,
        volatility: VolatilityLevel,
        session: MarketSession,
        score: float,
        reasons: List[str],
        data_quality: str = "good"
    ):
        self.symbol = symbol
        self.condition = condition
        self.volatility = volatility
        self.session = session
        self.score = score  # 0-100 trading favorability score
        self.reasons = reasons
        self.data_quality = data_quality
        self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'symbol': self.symbol,
            'condition': self.condition.value,
            'volatility': self.volatility.value,
            'session': self.session.value,
            'score': self.score,
            'reasons': self.reasons,
            'data_quality': self.data_quality,
            'timestamp': self.timestamp.isoformat()
        }


class MarketSessionAnalyzer:
    """Analyzes market sessions and their characteristics"""
    
    def get_current_session(self, now: Optional[datetime] = None) -> MarketSession:
        """Get current market session"""
        if now is None:
            now = datetime.now(timezone.utc)
        
        hour = now.hour
        
        # Define session hours (UTC)
        if 0 <= hour < 7:
            return MarketSession.ASIA
        elif 7 <= hour < 13:
            return MarketSession.EUROPE
        elif 13 <= hour < 15:
            return MarketSession.OVERLAP  # EU-NY overlap
        else:
            return MarketSession.NEW_YORK
    
    def get_session_characteristics(self, session: MarketSession) -> Dict[str, any]:
        """Get characteristics of a trading session"""
        characteristics = {
            MarketSession.ASIA: {
                'volatility': 'low_to_medium',
                'volume': 'medium',
                'best_pairs': ['USDJPY', 'AUDUSD', 'NZDUSD'],
                'avoid_pairs': ['GBPUSD'],
                'notes': 'Generally quieter, JPY pairs most active'
            },
            MarketSession.EUROPE: {
                'volatility': 'medium_to_high',
                'volume': 'high',
                'best_pairs': ['EURUSD', 'GBPUSD', 'EURGBP'],
                'avoid_pairs': ['USDJPY'],
                'notes': 'High activity, major EUR/GBP movements'
            },
            MarketSession.OVERLAP: {
                'volatility': 'highest',
                'volume': 'highest',
                'best_pairs': ['EURUSD', 'GBPUSD'],
                'avoid_pairs': [],
                'notes': 'Peak trading time, highest volatility'
            },
            MarketSession.NEW_YORK: {
                'volatility': 'medium_to_high',
                'volume': 'high',
                'best_pairs': ['EURUSD', 'GBPUSD', 'USDCAD'],
                'avoid_pairs': ['USDJPY'],
                'notes': 'Strong USD movements, news-driven'
            }
        }
        
        return characteristics.get(session, {})
    
    def is_symbol_session_optimal(self, symbol: str, session: MarketSession) -> bool:
        """Check if current session is optimal for symbol"""
        optimal_sessions = SYMBOL_SESSIONS.get(symbol, [])
        return session.value in optimal_sessions


class MarketConditionAnalyzer:
    """Analyzes overall market conditions for trading"""
    
    def __init__(self, data_provider: MT5DataProvider):
        self.data_provider = data_provider
        self.session_analyzer = MarketSessionAnalyzer()
    
    async def analyze_market_conditions(
        self, 
        symbol: str,
        include_multi_timeframe: bool = True
    ) -> MarketIntelligence:
        """
        Comprehensive analysis of market conditions for a symbol.
        
        Args:
            symbol: Trading symbol to analyze
            include_multi_timeframe: Whether to include H4 analysis
            
        Returns:
            MarketIntelligence object with complete assessment
        """
        with ErrorContext("Market condition analysis", symbol=symbol) as ctx:
            reasons = []
            score = 50  # Start with neutral score
            
            try:
                # Get current session
                current_session = self.session_analyzer.get_current_session()
                
                # Get market data
                h1_data = await self.data_provider.get_market_data(
                    symbol, TimeFrame.H1, bars=50
                )
                
                h4_data = None
                if include_multi_timeframe:
                    h4_data = await self.data_provider.get_market_data(
                        symbol, TimeFrame.H4, bars=50
                    )
                
                # Analyze volatility
                volatility_metrics = self.data_provider.calculate_volatility_metrics(h1_data)
                volatility = VolatilityLevel(volatility_metrics.get('volatility_level', 'medium'))
                
                # Session analysis
                session_score = self._analyze_session_suitability(symbol, current_session)
                score += session_score
                if session_score > 0:
                    reasons.append(f"Favorable session ({current_session.value})")
                elif session_score < 0:
                    reasons.append(f"Suboptimal session ({current_session.value})")
                
                # Volatility analysis
                volatility_score = self._analyze_volatility(volatility_metrics)
                score += volatility_score
                if volatility_score > 0:
                    reasons.append(f"Good volatility ({volatility.value})")
                elif volatility_score < 0:
                    reasons.append(f"Poor volatility ({volatility.value})")
                
                # Volume analysis
                volume_score = self._analyze_volume(h1_data)
                score += volume_score
                if volume_score > 0:
                    reasons.append("Strong volume activity")
                elif volume_score < 0:
                    reasons.append("Weak volume activity")
                
                # Trend analysis
                trend_score = self._analyze_trend_strength(h1_data, h4_data)
                score += trend_score
                if trend_score > 0:
                    reasons.append("Clear trend structure")
                elif trend_score < 0:
                    reasons.append("Choppy/sideways market")
                
                # Data quality check
                data_quality = self._assess_data_quality(h1_data)
                if data_quality != "good":
                    score -= 10
                    reasons.append(f"Data quality: {data_quality}")
                
                # Determine overall condition
                condition = self._score_to_condition(score)
                
                return MarketIntelligence(
                    symbol=symbol,
                    condition=condition,
                    volatility=volatility,
                    session=current_session,
                    score=max(0, min(100, score)),
                    reasons=reasons,
                    data_quality=data_quality
                )
                
            except Exception as e:
                logger.error(f"Market condition analysis failed for {symbol}: {e}")
                return MarketIntelligence(
                    symbol=symbol,
                    condition=MarketCondition.POOR,
                    volatility=VolatilityLevel.LOW,
                    session=MarketSession.ASIA,
                    score=0,
                    reasons=[f"Analysis failed: {str(e)}"],
                    data_quality="poor"
                )
    
    def _analyze_session_suitability(self, symbol: str, session: MarketSession) -> int:
        """Analyze session suitability for symbol (-20 to +20)"""
        if self.session_analyzer.is_symbol_session_optimal(symbol, session):
            return 15
        
        session_chars = self.session_analyzer.get_session_characteristics(session)
        best_pairs = session_chars.get('best_pairs', [])
        avoid_pairs = session_chars.get('avoid_pairs', [])
        
        if symbol in avoid_pairs:
            return -15
        elif symbol in best_pairs:
            return 10
        else:
            return 0
    
    def _analyze_volatility(self, volatility_metrics: Dict) -> int:
        """Analyze volatility conditions (-15 to +15)"""
        atr_ratio = volatility_metrics.get('atr_ratio', 1.0)
        volume_ratio = volatility_metrics.get('volume_ratio', 1.0)
        
        # ATR analysis
        if atr_ratio > 1.3:  # High volatility
            atr_score = 10
        elif atr_ratio < 0.7:  # Low volatility
            atr_score = -10
        else:
            atr_score = 0
        
        # Volume analysis
        if volume_ratio > 1.2:  # High volume
            vol_score = 5
        elif volume_ratio < 0.8:  # Low volume
            vol_score = -5
        else:
            vol_score = 0
        
        return atr_score + vol_score
    
    def _analyze_volume(self, market_data: MarketData) -> int:
        """Analyze volume patterns (-10 to +10)"""
        if not market_data.candles or len(market_data.candles) < 10:
            return -5
        
        recent_volumes = [c.volume for c in market_data.candles[-10:]]
        avg_volume = sum(recent_volumes) / len(recent_volumes)
        latest_volume = recent_volumes[-1]
        
        if latest_volume > avg_volume * 1.5:
            return 8  # Strong volume spike
        elif latest_volume < avg_volume * 0.5:
            return -8  # Very weak volume
        elif latest_volume > avg_volume * 1.2:
            return 5   # Good volume
        elif latest_volume < avg_volume * 0.8:
            return -5  # Weak volume
        else:
            return 0   # Normal volume
    
    def _analyze_trend_strength(
        self, 
        h1_data: MarketData, 
        h4_data: Optional[MarketData]
    ) -> int:
        """Analyze trend strength (-15 to +15)"""
        if not h1_data.candles:
            return -10
        
        score = 0
        
        # H1 trend analysis
        latest_h1 = h1_data.latest_candle
        if latest_h1 and latest_h1.ema50 and latest_h1.ema200:
            if latest_h1.ema50 > latest_h1.ema200:
                # Uptrend
                if latest_h1.close > latest_h1.ema50:
                    score += 5  # Price above fast EMA
                if latest_h1.rsi14 and 50 < latest_h1.rsi14 < 80:
                    score += 3  # RSI in bullish zone but not overbought
            elif latest_h1.ema50 < latest_h1.ema200:
                # Downtrend
                if latest_h1.close < latest_h1.ema50:
                    score += 5  # Price below fast EMA
                if latest_h1.rsi14 and 20 < latest_h1.rsi14 < 50:
                    score += 3  # RSI in bearish zone but not oversold
            else:
                score -= 5  # EMAs too close, choppy
        
        # H4 confirmation
        if h4_data and h4_data.latest_candle:
            latest_h4 = h4_data.latest_candle
            if latest_h4.ema50 and latest_h4.ema200:
                if (latest_h1.ema50 > latest_h1.ema200) == (latest_h4.ema50 > latest_h4.ema200):
                    score += 7  # H1 and H4 trends agree
                else:
                    score -= 5  # Trends conflict
        
        return score
    
    def _assess_data_quality(self, market_data: MarketData) -> str:
        """Assess data quality"""
        if not market_data.candles:
            return "poor"
        
        if len(market_data.candles) < 20:
            return "limited"
        
        # Check for gaps or unusual data
        recent_candles = market_data.candles[-10:]
        zero_volume_count = sum(1 for c in recent_candles if c.volume == 0)
        
        if zero_volume_count > 3:
            return "questionable"
        
        return "good"
    
    def _score_to_condition(self, score: float) -> MarketCondition:
        """Convert numeric score to market condition"""
        if score >= 75:
            return MarketCondition.EXCELLENT
        elif score >= 60:
            return MarketCondition.GOOD
        elif score >= 40:
            return MarketCondition.FAIR
        else:
            return MarketCondition.POOR


class MarketService:
    """
    High-level market service that coordinates market analysis and intelligence.
    """
    
    def __init__(
        self,
        data_provider: MT5DataProvider,
        mt5_client: MT5Client
    ):
        self.data_provider = data_provider
        self.mt5_client = mt5_client
        self.condition_analyzer = MarketConditionAnalyzer(data_provider)
        self.session_analyzer = MarketSessionAnalyzer()
    
    async def get_market_intelligence(self, symbols: List[str]) -> Dict[str, MarketIntelligence]:
        """
        Get comprehensive market intelligence for multiple symbols.
        
        Args:
            symbols: List of symbols to analyze
            
        Returns:
            Dictionary mapping symbols to their market intelligence
        """
        intelligence = {}
        
        for symbol in symbols:
            try:
                intel = await self.condition_analyzer.analyze_market_conditions(symbol)
                intelligence[symbol] = intel
                logger.debug(f"Market intelligence for {symbol}: {intel.condition.value} (Score: {intel.score})")
            except Exception as e:
                logger.error(f"Failed to get market intelligence for {symbol}: {e}")
                # Create fallback intelligence
                intelligence[symbol] = MarketIntelligence(
                    symbol=symbol,
                    condition=MarketCondition.POOR,
                    volatility=VolatilityLevel.LOW,
                    session=self.session_analyzer.get_current_session(),
                    score=0,
                    reasons=[f"Analysis failed: {str(e)}"],
                    data_quality="poor"
                )
        
        return intelligence
    
    async def filter_tradeable_symbols(
        self, 
        symbols: List[str], 
        min_score: float = 50
    ) -> List[str]:
        """
        Filter symbols that meet minimum trading conditions.
        
        Args:
            symbols: Symbols to evaluate
            min_score: Minimum score threshold
            
        Returns:
            List of symbols suitable for trading
        """
        intelligence = await self.get_market_intelligence(symbols)
        
        tradeable = []
        for symbol, intel in intelligence.items():
            if (intel.score >= min_score and 
                intel.condition not in [MarketCondition.POOR, MarketCondition.CLOSED] and
                intel.data_quality in ["good", "limited"]):
                tradeable.append(symbol)
        
        logger.info(f"Filtered {len(tradeable)}/{len(symbols)} symbols as tradeable")
        return tradeable
    
    def get_session_summary(self) -> Dict[str, any]:
        """Get current session summary"""
        current_session = self.session_analyzer.get_current_session()
        characteristics = self.session_analyzer.get_session_characteristics(current_session)
        
        return {
            'current_session': current_session.value,
            'characteristics': characteristics,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def get_market_overview(self, symbols: List[str]) -> Dict[str, any]:
        """
        Get comprehensive market overview.
        
        Args:
            symbols: Symbols to include in overview
            
        Returns:
            Complete market overview
        """
        intelligence = await self.get_market_intelligence(symbols)
        session_summary = self.get_session_summary()
        
        # Aggregate statistics
        total_symbols = len(intelligence)
        excellent_count = sum(1 for i in intelligence.values() if i.condition == MarketCondition.EXCELLENT)
        good_count = sum(1 for i in intelligence.values() if i.condition == MarketCondition.GOOD)
        avg_score = sum(i.score for i in intelligence.values()) / total_symbols if total_symbols > 0 else 0
        
        return {
            'session': session_summary,
            'symbol_analysis': {symbol: intel.to_dict() for symbol, intel in intelligence.items()},
            'summary': {
                'total_symbols': total_symbols,
                'excellent_conditions': excellent_count,
                'good_conditions': good_count,
                'average_score': round(avg_score, 1),
                'tradeable_symbols': excellent_count + good_count
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


# Export main classes
__all__ = [
    'MarketService', 
    'MarketIntelligence', 
    'MarketCondition',
    'MarketConditionAnalyzer',
    'MarketSessionAnalyzer'
]