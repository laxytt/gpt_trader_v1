"""
Pre-trade filtering service to eliminate obvious non-trading opportunities.
Reduces expensive GPT API calls by filtering out low-quality setups early.
"""

import logging
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
import numpy as np

from core.domain.models import MarketData, Candle
from core.domain.exceptions import ValidationError
from core.utils.error_handling import with_error_context, ErrorContext
from config.settings import TradingSettings

logger = logging.getLogger(__name__)


class FilterResult:
    """Result of pre-trade filtering"""
    def __init__(self, should_analyze: bool, reason: str, details: Dict[str, Any] = None):
        self.should_analyze = should_analyze
        self.reason = reason
        self.details = details or {}
        self.filter_scores = {}


class PreTradeFilter:
    """
    Fast pre-filters to eliminate obvious non-trades before expensive council analysis.
    """
    
    def __init__(self, trading_config: TradingSettings):
        self.trading_config = trading_config
        
        # Filter thresholds
        self.max_spread_atr_ratio = 0.3  # Spread should not exceed 30% of ATR
        self.min_atr_threshold = 0.0002  # Minimum ATR for tradeable volatility
        self.min_volume_ratio = 0.5  # Current volume vs average
        self.trend_alignment_threshold = 0.6  # Minimum trend score
        
        # Time filters
        self.restricted_hours = {
            'friday_close': [(20, 0), (23, 59)],  # Friday evening
            'sunday_open': [(0, 0), (2, 0)],      # Sunday opening
            'low_liquidity': [(22, 0), (1, 0)]     # General low liquidity
        }
        
        # News blackout window (minutes)
        self.news_blackout_before = 30
        self.news_blackout_after = 15
        
        # Statistics
        self.stats = {
            'total_filtered': 0,
            'filter_reasons': {},
            'pass_rate': 0.0
        }
    
    @with_error_context("pre_trade_filter")
    async def should_analyze(self, market_data: MarketData, news_events: List[Dict] = None) -> FilterResult:
        """
        Determine if market conditions warrant full council analysis.
        
        Args:
            market_data: Current market data
            news_events: Upcoming news events
            
        Returns:
            FilterResult with decision and reasoning
        """
        # Run all filter checks
        checks = [
            ('spread', self.check_spread_filter(market_data)),
            ('volatility', self.check_volatility_filter(market_data)),
            ('time', self.check_time_filter()),
            ('trend', self.check_trend_alignment(market_data)),
            ('volume', self.check_volume_filter(market_data)),
            ('structure', self.check_market_structure(market_data))
        ]
        
        if news_events:
            checks.append(('news', self.check_news_blackout(news_events)))
        
        # Compile results
        failed_checks = []
        filter_scores = {}
        total_score = 0
        
        for check_name, (passed, score, reason) in checks:
            filter_scores[check_name] = score
            total_score += score
            
            if not passed:
                failed_checks.append(f"{check_name}: {reason}")
        
        # Calculate overall quality score
        quality_score = total_score / len(checks)
        
        # Decision logic
        if len(failed_checks) > 2:  # Multiple failures
            result = FilterResult(
                should_analyze=False,
                reason=f"Multiple filter failures: {'; '.join(failed_checks[:2])}",
                details={
                    'failed_checks': failed_checks,
                    'quality_score': quality_score,
                    'filter_scores': filter_scores
                }
            )
        elif failed_checks and quality_score < 0.5:  # Low overall quality
            result = FilterResult(
                should_analyze=False,
                reason=f"Low market quality ({quality_score:.1%}): {failed_checks[0]}",
                details={
                    'failed_checks': failed_checks,
                    'quality_score': quality_score,
                    'filter_scores': filter_scores
                }
            )
        else:
            result = FilterResult(
                should_analyze=True,
                reason="Market conditions suitable for analysis",
                details={
                    'quality_score': quality_score,
                    'filter_scores': filter_scores
                }
            )
        
        # Update statistics
        self._update_stats(result)
        
        return result
    
    def check_spread_filter(self, market_data: MarketData) -> Tuple[bool, float, str]:
        """Check if spread is reasonable relative to ATR"""
        if not market_data.h1_candles:
            return False, 0.0, "No candle data"
        
        last_candle = market_data.h1_candles[-1]
        
        if last_candle.atr <= 0:
            return False, 0.0, "Invalid ATR"
        
        spread_atr_ratio = last_candle.spread / last_candle.atr
        
        if spread_atr_ratio > self.max_spread_atr_ratio:
            return False, 0.2, f"Spread too wide ({spread_atr_ratio:.1%} of ATR)"
        
        # Score based on how tight the spread is
        score = max(0, 1.0 - (spread_atr_ratio / self.max_spread_atr_ratio))
        
        return True, score, "Spread acceptable"
    
    def check_volatility_filter(self, market_data: MarketData) -> Tuple[bool, float, str]:
        """Check if volatility is sufficient for trading"""
        if not market_data.h1_candles:
            return False, 0.0, "No candle data"
        
        last_candle = market_data.h1_candles[-1]
        recent_candles = market_data.h1_candles[-10:] if len(market_data.h1_candles) >= 10 else market_data.h1_candles
        
        # Check ATR
        if last_candle.atr < self.min_atr_threshold:
            return False, 0.1, f"ATR too low ({last_candle.atr:.5f})"
        
        # Check recent price movement
        price_changes = []
        for i in range(1, len(recent_candles)):
            change = abs(recent_candles[i].close - recent_candles[i-1].close)
            price_changes.append(change)
        
        avg_movement = np.mean(price_changes) if price_changes else 0
        
        if avg_movement < self.min_atr_threshold * 0.5:
            return False, 0.2, "Insufficient price movement"
        
        # Score based on volatility level
        volatility_score = min(1.0, last_candle.atr / (self.min_atr_threshold * 5))
        
        return True, volatility_score, "Adequate volatility"
    
    def check_time_filter(self) -> Tuple[bool, float, str]:
        """Check if current time is suitable for trading"""
        now = datetime.now(timezone.utc)
        hour = now.hour
        minute = now.minute
        day_of_week = now.weekday()  # 0 = Monday, 6 = Sunday
        
        current_time = (hour, minute)
        
        # Check if within trading hours
        if not (self.trading_config.start_hour <= hour < self.trading_config.end_hour):
            return False, 0.0, f"Outside trading hours ({hour}:00 UTC)"
        
        # Friday close restrictions
        if day_of_week == 4:  # Friday
            for start, end in self.restricted_hours['friday_close']:
                if self._is_time_in_range(current_time, start, end):
                    return False, 0.1, "Friday close period"
        
        # Sunday open restrictions
        if day_of_week == 6:  # Sunday
            for start, end in self.restricted_hours['sunday_open']:
                if self._is_time_in_range(current_time, start, end):
                    return False, 0.1, "Sunday open period"
        
        # General low liquidity hours
        for start, end in self.restricted_hours['low_liquidity']:
            if self._is_time_in_range(current_time, start, end):
                return True, 0.5, "Low liquidity period (caution)"
        
        # Score based on optimal trading hours
        if 8 <= hour <= 16:  # London/NY overlap
            score = 1.0
        elif 7 <= hour <= 20:  # Major sessions
            score = 0.8
        else:
            score = 0.6
        
        return True, score, "Good trading time"
    
    def check_trend_alignment(self, market_data: MarketData) -> Tuple[bool, float, str]:
        """Check if H1 and H4 trends are aligned"""
        if not market_data.h1_candles or len(market_data.h1_candles) < 20:
            return False, 0.0, "Insufficient H1 data"
        
        # H1 trend analysis
        h1_last = market_data.h1_candles[-1]
        h1_trend_score = self._calculate_trend_score(market_data.h1_candles[-20:])
        
        # H4 trend analysis (if available)
        h4_trend_score = 0
        if market_data.h4_candles and len(market_data.h4_candles) >= 5:
            h4_trend_score = self._calculate_trend_score(market_data.h4_candles[-5:])
        
        # Check alignment
        if h1_trend_score == 0:  # No clear trend
            return True, 0.3, "No clear trend (ranging market)"
        
        if h4_trend_score != 0 and np.sign(h1_trend_score) != np.sign(h4_trend_score):
            return False, 0.2, "H1/H4 trend conflict"
        
        # Score based on trend strength
        trend_strength = abs(h1_trend_score)
        
        if trend_strength < self.trend_alignment_threshold:
            return True, 0.5, "Weak trend"
        
        return True, min(1.0, trend_strength), "Strong aligned trend"
    
    def check_volume_filter(self, market_data: MarketData) -> Tuple[bool, float, str]:
        """Check if volume is sufficient"""
        if not market_data.h1_candles or len(market_data.h1_candles) < 20:
            return True, 0.5, "Insufficient data for volume check"
        
        recent_candles = market_data.h1_candles[-20:]
        current_volume = recent_candles[-1].volume
        
        # Calculate average volume
        volumes = [c.volume for c in recent_candles[:-1]]
        avg_volume = np.mean(volumes) if volumes else current_volume
        
        if avg_volume == 0:
            return True, 0.5, "No volume data"
        
        volume_ratio = current_volume / avg_volume
        
        if volume_ratio < self.min_volume_ratio:
            return False, 0.3, f"Low volume ({volume_ratio:.1%} of average)"
        
        # Score based on volume level
        if volume_ratio > 2.0:  # High volume
            score = 1.0
        elif volume_ratio > 1.2:  # Above average
            score = 0.8
        else:
            score = 0.6
        
        return True, score, f"Volume {volume_ratio:.1f}x average"
    
    def check_market_structure(self, market_data: MarketData) -> Tuple[bool, float, str]:
        """Check for clean market structure"""
        if not market_data.h1_candles or len(market_data.h1_candles) < 10:
            return True, 0.5, "Insufficient data for structure check"
        
        recent_candles = market_data.h1_candles[-10:]
        
        # Check for choppy/noisy price action
        direction_changes = 0
        for i in range(1, len(recent_candles)):
            if i > 1:
                prev_direction = np.sign(recent_candles[i-1].close - recent_candles[i-2].close)
                curr_direction = np.sign(recent_candles[i].close - recent_candles[i-1].close)
                if prev_direction != curr_direction and prev_direction != 0:
                    direction_changes += 1
        
        choppiness_ratio = direction_changes / (len(recent_candles) - 2)
        
        if choppiness_ratio > 0.7:
            return False, 0.2, "Choppy market structure"
        
        # Check for clean swings
        swings = self._identify_swings(recent_candles)
        
        if len(swings) < 2:
            return True, 0.4, "No clear swings"
        
        # Score based on structure quality
        structure_score = 1.0 - choppiness_ratio
        
        return True, structure_score, "Clean market structure"
    
    def check_news_blackout(self, news_events: List[Dict]) -> Tuple[bool, float, str]:
        """Check if we're in news blackout period"""
        now = datetime.now(timezone.utc)
        
        for event in news_events:
            event_time = event.get('timestamp')
            impact = event.get('impact', 'low')
            
            if not event_time or impact == 'low':
                continue
            
            # Calculate time difference
            time_diff = (event_time - now).total_seconds() / 60  # In minutes
            
            # Check blackout windows
            if -self.news_blackout_after <= time_diff <= self.news_blackout_before:
                return False, 0.0, f"News blackout: {event.get('title', 'High impact news')}"
        
        return True, 1.0, "No imminent news"
    
    def _calculate_trend_score(self, candles: List[Candle]) -> float:
        """Calculate trend score (-1 to 1, negative for downtrend)"""
        if len(candles) < 3:
            return 0.0
        
        # Price position relative to EMAs
        last_candle = candles[-1]
        price_vs_ema50 = (last_candle.close - last_candle.ema_50) / last_candle.close
        price_vs_ema200 = (last_candle.close - last_candle.ema_200) / last_candle.close
        
        # EMA alignment
        ema_alignment = 1.0 if last_candle.ema_50 > last_candle.ema_200 else -1.0
        
        # Price momentum
        price_changes = [(candles[i].close - candles[i-1].close) / candles[i-1].close 
                        for i in range(1, len(candles))]
        momentum = np.mean(price_changes) * 100
        
        # Combine factors
        trend_score = (
            price_vs_ema50 * 0.3 +
            price_vs_ema200 * 0.2 +
            ema_alignment * 0.3 +
            np.sign(momentum) * min(abs(momentum), 1.0) * 0.2
        )
        
        return np.clip(trend_score, -1.0, 1.0)
    
    def _identify_swings(self, candles: List[Candle]) -> List[Dict]:
        """Identify price swings in candles"""
        swings = []
        
        for i in range(1, len(candles) - 1):
            # Swing high
            if candles[i].high > candles[i-1].high and candles[i].high > candles[i+1].high:
                swings.append({'type': 'high', 'index': i, 'price': candles[i].high})
            # Swing low
            elif candles[i].low < candles[i-1].low and candles[i].low < candles[i+1].low:
                swings.append({'type': 'low', 'index': i, 'price': candles[i].low})
        
        return swings
    
    def _is_time_in_range(self, current: Tuple[int, int], start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """Check if current time is within range"""
        current_minutes = current[0] * 60 + current[1]
        start_minutes = start[0] * 60 + start[1]
        end_minutes = end[0] * 60 + end[1]
        
        if start_minutes <= end_minutes:
            return start_minutes <= current_minutes <= end_minutes
        else:  # Crosses midnight
            return current_minutes >= start_minutes or current_minutes <= end_minutes
    
    def _update_stats(self, result: FilterResult):
        """Update filter statistics"""
        self.stats['total_filtered'] += 1
        
        if not result.should_analyze:
            reason_key = result.reason.split(':')[0]
            self.stats['filter_reasons'][reason_key] = self.stats['filter_reasons'].get(reason_key, 0) + 1
        
        # Calculate pass rate
        total = self.stats['total_filtered']
        passed = total - sum(self.stats['filter_reasons'].values())
        self.stats['pass_rate'] = passed / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get filter statistics"""
        return {
            'total_analyzed': self.stats['total_filtered'],
            'pass_rate': f"{self.stats['pass_rate']:.1%}",
            'filter_reasons': self.stats['filter_reasons'],
            'filters_active': {
                'spread': self.max_spread_atr_ratio,
                'min_atr': self.min_atr_threshold,
                'min_volume': self.min_volume_ratio,
                'trend_threshold': self.trend_alignment_threshold
            }
        }