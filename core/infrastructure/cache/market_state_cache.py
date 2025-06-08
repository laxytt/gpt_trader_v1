"""
Intelligent caching system for market states and trading decisions.
Reduces expensive GPT-4 API calls by caching similar market conditions.
"""

import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path
import logging
import threading
from collections import OrderedDict

from core.domain.models import MarketData, TradingSignal
from core.utils.error_handling import with_error_context, ErrorContext

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached trading decision"""
    key: str
    market_data_hash: str
    signal: TradingSignal
    council_decision: Dict[str, Any]
    timestamp: datetime
    hit_count: int = 0
    confidence_score: float = 0.0
    
    def is_valid(self, ttl_minutes: int) -> bool:
        """Check if cache entry is still valid"""
        age = datetime.now() - self.timestamp
        return age < timedelta(minutes=ttl_minutes)


class MarketStateCache:
    """
    Intelligent caching system that identifies similar market conditions
    to avoid redundant GPT-4 API calls.
    """
    
    def __init__(self, 
                 cache_dir: Path = Path("data/cache"),
                 max_size_mb: int = 500,
                 similarity_threshold: float = 0.85,
                 ttl_minutes: int = 60):
        """
        Initialize the cache system.
        
        Args:
            cache_dir: Directory to store cache files
            max_size_mb: Maximum cache size in MB
            similarity_threshold: Minimum similarity score to use cached decision
            ttl_minutes: Cache time-to-live in minutes
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.similarity_threshold = similarity_threshold
        self.ttl_minutes = ttl_minutes
        
        # In-memory cache for fast access
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'saves': 0,
            'evictions': 0
        }
        
        # Load existing cache
        self._load_cache()
        
        logger.info(f"MarketStateCache initialized: {len(self._cache)} entries loaded")
    
    def get_cache_key(self, market_data: MarketData) -> str:
        """
        Generate a cache key based on normalized market conditions.
        
        Key components:
        - Symbol
        - Price pattern (normalized)
        - Technical indicator ranges
        - Market regime
        - Time characteristics
        - News impact level
        """
        # Extract key features
        features = self._extract_key_features(market_data)
        
        # Create deterministic string representation
        key_string = json.dumps(features, sort_keys=True)
        
        # Generate hash
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]
    
    def _extract_key_features(self, market_data: MarketData) -> Dict[str, Any]:
        """Extract normalized features for cache key generation"""
        # Get recent candles (last 20 H1)
        h1_candles = market_data.h1_candles[-20:] if len(market_data.h1_candles) >= 20 else market_data.h1_candles
        
        if not h1_candles:
            return {}
        
        # Price normalization - convert to percentage changes
        price_changes = []
        for i in range(1, len(h1_candles)):
            change = (h1_candles[i].close - h1_candles[i-1].close) / h1_candles[i-1].close
            price_changes.append(round(change * 100, 1))  # Round to 0.1%
        
        # Technical indicators - convert to ranges
        rsi_range = "oversold" if h1_candles[-1].rsi < 30 else "overbought" if h1_candles[-1].rsi > 70 else "neutral"
        
        # ATR as percentage of price for volatility
        atr_percent = round((h1_candles[-1].atr / h1_candles[-1].close) * 100, 1)
        volatility_regime = "low" if atr_percent < 0.5 else "high" if atr_percent > 1.5 else "normal"
        
        # Trend based on EMAs
        ema50 = h1_candles[-1].ema_50
        ema200 = h1_candles[-1].ema_200
        price = h1_candles[-1].close
        
        if price > ema50 > ema200:
            trend = "strong_up"
        elif price < ema50 < ema200:
            trend = "strong_down"
        elif price > ema50:
            trend = "weak_up"
        elif price < ema50:
            trend = "weak_down"
        else:
            trend = "neutral"
        
        # Time characteristics
        hour = datetime.now().hour
        day_of_week = datetime.now().weekday()
        session = "asian" if 0 <= hour < 8 else "european" if 8 <= hour < 16 else "american"
        
        # Volume profile (normalized)
        avg_volume = np.mean([c.volume for c in h1_candles])
        recent_volume = h1_candles[-1].volume
        volume_ratio = round(recent_volume / avg_volume if avg_volume > 0 else 1.0, 1)
        
        return {
            'symbol': market_data.symbol,
            'price_pattern': self._categorize_price_pattern(price_changes),
            'rsi_range': rsi_range,
            'volatility': volatility_regime,
            'trend': trend,
            'session': session,
            'day_type': 'weekend' if day_of_week >= 5 else 'weekday',
            'volume_profile': 'high' if volume_ratio > 1.5 else 'low' if volume_ratio < 0.5 else 'normal',
            'news_impact': market_data.news_impact if hasattr(market_data, 'news_impact') else 'none'
        }
    
    def _categorize_price_pattern(self, price_changes: List[float]) -> str:
        """Categorize price pattern for caching"""
        if not price_changes:
            return "unknown"
        
        # Simple pattern recognition
        avg_change = np.mean(price_changes)
        volatility = np.std(price_changes)
        
        if avg_change > 0.5 and volatility < 0.3:
            return "steady_up"
        elif avg_change < -0.5 and volatility < 0.3:
            return "steady_down"
        elif abs(avg_change) < 0.2 and volatility < 0.2:
            return "ranging_tight"
        elif abs(avg_change) < 0.2 and volatility > 0.5:
            return "ranging_wide"
        elif avg_change > 0.5 and volatility > 0.5:
            return "volatile_up"
        elif avg_change < -0.5 and volatility > 0.5:
            return "volatile_down"
        else:
            return "mixed"
    
    @with_error_context("get_cached_decision")
    def get_cached_decision(self, market_data: MarketData) -> Optional[Tuple[TradingSignal, Dict[str, Any]]]:
        """
        Get cached decision for similar market conditions.
        
        Returns:
            Tuple of (signal, council_decision) if found, None otherwise
        """
        with self._lock:
            # Generate key for current market state
            current_key = self.get_cache_key(market_data)
            
            # First, try exact match
            if current_key in self._cache:
                entry = self._cache[current_key]
                if entry.is_valid(self.ttl_minutes):
                    # Update hit count and move to end (LRU)
                    entry.hit_count += 1
                    self._cache.move_to_end(current_key)
                    self.stats['hits'] += 1
                    
                    logger.info(f"Cache hit for {market_data.symbol}: {entry.signal.signal} "
                              f"(confidence: {entry.confidence_score:.1f}%, hits: {entry.hit_count})")
                    
                    return entry.signal, entry.council_decision
            
            # If no exact match, find similar market conditions
            best_match = self._find_similar_entry(market_data)
            if best_match:
                self.stats['hits'] += 1
                return best_match.signal, best_match.council_decision
            
            self.stats['misses'] += 1
            return None
    
    def _find_similar_entry(self, market_data: MarketData) -> Optional[CacheEntry]:
        """Find similar cached entry based on market similarity"""
        current_features = self._extract_detailed_features(market_data)
        best_similarity = 0.0
        best_entry = None
        
        for key, entry in self._cache.items():
            if not entry.is_valid(self.ttl_minutes):
                continue
            
            # Only compare same symbol
            if entry.signal.symbol != market_data.symbol:
                continue
            
            # Calculate similarity score
            similarity = self.calculate_similarity(current_features, entry)
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_entry = entry
        
        if best_entry:
            logger.info(f"Found similar cached decision for {market_data.symbol} "
                      f"(similarity: {best_similarity:.1%})")
            best_entry.hit_count += 1
        
        return best_entry
    
    def _extract_detailed_features(self, market_data: MarketData) -> np.ndarray:
        """Extract detailed features for similarity comparison"""
        features = []
        
        # Get recent candles
        h1_candles = market_data.h1_candles[-20:] if len(market_data.h1_candles) >= 20 else market_data.h1_candles
        
        if not h1_candles:
            return np.array([])
        
        # Price features (normalized)
        prices = [c.close for c in h1_candles]
        price_mean = np.mean(prices)
        price_std = np.std(prices) if len(prices) > 1 else 1.0
        
        # Normalize recent prices
        normalized_prices = [(p - price_mean) / price_std if price_std > 0 else 0 for p in prices[-5:]]
        features.extend(normalized_prices)
        
        # Technical indicators (last candle)
        last_candle = h1_candles[-1]
        features.extend([
            last_candle.rsi / 100.0,  # Normalize RSI
            (last_candle.close - last_candle.ema_50) / last_candle.close,  # Price vs EMA50
            (last_candle.close - last_candle.ema_200) / last_candle.close,  # Price vs EMA200
            last_candle.atr / last_candle.close,  # ATR as percentage
            last_candle.volume / np.mean([c.volume for c in h1_candles])  # Volume ratio
        ])
        
        # Price patterns
        returns = [(h1_candles[i].close - h1_candles[i-1].close) / h1_candles[i-1].close 
                  for i in range(1, len(h1_candles))]
        
        if returns:
            features.extend([
                np.mean(returns),  # Average return
                np.std(returns),   # Volatility
                max(returns),      # Maximum gain
                min(returns),      # Maximum loss
                sum(1 for r in returns if r > 0) / len(returns)  # Win rate
            ])
        
        return np.array(features)
    
    def calculate_similarity(self, current_features: np.ndarray, cached_entry: CacheEntry) -> float:
        """
        Calculate similarity between current market state and cached entry.
        Uses multiple similarity metrics combined.
        """
        # For now, use the basic feature similarity from the cache key
        # This can be enhanced with the detailed features
        current_key_features = self._extract_key_features(MarketData(
            symbol=cached_entry.signal.symbol,
            h1_candles=[],  # Not needed for key features
            h4_candles=[],
            h1_screenshot_path="",
            h4_screenshot_path=""
        ))
        
        # Count matching features
        matches = 0
        total = 0
        
        for key, value in current_key_features.items():
            if key in ['symbol', 'price_pattern', 'trend', 'rsi_range', 'volatility', 'session']:
                total += 1
                # Compare based on stored metadata if available
                # For now, simple comparison
                if key == 'symbol' and value == cached_entry.signal.symbol:
                    matches += 1
                # Add more sophisticated comparison logic here
        
        # Return similarity score
        return matches / total if total > 0 else 0.0
    
    @with_error_context("cache_decision")
    def cache_decision(self, 
                      market_data: MarketData, 
                      signal: TradingSignal,
                      council_decision: Dict[str, Any],
                      confidence_score: float):
        """Cache a trading decision for future use"""
        with self._lock:
            key = self.get_cache_key(market_data)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                market_data_hash=self._hash_market_data(market_data),
                signal=signal,
                council_decision=council_decision,
                timestamp=datetime.now(),
                confidence_score=confidence_score
            )
            
            # Add to cache (LRU eviction if needed)
            self._cache[key] = entry
            self._cache.move_to_end(key)
            
            # Check size and evict if necessary
            self._evict_if_needed()
            
            # Persist to disk
            self._save_cache()
            
            self.stats['saves'] += 1
            logger.info(f"Cached decision for {signal.symbol}: {signal.signal} "
                      f"(confidence: {confidence_score:.1f}%)")
    
    def _hash_market_data(self, market_data: MarketData) -> str:
        """Create hash of full market data for verification"""
        # Create a string representation of key market data
        data_str = f"{market_data.symbol}"
        
        if market_data.h1_candles:
            last_candle = market_data.h1_candles[-1]
            data_str += f"_{last_candle.close}_{last_candle.volume}_{last_candle.rsi}"
        
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def _evict_if_needed(self):
        """Evict old entries if cache is too large"""
        # Simple implementation - remove oldest entries
        while len(self._cache) > 1000:  # Max 1000 entries
            # Remove oldest (first) entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self.stats['evictions'] += 1
    
    def _save_cache(self):
        """Persist cache to disk"""
        try:
            cache_file = self.cache_dir / "market_state_cache.json"
            
            # Convert cache to serializable format
            cache_data = {
                'entries': [],
                'stats': self.stats,
                'timestamp': datetime.now().isoformat()
            }
            
            for key, entry in self._cache.items():
                # Serialize entry (skip complex objects)
                entry_data = {
                    'key': entry.key,
                    'symbol': entry.signal.symbol,
                    'signal': entry.signal.signal,
                    'timestamp': entry.timestamp.isoformat(),
                    'hit_count': entry.hit_count,
                    'confidence_score': entry.confidence_score
                }
                cache_data['entries'].append(entry_data)
            
            # Write to file
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _load_cache(self):
        """Load cache from disk"""
        try:
            cache_file = self.cache_dir / "market_state_cache.json"
            
            if not cache_file.exists():
                return
            
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Load stats
            self.stats.update(cache_data.get('stats', {}))
            
            # Note: We're not loading full entries to avoid stale data
            # Just log what was there
            logger.info(f"Cache file found with {len(cache_data.get('entries', []))} entries")
            
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'hit_rate': hit_rate,
                'total_requests': total_requests,
                'cache_size': len(self._cache),
                'saves': self.stats['saves'],
                'evictions': self.stats['evictions'],
                'hits': self.stats['hits'],
                'misses': self.stats['misses']
            }
    
    def should_skip_analysis(self, market_data: MarketData) -> Tuple[bool, str]:
        """
        Quick check if we should skip analysis entirely.
        Used for obvious non-trading conditions.
        """
        # Get recent candles
        h1_candles = market_data.h1_candles[-5:] if len(market_data.h1_candles) >= 5 else market_data.h1_candles
        
        if not h1_candles:
            return True, "No candle data available"
        
        last_candle = h1_candles[-1]
        
        # Check for extreme spread
        spread_percent = (last_candle.spread / last_candle.close) * 100
        if spread_percent > 0.5:  # 0.5% spread is very high
            return True, f"Spread too high: {spread_percent:.2f}%"
        
        # Check for extremely low volatility
        if last_candle.atr < last_candle.close * 0.0001:  # ATR less than 0.01%
            return True, "Volatility too low for trading"
        
        # Check for data quality issues
        if last_candle.volume == 0:
            return True, "No volume data"
        
        # Check for weekend/holiday thin markets
        if datetime.now().weekday() >= 5:  # Saturday or Sunday
            hour = datetime.now().hour
            if hour < 22:  # Before Sunday 22:00 UTC
                return True, "Weekend - market closed"
        
        return False, ""