"""
Production-grade OpenAI Rate Limiter
Supports different OpenAI tiers and implements token bucket algorithm
"""

import asyncio
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class RateLimitBucket:
    """Token bucket for rate limiting"""
    capacity: int  # Max tokens
    tokens: float  # Current tokens
    refill_rate: float  # Tokens per second
    last_refill: float  # Last refill timestamp
    
    def refill(self):
        """Refill tokens based on time elapsed"""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        
    def consume(self, count: int = 1) -> bool:
        """Try to consume tokens"""
        self.refill()
        if self.tokens >= count:
            self.tokens -= count
            return True
        return False
        
    def time_until_available(self, count: int = 1) -> float:
        """Time in seconds until tokens are available"""
        self.refill()
        if self.tokens >= count:
            return 0
        needed = count - self.tokens
        return needed / self.refill_rate


class OpenAIRateLimiter:
    """
    Production rate limiter for OpenAI API
    Implements token bucket algorithm with tier support
    """
    
    # OpenAI Tier Limits (from production_settings.py)
    TIER_LIMITS = {
        "tier_1": {
            "gpt-4": {"rpm": 500, "tpm": 10000, "rpd": 10000},
            "gpt-4-turbo": {"rpm": 500, "tpm": 30000, "rpd": 10000},
            "gpt-3.5-turbo": {"rpm": 3500, "tpm": 60000, "rpd": 10000},
            "gpt-4o-mini": {"rpm": 500, "tpm": 30000, "rpd": 10000},
            "gpt-4o": {"rpm": 500, "tpm": 30000, "rpd": 10000},
        },
        "tier_2": {
            "gpt-4": {"rpm": 5000, "tpm": 40000, "rpd": None},
            "gpt-4-turbo": {"rpm": 5000, "tpm": 80000, "rpd": None},
            "gpt-3.5-turbo": {"rpm": 10000, "tpm": 200000, "rpd": None},
            "gpt-4o-mini": {"rpm": 5000, "tpm": 80000, "rpd": None},
            "gpt-4o": {"rpm": 5000, "tpm": 80000, "rpd": None},
        },
        "tier_3": {
            "gpt-4": {"rpm": 10000, "tpm": 150000, "rpd": None},
            "gpt-4-turbo": {"rpm": 10000, "tpm": 300000, "rpd": None},
            "gpt-3.5-turbo": {"rpm": 10000, "tpm": 1000000, "rpd": None},
            "gpt-4o-mini": {"rpm": 10000, "tpm": 300000, "rpd": None},
            "gpt-4o": {"rpm": 10000, "tpm": 300000, "rpd": None},
        }
    }
    
    def __init__(
        self,
        tier: str = "tier_1",
        safety_margin: float = 0.8,
        burst_limit: int = 10,
        burst_window: int = 10
    ):
        """
        Initialize rate limiter
        
        Args:
            tier: OpenAI tier (tier_1, tier_2, tier_3)
            safety_margin: Use only this fraction of actual limits (0.8 = 80%)
            burst_limit: Max requests in burst window
            burst_window: Burst window in seconds
        """
        self.tier = tier
        self.safety_margin = safety_margin
        self.burst_limit = burst_limit
        self.burst_window = burst_window
        
        # Rate limit buckets per model
        self.request_buckets: Dict[str, RateLimitBucket] = {}
        self.token_buckets: Dict[str, RateLimitBucket] = {}
        self.daily_buckets: Dict[str, RateLimitBucket] = {}
        
        # Burst control
        self.burst_timestamps: list[float] = []
        
        # Statistics
        self.stats = {
            'requests_allowed': 0,
            'requests_throttled': 0,
            'total_wait_time': 0,
            'rate_limit_hits': 0
        }
        
        self._locks = {}  # Will store locks per thread/loop
        self._init_buckets()
        
    def _init_buckets(self):
        """Initialize rate limit buckets for each model"""
        if self.tier not in self.TIER_LIMITS:
            logger.warning(f"Unknown tier {self.tier}, using tier_1")
            self.tier = "tier_1"
            
        for model, limits in self.TIER_LIMITS[self.tier].items():
            # Requests per minute bucket
            rpm = int(limits['rpm'] * self.safety_margin)
            self.request_buckets[model] = RateLimitBucket(
                capacity=rpm,
                tokens=rpm,
                refill_rate=rpm / 60.0,
                last_refill=time.time()
            )
            
            # Tokens per minute bucket
            tpm = int(limits['tpm'] * self.safety_margin)
            self.token_buckets[model] = RateLimitBucket(
                capacity=tpm,
                tokens=tpm,
                refill_rate=tpm / 60.0,
                last_refill=time.time()
            )
            
            # Requests per day bucket (if applicable)
            if limits.get('rpd'):
                rpd = int(limits['rpd'] * self.safety_margin)
                self.daily_buckets[model] = RateLimitBucket(
                    capacity=rpd,
                    tokens=rpd,
                    refill_rate=rpd / 86400.0,  # per second
                    last_refill=time.time()
                )
                
    async def acquire(
        self, 
        model: str, 
        estimated_tokens: int = 1000,
        priority: int = 1
    ) -> Tuple[bool, float]:
        """
        Acquire permission to make an API call
        
        Args:
            model: OpenAI model name
            estimated_tokens: Estimated tokens for the request
            priority: Request priority (higher = more important)
            
        Returns:
            (allowed, wait_time) - whether allowed and wait time if not
        """
        # Get or create lock for current event loop/thread
        import threading
        thread_id = threading.get_ident()
        
        if thread_id not in self._locks:
            try:
                # Try to get current event loop
                loop = asyncio.get_event_loop()
                self._locks[thread_id] = asyncio.Lock()
            except RuntimeError:
                # No event loop, use threading lock
                import threading
                self._locks[thread_id] = threading.Lock()
        
        lock = self._locks[thread_id]
        
        # Handle both async and sync locks
        if isinstance(lock, asyncio.Lock):
            async with lock:
                return await self._do_acquire(model, estimated_tokens, priority)
        else:
            with lock:
                return await self._do_acquire(model, estimated_tokens, priority)
                
    async def _do_acquire(
        self, 
        model: str, 
        estimated_tokens: int = 1000,
        priority: int = 1
    ) -> Tuple[bool, float]:
            # Check burst limit
            now = time.time()
            self.burst_timestamps = [
                ts for ts in self.burst_timestamps 
                if ts > now - self.burst_window
            ]
            
            if len(self.burst_timestamps) >= self.burst_limit:
                wait_time = self.burst_window - (now - self.burst_timestamps[0])
                self.stats['requests_throttled'] += 1
                return False, wait_time
                
            # Get model buckets (fallback to gpt-4 limits if model unknown)
            if model not in self.request_buckets:
                logger.warning(f"Unknown model {model}, using gpt-4 limits")
                model = "gpt-4"
                
            # Check all rate limits
            request_bucket = self.request_buckets[model]
            token_bucket = self.token_buckets[model]
            daily_bucket = self.daily_buckets.get(model)
            
            # Calculate wait times
            request_wait = request_bucket.time_until_available(1)
            token_wait = token_bucket.time_until_available(estimated_tokens)
            daily_wait = daily_bucket.time_until_available(1) if daily_bucket else 0
            
            max_wait = max(request_wait, token_wait, daily_wait)
            
            if max_wait > 0:
                self.stats['requests_throttled'] += 1
                self.stats['total_wait_time'] += max_wait
                
                # High priority requests get shorter wait
                if priority > 1:
                    max_wait = max_wait / priority
                    
                return False, max_wait
                
            # Consume tokens
            request_bucket.consume(1)
            token_bucket.consume(estimated_tokens)
            if daily_bucket:
                daily_bucket.consume(1)
                
            # Record burst timestamp
            self.burst_timestamps.append(now)
            
            self.stats['requests_allowed'] += 1
            return True, 0
            
    async def wait_if_needed(
        self,
        model: str,
        estimated_tokens: int = 1000,
        max_wait: float = 60.0,
        priority: int = 1
    ) -> bool:
        """
        Wait if rate limited
        
        Args:
            model: OpenAI model
            estimated_tokens: Estimated tokens
            max_wait: Maximum wait time in seconds
            priority: Request priority
            
        Returns:
            True if acquired, False if timed out
        """
        start_time = time.time()
        
        while True:
            allowed, wait_time = await self.acquire(model, estimated_tokens, priority)
            
            if allowed:
                return True
                
            if wait_time > max_wait:
                logger.warning(f"Rate limit wait time {wait_time}s exceeds max {max_wait}s")
                return False
                
            # Add jitter to prevent thundering herd
            jitter = 0.1 * wait_time
            actual_wait = wait_time + (jitter * (0.5 - time.time() % 1))
            
            logger.info(f"Rate limited, waiting {actual_wait:.1f}s (priority={priority})")
            await asyncio.sleep(actual_wait)
            
            # Check timeout
            if time.time() - start_time > max_wait:
                return False
                
    def report_actual_usage(self, model: str, actual_tokens: int):
        """Report actual token usage after completion"""
        # Could be used to adjust estimates in the future
        pass
        
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        return {
            **self.stats,
            'tier': self.tier,
            'safety_margin': self.safety_margin,
            'current_buckets': {
                model: {
                    'requests_available': bucket.tokens,
                    'tokens_available': self.token_buckets[model].tokens
                }
                for model, bucket in self.request_buckets.items()
            }
        }
        
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'requests_allowed': 0,
            'requests_throttled': 0,
            'total_wait_time': 0,
            'rate_limit_hits': 0
        }


# Global rate limiter instance
_rate_limiter: Optional[OpenAIRateLimiter] = None


def get_rate_limiter(
    tier: Optional[str] = None,
    safety_margin: Optional[float] = None
) -> OpenAIRateLimiter:
    """Get the global rate limiter instance"""
    global _rate_limiter
    
    if _rate_limiter is None:
        # Load from settings or use defaults
        import os
        tier = tier or os.getenv('OPENAI_TIER', 'tier_1')
        safety_margin = safety_margin or float(os.getenv('RATE_LIMIT_SAFETY_MARGIN', '0.8'))
        
        _rate_limiter = OpenAIRateLimiter(
            tier=tier,
            safety_margin=safety_margin
        )
        
    return _rate_limiter