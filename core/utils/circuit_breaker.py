"""
Circuit Breaker pattern implementation for fault-tolerant external service calls.
Prevents cascading failures by temporarily disabling calls to failing services.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Callable, Any, Dict
from functools import wraps
import time

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker implementation for protecting external service calls.
    
    States:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Service is failing, calls are rejected immediately
    - HALF_OPEN: Testing if service has recovered
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception,
        success_threshold: int = 2
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to catch (others are re-raised)
            success_threshold: Successes needed in half-open state to close circuit
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.success_threshold = success_threshold
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._last_state_change = datetime.now()
        
    @property
    def state(self) -> CircuitState:
        """Get current circuit state"""
        return self._state
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)"""
        return self._state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing)"""
        return self._state == CircuitState.OPEN
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Raises:
            Exception: If circuit is open or function fails
        """
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    async def async_call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute async function through circuit breaker.
        
        Raises:
            Exception: If circuit is open or function fails
        """
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        return (
            self._last_failure_time and
            datetime.now() >= self._last_failure_time + timedelta(seconds=self.recovery_timeout)
        )
    
    def _on_success(self):
        """Handle successful call"""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._transition_to_closed()
        else:
            self._failure_count = 0
    
    def _on_failure(self):
        """Handle failed call"""
        self._failure_count += 1
        self._last_failure_time = datetime.now()
        
        if self._state == CircuitState.HALF_OPEN:
            self._transition_to_open()
        elif self._failure_count >= self.failure_threshold:
            self._transition_to_open()
    
    def _transition_to_open(self):
        """Transition to OPEN state"""
        self._state = CircuitState.OPEN
        self._last_state_change = datetime.now()
        logger.warning(f"Circuit breaker opened after {self._failure_count} failures")
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state"""
        self._state = CircuitState.HALF_OPEN
        self._last_state_change = datetime.now()
        self._success_count = 0
        self._failure_count = 0
        logger.info("Circuit breaker transitioned to HALF_OPEN")
    
    def _transition_to_closed(self):
        """Transition to CLOSED state"""
        self._state = CircuitState.CLOSED
        self._last_state_change = datetime.now()
        self._failure_count = 0
        self._success_count = 0
        logger.info("Circuit breaker closed after successful recovery")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            'state': self._state.value,
            'failure_count': self._failure_count,
            'success_count': self._success_count,
            'last_failure_time': self._last_failure_time.isoformat() if self._last_failure_time else None,
            'last_state_change': self._last_state_change.isoformat()
        }


class CircuitBreakerManager:
    """Manages circuit breakers for different services"""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
    
    def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception,
        success_threshold: int = 2
    ) -> CircuitBreaker:
        """Get existing circuit breaker or create new one"""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                expected_exception=expected_exception,
                success_threshold=success_threshold
            )
        return self._breakers[name]
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers"""
        return {
            name: breaker.get_stats()
            for name, breaker in self._breakers.items()
        }


# Global circuit breaker manager
_manager = CircuitBreakerManager()


def circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    expected_exception: type = Exception,
    success_threshold: int = 2
):
    """
    Decorator for applying circuit breaker pattern to functions.
    
    Usage:
        @circuit_breaker(name="gpt_api", failure_threshold=3, recovery_timeout=30)
        async def call_gpt_api():
            # API call implementation
            pass
    """
    def decorator(func):
        # Use function name if no name provided
        breaker_name = name or func.__name__
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            breaker = _manager.get_or_create(
                breaker_name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                expected_exception=expected_exception,
                success_threshold=success_threshold
            )
            return await breaker.async_call(func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            breaker = _manager.get_or_create(
                breaker_name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                expected_exception=expected_exception,
                success_threshold=success_threshold
            )
            return breaker.call(func, *args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def get_circuit_breaker_stats() -> Dict[str, Dict[str, Any]]:
    """Get global circuit breaker statistics"""
    return _manager.get_all_stats()


# Specialized circuit breakers for common services
def gpt_circuit_breaker(func):
    """Circuit breaker specifically for GPT API calls"""
    return circuit_breaker(
        name="gpt_api",
        failure_threshold=3,
        recovery_timeout=30,
        expected_exception=Exception,
        success_threshold=2
    )(func)


def mt5_circuit_breaker(func):
    """Circuit breaker specifically for MT5 operations"""
    return circuit_breaker(
        name="mt5_operations",
        failure_threshold=5,
        recovery_timeout=60,
        expected_exception=Exception,
        success_threshold=3
    )(func)


def database_circuit_breaker(func):
    """Circuit breaker specifically for database operations"""
    return circuit_breaker(
        name="database",
        failure_threshold=10,
        recovery_timeout=30,
        expected_exception=Exception,
        success_threshold=5
    )(func)