# core/utils/error_handling.py
"""Standardized error handling utilities"""

import asyncio
import logging
import functools
import time
from typing import Dict, TypeVar, Callable, Optional, Any
from datetime import datetime, timezone

from core.domain.exceptions import TradingSystemError

logger = logging.getLogger(__name__)

T = TypeVar('T')


def with_error_recovery(
    default_return: Optional[T] = None,
    log_level: str = 'error',
    max_retries: int = 0,
    raise_on_critical: bool = True
) -> Callable:
    """
    Decorator for standardized error recovery.
    
    Args:
        default_return: Default value to return on error
        log_level: Logging level for errors
        max_retries: Number of retry attempts
        raise_on_critical: Whether to raise on critical errors
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except TradingSystemError as e:
                    last_exception = e
                    if raise_on_critical and e.__class__.__name__ in [
                        'ConfigurationError', 'RiskManagementError', 'InsufficientFundsError'
                    ]:
                        raise
                    
                    log_func = getattr(logger, log_level)
                    log_func(f"{func.__name__} failed (attempt {attempt + 1}): {e}")
                    
                    if attempt < max_retries:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                except Exception as e:
                    last_exception = e
                    logger.exception(f"Unexpected error in {func.__name__}: {e}")
                    
                    if raise_on_critical:
                        raise
                    
                    if attempt < max_retries:
                        await asyncio.sleep(2 ** attempt)
            
            # All attempts failed
            logger.error(f"{func.__name__} failed after {max_retries + 1} attempts")
            return default_return
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except TradingSystemError as e:
                    last_exception = e
                    if raise_on_critical and e.__class__.__name__ in [
                        'ConfigurationError', 'RiskManagementError', 'InsufficientFundsError'
                    ]:
                        raise
                    
                    log_func = getattr(logger, log_level)
                    log_func(f"{func.__name__} failed (attempt {attempt + 1}): {e}")
                    
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)
                except Exception as e:
                    last_exception = e
                    logger.exception(f"Unexpected error in {func.__name__}: {e}")
                    
                    if raise_on_critical:
                        raise
                    
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)
            
            # All attempts failed
            logger.error(f"{func.__name__} failed after {max_retries + 1} attempts")
            return default_return
        
        # Return the appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class ErrorAggregator:
    """Aggregates errors for batch operations"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def add_error(self, context: str, error: Exception):
        """Add an error to the aggregator"""
        self.errors.append({
            'timestamp': datetime.now(timezone.utc),
            'context': context,
            'error': str(error),
            'type': type(error).__name__
        })
    
    def add_warning(self, context: str, message: str):
        """Add a warning to the aggregator"""
        self.warnings.append({
            'timestamp': datetime.now(timezone.utc),
            'context': context,
            'message': message
        })
    
    def has_errors(self) -> bool:
        """Check if any errors occurred"""
        return len(self.errors) > 0
    
    def has_critical_errors(self) -> bool:
        """Check if any critical errors occurred"""
        critical_types = {'ConfigurationError', 'RiskManagementError', 'InsufficientFundsError'}
        return any(e['type'] in critical_types for e in self.errors)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get error summary"""
        return {
            'total_errors': len(self.errors),
            'total_warnings': len(self.warnings),
            'has_critical': self.has_critical_errors(),
            'errors': self.errors,
            'warnings': self.warnings
        }