"""
Retry utilities with exponential backoff for MT5 operations
"""

import asyncio
import time
import logging
from typing import TypeVar, Callable, Optional, Union, Any
from functools import wraps
import random

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryConfig:
    """Configuration for retry behavior"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 0.1,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_errors: Optional[tuple] = None
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_errors = retryable_errors or (Exception,)


def calculate_backoff_delay(
    attempt: int,
    config: RetryConfig
) -> float:
    """Calculate delay with exponential backoff and optional jitter"""
    # Exponential backoff: delay = initial * (base ^ attempt)
    delay = config.initial_delay * (config.exponential_base ** attempt)
    
    # Cap at max delay
    delay = min(delay, config.max_delay)
    
    # Add jitter to prevent thundering herd
    if config.jitter:
        delay = delay * (0.5 + random.random() * 0.5)
    
    return delay


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Decorator for retrying functions with exponential backoff
    
    Args:
        config: Retry configuration
        on_retry: Callback function called on each retry (error, attempt_number)
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except config.retryable_errors as e:
                    last_exception = e
                    
                    if attempt < config.max_attempts - 1:
                        delay = calculate_backoff_delay(attempt, config)
                        
                        logger.warning(
                            f"Retry {attempt + 1}/{config.max_attempts} for {func.__name__} "
                            f"after {delay:.2f}s delay. Error: {str(e)}"
                        )
                        
                        if on_retry:
                            on_retry(e, attempt + 1)
                        
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"Failed after {config.max_attempts} attempts: {func.__name__}"
                        )
            
            raise last_exception
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except config.retryable_errors as e:
                    last_exception = e
                    
                    if attempt < config.max_attempts - 1:
                        delay = calculate_backoff_delay(attempt, config)
                        
                        logger.warning(
                            f"Retry {attempt + 1}/{config.max_attempts} for {func.__name__} "
                            f"after {delay:.2f}s delay. Error: {str(e)}"
                        )
                        
                        if on_retry:
                            on_retry(e, attempt + 1)
                        
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"Failed after {config.max_attempts} attempts: {func.__name__}"
                        )
            
            raise last_exception
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# MT5-specific retry configurations
MT5_ORDER_RETRY_CONFIG = RetryConfig(
    max_attempts=5,
    initial_delay=0.1,
    max_delay=5.0,
    exponential_base=2.0,
    jitter=True,
    retryable_errors=(Exception,)  # Could be more specific
)

MT5_DATA_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    initial_delay=0.5,
    max_delay=10.0,
    exponential_base=2.0,
    jitter=True
)

MT5_MODIFY_RETRY_CONFIG = RetryConfig(
    max_attempts=4,
    initial_delay=0.2,
    max_delay=3.0,
    exponential_base=1.5,
    jitter=True
)


class MT5RetryHelper:
    """Helper class for MT5-specific retry logic"""
    
    @staticmethod
    def should_retry_error(error_code: int) -> bool:
        """Determine if MT5 error code is retryable"""
        # Retryable MT5 error codes
        RETRYABLE_CODES = {
            10004,  # Requote
            10006,  # Request rejected
            10007,  # Request canceled by trader
            10010,  # Request processing
            10012,  # Request expired
            10013,  # Invalid request
            10014,  # Invalid volume
            10019,  # No quotes to process request
            10020,  # Too frequent requests
            10021,  # Market closed
            10044,  # Request queue full
            10045,  # Position with this ID already closed
            10051,  # Invalid ticket
        }
        
        return error_code in RETRYABLE_CODES
    
    @staticmethod
    def get_retry_config_for_error(error_code: int) -> Optional[RetryConfig]:
        """Get appropriate retry config based on error code"""
        if error_code == 10020:  # Too frequent requests
            return RetryConfig(
                max_attempts=3,
                initial_delay=2.0,  # Longer initial delay
                max_delay=10.0,
                exponential_base=2.0
            )
        elif error_code in [10004, 10019]:  # Requote or no quotes
            return RetryConfig(
                max_attempts=5,
                initial_delay=0.5,
                max_delay=5.0,
                exponential_base=1.5
            )
        elif error_code == 10021:  # Market closed
            return None  # No point retrying
        else:
            return MT5_ORDER_RETRY_CONFIG


def retry_mt5_operation(
    func: Callable[..., T],
    *args,
    config: Optional[RetryConfig] = None,
    check_result: Optional[Callable[[Any], bool]] = None,
    **kwargs
) -> T:
    """
    Execute MT5 operation with retry logic
    
    Args:
        func: Function to retry
        config: Retry configuration
        check_result: Function to check if result is successful
        
    Returns:
        Result from successful execution
    """
    if config is None:
        config = MT5_ORDER_RETRY_CONFIG
    
    last_exception = None
    last_result = None
    
    for attempt in range(config.max_attempts):
        try:
            result = func(*args, **kwargs)
            
            # Check if result indicates success
            if check_result and not check_result(result):
                # Check if we should retry based on error code
                if hasattr(result, 'retcode'):
                    if not MT5RetryHelper.should_retry_error(result.retcode):
                        return result  # Non-retryable error
                    
                    # Get specific config for this error
                    error_config = MT5RetryHelper.get_retry_config_for_error(result.retcode)
                    if error_config and attempt == 0:
                        config = error_config  # Switch to error-specific config
                
                if attempt < config.max_attempts - 1:
                    delay = calculate_backoff_delay(attempt, config)
                    logger.warning(
                        f"MT5 operation retry {attempt + 1}/{config.max_attempts} "
                        f"after {delay:.2f}s. Result: {getattr(result, 'comment', 'Unknown error')}"
                    )
                    time.sleep(delay)
                    continue
                else:
                    return result  # Final attempt failed
            
            return result  # Success
            
        except Exception as e:
            last_exception = e
            
            if attempt < config.max_attempts - 1:
                delay = calculate_backoff_delay(attempt, config)
                logger.warning(
                    f"MT5 operation retry {attempt + 1}/{config.max_attempts} "
                    f"after {delay:.2f}s. Error: {str(e)}"
                )
                time.sleep(delay)
            else:
                logger.error(f"MT5 operation failed after {config.max_attempts} attempts")
                raise
    
    if last_exception:
        raise last_exception
    
    return last_result