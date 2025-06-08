"""
Exception handling utilities and decorators for the trading system.
Provides common patterns for handling specific exceptions.
"""

import functools
import logging
import asyncio
from typing import Callable, Type, Tuple, Optional, Any
import sqlite3
import json
import time

from core.domain.exceptions import (
    DatabaseError, RepositoryError, SerializationError,
    MT5ConnectionError, MT5DataError, MT5OrderError,
    GPTAPIError, GPTResponseError, NetworkError,
    ConfigurationError, ValidationError, TradingSystemError
)

logger = logging.getLogger(__name__)


def handle_mt5_exceptions(func: Callable) -> Callable:
    """
    Decorator to handle MT5-specific exceptions with proper error mapping.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ConnectionError as e:
            logger.error(f"MT5 connection error in {func.__name__}: {e}")
            raise MT5ConnectionError(f"MT5 connection failed: {str(e)}") from e
        except TimeoutError as e:
            logger.error(f"MT5 timeout in {func.__name__}: {e}")
            raise MT5ConnectionError(f"MT5 operation timed out: {str(e)}") from e
        except (ValueError, TypeError) as e:
            logger.error(f"MT5 data error in {func.__name__}: {e}")
            raise MT5DataError(f"Invalid MT5 data: {str(e)}") from e
        except Exception as e:
            # Check for MT5-specific error patterns
            error_msg = str(e).lower()
            if any(pattern in error_msg for pattern in ['mt5', 'metatrader', 'terminal']):
                logger.error(f"MT5 error in {func.__name__}: {e}")
                raise MT5ConnectionError(f"MT5 operation failed: {str(e)}") from e
            raise
    
    return wrapper


def handle_mt5_exceptions_async(func: Callable) -> Callable:
    """
    Async version of handle_mt5_exceptions decorator.
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ConnectionError as e:
            logger.error(f"MT5 connection error in {func.__name__}: {e}")
            raise MT5ConnectionError(f"MT5 connection failed: {str(e)}") from e
        except asyncio.TimeoutError as e:
            logger.error(f"MT5 timeout in {func.__name__}: {e}")
            raise MT5ConnectionError(f"MT5 operation timed out: {str(e)}") from e
        except (ValueError, TypeError) as e:
            logger.error(f"MT5 data error in {func.__name__}: {e}")
            raise MT5DataError(f"Invalid MT5 data: {str(e)}") from e
        except Exception as e:
            error_msg = str(e).lower()
            if any(pattern in error_msg for pattern in ['mt5', 'metatrader', 'terminal']):
                logger.error(f"MT5 error in {func.__name__}: {e}")
                raise MT5ConnectionError(f"MT5 operation failed: {str(e)}") from e
            raise
    
    return wrapper


def handle_database_exceptions(func: Callable) -> Callable:
    """
    Decorator to handle database-specific exceptions.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except sqlite3.IntegrityError as e:
            logger.error(f"Database integrity error in {func.__name__}: {e}")
            raise DatabaseError(f"Database integrity violation: {str(e)}") from e
        except sqlite3.OperationalError as e:
            logger.error(f"Database operational error in {func.__name__}: {e}")
            raise DatabaseError(f"Database operation failed: {str(e)}") from e
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {func.__name__}: {e}")
            raise SerializationError(f"Failed to parse JSON data: {str(e)}") from e
        except Exception as e:
            error_msg = str(e).lower()
            if any(pattern in error_msg for pattern in ['sqlite', 'database', 'sql']):
                logger.error(f"Database error in {func.__name__}: {e}")
                raise DatabaseError(f"Database operation failed: {str(e)}") from e
            raise
    
    return wrapper


def handle_gpt_exceptions(func: Callable) -> Callable:
    """
    Decorator to handle GPT/OpenAI API exceptions.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except json.JSONDecodeError as e:
            logger.error(f"GPT response parsing error in {func.__name__}: {e}")
            raise GPTResponseError(f"Failed to parse GPT response: {str(e)}") from e
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"GPT API connection error in {func.__name__}: {e}")
            raise GPTAPIError(f"GPT API connection failed: {str(e)}") from e
        except Exception as e:
            error_msg = str(e).lower()
            if any(pattern in error_msg for pattern in ['openai', 'gpt', 'api key', 'rate limit']):
                logger.error(f"GPT API error in {func.__name__}: {e}")
                # Try to extract status code if available
                status_code = None
                if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                    status_code = e.response.status_code
                raise GPTAPIError(f"GPT API error: {str(e)}", status_code=status_code) from e
            raise
    
    return wrapper


def handle_gpt_exceptions_async(func: Callable) -> Callable:
    """
    Async version of handle_gpt_exceptions decorator.
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except json.JSONDecodeError as e:
            logger.error(f"GPT response parsing error in {func.__name__}: {e}")
            raise GPTResponseError(f"Failed to parse GPT response: {str(e)}") from e
        except (ConnectionError, asyncio.TimeoutError) as e:
            logger.error(f"GPT API connection error in {func.__name__}: {e}")
            raise GPTAPIError(f"GPT API connection failed: {str(e)}") from e
        except Exception as e:
            error_msg = str(e).lower()
            if any(pattern in error_msg for pattern in ['openai', 'gpt', 'api key', 'rate limit']):
                logger.error(f"GPT API error in {func.__name__}: {e}")
                status_code = None
                if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                    status_code = e.response.status_code
                raise GPTAPIError(f"GPT API error: {str(e)}", status_code=status_code) from e
            raise
    
    return wrapper


def retry_on_exceptions(
    exceptions: Tuple[Type[Exception], ...] = (NetworkError, ConnectionError),
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0
) -> Callable:
    """
    Decorator to retry function on specific exceptions with exponential backoff.
    
    Args:
        exceptions: Tuple of exception types to retry on
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for each retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts")
            
            # Re-raise the last exception if all attempts failed
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


def retry_on_exceptions_async(
    exceptions: Tuple[Type[Exception], ...] = (NetworkError, ConnectionError),
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0
) -> Callable:
    """
    Async version of retry_on_exceptions decorator.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts")
            
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    default_value: Any = None,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    log_error: bool = True
) -> Any:
    """
    Execute a function safely, returning a default value on specified exceptions.
    
    Args:
        func: Function to execute
        default_value: Value to return if exception occurs
        exceptions: Tuple of exception types to catch
        log_error: Whether to log the error
        
    Returns:
        Function result or default_value on exception
    """
    try:
        return func()
    except exceptions as e:
        if log_error:
            logger.error(f"Safe execution failed for {func.__name__}: {e}")
        return default_value


async def safe_execute_async(
    func: Callable,
    default_value: Any = None,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    log_error: bool = True
) -> Any:
    """
    Async version of safe_execute.
    """
    try:
        return await func()
    except exceptions as e:
        if log_error:
            logger.error(f"Safe async execution failed for {func.__name__}: {e}")
        return default_value


class ExceptionAggregator:
    """
    Aggregates multiple exceptions that occur during parallel operations.
    Useful for gathering errors from multiple async tasks.
    """
    
    def __init__(self):
        self.exceptions: List[Tuple[str, Exception]] = []
    
    def add(self, context: str, exception: Exception):
        """Add an exception with context"""
        self.exceptions.append((context, exception))
        logger.error(f"Exception in {context}: {exception}")
    
    def has_errors(self) -> bool:
        """Check if any exceptions were collected"""
        return len(self.exceptions) > 0
    
    def raise_first(self):
        """Raise the first exception if any were collected"""
        if self.exceptions:
            _, first_exception = self.exceptions[0]
            raise first_exception
    
    def get_summary(self) -> str:
        """Get a summary of all collected exceptions"""
        if not self.exceptions:
            return "No exceptions"
        
        summary_parts = []
        for context, exc in self.exceptions:
            summary_parts.append(f"{context}: {type(exc).__name__} - {str(exc)}")
        
        return "; ".join(summary_parts)
    
    def create_aggregate_exception(self) -> TradingSystemError:
        """Create an aggregate exception with all collected errors"""
        if not self.exceptions:
            return None
        
        details = {
            'error_count': len(self.exceptions),
            'errors': [
                {'context': ctx, 'type': type(exc).__name__, 'message': str(exc)}
                for ctx, exc in self.exceptions
            ]
        }
        
        return TradingSystemError(
            f"Multiple errors occurred ({len(self.exceptions)} total)",
            details=details
        )


# Export all utilities
__all__ = [
    'handle_mt5_exceptions',
    'handle_mt5_exceptions_async',
    'handle_database_exceptions',
    'handle_gpt_exceptions',
    'handle_gpt_exceptions_async',
    'retry_on_exceptions',
    'retry_on_exceptions_async',
    'safe_execute',
    'safe_execute_async',
    'ExceptionAggregator'
]