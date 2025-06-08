"""
Comprehensive error handling framework for the trading system.
Provides consistent error handling, logging, and recovery strategies.
"""

import logging
import traceback
import functools
import asyncio
from typing import Optional, Type, Callable, Any, Union, Dict, List
from datetime import datetime, timezone
from contextlib import contextmanager, asynccontextmanager
from enum import Enum

from core.domain.exceptions import (
    TradingSystemError, ErrorContext, ServiceError,
    ConfigurationError, ValidationError
)


logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"          # Log and continue
    MEDIUM = "medium"    # Log, alert, and continue with caution
    HIGH = "high"        # Log, alert, and consider stopping
    CRITICAL = "critical" # Log, alert, and stop immediately


class ErrorCategory(Enum):
    """Error categories for better classification"""
    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    NETWORK = "network"
    DATABASE = "database"
    API_CALL = "api_call"
    TRADING = "trading"
    PARSING = "parsing"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class ErrorHandler:
    """Centralized error handler for consistent error management"""
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.error_history: List[Dict[str, Any]] = []
        self.max_history = 1000
        self.error_callbacks: Dict[ErrorSeverity, List[Callable]] = {
            severity: [] for severity in ErrorSeverity
        }
    
    def handle_error(
        self,
        error: Exception,
        context: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        reraise: bool = True
    ) -> Optional[Any]:
        """
        Handle an error with proper logging and tracking.
        
        Args:
            error: The exception that occurred
            context: Description of what was being done
            severity: Error severity level
            category: Error category for classification
            operation: Specific operation that failed
            details: Additional error details
            reraise: Whether to re-raise the exception
            
        Returns:
            None if reraise=True, otherwise returns based on severity
        """
        # Create error record
        error_record = {
            'timestamp': datetime.now(timezone.utc),
            'context': context,
            'operation': operation or 'unknown',
            'error_type': type(error).__name__,
            'error_message': str(error),
            'severity': severity.value,
            'category': category.value,
            'details': details or {},
            'traceback': traceback.format_exc()
        }
        
        # Track error
        self._track_error(error_record)
        
        # Log error with appropriate level
        self._log_error(error_record)
        
        # Execute callbacks
        self._execute_callbacks(severity, error_record)
        
        # Handle based on severity
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR in {context}: System stability at risk!")
            # Critical errors should always be re-raised
            if isinstance(error, TradingSystemError):
                raise error
            else:
                raise ServiceError(f"Critical error in {context}: {error}") from error
                
        elif reraise:
            # Wrap in appropriate exception type if needed
            if isinstance(error, TradingSystemError):
                raise error
            else:
                raise ServiceError(f"Error in {context}: {error}") from error
        
        # Return None for non-critical errors when not re-raising
        return None
    
    def _track_error(self, error_record: Dict[str, Any]):
        """Track error for statistics"""
        # Update counts
        error_key = f"{error_record['category']}.{error_record['error_type']}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Add to history
        self.error_history.append(error_record)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
    
    def _log_error(self, error_record: Dict[str, Any]):
        """Log error with appropriate level and formatting"""
        message = (
            f"[{error_record['category'].upper()}] "
            f"Error in {error_record['context']}: "
            f"{error_record['error_message']}"
        )
        
        if error_record['operation'] != 'unknown':
            message += f" (Operation: {error_record['operation']})"
        
        # Add details if present
        if error_record['details']:
            details_str = ", ".join(f"{k}={v}" for k, v in error_record['details'].items())
            message += f" | Details: {details_str}"
        
        # Log based on severity
        severity = ErrorSeverity(error_record['severity'])
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(message, exc_info=True)
        elif severity == ErrorSeverity.HIGH:
            logger.error(message, exc_info=True)
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(message)
        else:
            logger.info(message)
    
    def _execute_callbacks(self, severity: ErrorSeverity, error_record: Dict[str, Any]):
        """Execute registered callbacks for error severity"""
        for callback in self.error_callbacks[severity]:
            try:
                callback(error_record)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")
    
    def register_callback(self, severity: ErrorSeverity, callback: Callable):
        """Register a callback for specific error severity"""
        self.error_callbacks[severity].append(callback)
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        return {
            'total_errors': sum(self.error_counts.values()),
            'by_category': self._group_by_category(),
            'by_severity': self._group_by_severity(),
            'top_errors': self._get_top_errors(10),
            'recent_errors': self._get_recent_errors(10)
        }
    
    def _group_by_category(self) -> Dict[str, int]:
        """Group errors by category"""
        result = {}
        for key, count in self.error_counts.items():
            category = key.split('.')[0]
            result[category] = result.get(category, 0) + count
        return result
    
    def _group_by_severity(self) -> Dict[str, int]:
        """Group errors by severity"""
        result = {s.value: 0 for s in ErrorSeverity}
        for record in self.error_history:
            severity = record['severity']
            result[severity] = result.get(severity, 0) + 1
        return result
    
    def _get_top_errors(self, limit: int) -> List[Dict[str, Any]]:
        """Get most frequent errors"""
        sorted_errors = sorted(
            self.error_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [
            {'error': key, 'count': count}
            for key, count in sorted_errors[:limit]
        ]
    
    def _get_recent_errors(self, limit: int) -> List[Dict[str, Any]]:
        """Get recent errors"""
        return [
            {
                'timestamp': r['timestamp'].isoformat(),
                'context': r['context'],
                'error': r['error_message'],
                'severity': r['severity']
            }
            for r in self.error_history[-limit:]
        ]


# Global error handler instance
_error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance"""
    return _error_handler


# Decorators for consistent error handling

def handle_errors(
    context: str,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    default_return: Any = None,
    reraise: bool = True
):
    """
    Decorator for consistent error handling in functions.
    
    Args:
        context: Description of what the function does
        severity: Default error severity
        category: Error category
        default_return: Value to return on error (if not re-raising)
        reraise: Whether to re-raise exceptions
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handler = get_error_handler()
                result = handler.handle_error(
                    error=e,
                    context=context,
                    severity=severity,
                    category=category,
                    operation=func.__name__,
                    reraise=reraise
                )
                return result if result is not None else default_return
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                handler = get_error_handler()
                result = handler.handle_error(
                    error=e,
                    context=context,
                    severity=severity,
                    category=category,
                    operation=func.__name__,
                    reraise=reraise
                )
                return result if result is not None else default_return
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    
    return decorator


@contextmanager
def error_context(
    context: str,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    reraise: bool = True
):
    """
    Context manager for consistent error handling in code blocks.
    
    Usage:
        with error_context("processing trade", severity=ErrorSeverity.HIGH):
            # code that might raise exceptions
    """
    try:
        yield
    except Exception as e:
        handler = get_error_handler()
        handler.handle_error(
            error=e,
            context=context,
            severity=severity,
            category=category,
            reraise=reraise
        )


@asynccontextmanager
async def async_error_context(
    context: str,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    reraise: bool = True
):
    """Async context manager for error handling"""
    try:
        yield
    except Exception as e:
        handler = get_error_handler()
        handler.handle_error(
            error=e,
            context=context,
            severity=severity,
            category=category,
            reraise=reraise
        )


def safe_parse(
    parse_func: Callable,
    data: Any,
    default: Any = None,
    context: str = "parsing data"
) -> Any:
    """
    Safely parse data with error handling.
    
    Args:
        parse_func: Function to parse the data
        data: Data to parse
        default: Default value on error
        context: Context for error messages
        
    Returns:
        Parsed value or default on error
    """
    try:
        return parse_func(data)
    except Exception as e:
        logger.warning(f"Parse error in {context}: {e}")
        return default


def validate_and_handle(
    value: Any,
    validator: Callable[[Any], bool],
    error_message: str,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
) -> Any:
    """
    Validate a value and handle validation errors.
    
    Args:
        value: Value to validate
        validator: Validation function
        error_message: Error message if validation fails
        severity: Error severity
        
    Returns:
        The value if valid
        
    Raises:
        ValidationError if validation fails
    """
    try:
        if not validator(value):
            raise ValidationError(error_message)
        return value
    except Exception as e:
        handler = get_error_handler()
        handler.handle_error(
            error=e,
            context="validation",
            severity=severity,
            category=ErrorCategory.VALIDATION,
            details={'value': str(value)[:100]}
        )


# Specific error handlers for common scenarios

def handle_api_error(
    error: Exception,
    api_name: str,
    operation: str,
    retry_count: int = 0,
    max_retries: int = 3
) -> bool:
    """
    Handle API errors with retry logic.
    
    Returns:
        True if should retry, False otherwise
    """
    handler = get_error_handler()
    
    # Determine if retryable
    retryable = (
        retry_count < max_retries and
        isinstance(error, (ConnectionError, TimeoutError))
    )
    
    severity = ErrorSeverity.MEDIUM if retryable else ErrorSeverity.HIGH
    
    handler.handle_error(
        error=error,
        context=f"{api_name} API call",
        severity=severity,
        category=ErrorCategory.API_CALL,
        operation=operation,
        details={
            'retry_count': retry_count,
            'max_retries': max_retries,
            'retryable': retryable
        },
        reraise=False
    )
    
    return retryable


def handle_database_error(
    error: Exception,
    operation: str,
    table: Optional[str] = None,
    critical: bool = False
) -> None:
    """Handle database errors"""
    handler = get_error_handler()
    
    severity = ErrorSeverity.CRITICAL if critical else ErrorSeverity.HIGH
    
    handler.handle_error(
        error=error,
        context="database operation",
        severity=severity,
        category=ErrorCategory.DATABASE,
        operation=operation,
        details={'table': table} if table else None
    )


def handle_trading_error(
    error: Exception,
    symbol: str,
    operation: str,
    trade_id: Optional[str] = None,
    critical: bool = True
) -> None:
    """Handle trading-related errors"""
    handler = get_error_handler()
    
    severity = ErrorSeverity.CRITICAL if critical else ErrorSeverity.HIGH
    
    details = {'symbol': symbol}
    if trade_id:
        details['trade_id'] = trade_id
    
    handler.handle_error(
        error=error,
        context="trading operation",
        severity=severity,
        category=ErrorCategory.TRADING,
        operation=operation,
        details=details
    )