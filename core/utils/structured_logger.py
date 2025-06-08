"""
Structured logging framework for the trading system.
Provides consistent, performant, and secure logging with context awareness.
"""

import logging
import json
import time
import asyncio
import functools
from typing import Any, Dict, Optional, Union, Callable, List
from datetime import datetime, timezone
from contextvars import ContextVar
from contextlib import contextmanager
from pathlib import Path
import sys
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
from queue import Queue
import threading

# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
trading_context_var: ContextVar[Optional[Dict[str, Any]]] = ContextVar('trading_context', default={})


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter with context awareness"""
    
    def __init__(self, include_context: bool = True, pretty_print: bool = False):
        super().__init__()
        self.include_context = include_context
        self.pretty_print = pretty_print
        self.hostname = self._get_hostname()
    
    def _get_hostname(self) -> str:
        """Get hostname for log records"""
        import socket
        try:
            return socket.gethostname()
        except:
            return "unknown"
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        # Base log structure
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'hostname': self.hostname,
        }
        
        # Add context if available
        if self.include_context:
            request_id = request_id_var.get()
            if request_id:
                log_data['request_id'] = request_id
            
            trading_context = trading_context_var.get()
            if trading_context:
                log_data['context'] = trading_context
        
        # Add extra fields
        if hasattr(record, '__dict__'):
            extras = {k: v for k, v in record.__dict__.items()
                     if k not in ('name', 'msg', 'args', 'created', 'filename',
                                'funcName', 'levelname', 'levelno', 'lineno',
                                'module', 'msecs', 'message', 'pathname', 'process',
                                'processName', 'relativeCreated', 'stack_info',
                                'thread', 'threadName', 'exc_info', 'exc_text')}
            if extras:
                log_data['extra'] = extras
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add code location for errors and warnings
        if record.levelno >= logging.WARNING:
            log_data['location'] = {
                'file': record.pathname,
                'line': record.lineno,
                'function': record.funcName
            }
        
        # Format as JSON
        if self.pretty_print:
            return json.dumps(log_data, indent=2, default=str)
        else:
            return json.dumps(log_data, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable console formatter with colors and emojis"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    # Level emojis
    EMOJIS = {
        'DEBUG': '🔍',
        'INFO': '📌',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🚨'
    }
    
    def __init__(self, use_colors: bool = True, use_emojis: bool = True):
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()
        self.use_emojis = use_emojis
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record for console output"""
        # Time
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        
        # Level
        level = record.levelname
        if self.use_emojis and level in self.EMOJIS:
            level_str = f"{self.EMOJIS[level]} {level}"
        else:
            level_str = f"[{level}]"
        
        # Color
        if self.use_colors and level in self.COLORS:
            level_str = f"{self.COLORS[level]}{level_str}{self.RESET}"
        
        # Logger name (shortened)
        logger_name = record.name
        if '.' in logger_name:
            parts = logger_name.split('.')
            logger_name = '.'.join(p[0] for p in parts[:-1]) + '.' + parts[-1]
        
        # Message
        message = record.getMessage()
        
        # Add context if available
        context_parts = []
        request_id = request_id_var.get()
        if request_id:
            context_parts.append(f"req:{request_id[:8]}")
        
        trading_context = trading_context_var.get()
        if trading_context and 'symbol' in trading_context:
            context_parts.append(f"sym:{trading_context['symbol']}")
        
        context_str = f"[{' '.join(context_parts)}] " if context_parts else ""
        
        # Build final message
        formatted = f"{timestamp} {level_str:<12} {logger_name:<20} {context_str}{message}"
        
        # Add exception if present
        if record.exc_info:
            formatted += '\n' + self.formatException(record.exc_info)
        
        return formatted


class PerformanceFilter(logging.Filter):
    """Filter to prevent excessive logging in tight loops"""
    
    def __init__(self, rate_limit: float = 1.0):
        """
        Args:
            rate_limit: Minimum seconds between identical log messages
        """
        super().__init__()
        self.rate_limit = rate_limit
        self.last_log_times: Dict[str, float] = {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter out rapid duplicate messages"""
        # Create key from logger name and message
        key = f"{record.name}:{record.msg}"
        
        current_time = time.time()
        last_time = self.last_log_times.get(key, 0)
        
        if current_time - last_time < self.rate_limit:
            return False
        
        self.last_log_times[key] = current_time
        
        # Clean old entries periodically
        if len(self.last_log_times) > 1000:
            cutoff = current_time - 300  # 5 minutes
            self.last_log_times = {
                k: v for k, v in self.last_log_times.items()
                if v > cutoff
            }
        
        return True


class SensitiveDataFilter(logging.Filter):
    """Filter to redact sensitive data from logs"""
    
    # Patterns to redact
    PATTERNS = [
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
        (r'\b(?:\d{4}[-\s]?){3}\d{4}\b', '[CARD]'),
        (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),
        (r'(api[_-]?key|token|password|secret)["\']?\s*[:=]\s*["\']?([^"\'\s]+)',
         r'\1=[REDACTED]'),
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Redact sensitive data from log messages"""
        import re
        
        message = record.getMessage()
        for pattern, replacement in self.PATTERNS:
            message = re.sub(pattern, replacement, message, flags=re.IGNORECASE)
        
        record.msg = message
        record.args = ()
        
        return True


class AsyncRotatingFileHandler(RotatingFileHandler):
    """Async-friendly rotating file handler using queue"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue: Queue = Queue()
        self.queue_handler = QueueHandler(self.queue)
        self.listener = QueueListener(self.queue, self)
        self.listener.start()
    
    def emit(self, record):
        """Queue the record for async processing"""
        self.queue_handler.emit(record)
    
    def close(self):
        """Stop the queue listener and close handler"""
        self.listener.stop()
        super().close()


class TradingLogger:
    """Enhanced logger for trading operations with context management"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._context_stack: List[Dict[str, Any]] = []
    
    @contextmanager
    def context(self, **kwargs):
        """Add context to all logs within this block"""
        current_context = trading_context_var.get() or {}
        new_context = {**current_context, **kwargs}
        token = trading_context_var.set(new_context)
        
        try:
            yield self
        finally:
            trading_context_var.reset(token)
    
    @contextmanager
    def request_context(self, request_id: str):
        """Set request ID for all logs within this block"""
        token = request_id_var.set(request_id)
        try:
            yield self
        finally:
            request_id_var.reset(token)
    
    def _log_with_context(self, level: int, msg: str, *args, **kwargs):
        """Log with current context"""
        extra = kwargs.pop('extra', {})
        
        # Add performance metrics if available
        if 'duration' in kwargs:
            extra['duration_ms'] = int(kwargs.pop('duration') * 1000)
        
        if 'symbol' in kwargs:
            extra['symbol'] = kwargs.pop('symbol')
        
        if 'error_type' in kwargs:
            extra['error_type'] = kwargs.pop('error_type')
        
        self.logger.log(level, msg, *args, extra=extra, **kwargs)
    
    # Convenience methods
    def debug(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.CRITICAL, msg, *args, **kwargs)
    
    def trade_signal(self, signal: Dict[str, Any], agents: List[str] = None):
        """Log trade signal generation"""
        self.info(
            f"Signal generated for {signal['symbol']}: {signal['type']}",
            extra={
                'signal': signal,
                'agent_count': len(agents) if agents else 0,
                'event_type': 'trade_signal'
            }
        )
    
    def trade_execution(self, trade: Dict[str, Any], success: bool):
        """Log trade execution"""
        level = logging.INFO if success else logging.ERROR
        self._log_with_context(
            level,
            f"Trade {'executed' if success else 'failed'}: {trade.get('id', 'unknown')}",
            extra={
                'trade': trade,
                'event_type': 'trade_execution',
                'success': success
            }
        )
    
    def performance_metric(self, operation: str, duration: float, success: bool = True):
        """Log performance metrics"""
        level = logging.WARNING if duration > 5.0 else logging.DEBUG
        self._log_with_context(
            level,
            f"{operation} completed in {duration:.2f}s",
            duration=duration,
            extra={
                'operation': operation,
                'success': success,
                'event_type': 'performance'
            }
        )


def setup_logging(
    settings: Dict[str, Any],
    log_dir: Path = None,
    structured_logs: bool = True,
    async_handler: bool = True
) -> None:
    """
    Setup comprehensive logging for the trading system.
    
    Args:
        settings: Configuration settings
        log_dir: Directory for log files
        structured_logs: Use structured JSON logging for files
        async_handler: Use async file handler
    """
    # Create log directory
    if log_dir is None:
        log_dir = Path(settings.get('paths', {}).get('logs_dir', 'logs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler with human-readable format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.get('log_level', 'INFO')))
    console_handler.setFormatter(ConsoleFormatter())
    console_handler.addFilter(PerformanceFilter(rate_limit=1.0))
    root_logger.addHandler(console_handler)
    
    # File handler with structured format
    if structured_logs:
        # Main log file
        if async_handler:
            file_handler = AsyncRotatingFileHandler(
                log_dir / 'trading_system.json',
                maxBytes=50*1024*1024,  # 50MB
                backupCount=10,
                encoding='utf-8'
            )
        else:
            file_handler = RotatingFileHandler(
                log_dir / 'trading_system.json',
                maxBytes=50*1024*1024,  # 50MB
                backupCount=10,
                encoding='utf-8'
            )
        
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(StructuredFormatter())
        file_handler.addFilter(SensitiveDataFilter())
        root_logger.addHandler(file_handler)
        
        # Error log file
        error_handler = RotatingFileHandler(
            log_dir / 'errors.json',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter(pretty_print=True))
        root_logger.addHandler(error_handler)
    
    # Configure third-party loggers
    if settings.get('production_mode', False):
        # Suppress noisy loggers in production
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('asyncio').setLevel(logging.WARNING)
        logging.getLogger('aiohttp').setLevel(logging.WARNING)
        logging.getLogger('openai').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('httpcore').setLevel(logging.WARNING)
    
    # Trading-specific loggers with appropriate levels
    logging.getLogger('core.services').setLevel(logging.INFO)
    logging.getLogger('core.agents').setLevel(logging.INFO)
    logging.getLogger('core.infrastructure').setLevel(logging.INFO)
    
    # Performance-critical loggers
    logging.getLogger('core.infrastructure.mt5').setLevel(logging.INFO)
    logging.getLogger('core.services.market_service').setLevel(logging.INFO)


# Performance logging decorator
def log_performance(
    logger: Union[logging.Logger, TradingLogger] = None,
    operation: str = None,
    threshold: float = 1.0
):
    """
    Decorator to log function performance.
    
    Args:
        logger: Logger instance (uses function's module logger if None)
        operation: Operation name (uses function name if None)
        threshold: Only log if duration exceeds this threshold
    """
    def decorator(func):
        nonlocal logger, operation
        
        if logger is None:
            logger = logging.getLogger(func.__module__)
        
        if operation is None:
            operation = func.__name__
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                if duration > threshold:
                    if isinstance(logger, TradingLogger):
                        logger.performance_metric(operation, duration, success=True)
                    else:
                        logger.warning(f"{operation} took {duration:.2f}s", extra={
                            'duration_ms': int(duration * 1000),
                            'operation': operation
                        })
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                if isinstance(logger, TradingLogger):
                    logger.performance_metric(operation, duration, success=False)
                else:
                    logger.error(f"{operation} failed after {duration:.2f}s: {e}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                if duration > threshold:
                    if isinstance(logger, TradingLogger):
                        logger.performance_metric(operation, duration, success=True)
                    else:
                        logger.warning(f"{operation} took {duration:.2f}s", extra={
                            'duration_ms': int(duration * 1000),
                            'operation': operation
                        })
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                if isinstance(logger, TradingLogger):
                    logger.performance_metric(operation, duration, success=False)
                else:
                    logger.error(f"{operation} failed after {duration:.2f}s: {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Create specialized loggers
def get_logger(name: str) -> TradingLogger:
    """Get a trading logger instance"""
    return TradingLogger(name)