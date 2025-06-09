"""
Custom exceptions for the GPT Trading System.
Provides a clear hierarchy of exceptions for different error types.
"""

from typing import Optional, Dict, Any


class TradingSystemError(Exception):
    """Base exception for all trading system errors"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class ConfigurationError(TradingSystemError):
    """Raised when there are configuration issues"""
    pass


class ValidationError(TradingSystemError):
    """Raised when data validation fails"""
    pass


# MT5 Related Exceptions
class MT5Error(TradingSystemError):
    """Base exception for MT5 related errors"""
    pass


class MT5ConnectionError(MT5Error):
    """Raised when MT5 connection fails"""
    pass


class MT5InitializationError(MT5Error):
    """Raised when MT5 initialization fails"""
    pass


class MT5DataError(MT5Error):
    """Raised when MT5 data retrieval fails"""
    pass


class MT5OrderError(MT5Error):
    """Raised when MT5 order operations fail"""
    
    def __init__(self, message: str, retcode: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        self.retcode = retcode
        if retcode:
            details = details or {}
            details['retcode'] = retcode
        super().__init__(message, details)


class SymbolNotFoundError(MT5Error):
    """Raised when a trading symbol is not found"""
    pass


class InsufficientDataError(MT5Error):
    """Raised when insufficient market data is available"""
    pass


# GPT/AI Related Exceptions
class GPTError(TradingSystemError):
    """Base exception for GPT related errors"""
    pass


class GPTAPIError(GPTError):
    """Raised when GPT API calls fail"""
    
    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        self.status_code = status_code
        if status_code:
            details = details or {}
            details['status_code'] = status_code
        super().__init__(message, details)


class GPTResponseError(GPTError):
    """Raised when GPT response is invalid or cannot be parsed"""
    pass


class SignalGenerationError(GPTError):
    """Raised when signal generation fails"""
    pass


class ReflectionGenerationError(GPTError):
    """Raised when trade reflection generation fails"""
    pass


# Agent Related Exceptions
class AgentError(TradingSystemError):
    """Base exception for agent-related errors"""
    pass


class CouncilDebateError(AgentError):
    """Raised when council debate process fails"""
    pass


class AgentResponseError(AgentError):
    """Raised when agent response is invalid or cannot be parsed"""
    pass


# Trading Logic Exceptions
class TradingError(TradingSystemError):
    """Base exception for trading logic errors"""
    pass


class InvalidSignalError(TradingError):
    """Raised when a trading signal is invalid"""
    pass


class TradeExecutionError(TradingError):
    """Raised when trade execution fails"""
    pass


class TradeManagementError(TradingError):
    """Raised when trade management operations fail"""
    pass


class RiskManagementError(TradingError):
    """Raised when risk management rules are violated"""
    pass


class InsufficientFundsError(TradingError):
    """Raised when there are insufficient funds for trading"""
    pass


class MaxTradesExceededError(TradingError):
    """Raised when maximum number of trades is exceeded"""
    pass


# Data and Database Related Exceptions
class DataError(TradingSystemError):
    """Base exception for data related errors"""
    pass


class DatabaseError(DataError):
    """Raised when database operations fail"""
    pass


class RepositoryError(DataError):
    """Raised when repository operations fail"""
    pass


class MemoryError(DataError):
    """Raised when RAG memory operations fail"""
    pass


class SerializationError(DataError):
    """Raised when data serialization/deserialization fails"""
    pass


# News and Market Data Exceptions
class NewsError(TradingSystemError):
    """Base exception for news related errors"""
    pass


class NewsRestrictionError(NewsError):
    """Raised when trading is restricted due to news events"""
    pass


class NewsDataError(NewsError):
    """Raised when news data is invalid or missing"""
    pass


class MarketDataError(TradingSystemError):
    """Raised when market data operations fail"""
    pass


class ChartGenerationError(TradingSystemError):
    """Raised when chart generation fails"""
    pass


# Service and Infrastructure Exceptions
class ServiceError(TradingSystemError):
    """Base exception for service layer errors"""
    pass

class BacktestingError(ServiceError):
    """Raised when backtesting operations fail"""
    pass


class AuthenticationError(ServiceError):
    """Raised when authentication fails"""
    pass


class InitializationError(ServiceError):
    """Raised when service initialization fails"""
    pass


class ExternalAPIError(ServiceError):
    """Raised when external API calls fail"""
    
    def __init__(self, message: str, api_name: str, status_code: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        self.api_name = api_name
        self.status_code = status_code
        details = details or {}
        details['api_name'] = api_name
        if status_code:
            details['status_code'] = status_code
        super().__init__(message, details)


class ServiceNotAvailableError(ServiceError):
    """Raised when a required service is not available"""
    pass


class TimeoutError(TradingSystemError):
    """Raised when operations timeout"""
    pass


class NetworkError(TradingSystemError):
    """Raised when network operations fail"""
    pass


# Notification Exceptions
class NotificationError(TradingSystemError):
    """Base exception for notification errors"""
    pass


class TelegramError(NotificationError):
    """Raised when Telegram notifications fail"""
    pass


# Utility functions for exception handling
def wrap_mt5_error(func):
    """Decorator to wrap MT5 exceptions"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "MT5" in str(e) or "MetaTrader" in str(e):
                raise MT5Error(f"MT5 operation failed: {str(e)}")
            raise
    return wrapper


def wrap_gpt_error(func):
    """Decorator to wrap GPT API exceptions"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "openai" in str(e).lower() or "gpt" in str(e).lower():
                raise GPTError(f"GPT operation failed: {str(e)}")
            raise
    return wrapper


def handle_database_errors(func):
    """Decorator to handle database exceptions"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "sqlite" in str(e).lower() or "database" in str(e).lower():
                raise DatabaseError(f"Database operation failed: {str(e)}")
            raise
    return wrapper


class ErrorContext:
    """Context manager for error handling with additional details"""
    
    def __init__(self, operation: str, symbol: Optional[str] = None):
        self.operation = operation
        self.symbol = symbol
        self.details = {}
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and issubclass(exc_type, TradingSystemError):
            # Add context to existing trading system errors
            if hasattr(exc_val, 'details'):
                exc_val.details.update({
                    'operation': self.operation,
                    'symbol': self.symbol,
                    **self.details
                })
        elif exc_type:
            # Wrap other exceptions
            details = {
                'operation': self.operation,
                'symbol': self.symbol,
                'original_error': str(exc_val),
                'error_type': exc_type.__name__,
                **self.details
            }
            raise TradingSystemError(
                f"Operation '{self.operation}' failed",
                details
            ) from exc_val
        return False
    
    def add_detail(self, key: str, value: Any):
        """Add additional detail to error context"""
        self.details[key] = value
        return self


# Common error messages
class ErrorMessages:
    """Common error messages used throughout the system"""
    
    # MT5 Errors
    MT5_NOT_INITIALIZED = "MetaTrader 5 is not initialized"
    MT5_CONNECTION_FAILED = "Failed to connect to MetaTrader 5"
    MT5_SYMBOL_NOT_FOUND = "Symbol not found in MetaTrader 5"
    MT5_INSUFFICIENT_DATA = "Insufficient market data from MetaTrader 5"
    MT5_ORDER_FAILED = "Order execution failed in MetaTrader 5"
    
    # GPT Errors
    GPT_API_KEY_MISSING = "OpenAI API key is not configured"
    GPT_RESPONSE_INVALID = "GPT response is invalid or malformed"
    GPT_QUOTA_EXCEEDED = "OpenAI API quota exceeded"
    GPT_TIMEOUT = "GPT API request timed out"
    
    # Trading Errors
    INVALID_SIGNAL = "Trading signal is invalid or incomplete"
    TRADE_EXECUTION_FAILED = "Trade execution failed"
    INSUFFICIENT_FUNDS = "Insufficient funds for trading"
    MAX_TRADES_EXCEEDED = "Maximum number of open trades exceeded"
    RISK_LIMIT_EXCEEDED = "Risk management limits exceeded"
    
    # Data Errors
    DATABASE_CONNECTION_FAILED = "Database connection failed"
    DATA_SERIALIZATION_FAILED = "Data serialization failed"
    MEMORY_OPERATION_FAILED = "RAG memory operation failed"
    
    # News Errors
    NEWS_DATA_OUTDATED = "News data is outdated"
    TRADING_RESTRICTED_BY_NEWS = "Trading restricted due to high-impact news"
    
    # Configuration Errors
    CONFIG_VALIDATION_FAILED = "Configuration validation failed"
    REQUIRED_SETTING_MISSING = "Required configuration setting is missing"


# Export all exceptions and utilities
__all__ = [
    # Base exceptions
    'TradingSystemError', 'ConfigurationError', 'ValidationError',
    
    # MT5 exceptions
    'MT5Error', 'MT5ConnectionError', 'MT5InitializationError', 
    'MT5DataError', 'MT5OrderError', 'SymbolNotFoundError', 'InsufficientDataError',
    
    # GPT exceptions
    'GPTError', 'GPTAPIError', 'GPTResponseError', 'SignalGenerationError', 'ReflectionGenerationError',
    
    # Agent exceptions
    'AgentError', 'CouncilDebateError', 'AgentResponseError',
    
    # Trading exceptions
    'TradingError', 'InvalidSignalError', 'TradeExecutionError', 'TradeManagementError',
    'RiskManagementError', 'InsufficientFundsError', 'MaxTradesExceededError',
    
    # Data exceptions
    'DataError', 'DatabaseError', 'RepositoryError', 'MemoryError', 'SerializationError',
    
    # News and market data exceptions
    'NewsError', 'NewsRestrictionError', 'NewsDataError', 'MarketDataError', 'ChartGenerationError',
    
    # Service exceptions
    'ServiceError', 'ServiceNotAvailableError', 'TimeoutError', 'NetworkError', 'BacktestingError', 'ExternalAPIError',
    'AuthenticationError', 'InitializationError',
    
    # Notification exceptions
    'NotificationError', 'TelegramError',
    
    # Utilities
    'wrap_mt5_error', 'wrap_gpt_error', 'handle_database_errors', 'ErrorContext', 'ErrorMessages'
]