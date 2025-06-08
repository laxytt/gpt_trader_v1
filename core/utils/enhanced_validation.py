"""
Enhanced validation framework for comprehensive input and data validation.
Provides decorators, validators, and utilities for consistent validation.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Callable, Union, Type
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import functools
from dataclasses import dataclass

from core.domain.exceptions import ValidationError
from core.utils.error_handler import ErrorSeverity, ErrorCategory, get_error_handler


logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """Defines a validation rule"""
    name: str
    validator: Callable[[Any], bool]
    error_message: str
    severity: ErrorSeverity = ErrorSeverity.MEDIUM


class ValidationContext:
    """Context for validation with detailed error tracking"""
    
    def __init__(self, context_name: str):
        self.context_name = context_name
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
    
    def add_error(self, field: str, message: str, value: Any = None):
        """Add a validation error"""
        self.errors.append({
            'field': field,
            'message': message,
            'value': str(value)[:100] if value is not None else None
        })
    
    def add_warning(self, field: str, message: str, value: Any = None):
        """Add a validation warning"""
        self.warnings.append({
            'field': field,
            'message': message,
            'value': str(value)[:100] if value is not None else None
        })
    
    def is_valid(self) -> bool:
        """Check if validation passed"""
        return len(self.errors) == 0
    
    def raise_if_invalid(self):
        """Raise ValidationError if validation failed"""
        if not self.is_valid():
            error_details = "; ".join(
                f"{e['field']}: {e['message']}" for e in self.errors
            )
            raise ValidationError(
                f"Validation failed in {self.context_name}: {error_details}"
            )


class RangeValidator:
    """Validates numeric ranges"""
    
    @staticmethod
    def validate_range(
        value: Union[int, float, Decimal],
        min_value: Optional[Union[int, float, Decimal]] = None,
        max_value: Optional[Union[int, float, Decimal]] = None,
        field_name: str = "value"
    ) -> Union[int, float, Decimal]:
        """Validate a numeric value is within range"""
        if min_value is not None and value < min_value:
            raise ValidationError(
                f"{field_name} must be >= {min_value}, got {value}"
            )
        if max_value is not None and value > max_value:
            raise ValidationError(
                f"{field_name} must be <= {max_value}, got {value}"
            )
        return value
    
    @staticmethod
    def validate_positive(
        value: Union[int, float, Decimal],
        field_name: str = "value",
        allow_zero: bool = False
    ) -> Union[int, float, Decimal]:
        """Validate a value is positive"""
        min_val = 0 if allow_zero else 0.0000001
        if value < min_val:
            raise ValidationError(
                f"{field_name} must be {'non-negative' if allow_zero else 'positive'}, got {value}"
            )
        return value
    
    @staticmethod
    def validate_percentage(
        value: Union[int, float],
        field_name: str = "percentage"
    ) -> float:
        """Validate a percentage value (0-100)"""
        return RangeValidator.validate_range(
            float(value), 0.0, 100.0, field_name
        )


class StringValidator:
    """Validates string inputs"""
    
    # Regex patterns
    SYMBOL_PATTERN = re.compile(r'^[A-Z]{6}$')
    ALPHANUMERIC_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
    
    @staticmethod
    def validate_not_empty(
        value: str,
        field_name: str = "value"
    ) -> str:
        """Validate string is not empty"""
        if not value or not value.strip():
            raise ValidationError(f"{field_name} cannot be empty")
        return value.strip()
    
    @staticmethod
    def validate_length(
        value: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        field_name: str = "value"
    ) -> str:
        """Validate string length"""
        if min_length is not None and len(value) < min_length:
            raise ValidationError(
                f"{field_name} must be at least {min_length} characters, got {len(value)}"
            )
        if max_length is not None and len(value) > max_length:
            raise ValidationError(
                f"{field_name} must be at most {max_length} characters, got {len(value)}"
            )
        return value
    
    @staticmethod
    def validate_pattern(
        value: str,
        pattern: re.Pattern,
        field_name: str = "value",
        pattern_description: str = "pattern"
    ) -> str:
        """Validate string matches pattern"""
        if not pattern.match(value):
            raise ValidationError(
                f"{field_name} must match {pattern_description}, got '{value}'"
            )
        return value
    
    @staticmethod
    def validate_forex_symbol(symbol: str) -> str:
        """Validate forex symbol format"""
        symbol = symbol.upper().strip()
        if not StringValidator.SYMBOL_PATTERN.match(symbol):
            raise ValidationError(
                f"Invalid forex symbol format: {symbol}. Expected 6 uppercase letters (e.g., EURUSD)"
            )
        return symbol
    
    @staticmethod
    def validate_enum_value(
        value: str,
        allowed_values: List[str],
        field_name: str = "value"
    ) -> str:
        """Validate string is in allowed values"""
        if value not in allowed_values:
            raise ValidationError(
                f"{field_name} must be one of {allowed_values}, got '{value}'"
            )
        return value


class DateTimeValidator:
    """Validates datetime inputs"""
    
    @staticmethod
    def validate_datetime_range(
        start: datetime,
        end: datetime,
        max_range_days: Optional[int] = None,
        field_name: str = "date range"
    ) -> tuple[datetime, datetime]:
        """Validate datetime range"""
        if start >= end:
            raise ValidationError(
                f"{field_name}: start time must be before end time"
            )
        
        if max_range_days is not None:
            range_days = (end - start).days
            if range_days > max_range_days:
                raise ValidationError(
                    f"{field_name}: range cannot exceed {max_range_days} days, got {range_days}"
                )
        
        return start, end
    
    @staticmethod
    def validate_not_future(
        dt: datetime,
        field_name: str = "datetime"
    ) -> datetime:
        """Validate datetime is not in the future"""
        now = datetime.now(timezone.utc)
        if dt > now:
            raise ValidationError(
                f"{field_name} cannot be in the future"
            )
        return dt


class TradingValidator:
    """Trading-specific validations"""
    
    @staticmethod
    def validate_lot_size(
        lot_size: float,
        min_lot: float = 0.01,
        max_lot: float = 100.0,
        step: float = 0.01
    ) -> float:
        """Validate lot size"""
        # Check range
        RangeValidator.validate_range(
            lot_size, min_lot, max_lot, "lot size"
        )
        
        # Check step
        steps = round((lot_size - min_lot) / step)
        adjusted = min_lot + (steps * step)
        
        if abs(adjusted - lot_size) > 0.00001:
            raise ValidationError(
                f"Lot size must be in steps of {step}, got {lot_size}"
            )
        
        return round(adjusted, 2)
    
    @staticmethod
    def validate_price(
        price: float,
        symbol: str,
        price_type: str = "price"
    ) -> float:
        """Validate price for a symbol"""
        if price <= 0:
            raise ValidationError(
                f"{price_type} must be positive, got {price}"
            )
        
        # Symbol-specific validation
        if symbol.endswith("JPY"):
            # JPY pairs typically 2-3 decimal places
            if price < 1.0 or price > 1000.0:
                logger.warning(
                    f"Unusual {price_type} for {symbol}: {price}"
                )
        else:
            # Most pairs 4-5 decimal places
            if price < 0.0001 or price > 100.0:
                logger.warning(
                    f"Unusual {price_type} for {symbol}: {price}"
                )
        
        return price
    
    @staticmethod
    def validate_stop_loss(
        entry_price: float,
        stop_loss: float,
        is_buy: bool,
        min_distance_pips: float = 5.0,
        symbol: str = "EURUSD"
    ) -> float:
        """Validate stop loss placement"""
        pip_size = 0.01 if symbol.endswith("JPY") else 0.0001
        
        if is_buy:
            if stop_loss >= entry_price:
                raise ValidationError(
                    f"Buy stop loss ({stop_loss}) must be below entry ({entry_price})"
                )
            distance_pips = (entry_price - stop_loss) / pip_size
        else:
            if stop_loss <= entry_price:
                raise ValidationError(
                    f"Sell stop loss ({stop_loss}) must be above entry ({entry_price})"
                )
            distance_pips = (stop_loss - entry_price) / pip_size
        
        if distance_pips < min_distance_pips:
            raise ValidationError(
                f"Stop loss too close: {distance_pips:.1f} pips (minimum {min_distance_pips})"
            )
        
        return stop_loss
    
    @staticmethod
    def validate_risk_reward_ratio(
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        is_buy: bool,
        min_ratio: float = 1.0
    ) -> float:
        """Validate risk/reward ratio"""
        if is_buy:
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
        
        if risk <= 0:
            raise ValidationError("Invalid risk calculation")
        
        ratio = reward / risk
        
        if ratio < min_ratio:
            raise ValidationError(
                f"Risk/reward ratio ({ratio:.2f}) below minimum ({min_ratio})"
            )
        
        return ratio


class DataValidator:
    """Validates data structures"""
    
    @staticmethod
    def validate_required_fields(
        data: Dict[str, Any],
        required_fields: List[str],
        context: str = "data"
    ) -> Dict[str, Any]:
        """Validate required fields exist"""
        missing = [f for f in required_fields if f not in data or data[f] is None]
        if missing:
            raise ValidationError(
                f"{context} missing required fields: {missing}"
            )
        return data
    
    @staticmethod
    def validate_not_none(
        value: Any,
        field_name: str = "value"
    ) -> Any:
        """Validate value is not None"""
        if value is None:
            raise ValidationError(f"{field_name} cannot be None")
        return value
    
    @staticmethod
    def validate_list_not_empty(
        lst: List[Any],
        field_name: str = "list"
    ) -> List[Any]:
        """Validate list is not empty"""
        if not lst:
            raise ValidationError(f"{field_name} cannot be empty")
        return lst


# Validation decorators

def validate_params(**validators):
    """
    Decorator to validate function parameters.
    
    Usage:
        @validate_params(
            symbol=StringValidator.validate_forex_symbol,
            lot_size=lambda x: RangeValidator.validate_range(x, 0.01, 10.0)
        )
        def place_order(symbol: str, lot_size: float):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in bound.arguments:
                    try:
                        bound.arguments[param_name] = validator(
                            bound.arguments[param_name]
                        )
                    except ValidationError as e:
                        error_handler = get_error_handler()
                        error_handler.handle_error(
                            error=e,
                            context=f"validating {func.__name__}",
                            severity=ErrorSeverity.MEDIUM,
                            category=ErrorCategory.VALIDATION,
                            operation=func.__name__,
                            details={'parameter': param_name}
                        )
            
            return func(*bound.args, **bound.kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Same validation for async functions
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            for param_name, validator in validators.items():
                if param_name in bound.arguments:
                    try:
                        bound.arguments[param_name] = validator(
                            bound.arguments[param_name]
                        )
                    except ValidationError as e:
                        error_handler = get_error_handler()
                        error_handler.handle_error(
                            error=e,
                            context=f"validating {func.__name__}",
                            severity=ErrorSeverity.MEDIUM,
                            category=ErrorCategory.VALIDATION,
                            operation=func.__name__,
                            details={'parameter': param_name}
                        )
            
            return await func(*bound.args, **bound.kwargs)
        
        import asyncio
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    
    return decorator


def validate_output(validator: Callable[[Any], Any]):
    """
    Decorator to validate function output.
    
    Usage:
        @validate_output(lambda x: RangeValidator.validate_positive(x))
        def calculate_position_size():
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            try:
                return validator(result)
            except ValidationError as e:
                error_handler = get_error_handler()
                error_handler.handle_error(
                    error=e,
                    context=f"validating output of {func.__name__}",
                    severity=ErrorSeverity.MEDIUM,
                    category=ErrorCategory.VALIDATION,
                    operation=func.__name__
                )
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            try:
                return validator(result)
            except ValidationError as e:
                error_handler = get_error_handler()
                error_handler.handle_error(
                    error=e,
                    context=f"validating output of {func.__name__}",
                    severity=ErrorSeverity.MEDIUM,
                    category=ErrorCategory.VALIDATION,
                    operation=func.__name__
                )
        
        import asyncio
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    
    return decorator


# Composite validators for complex validation scenarios

class TradingSignalValidator:
    """Comprehensive trading signal validation"""
    
    @staticmethod
    def validate_signal(signal: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a complete trading signal"""
        ctx = ValidationContext("trading signal")
        
        # Required fields
        required = ['symbol', 'signal_type', 'entry_price', 'stop_loss', 'take_profit']
        try:
            DataValidator.validate_required_fields(signal, required)
        except ValidationError as e:
            ctx.add_error('structure', str(e))
            ctx.raise_if_invalid()
        
        # Symbol
        try:
            signal['symbol'] = StringValidator.validate_forex_symbol(signal['symbol'])
        except ValidationError as e:
            ctx.add_error('symbol', str(e))
        
        # Signal type
        try:
            signal['signal_type'] = StringValidator.validate_enum_value(
                signal['signal_type'],
                ['BUY', 'SELL', 'WAIT'],
                'signal_type'
            )
        except ValidationError as e:
            ctx.add_error('signal_type', str(e))
        
        # Prices
        if signal['signal_type'] != 'WAIT':
            try:
                signal['entry_price'] = TradingValidator.validate_price(
                    signal['entry_price'],
                    signal['symbol'],
                    'entry price'
                )
                
                signal['stop_loss'] = TradingValidator.validate_stop_loss(
                    signal['entry_price'],
                    signal['stop_loss'],
                    signal['signal_type'] == 'BUY',
                    symbol=signal['symbol']
                )
                
                signal['take_profit'] = TradingValidator.validate_price(
                    signal['take_profit'],
                    signal['symbol'],
                    'take profit'
                )
                
                # Validate R:R ratio
                TradingValidator.validate_risk_reward_ratio(
                    signal['entry_price'],
                    signal['stop_loss'],
                    signal['take_profit'],
                    signal['signal_type'] == 'BUY'
                )
            except ValidationError as e:
                ctx.add_error('prices', str(e))
        
        ctx.raise_if_invalid()
        return signal