"""
Validation utilities for trading system data and business rules.
Provides comprehensive validation for signals, trades, and market data.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import re

from core.domain.models import (
    TradingSignal, Trade, MarketData, NewsEvent, SignalType, 
    RiskClass, TradeStatus
)
from core.domain.exceptions import (
    ValidationError, InvalidSignalError, RiskManagementError,
    ErrorContext
)
from core.domain.enums import TradingConstants, SymbolConfig


logger = logging.getLogger(__name__)


class SignalValidator:
    """Validates trading signals for completeness and business rules"""
    
    def __init__(self):
        self.min_risk_reward = TradingConstants.MIN_RISK_REWARD_RATIO
        self.symbol_configs = SymbolConfig.SYMBOL_SPECIFICATIONS
    
    def validate_signal(self, signal: TradingSignal) -> bool:
        """
        Comprehensive validation of a trading signal.
        
        Args:
            signal: TradingSignal to validate
            
        Returns:
            True if valid
            
        Raises:
            InvalidSignalError: If signal is invalid
        """
        with ErrorContext("Signal validation", symbol=signal.symbol) as ctx:
            ctx.add_detail("signal_type", signal.signal.value)
            ctx.add_detail("risk_class", signal.risk_class.value)
            
            # Basic completeness check
            self._validate_signal_completeness(signal)
            
            # Skip further validation for WAIT signals
            if signal.signal == SignalType.WAIT:
                return True
            
            # Price level validation
            self._validate_price_levels(signal)
            
            # Risk/reward validation
            self._validate_risk_reward(signal)
            
            # Symbol-specific validation
            self._validate_symbol_specific(signal)
            
            # Timing validation
            self._validate_timing(signal)
            
            logger.debug(f"Signal validation passed for {signal.symbol}")
            return True
    
    def _validate_signal_completeness(self, signal: TradingSignal):
        """Validate signal has all required fields"""
        if not signal.symbol:
            raise InvalidSignalError("Signal missing symbol")
        
        if not signal.signal:
            raise InvalidSignalError("Signal missing signal type")
        
        if not signal.reason:
            raise InvalidSignalError("Signal missing reason")
        
        if not signal.risk_class:
            raise InvalidSignalError("Signal missing risk class")
        
        # For actionable signals, require price levels
        if signal.is_actionable:
            if signal.entry is None:
                raise InvalidSignalError("Actionable signal missing entry price")
            
            if signal.stop_loss is None:
                raise InvalidSignalError("Actionable signal missing stop loss")
            
            if signal.take_profit is None:
                raise InvalidSignalError("Actionable signal missing take profit")
    
    def _validate_price_levels(self, signal: TradingSignal):
        """Validate price levels are logical"""
        if not signal.is_actionable:
            return
        
        entry = signal.entry
        sl = signal.stop_loss
        tp = signal.take_profit
        
        # Basic price validation
        if entry <= 0 or sl <= 0 or tp <= 0:
            raise InvalidSignalError("Price levels must be positive")
        
        # Directional validation
        if signal.signal == SignalType.BUY:
            if sl >= entry:
                raise InvalidSignalError("BUY signal: stop loss must be below entry")
            if tp <= entry:
                raise InvalidSignalError("BUY signal: take profit must be above entry")
        
        elif signal.signal == SignalType.SELL:
            if sl <= entry:
                raise InvalidSignalError("SELL signal: stop loss must be above entry")
            if tp >= entry:
                raise InvalidSignalError("SELL signal: take profit must be below entry")
        
        # Minimum distance validation
        symbol_config = self.symbol_configs.get(signal.symbol, {})
        min_distance = symbol_config.get('pip_value', 0.0001) * 10  # 10 pips minimum
        
        sl_distance = abs(entry - sl)
        tp_distance = abs(entry - tp)
        
        if sl_distance < min_distance:
            raise InvalidSignalError(f"Stop loss too close to entry (min {min_distance})")
        
        if tp_distance < min_distance:
            raise InvalidSignalError(f"Take profit too close to entry (min {min_distance})")
    
    def _validate_risk_reward(self, signal: TradingSignal):
        """Validate risk/reward ratio"""
        if not signal.is_actionable or not signal.risk_reward:
            return
        
        if signal.risk_reward < self.min_risk_reward:
            raise RiskManagementError(
                f"Risk/reward ratio {signal.risk_reward:.2f} below minimum {self.min_risk_reward}"
            )
        
        # Calculate actual R/R from price levels for verification
        entry = signal.entry
        sl = signal.stop_loss
        tp = signal.take_profit
        
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        
        if risk > 0:
            calculated_rr = reward / risk
            
            # Allow small tolerance for rounding
            tolerance = 0.1
            if abs(calculated_rr - signal.risk_reward) > tolerance:
                logger.warning(
                    f"Calculated R/R {calculated_rr:.2f} differs from stated R/R {signal.risk_reward:.2f}"
                )
    
    def _validate_symbol_specific(self, signal: TradingSignal):
        """Validate symbol-specific requirements"""
        symbol_config = self.symbol_configs.get(signal.symbol)
        
        if not symbol_config:
            logger.warning(f"No configuration found for symbol {signal.symbol}")
            return
        
        # Validate minimum spread requirements if available
        typical_spread = symbol_config.get('typical_spread', 0)
        if typical_spread > 0:
            # This would require current market data to validate
            # For now, just log the requirement
            logger.debug(f"Symbol {signal.symbol} typical spread: {typical_spread}")
    
    def _validate_timing(self, signal: TradingSignal):
        """Validate signal timing"""
        # Check if signal is too old
        age_minutes = (datetime.now(timezone.utc) - signal.timestamp).total_seconds() / 60
        
        if age_minutes > 60:  # 1 hour maximum age
            raise InvalidSignalError(f"Signal too old: {age_minutes:.1f} minutes")
        
        # Check for weekend trading (basic check)
        if signal.timestamp.weekday() >= 5:  # Saturday or Sunday
            logger.warning(f"Signal generated on weekend: {signal.timestamp}")


class TradeValidator:
    """Validates trade data and business rules"""
    
    def validate_trade(self, trade: Trade) -> bool:
        """
        Validate trade object for completeness and business rules.
        
        Args:
            trade: Trade to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If trade is invalid
        """
        with ErrorContext("Trade validation", symbol=trade.symbol) as ctx:
            ctx.add_detail("trade_id", trade.id)
            ctx.add_detail("status", trade.status.value)
            
            # Basic completeness
            self._validate_trade_completeness(trade)
            
            # Price level validation
            self._validate_trade_prices(trade)
            
            # Status validation
            self._validate_trade_status(trade)
            
            # Risk validation
            self._validate_trade_risk(trade)
            
            return True
    
    def _validate_trade_completeness(self, trade: Trade):
        """Validate trade has required fields"""
        if not trade.id:
            raise ValidationError("Trade missing ID")
        
        if not trade.symbol:
            raise ValidationError("Trade missing symbol")
        
        if not trade.side:
            raise ValidationError("Trade missing side")
        
        if trade.entry_price <= 0:
            raise ValidationError("Trade missing or invalid entry price")
        
        if trade.stop_loss <= 0:
            raise ValidationError("Trade missing or invalid stop loss")
        
        if trade.take_profit <= 0:
            raise ValidationError("Trade missing or invalid take profit")
    
    def _validate_trade_prices(self, trade: Trade):
        """Validate trade price levels"""
        entry = trade.entry_price
        sl = trade.stop_loss
        tp = trade.take_profit
        
        # Directional validation
        if trade.side == SignalType.BUY:
            if sl >= entry:
                raise ValidationError("BUY trade: stop loss must be below entry")
            if tp <= entry:
                raise ValidationError("BUY trade: take profit must be above entry")
        
        elif trade.side == SignalType.SELL:
            if sl <= entry:
                raise ValidationError("SELL trade: stop loss must be above entry")
            if tp >= entry:
                raise ValidationError("SELL trade: take profit must be below entry")
        
        # Exit price validation for closed trades
        if trade.exit_price is not None:
            if trade.exit_price <= 0:
                raise ValidationError("Invalid exit price")
    
    def _validate_trade_status(self, trade: Trade):
        """Validate trade status consistency"""
        if trade.status == TradeStatus.CLOSED:
            if trade.exit_price is None:
                raise ValidationError("Closed trade missing exit price")
            
            if trade.exit_timestamp is None:
                raise ValidationError("Closed trade missing exit timestamp")
            
            if trade.result is None:
                raise ValidationError("Closed trade missing result")
        
        elif trade.status == TradeStatus.OPEN:
            if trade.exit_price is not None:
                raise ValidationError("Open trade should not have exit price")
            
            if trade.ticket is None:
                logger.warning(f"Open trade {trade.id} missing ticket number")
    
    def _validate_trade_risk(self, trade: Trade):
        """Validate trade risk parameters"""
        if trade.risk_amount_usd is not None and trade.risk_amount_usd <= 0:
            raise ValidationError("Invalid risk amount")
        
        if trade.risk_reward_ratio is not None:
            if trade.risk_reward_ratio < TradingConstants.MIN_RISK_REWARD_RATIO:
                raise RiskManagementError(
                    f"Trade R/R {trade.risk_reward_ratio} below minimum"
                )


class MarketDataValidator:
    """Validates market data quality and completeness"""
    
    def validate_market_data(self, market_data: MarketData) -> bool:
        """
        Validate market data quality.
        
        Args:
            market_data: MarketData to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If data is invalid
        """
        with ErrorContext("Market data validation", symbol=market_data.symbol):
            # Basic completeness
            if not market_data.symbol:
                raise ValidationError("Market data missing symbol")
            
            if not market_data.timeframe:
                raise ValidationError("Market data missing timeframe")
            
            if not market_data.candles:
                raise ValidationError("Market data has no candles")
            
            # Validate individual candles
            self._validate_candles(market_data.candles)
            
            # Validate data consistency
            self._validate_data_consistency(market_data.candles)
            
            return True
    
    def _validate_candles(self, candles: List):
        """Validate individual candles"""
        for i, candle in enumerate(candles):
            try:
                # OHLC validation
                if candle.high < candle.low:
                    raise ValidationError(f"Candle {i}: high < low")
                
                if candle.open < candle.low or candle.open > candle.high:
                    raise ValidationError(f"Candle {i}: open outside high/low range")
                
                if candle.close < candle.low or candle.close > candle.high:
                    raise ValidationError(f"Candle {i}: close outside high/low range")
                
                # Volume validation
                if candle.volume < 0:
                    raise ValidationError(f"Candle {i}: negative volume")
                
                # Indicator validation (if present)
                if candle.rsi14 is not None:
                    if candle.rsi14 < 0 or candle.rsi14 > 100:
                        raise ValidationError(f"Candle {i}: RSI out of range (0-100)")
                
            except AttributeError as e:
                raise ValidationError(f"Candle {i} missing required attribute: {e}")
    
    def _validate_data_consistency(self, candles: List):
        """Validate data consistency across candles"""
        if len(candles) < 2:
            return
        
        # Check timestamp ordering
        for i in range(1, len(candles)):
            if candles[i].timestamp <= candles[i-1].timestamp:
                logger.warning(f"Candles not in chronological order at index {i}")
        
        # Check for gaps or duplicates
        timestamps = [candle.timestamp for candle in candles]
        unique_timestamps = set(timestamps)
        
        if len(unique_timestamps) != len(timestamps):
            raise ValidationError("Duplicate timestamps in market data")


class NewsValidator:
    """Validates news event data"""
    
    def validate_news_event(self, event: NewsEvent) -> bool:
        """
        Validate news event data.
        
        Args:
            event: NewsEvent to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If event is invalid
        """
        if not event.country:
            raise ValidationError("News event missing country")
        
        if not event.title:
            raise ValidationError("News event missing title")
        
        if not event.impact:
            raise ValidationError("News event missing impact level")
        
        # Validate impact level
        valid_impacts = ['low', 'medium', 'high']
        if event.impact.lower() not in valid_impacts:
            raise ValidationError(f"Invalid impact level: {event.impact}")
        
        # Validate timestamp
        if event.timestamp < datetime.now(timezone.utc):
            logger.warning(f"News event is in the past: {event.title}")
        
        return True


class InputSanitizer:
    """Sanitizes user inputs and external data"""
    
    @staticmethod
    def sanitize_symbol(symbol: str) -> str:
        """
        Sanitize trading symbol input.
        
        Args:
            symbol: Raw symbol string
            
        Returns:
            Sanitized symbol
            
        Raises:
            ValidationError: If symbol is invalid
        """
        return SymbolValidator.validate_and_sanitize(symbol)
    
    @staticmethod
    def sanitize_price(price: Any) -> float:
        """
        Sanitize price input.
        
        Args:
            price: Raw price value
            
        Returns:
            Sanitized price as float
            
        Raises:
            ValidationError: If price is invalid
        """
        try:
            clean_price = float(price)
        except (ValueError, TypeError):
            raise ValidationError(f"Invalid price format: {price}")
        
        if clean_price <= 0:
            raise ValidationError(f"Price must be positive: {price}")
        
        if clean_price > 1e10:  # Arbitrary large number check
            raise ValidationError(f"Price unreasonably large: {price}")
        
        return clean_price
    
    @staticmethod
    def sanitize_json_response(json_str: str) -> Dict[str, Any]:
        """
        Sanitize and validate JSON response from external APIs.
        
        Args:
            json_str: Raw JSON string
            
        Returns:
            Parsed and sanitized dictionary
            
        Raises:
            ValidationError: If JSON is invalid
        """
        if not json_str or not isinstance(json_str, str):
            raise ValidationError("JSON input must be a non-empty string")
        
        try:
            import json
            data = json.loads(json_str.strip())
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON format: {e}")
        
        if not isinstance(data, dict):
            raise ValidationError("JSON must represent an object")
        
        # Basic size limit
        if len(json_str) > 100000:  # 100KB limit
            raise ValidationError("JSON response too large")
        
        return data

class SymbolValidator:
    """Centralized symbol validation"""
    
    VALID_SYMBOL_PATTERN = re.compile(r'^[A-Z0-9._]+$')
    MIN_SYMBOL_LENGTH = 2
    MAX_SYMBOL_LENGTH = 20
    
    @classmethod
    def validate_and_sanitize(cls, symbol: str) -> str:
        """
        Validate and sanitize trading symbol.
        
        Args:
            symbol: Raw symbol string
            
        Returns:
            Sanitized and validated symbol
            
        Raises:
            ValidationError: If symbol is invalid
        """
        if not symbol or not isinstance(symbol, str):
            raise ValidationError("Symbol must be a non-empty string")
        
        # Sanitize: remove whitespace and convert to uppercase
        clean_symbol = symbol.strip().upper()
        
        # Validate format
        if not cls.VALID_SYMBOL_PATTERN.match(clean_symbol):
            raise ValidationError(f"Invalid symbol format: {symbol}")
        
        # Validate length
        if len(clean_symbol) < cls.MIN_SYMBOL_LENGTH:
            raise ValidationError(f"Symbol too short: {symbol}")
        
        if len(clean_symbol) > cls.MAX_SYMBOL_LENGTH:
            raise ValidationError(f"Symbol too long: {symbol}")
        
        return clean_symbol
    
    @classmethod
    def is_valid(cls, symbol: str) -> bool:
        """
        Check if symbol is valid without raising exceptions.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            True if valid, False otherwise
        """
        try:
            cls.validate_and_sanitize(symbol)
            return True
        except ValidationError:
            return False

# Validation helper functions
def validate_risk_parameters(
    risk_amount: float, 
    account_balance: float,
    max_risk_percent: float = TradingConstants.MAX_RISK_PERCENT
) -> bool:
    """
    Validate risk parameters against account limits.
    
    Args:
        risk_amount: Risk amount in account currency
        account_balance: Current account balance
        max_risk_percent: Maximum risk percentage allowed
        
    Returns:
        True if risk parameters are valid
        
    Raises:
        RiskManagementError: If risk parameters exceed limits
    """
    if risk_amount <= 0:
        raise RiskManagementError("Risk amount must be positive")
    
    if account_balance <= 0:
        raise RiskManagementError("Account balance must be positive")
    
    risk_percent = (risk_amount / account_balance) * 100
    
    if risk_percent > max_risk_percent:
        raise RiskManagementError(
            f"Risk {risk_percent:.2f}% exceeds maximum {max_risk_percent}%"
        )
    
    return True


def validate_symbol_format(symbol: str) -> bool:
    """
    Validate trading symbol format.
    
    Args:
        symbol: Symbol to validate
        
    Returns:
        True if format is valid
        
    Raises:
        ValidationError: If symbol format is invalid
    """
    SymbolValidator.validate_and_sanitize(symbol)
    return True


# Export all validators and utilities
__all__ = [
    'SignalValidator', 'TradeValidator', 'MarketDataValidator', 
    'NewsValidator', 'InputSanitizer',
    'validate_risk_parameters', 'validate_symbol_format'
]