"""
MetaTrader 5 client for connection management and basic operations.
Handles MT5 initialization, connection monitoring, and basic API calls.
"""

import time
import logging
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

import MetaTrader5 as mt5

from config.settings import MT5Settings
from core.domain.exceptions import (
    MT5Error, MT5ConnectionError, MT5InitializationError, 
    ErrorContext, ErrorMessages
)
from core.domain.enums import ReturnCode, MT5TimeFrame


logger = logging.getLogger(__name__)


class MT5Client:
    """
    MetaTrader 5 client for managing connections and basic operations.
    Provides a clean interface to MT5 API with proper error handling.
    """
    
    def __init__(self, config: MT5Settings):
        self.config = config
        self._is_initialized = False
        self._connection_attempts = 0
        self._max_connection_attempts = config.max_retries
        
    @property
    def is_initialized(self) -> bool:
        """Check if MT5 is currently initialized"""
        return self._is_initialized and mt5.initialize()
    
    def initialize(self) -> bool:
        """
        Initialize MT5 connection with retry logic.
        
        Returns:
            bool: True if initialization successful, False otherwise
            
        Raises:
            MT5InitializationError: If initialization fails after all retries
        """
        if self._is_initialized and mt5.initialize():
            return True
            
        with ErrorContext("MT5 initialization"):
            for attempt in range(self._max_connection_attempts):
                try:
                    logger.info(f"Initializing MT5 connection (attempt {attempt + 1}/{self._max_connection_attempts})")
                    
                    if mt5.initialize():
                        self._is_initialized = True
                        self._connection_attempts = 0
                        
                        # Log connection info
                        account_info = self.get_account_info()
                        if account_info:
                            logger.info(f"MT5 connected - Account: {account_info.get('login', 'Unknown')}")
                        else:
                            logger.warning("MT5 connected but account info unavailable")
                            
                        return True
                    else:
                        error_code = mt5.last_error()
                        logger.warning(f"MT5 initialization attempt {attempt + 1} failed: {error_code}")
                        
                except Exception as e:
                    logger.error(f"MT5 initialization attempt {attempt + 1} exception: {e}")
                
                if attempt < self._max_connection_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            self._is_initialized = False
            raise MT5InitializationError(ErrorMessages.MT5_CONNECTION_FAILED)
    
    def shutdown(self):
        """Properly shutdown MT5 connection"""
        try:
            if self._is_initialized:
                mt5.shutdown()
                self._is_initialized = False
                logger.info("MT5 connection shutdown")
        except Exception as e:
            logger.error(f"Error during MT5 shutdown: {e}")
    
    def ensure_connection(self) -> bool:
        """
        Ensure MT5 connection is active, reinitialize if needed.
        
        Returns:
            bool: True if connection is active
            
        Raises:
            MT5ConnectionError: If connection cannot be established
        """
        if not self.is_initialized:
            return self.initialize()
        return True
    
    @contextmanager
    def connection_context(self):
        """Context manager for MT5 operations with automatic connection management"""
        try:
            if not self.ensure_connection():
                raise MT5ConnectionError(ErrorMessages.MT5_CONNECTION_FAILED)
            yield self
        except Exception as e:
            logger.error(f"MT5 operation failed: {e}")
            raise
        # Note: We don't shutdown here as other operations might need the connection
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        Get MT5 account information.
        
        Returns:
            Dict with account info or None if failed
        """
        if not self.ensure_connection():
            return None
            
        try:
            account_info = mt5.account_info()
            if account_info:
                return account_info._asdict()
            return None
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return None
    
    def get_terminal_info(self) -> Optional[Dict[str, Any]]:
        """
        Get MT5 terminal information.
        
        Returns:
            Dict with terminal info or None if failed
        """
        if not self.ensure_connection():
            return None
            
        try:
            terminal_info = mt5.terminal_info()
            if terminal_info:
                return terminal_info._asdict()
            return None
        except Exception as e:
            logger.error(f"Failed to get terminal info: {e}")
            return None
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get symbol information from MT5.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with symbol info or None if failed
        """
        if not self.ensure_connection():
            return None
            
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                return symbol_info._asdict()
            else:
                logger.warning(f"Symbol {symbol} not found")
                return None
        except Exception as e:
            logger.error(f"Failed to get symbol info for {symbol}: {e}")
            return None
    
    def get_symbol_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get latest tick for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with tick info or None if failed
        """
        if not self.ensure_connection():
            return None
            
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                return tick._asdict()
            else:
                logger.warning(f"No tick data for {symbol}")
                return None
        except Exception as e:
            logger.error(f"Failed to get tick for {symbol}: {e}")
            return None
    
    def copy_rates(
        self, 
        symbol: str, 
        timeframe: MT5TimeFrame, 
        start_pos: int = 0, 
        count: int = 100
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Copy rates from MT5.
        
        Args:
            symbol: Trading symbol
            timeframe: MT5 timeframe
            start_pos: Start position
            count: Number of bars
            
        Returns:
            List of rate dictionaries or None if failed
        """
        if not self.ensure_connection():
            return None
            
        try:
            # Get the numeric value for MT5
            timeframe_value = timeframe.value if hasattr(timeframe, 'value') else timeframe
            
            rates = mt5.copy_rates_from_pos(symbol, timeframe_value, start_pos, count)
            
            if rates is not None and len(rates) > 0:
                # Convert numpy array to list of dictionaries properly
                result = []
                for rate in rates:
                    rate_dict = {
                        'time': int(rate[0]),
                        'open': float(rate[1]),
                        'high': float(rate[2]),
                        'low': float(rate[3]),
                        'close': float(rate[4]),
                        'tick_volume': int(rate[5]),
                        'spread': int(rate[6]),
                        'real_volume': int(rate[7])
                    }
                    result.append(rate_dict)
                return result
            else:
                logger.warning(f"No rates data for {symbol} {timeframe}")
                return None
        except Exception as e:
            logger.error(f"Failed to copy rates for {symbol}: {e}")
            return None
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get open positions.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of position dictionaries
        """
        if not self.ensure_connection():
            return []
            
        try:
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()
                
            if positions:
                return [pos._asdict() for pos in positions]
            return []
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def get_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get pending orders.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of order dictionaries
        """
        if not self.ensure_connection():
            return []
            
        try:
            if symbol:
                orders = mt5.orders_get(symbol=symbol)
            else:
                orders = mt5.orders_get()
                
            if orders:
                return [order._asdict() for order in orders]
            return []
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []
    
    def send_order(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send trading request to MT5.
        
        Args:
            request: Order request dictionary
            
        Returns:
            Result dictionary with retcode and details
            
        Raises:
            MT5Error: If order fails
        """
        if not self.ensure_connection():
            raise MT5ConnectionError(ErrorMessages.MT5_CONNECTION_FAILED)
        
        with ErrorContext("MT5 order execution", symbol=request.get('symbol')):
            try:
                # Add magic number if not specified
                if 'magic' not in request:
                    request['magic'] = self.config.magic_number
                
                logger.info(f"Sending MT5 order: {request}")
                result = mt5.order_send(request)
                
                if result:
                    result_dict = result._asdict()
                    
                    if result.retcode == ReturnCode.DONE:
                        logger.info(f"Order successful: {result_dict}")
                    else:
                        logger.error(f"Order failed: {result_dict}")
                        
                    return result_dict
                else:
                    error = mt5.last_error()
                    raise MT5Error(f"Order send failed: {error}")
                    
            except Exception as e:
                logger.error(f"Order execution error: {e}")
                raise MT5Error(f"Order execution failed: {str(e)}")
    
    def check_order_result(self, result: Dict[str, Any]) -> bool:
        """
        Check if order result indicates success.
        
        Args:
            result: Order result from send_order
            
        Returns:
            bool: True if order was successful
        """
        return result.get('retcode') == ReturnCode.DONE
    
    def get_last_error(self) -> tuple:
        """
        Get last MT5 error.
        
        Returns:
            Tuple of (error_code, error_description)
        """
        return mt5.last_error()
    
    def get_spread(self, symbol: str) -> Optional[float]:
        """
        Get current spread for symbol in pips.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Spread in pips or None if failed
        """
        tick = self.get_symbol_tick(symbol)
        symbol_info = self.get_symbol_info(symbol)
        
        if not tick or not symbol_info:
            return None
            
        try:
            spread_points = abs(tick['ask'] - tick['bid'])
            point = symbol_info['point']
            
            # Convert to pips (for most pairs, 1 pip = 10 points)
            if 'JPY' in symbol:
                spread_pips = spread_points / point
            else:
                spread_pips = spread_points / point / 10
                
            return round(spread_pips, 1)
        except Exception as e:
            logger.error(f"Failed to calculate spread for {symbol}: {e}")
            return None
    
    def is_market_open(self, symbol: str) -> bool:
        """
        Check if market is open for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            bool: True if market is open
        """
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            return False
            
        return symbol_info.get('trade_mode', 0) in [2, 3, 4]  # Full trading modes
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if exc_type:
            logger.error(f"MT5 context error: {exc_val}")
        # Don't shutdown automatically - let the application manage this
        return False


# Export main class
__all__ = ['MT5Client']