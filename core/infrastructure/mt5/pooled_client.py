"""
MT5 client using connection pooling for improved reliability and performance.
This client uses the MT5 connection pool instead of managing its own connection.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import MetaTrader5 as mt5

from core.domain.exceptions import (
    MT5Error, MT5ConnectionError, MT5DataError, SymbolNotFoundError,
    InsufficientDataError, ErrorMessages
)
from core.infrastructure.mt5.connection_pool import (
    MT5ConnectionPool, MT5ConnectionConfig, get_mt5_connection_pool
)
from core.utils.enhanced_validation import (
    validate_params, StringValidator, NumericValidator
)
from config.settings import MT5Settings

logger = logging.getLogger(__name__)


class PooledMT5Client:
    """
    MT5 client that uses connection pooling for all operations.
    Provides the same interface as MT5Client but with better reliability.
    """
    
    def __init__(self, settings: MT5Settings, pool_config: Optional[MT5ConnectionConfig] = None):
        """
        Initialize MT5 client with connection pooling.
        
        Args:
            settings: MT5 configuration settings
            pool_config: Optional custom pool configuration
        """
        self.settings = settings
        
        # Create pool configuration from settings
        if pool_config is None:
            pool_config = MT5ConnectionConfig(
                login=settings.login,
                password=settings.password,
                server=settings.server,
                terminal_path=settings.path,
                max_reconnect_attempts=5,
                reconnect_delay=2.0,
                health_check_interval=30.0,
                enable_pooling=True,
                pool_size=1  # MT5 typically uses single connection
            )
        
        # Get or create the global pool
        self._pool = get_mt5_connection_pool(pool_config)
        self._symbol_info_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 300  # 5 minutes cache
        self._last_cache_time: Dict[str, datetime] = {}
        
        logger.info("PooledMT5Client initialized with connection pooling")
    
    def initialize(self) -> bool:
        """
        Initialize MT5 connection through the pool.
        
        Returns:
            bool: True if at least one connection is available
        """
        try:
            # Pool handles initialization automatically
            # Just verify we can get a connection
            with self._pool.get_connection() as conn:
                return conn.is_healthy()
        except Exception as e:
            logger.error(f"Failed to initialize MT5 through pool: {e}")
            return False
    
    @property
    def is_initialized(self) -> bool:
        """Check if client has access to MT5 connections"""
        try:
            metrics = self._pool.get_metrics()
            # Check if we have any connected connections
            return metrics.get('connection_states', {}).get('connected', 0) > 0
        except:
            return False
    
    def shutdown(self):
        """Shutdown is handled by the pool lifecycle"""
        logger.info("PooledMT5Client shutdown called (handled by pool)")
    
    def ensure_connection(self) -> bool:
        """Ensure a connection is available from the pool"""
        return self.is_initialized
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get MT5 account information using pooled connection"""
        try:
            def get_info():
                account_info = mt5.account_info()
                if account_info:
                    return account_info._asdict()
                return None
            
            return self._pool.execute_with_connection(get_info)
            
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise MT5ConnectionError(f"Failed to retrieve account info: {str(e)}") from e
    
    def get_terminal_info(self) -> Optional[Dict[str, Any]]:
        """Get MT5 terminal information using pooled connection"""
        try:
            def get_info():
                terminal_info = mt5.terminal_info()
                if terminal_info:
                    return terminal_info._asdict()
                return None
            
            return self._pool.execute_with_connection(get_info)
            
        except Exception as e:
            logger.error(f"Failed to get terminal info: {e}")
            return None
    
    @validate_params(symbol=StringValidator.validate_forex_symbol)
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol information with caching"""
        # Check cache first
        if symbol in self._symbol_info_cache:
            cache_time = self._last_cache_time.get(symbol)
            if cache_time and (datetime.now() - cache_time).total_seconds() < self._cache_ttl:
                return self._symbol_info_cache[symbol]
        
        try:
            def get_info():
                info = mt5.symbol_info(symbol)
                if info:
                    return info._asdict()
                return None
            
            symbol_info = self._pool.execute_with_connection(get_info)
            
            if symbol_info:
                # Update cache
                self._symbol_info_cache[symbol] = symbol_info
                self._last_cache_time[symbol] = datetime.now()
                return symbol_info
            else:
                raise SymbolNotFoundError(f"Symbol {symbol} not found in MT5")
                
        except Exception as e:
            logger.error(f"Failed to get symbol info for {symbol}: {e}")
            if "not found" in str(e).lower():
                raise SymbolNotFoundError(f"Symbol {symbol} not found") from e
            raise MT5DataError(f"Failed to get symbol info: {str(e)}") from e
    
    @validate_params(symbol=StringValidator.validate_forex_symbol)
    def get_symbol_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current tick for symbol"""
        try:
            def get_tick():
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    return tick._asdict()
                return None
            
            return self._pool.execute_with_connection(get_tick)
            
        except Exception as e:
            logger.error(f"Failed to get tick for {symbol}: {e}")
            raise MT5DataError(f"Failed to get tick data: {str(e)}") from e
    
    @validate_params(
        symbol=StringValidator.validate_forex_symbol,
        count=lambda x: NumericValidator.validate_positive_integer(x, "count")
    )
    def get_candles(
        self, 
        symbol: str, 
        timeframe: int, 
        count: int,
        start_pos: int = 0
    ) -> Optional[Any]:
        """Get historical candle data"""
        try:
            def get_data():
                rates = mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)
                return rates
            
            rates = self._pool.execute_with_connection(get_data)
            
            if rates is None or len(rates) == 0:
                raise InsufficientDataError(
                    f"No candle data available for {symbol} on timeframe {timeframe}"
                )
            
            return rates
            
        except Exception as e:
            logger.error(f"Failed to get candles for {symbol}: {e}")
            if isinstance(e, InsufficientDataError):
                raise
            raise MT5DataError(f"Failed to get candle data: {str(e)}") from e
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open positions"""
        try:
            def get_pos():
                if symbol:
                    positions = mt5.positions_get(symbol=symbol)
                else:
                    positions = mt5.positions_get()
                
                if positions is None:
                    return []
                
                return [pos._asdict() for pos in positions]
            
            return self._pool.execute_with_connection(get_pos)
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def send_order(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send order to MT5"""
        try:
            def send():
                result = mt5.order_send(request)
                if result:
                    return result._asdict()
                return {'retcode': -1, 'comment': 'Order send failed'}
            
            return self._pool.execute_with_connection(send)
            
        except Exception as e:
            logger.error(f"Failed to send order: {e}")
            raise MT5Error(f"Order send failed: {str(e)}") from e
    
    def check_order_result(self, result: Dict[str, Any]) -> bool:
        """Check if order result indicates success"""
        if not result:
            return False
        
        retcode = result.get('retcode', -1)
        return retcode == mt5.TRADE_RETCODE_DONE
    
    @validate_params(symbol=StringValidator.validate_forex_symbol)
    def is_market_open(self, symbol: str) -> bool:
        """Check if market is open for trading"""
        try:
            def check_market():
                # Get symbol info
                info = mt5.symbol_info(symbol)
                if not info:
                    return False
                
                # Check if trading is allowed
                if not info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
                    return False
                
                # Get current tick to verify market is active
                tick = mt5.symbol_info_tick(symbol)
                if not tick:
                    return False
                
                # Check if we have recent prices
                current_time = datetime.now().timestamp()
                tick_time = tick.time
                
                # If tick is older than 60 seconds, market might be closed
                if current_time - tick_time > 60:
                    return False
                
                return True
            
            return self._pool.execute_with_connection(check_market)
            
        except Exception as e:
            logger.error(f"Failed to check market status for {symbol}: {e}")
            return False
    
    def get_spread(self, symbol: str) -> Optional[float]:
        """Get current spread for symbol in points"""
        try:
            tick = self.get_symbol_tick(symbol)
            if tick:
                symbol_info = self.get_symbol_info(symbol)
                if symbol_info:
                    point = symbol_info.get('point', 0)
                    if point > 0:
                        return (tick['ask'] - tick['bid']) / point
            return None
        except Exception:
            return None
    
    def get_connection_metrics(self) -> Dict[str, Any]:
        """Get connection pool metrics"""
        return self._pool.get_metrics()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        # Pool manages its own lifecycle
        pass


# Compatibility function to replace MT5Client with PooledMT5Client
def create_pooled_mt5_client(settings: MT5Settings) -> PooledMT5Client:
    """
    Factory function to create a pooled MT5 client.
    
    Args:
        settings: MT5 configuration settings
        
    Returns:
        PooledMT5Client: Client instance using connection pooling
    """
    return PooledMT5Client(settings)


# Export public API
__all__ = ['PooledMT5Client', 'create_pooled_mt5_client']