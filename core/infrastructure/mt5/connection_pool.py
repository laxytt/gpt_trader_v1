"""
MT5 connection pooling and management.
Handles MT5 terminal connections with retry logic and health monitoring.
"""

import logging
import threading
import time
import MetaTrader5 as mt5
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Callable
import queue
from enum import Enum

from core.domain.exceptions import (
    MT5ConnectionError, MT5InitializationError, 
    ServiceNotAvailableError, ConfigurationError
)
from core.utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpen

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """MT5 connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    TERMINATED = "terminated"


@dataclass
class MT5ConnectionConfig:
    """Configuration for MT5 connection management"""
    login: Optional[int] = None
    password: Optional[str] = None
    server: Optional[str] = None
    terminal_path: Optional[str] = None
    
    # Connection settings
    max_reconnect_attempts: int = 5
    reconnect_delay: float = 2.0
    reconnect_backoff: float = 2.0
    
    # Health check settings
    health_check_interval: float = 30.0
    health_check_timeout: float = 5.0
    
    # Circuit breaker settings
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: float = 60.0
    circuit_breaker_expected_exception: type = MT5ConnectionError
    
    # Pool settings
    enable_pooling: bool = True
    pool_size: int = 1  # MT5 typically uses single connection
    connection_timeout: float = 30.0


@dataclass
class ConnectionMetrics:
    """Metrics for MT5 connection monitoring"""
    connection_attempts: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    uptime_seconds: float = 0.0
    last_health_check: Optional[datetime] = None
    
    def record_connection_attempt(self, success: bool, error: Optional[str] = None):
        """Record a connection attempt"""
        self.connection_attempts += 1
        if success:
            self.successful_connections += 1
        else:
            self.failed_connections += 1
            self.last_error = error
            self.last_error_time = datetime.now(timezone.utc)
    
    def record_request(self, success: bool):
        """Record an MT5 API request"""
        self.total_requests += 1
        if not success:
            self.failed_requests += 1
    
    def get_success_rate(self) -> float:
        """Calculate request success rate"""
        if self.total_requests == 0:
            return 0.0
        return (self.total_requests - self.failed_requests) / self.total_requests


class MT5Connection:
    """
    Wrapper for a single MT5 terminal connection.
    Handles connection lifecycle and health monitoring.
    """
    
    def __init__(self, config: MT5ConnectionConfig, connection_id: int = 0):
        self.config = config
        self.connection_id = connection_id
        self.state = ConnectionState.DISCONNECTED
        self.metrics = ConnectionMetrics()
        self._lock = threading.Lock()
        self._connected_at: Optional[datetime] = None
        
        # Circuit breaker for connection failures
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker_failure_threshold,
            recovery_timeout=config.circuit_breaker_recovery_timeout,
            expected_exception=config.circuit_breaker_expected_exception
        )
    
    def connect(self) -> bool:
        """
        Establish connection to MT5 terminal.
        
        Returns:
            bool: True if connection successful
            
        Raises:
            MT5InitializationError: If connection fails after all retries
            CircuitBreakerOpen: If circuit breaker is open
        """
        with self._lock:
            if self.state == ConnectionState.CONNECTED:
                return True
            
            self.state = ConnectionState.CONNECTING
            attempt = 0
            delay = self.config.reconnect_delay
            
            while attempt < self.config.max_reconnect_attempts:
                try:
                    # Use circuit breaker to prevent rapid reconnection attempts
                    with self.circuit_breaker:
                        # Initialize MT5
                        if self.config.terminal_path:
                            result = mt5.initialize(
                                path=self.config.terminal_path,
                                login=self.config.login,
                                password=self.config.password,
                                server=self.config.server,
                                timeout=int(self.config.connection_timeout * 1000)
                            )
                        else:
                            result = mt5.initialize(
                                login=self.config.login,
                                password=self.config.password,
                                server=self.config.server,
                                timeout=int(self.config.connection_timeout * 1000)
                            )
                        
                        if result:
                            # Verify connection is working
                            terminal_info = mt5.terminal_info()
                            if terminal_info:
                                self.state = ConnectionState.CONNECTED
                                self._connected_at = datetime.now(timezone.utc)
                                self.metrics.record_connection_attempt(True)
                                logger.info(f"MT5 connection {self.connection_id} established")
                                return True
                        
                        # Connection failed
                        error_code = mt5.last_error()
                        error_msg = f"MT5 initialization failed: {error_code}"
                        self.metrics.record_connection_attempt(False, error_msg)
                        
                        if attempt < self.config.max_reconnect_attempts - 1:
                            logger.warning(f"MT5 connection attempt {attempt + 1} failed, retrying in {delay}s")
                            time.sleep(delay)
                            delay *= self.config.reconnect_backoff
                            attempt += 1
                        else:
                            break
                        
                except CircuitBreakerOpen:
                    self.state = ConnectionState.ERROR
                    raise
                except Exception as e:
                    error_msg = f"MT5 connection error: {str(e)}"
                    self.metrics.record_connection_attempt(False, error_msg)
                    logger.error(error_msg)
                    
                    if attempt < self.config.max_reconnect_attempts - 1:
                        time.sleep(delay)
                        delay *= self.config.reconnect_backoff
                        attempt += 1
                    else:
                        break
            
            self.state = ConnectionState.ERROR
            raise MT5InitializationError(
                f"Failed to connect to MT5 after {self.config.max_reconnect_attempts} attempts"
            )
    
    def disconnect(self):
        """Disconnect from MT5 terminal"""
        with self._lock:
            if self.state == ConnectionState.CONNECTED:
                try:
                    mt5.shutdown()
                    logger.info(f"MT5 connection {self.connection_id} closed")
                except Exception as e:
                    logger.error(f"Error closing MT5 connection: {e}")
                finally:
                    self.state = ConnectionState.DISCONNECTED
                    self._connected_at = None
    
    def is_healthy(self) -> bool:
        """
        Check if connection is healthy.
        
        Returns:
            bool: True if connection is healthy and responsive
        """
        if self.state != ConnectionState.CONNECTED:
            return False
        
        try:
            # Test connection with a simple query
            terminal_info = mt5.terminal_info()
            if not terminal_info:
                return False
            
            # Check if terminal is connected to trade server
            if not terminal_info.connected:
                logger.warning(f"MT5 terminal {self.connection_id} disconnected from trade server")
                return False
            
            # Update metrics
            self.metrics.last_health_check = datetime.now(timezone.utc)
            if self._connected_at:
                self.metrics.uptime_seconds = (
                    datetime.now(timezone.utc) - self._connected_at
                ).total_seconds()
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed for connection {self.connection_id}: {e}")
            return False
    
    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute an MT5 function with automatic retry on connection failure.
        
        Args:
            func: MT5 function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result
            
        Raises:
            MT5ConnectionError: If execution fails after retry
        """
        if self.state != ConnectionState.CONNECTED:
            raise MT5ConnectionError(f"Connection {self.connection_id} not available")
        
        try:
            result = func(*args, **kwargs)
            self.metrics.record_request(True)
            return result
        except Exception as e:
            self.metrics.record_request(False)
            
            # Check if it's a connection error
            error_msg = str(e).lower()
            if any(term in error_msg for term in ['not connected', 'terminal', 'initialize']):
                # Try to reconnect once
                logger.warning(f"Connection lost, attempting reconnect for {self.connection_id}")
                try:
                    self.disconnect()
                    self.connect()
                    result = func(*args, **kwargs)
                    self.metrics.record_request(True)
                    return result
                except Exception as reconnect_error:
                    raise MT5ConnectionError(
                        f"Failed to execute after reconnect: {str(reconnect_error)}"
                    ) from reconnect_error
            else:
                raise


class MT5ConnectionPool:
    """
    Connection pool for MT5 terminal connections.
    Manages connection lifecycle, health monitoring, and load distribution.
    """
    
    def __init__(self, config: MT5ConnectionConfig):
        self.config = config
        self._connections: List[MT5Connection] = []
        self._connection_queue: queue.Queue[MT5Connection] = queue.Queue()
        self._lock = threading.Lock()
        self._closed = False
        self._health_check_thread: Optional[threading.Thread] = None
        
        # Initialize connections
        self._initialize_pool()
        
        # Start health monitoring
        if config.health_check_interval > 0:
            self._start_health_monitoring()
    
    def _initialize_pool(self):
        """Initialize the connection pool"""
        pool_size = self.config.pool_size if self.config.enable_pooling else 1
        
        for i in range(pool_size):
            conn = MT5Connection(self.config, connection_id=i)
            self._connections.append(conn)
            
            # Try to connect
            try:
                conn.connect()
                self._connection_queue.put(conn)
            except Exception as e:
                logger.error(f"Failed to initialize connection {i}: {e}")
    
    def _start_health_monitoring(self):
        """Start background health monitoring thread"""
        def health_check_loop():
            while not self._closed:
                try:
                    time.sleep(self.config.health_check_interval)
                    
                    if self._closed:
                        break
                    
                    # Check health of all connections
                    with self._lock:
                        for conn in self._connections:
                            if conn.state == ConnectionState.CONNECTED:
                                if not conn.is_healthy():
                                    logger.warning(f"Connection {conn.connection_id} unhealthy, reconnecting")
                                    try:
                                        conn.disconnect()
                                        conn.connect()
                                    except Exception as e:
                                        logger.error(f"Failed to reconnect {conn.connection_id}: {e}")
                            elif conn.state == ConnectionState.ERROR:
                                # Try to recover error connections
                                try:
                                    conn.connect()
                                    # Add back to queue if successful
                                    self._connection_queue.put(conn)
                                except Exception as e:
                                    logger.debug(f"Connection {conn.connection_id} still in error: {e}")
                    
                except Exception as e:
                    logger.error(f"Error in health check loop: {e}")
        
        self._health_check_thread = threading.Thread(
            target=health_check_loop,
            daemon=True,
            name="MT5Pool-HealthCheck"
        )
        self._health_check_thread.start()
    
    @contextmanager
    def get_connection(self, timeout: Optional[float] = None):
        """
        Get a connection from the pool.
        
        Args:
            timeout: Maximum time to wait for a connection
            
        Yields:
            MT5Connection: An MT5 connection
            
        Raises:
            ServiceNotAvailableError: If no connection available within timeout
        """
        if self._closed:
            raise ServiceNotAvailableError("MT5 connection pool is closed")
        
        timeout = timeout or self.config.connection_timeout
        conn = None
        
        try:
            # Get connection from queue
            conn = self._connection_queue.get(timeout=timeout)
            
            # Verify connection is healthy
            if not conn.is_healthy():
                # Try to reconnect
                try:
                    conn.disconnect()
                    conn.connect()
                except Exception as e:
                    logger.error(f"Failed to revive connection {conn.connection_id}: {e}")
                    # Put it back and try another
                    self._connection_queue.put(conn)
                    raise ServiceNotAvailableError("No healthy MT5 connections available")
            
            yield conn
            
        except queue.Empty:
            raise ServiceNotAvailableError(
                f"No MT5 connection available within {timeout}s timeout"
            )
        finally:
            # Return connection to pool
            if conn and not self._closed:
                self._connection_queue.put(conn)
    
    def execute_with_connection(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with a pooled connection.
        
        Args:
            func: Function to execute with MT5
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
        """
        with self.get_connection() as conn:
            return conn.execute_with_retry(func, *args, **kwargs)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pool metrics and statistics"""
        total_metrics = {
            'pool_size': len(self._connections),
            'available_connections': self._connection_queue.qsize(),
            'connection_states': {},
            'total_requests': 0,
            'failed_requests': 0,
            'average_success_rate': 0.0,
            'connections': []
        }
        
        # Aggregate metrics from all connections
        success_rates = []
        
        for conn in self._connections:
            state_key = conn.state.value
            total_metrics['connection_states'][state_key] = \
                total_metrics['connection_states'].get(state_key, 0) + 1
            
            total_metrics['total_requests'] += conn.metrics.total_requests
            total_metrics['failed_requests'] += conn.metrics.failed_requests
            
            if conn.metrics.total_requests > 0:
                success_rates.append(conn.metrics.get_success_rate())
            
            # Add individual connection metrics
            total_metrics['connections'].append({
                'id': conn.connection_id,
                'state': conn.state.value,
                'uptime_seconds': conn.metrics.uptime_seconds,
                'requests': conn.metrics.total_requests,
                'success_rate': conn.metrics.get_success_rate(),
                'last_error': conn.metrics.last_error,
                'circuit_breaker_state': 'open' if conn.circuit_breaker.is_open else 'closed'
            })
        
        # Calculate average success rate
        if success_rates:
            total_metrics['average_success_rate'] = sum(success_rates) / len(success_rates)
        
        return total_metrics
    
    def close(self):
        """Close all connections and shut down the pool"""
        self._closed = True
        
        # Stop health monitoring
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5.0)
        
        # Close all connections
        for conn in self._connections:
            try:
                conn.disconnect()
            except Exception as e:
                logger.error(f"Error closing connection {conn.connection_id}: {e}")
        
        logger.info("MT5 connection pool closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Global pool instance (singleton)
_mt5_pool: Optional[MT5ConnectionPool] = None
_pool_lock = threading.Lock()


def get_mt5_connection_pool(config: Optional[MT5ConnectionConfig] = None) -> MT5ConnectionPool:
    """
    Get or create the global MT5 connection pool.
    
    Args:
        config: Configuration for the pool (used only on first call)
        
    Returns:
        MT5ConnectionPool: The global connection pool
        
    Raises:
        ConfigurationError: If pool exists but config provided
    """
    global _mt5_pool
    
    with _pool_lock:
        if _mt5_pool is None:
            if config is None:
                raise ConfigurationError("MT5 connection config required for first initialization")
            _mt5_pool = MT5ConnectionPool(config)
        elif config is not None:
            logger.warning("MT5 connection pool already exists, ignoring new config")
        
        return _mt5_pool


def close_mt5_connection_pool():
    """Close the global MT5 connection pool"""
    global _mt5_pool
    
    with _pool_lock:
        if _mt5_pool:
            _mt5_pool.close()
            _mt5_pool = None


# Export public API
__all__ = [
    'MT5ConnectionConfig',
    'MT5ConnectionPool',
    'ConnectionState',
    'ConnectionMetrics',
    'get_mt5_connection_pool',
    'close_mt5_connection_pool'
]