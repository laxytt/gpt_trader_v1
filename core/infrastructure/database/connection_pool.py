"""
Database connection pooling for improved performance and resource management.
Provides both sync and async connection pools with proper lifecycle management.
"""

import sqlite3
import aiosqlite
import asyncio
import logging
import threading
import time
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue, Empty, Full
from typing import Optional, Dict, Any, AsyncGenerator, Generator
import weakref

from core.domain.exceptions import DatabaseError, ServiceNotAvailableError

logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """Configuration for connection pools"""
    min_connections: int = 2
    max_connections: int = 10
    connection_timeout: float = 30.0  # seconds to wait for a connection
    idle_timeout: float = 300.0  # seconds before closing idle connections
    max_lifetime: float = 3600.0  # maximum lifetime of a connection in seconds
    validation_interval: float = 60.0  # seconds between connection validations
    retry_on_error: bool = True
    retry_attempts: int = 3
    retry_delay: float = 0.1


class PooledConnection:
    """Wrapper for a database connection with metadata"""
    
    def __init__(self, connection, pool_ref):
        self.connection = connection
        self.pool_ref = weakref.ref(pool_ref)
        self.created_at = time.time()
        self.last_used_at = time.time()
        self.in_use = False
        self.id = id(connection)
    
    def is_expired(self, max_lifetime: float, idle_timeout: float) -> bool:
        """Check if connection has expired"""
        now = time.time()
        if now - self.created_at > max_lifetime:
            return True
        if not self.in_use and now - self.last_used_at > idle_timeout:
            return True
        return False
    
    def validate(self) -> bool:
        """Validate the connection is still alive"""
        try:
            self.connection.execute("SELECT 1")
            return True
        except Exception:
            return False
    
    def close(self):
        """Close the underlying connection"""
        try:
            self.connection.close()
        except Exception:
            pass


class SqliteConnectionPool:
    """
    Thread-safe connection pool for SQLite databases.
    Manages connection lifecycle and provides efficient connection reuse.
    """
    
    def __init__(self, database_path: str, config: Optional[PoolConfig] = None):
        self.database_path = str(database_path)
        self.config = config or PoolConfig()
        self._lock = threading.Lock()
        self._connections: Queue[PooledConnection] = Queue(maxsize=self.config.max_connections)
        self._all_connections: weakref.WeakSet[PooledConnection] = weakref.WeakSet()
        self._active_count = 0
        self._closed = False
        self._stats = {
            'connections_created': 0,
            'connections_reused': 0,
            'connections_expired': 0,
            'connection_errors': 0,
            'wait_timeouts': 0
        }
        
        # Start maintenance thread
        self._maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            daemon=True,
            name="SqlitePool-Maintenance"
        )
        self._maintenance_thread.start()
        
        # Pre-create minimum connections
        self._ensure_min_connections()
        
        logger.info(f"SQLite connection pool initialized for {database_path}")
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with optimized settings"""
        conn = sqlite3.connect(
            self.database_path,
            check_same_thread=False,  # Allow multi-threaded access
            timeout=30.0
        )
        conn.row_factory = sqlite3.Row
        
        # Optimize SQLite settings for better performance
        conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
        conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
        conn.execute("PRAGMA temp_store=MEMORY")  # Use memory for temp tables
        conn.execute("PRAGMA mmap_size=30000000000")  # 30GB memory-mapped I/O
        conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        conn.execute("PRAGMA busy_timeout=30000")  # 30 second timeout
        
        return conn
    
    def _ensure_min_connections(self):
        """Ensure minimum number of connections are available"""
        with self._lock:
            while self._active_count < self.config.min_connections and not self._closed:
                try:
                    conn = self._create_connection()
                    pooled = PooledConnection(conn, self)
                    self._connections.put_nowait(pooled)
                    self._all_connections.add(pooled)
                    self._active_count += 1
                    self._stats['connections_created'] += 1
                except Exception as e:
                    logger.error(f"Failed to create connection: {e}")
                    self._stats['connection_errors'] += 1
                    break
    
    @contextmanager
    def get_connection(self, timeout: Optional[float] = None) -> Generator[sqlite3.Connection, None, None]:
        """
        Get a connection from the pool.
        
        Args:
            timeout: Maximum time to wait for a connection (uses pool config if not specified)
            
        Yields:
            sqlite3.Connection: A database connection
            
        Raises:
            ServiceNotAvailableError: If pool is closed or timeout exceeded
            DatabaseError: If connection cannot be established
        """
        if self._closed:
            raise ServiceNotAvailableError("Connection pool is closed")
        
        timeout = timeout or self.config.connection_timeout
        pooled_conn = None
        start_time = time.time()
        
        try:
            # Try to get an existing connection
            while True:
                try:
                    pooled_conn = self._connections.get(timeout=0.1)
                    
                    # Validate connection
                    if pooled_conn.is_expired(self.config.max_lifetime, self.config.idle_timeout):
                        self._close_connection(pooled_conn)
                        pooled_conn = None
                        continue
                    
                    if not pooled_conn.validate():
                        self._close_connection(pooled_conn)
                        pooled_conn = None
                        continue
                    
                    # Connection is good
                    pooled_conn.in_use = True
                    pooled_conn.last_used_at = time.time()
                    self._stats['connections_reused'] += 1
                    break
                    
                except Empty:
                    # No available connections, try to create a new one
                    with self._lock:
                        if self._active_count < self.config.max_connections:
                            try:
                                conn = self._create_connection()
                                pooled_conn = PooledConnection(conn, self)
                                pooled_conn.in_use = True
                                self._all_connections.add(pooled_conn)
                                self._active_count += 1
                                self._stats['connections_created'] += 1
                                break
                            except Exception as e:
                                logger.error(f"Failed to create new connection: {e}")
                                self._stats['connection_errors'] += 1
                                raise DatabaseError(f"Cannot create database connection: {str(e)}") from e
                
                # Check timeout
                if time.time() - start_time > timeout:
                    self._stats['wait_timeouts'] += 1
                    raise ServiceNotAvailableError(
                        f"Connection pool timeout after {timeout}s "
                        f"(active: {self._active_count}/{self.config.max_connections})"
                    )
            
            yield pooled_conn.connection
            
        finally:
            # Return connection to pool
            if pooled_conn:
                pooled_conn.in_use = False
                pooled_conn.last_used_at = time.time()
                
                try:
                    self._connections.put_nowait(pooled_conn)
                except Full:
                    # Pool is full, close this connection
                    self._close_connection(pooled_conn)
    
    def _close_connection(self, pooled_conn: PooledConnection):
        """Close a connection and update tracking"""
        try:
            pooled_conn.close()
            with self._lock:
                self._active_count -= 1
                self._stats['connections_expired'] += 1
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
    
    def _maintenance_loop(self):
        """Background thread that maintains the connection pool"""
        while not self._closed:
            try:
                # Sleep for validation interval
                time.sleep(self.config.validation_interval)
                
                if self._closed:
                    break
                
                # Validate and clean up connections
                expired_connections = []
                
                # Check all connections in the pool
                temp_connections = []
                while True:
                    try:
                        pooled_conn = self._connections.get_nowait()
                        
                        if pooled_conn.is_expired(self.config.max_lifetime, self.config.idle_timeout):
                            expired_connections.append(pooled_conn)
                        elif not pooled_conn.validate():
                            expired_connections.append(pooled_conn)
                        else:
                            temp_connections.append(pooled_conn)
                    except Empty:
                        break
                
                # Return valid connections to pool
                for conn in temp_connections:
                    try:
                        self._connections.put_nowait(conn)
                    except Full:
                        expired_connections.append(conn)
                
                # Close expired connections
                for conn in expired_connections:
                    self._close_connection(conn)
                
                # Ensure minimum connections
                self._ensure_min_connections()
                
            except Exception as e:
                logger.error(f"Error in pool maintenance: {e}")
    
    def close(self):
        """Close all connections and shut down the pool"""
        self._closed = True
        
        # Close all connections
        closed_count = 0
        while True:
            try:
                pooled_conn = self._connections.get_nowait()
                pooled_conn.close()
                closed_count += 1
            except Empty:
                break
        
        # Close any remaining connections
        for pooled_conn in list(self._all_connections):
            try:
                pooled_conn.close()
                closed_count += 1
            except Exception:
                pass
        
        logger.info(f"Connection pool closed. Stats: {self._stats}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        return {
            **self._stats,
            'active_connections': self._active_count,
            'available_connections': self._connections.qsize(),
            'pool_size': f"{self._active_count}/{self.config.max_connections}"
        }


class AsyncSqliteConnectionPool:
    """
    Async connection pool for SQLite using aiosqlite.
    Provides efficient async connection management.
    """
    
    def __init__(self, database_path: str, config: Optional[PoolConfig] = None):
        self.database_path = str(database_path)
        self.config = config or PoolConfig()
        self._connections: asyncio.Queue[PooledConnection] = asyncio.Queue(maxsize=self.config.max_connections)
        self._all_connections: weakref.WeakSet[PooledConnection] = weakref.WeakSet()
        self._active_count = 0
        self._closed = False
        self._lock = asyncio.Lock()
        self._stats = {
            'connections_created': 0,
            'connections_reused': 0,
            'connections_expired': 0,
            'connection_errors': 0,
            'wait_timeouts': 0
        }
        
        # Will start maintenance task when first connection is requested
        self._maintenance_task: Optional[asyncio.Task] = None
    
    async def _create_connection(self) -> aiosqlite.Connection:
        """Create a new async SQLite connection"""
        conn = await aiosqlite.connect(
            self.database_path,
            timeout=30.0
        )
        
        # Set row factory
        conn.row_factory = aiosqlite.Row
        
        # Optimize SQLite settings
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute("PRAGMA temp_store=MEMORY")
        await conn.execute("PRAGMA mmap_size=30000000000")
        await conn.execute("PRAGMA cache_size=-64000")
        await conn.execute("PRAGMA busy_timeout=30000")
        
        return conn
    
    async def _ensure_min_connections(self):
        """Ensure minimum number of connections are available"""
        async with self._lock:
            while self._active_count < self.config.min_connections and not self._closed:
                try:
                    conn = await self._create_connection()
                    pooled = PooledConnection(conn, self)
                    await self._connections.put(pooled)
                    self._all_connections.add(pooled)
                    self._active_count += 1
                    self._stats['connections_created'] += 1
                except Exception as e:
                    logger.error(f"Failed to create async connection: {e}")
                    self._stats['connection_errors'] += 1
                    break
    
    async def _validate_connection_async(self, pooled_conn: PooledConnection) -> bool:
        """Validate an async connection"""
        try:
            await pooled_conn.connection.execute("SELECT 1")
            return True
        except Exception:
            return False
    
    @asynccontextmanager
    async def get_connection(self, timeout: Optional[float] = None) -> AsyncGenerator[aiosqlite.Connection, None]:
        """Get an async connection from the pool"""
        if self._closed:
            raise ServiceNotAvailableError("Async connection pool is closed")
        
        # Start maintenance task if not running
        if self._maintenance_task is None:
            self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        
        timeout = timeout or self.config.connection_timeout
        pooled_conn = None
        start_time = time.time()
        
        try:
            # Try to get an existing connection
            while True:
                try:
                    pooled_conn = await asyncio.wait_for(
                        self._connections.get(),
                        timeout=0.1
                    )
                    
                    # Validate connection
                    if pooled_conn.is_expired(self.config.max_lifetime, self.config.idle_timeout):
                        await self._close_connection_async(pooled_conn)
                        pooled_conn = None
                        continue
                    
                    if not await self._validate_connection_async(pooled_conn):
                        await self._close_connection_async(pooled_conn)
                        pooled_conn = None
                        continue
                    
                    # Connection is good
                    pooled_conn.in_use = True
                    pooled_conn.last_used_at = time.time()
                    self._stats['connections_reused'] += 1
                    break
                    
                except asyncio.TimeoutError:
                    # No available connections, try to create a new one
                    async with self._lock:
                        if self._active_count < self.config.max_connections:
                            try:
                                conn = await self._create_connection()
                                pooled_conn = PooledConnection(conn, self)
                                pooled_conn.in_use = True
                                self._all_connections.add(pooled_conn)
                                self._active_count += 1
                                self._stats['connections_created'] += 1
                                break
                            except Exception as e:
                                logger.error(f"Failed to create new async connection: {e}")
                                self._stats['connection_errors'] += 1
                                raise DatabaseError(f"Cannot create async database connection: {str(e)}") from e
                
                # Check timeout
                if time.time() - start_time > timeout:
                    self._stats['wait_timeouts'] += 1
                    raise ServiceNotAvailableError(
                        f"Async connection pool timeout after {timeout}s "
                        f"(active: {self._active_count}/{self.config.max_connections})"
                    )
            
            yield pooled_conn.connection
            
        finally:
            # Return connection to pool
            if pooled_conn:
                pooled_conn.in_use = False
                pooled_conn.last_used_at = time.time()
                
                try:
                    self._connections.put_nowait(pooled_conn)
                except asyncio.QueueFull:
                    # Pool is full, close this connection
                    await self._close_connection_async(pooled_conn)
    
    async def _close_connection_async(self, pooled_conn: PooledConnection):
        """Close an async connection"""
        try:
            await pooled_conn.connection.close()
            async with self._lock:
                self._active_count -= 1
                self._stats['connections_expired'] += 1
        except Exception as e:
            logger.error(f"Error closing async connection: {e}")
    
    async def _maintenance_loop(self):
        """Async maintenance loop"""
        while not self._closed:
            try:
                await asyncio.sleep(self.config.validation_interval)
                
                if self._closed:
                    break
                
                # Similar maintenance logic as sync version
                await self._ensure_min_connections()
                
            except Exception as e:
                logger.error(f"Error in async pool maintenance: {e}")
    
    async def close(self):
        """Close all async connections"""
        self._closed = True
        
        # Cancel maintenance task
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        closed_count = 0
        while not self._connections.empty():
            try:
                pooled_conn = self._connections.get_nowait()
                await pooled_conn.connection.close()
                closed_count += 1
            except asyncio.QueueEmpty:
                break
        
        logger.info(f"Async connection pool closed. Stats: {self._stats}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        return {
            **self._stats,
            'active_connections': self._active_count,
            'available_connections': self._connections.qsize(),
            'pool_size': f"{self._active_count}/{self.config.max_connections}"
        }


# Global pool instances (singleton pattern)
_sync_pools: Dict[str, SqliteConnectionPool] = {}
_async_pools: Dict[str, AsyncSqliteConnectionPool] = {}
_pools_lock = threading.Lock()


def get_sync_connection_pool(
    database_path: str,
    config: Optional[PoolConfig] = None
) -> SqliteConnectionPool:
    """
    Get or create a sync connection pool for the given database.
    Uses singleton pattern to ensure one pool per database.
    """
    path_str = str(Path(database_path).absolute())
    
    with _pools_lock:
        if path_str not in _sync_pools:
            _sync_pools[path_str] = SqliteConnectionPool(path_str, config)
        return _sync_pools[path_str]


def get_async_connection_pool(
    database_path: str,
    config: Optional[PoolConfig] = None
) -> AsyncSqliteConnectionPool:
    """
    Get or create an async connection pool for the given database.
    Uses singleton pattern to ensure one pool per database.
    """
    path_str = str(Path(database_path).absolute())
    
    with _pools_lock:
        if path_str not in _async_pools:
            _async_pools[path_str] = AsyncSqliteConnectionPool(path_str, config)
        return _async_pools[path_str]


def close_all_pools():
    """Close all connection pools (useful for cleanup)"""
    with _pools_lock:
        # Close sync pools
        for pool in _sync_pools.values():
            try:
                pool.close()
            except Exception as e:
                logger.error(f"Error closing sync pool: {e}")
        _sync_pools.clear()
        
        # Note: Async pools need to be closed in an async context
        # This just clears the references
        _async_pools.clear()


async def close_all_async_pools():
    """Close all async connection pools"""
    with _pools_lock:
        pools = list(_async_pools.values())
        _async_pools.clear()
    
    # Close all async pools
    for pool in pools:
        try:
            await pool.close()
        except Exception as e:
            logger.error(f"Error closing async pool: {e}")


# Export public API
__all__ = [
    'PoolConfig',
    'SqliteConnectionPool',
    'AsyncSqliteConnectionPool',
    'get_sync_connection_pool',
    'get_async_connection_pool',
    'close_all_pools',
    'close_all_async_pools'
]