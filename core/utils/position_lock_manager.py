"""
Position lock manager to prevent race conditions in position management
"""

import asyncio
import threading
import time
import logging
from typing import Dict, Optional, Set
from contextlib import contextmanager
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PositionLockManager:
    """
    Manages locks for position operations to prevent race conditions.
    Ensures only one operation can modify a position at a time.
    """
    
    def __init__(self, lock_timeout: float = 30.0):
        self._locks: Dict[int, threading.RLock] = {}  # ticket -> lock
        self._lock_holders: Dict[int, str] = {}  # ticket -> operation_id
        self._lock_times: Dict[int, datetime] = {}  # ticket -> lock acquisition time
        self._global_lock = threading.RLock()
        self.lock_timeout = lock_timeout
        
    @contextmanager
    def position_lock(self, ticket: int, operation: str):
        """
        Context manager for position locking.
        
        Args:
            ticket: Position ticket number
            operation: Operation description (for debugging)
        """
        lock_acquired = False
        start_time = time.time()
        
        try:
            # Get or create lock for this ticket
            with self._global_lock:
                if ticket not in self._locks:
                    self._locks[ticket] = threading.RLock()
                lock = self._locks[ticket]
            
            # Try to acquire lock with timeout
            lock_acquired = lock.acquire(timeout=self.lock_timeout)
            
            if not lock_acquired:
                current_holder = self._lock_holders.get(ticket, "unknown")
                raise TimeoutError(
                    f"Failed to acquire lock for position {ticket} after {self.lock_timeout}s. "
                    f"Currently held by: {current_holder}"
                )
            
            # Record lock holder
            with self._global_lock:
                self._lock_holders[ticket] = operation
                self._lock_times[ticket] = datetime.now()
            
            logger.debug(f"Lock acquired for position {ticket} by {operation}")
            
            yield
            
        finally:
            if lock_acquired:
                with self._global_lock:
                    self._lock_holders.pop(ticket, None)
                    self._lock_times.pop(ticket, None)
                lock.release()
                
                duration = time.time() - start_time
                logger.debug(f"Lock released for position {ticket} after {duration:.2f}s")
                
                # Clean up if no longer needed
                with self._global_lock:
                    if ticket in self._locks and not self._locks[ticket].locked():
                        # Check if position is likely closed
                        if duration > 5.0:  # Long operation might mean position was closed
                            self._locks.pop(ticket, None)
    
    def is_locked(self, ticket: int) -> bool:
        """Check if a position is currently locked"""
        with self._global_lock:
            lock = self._locks.get(ticket)
            return lock.locked() if lock else False
    
    def get_lock_info(self, ticket: int) -> Optional[Dict[str, any]]:
        """Get information about current lock holder"""
        with self._global_lock:
            if ticket in self._lock_holders:
                return {
                    'holder': self._lock_holders[ticket],
                    'acquired_at': self._lock_times.get(ticket),
                    'duration': (datetime.now() - self._lock_times[ticket]).total_seconds()
                    if ticket in self._lock_times else 0
                }
        return None
    
    def cleanup_stale_locks(self, max_age_seconds: float = 300):
        """Clean up locks older than max_age_seconds"""
        with self._global_lock:
            now = datetime.now()
            stale_tickets = []
            
            for ticket, lock_time in self._lock_times.items():
                if (now - lock_time).total_seconds() > max_age_seconds:
                    stale_tickets.append(ticket)
            
            for ticket in stale_tickets:
                logger.warning(f"Cleaning up stale lock for position {ticket}")
                self._locks.pop(ticket, None)
                self._lock_holders.pop(ticket, None)
                self._lock_times.pop(ticket, None)


class AsyncPositionLockManager:
    """Async version of position lock manager"""
    
    def __init__(self, lock_timeout: float = 30.0):
        self._locks: Dict[int, asyncio.Lock] = {}
        self._lock_holders: Dict[int, str] = {}
        self._lock_times: Dict[int, datetime] = {}
        self._global_lock = asyncio.Lock()
        self.lock_timeout = lock_timeout
    
    @contextmanager
    async def position_lock(self, ticket: int, operation: str):
        """Async context manager for position locking"""
        lock_acquired = False
        start_time = time.time()
        
        try:
            # Get or create lock
            async with self._global_lock:
                if ticket not in self._locks:
                    self._locks[ticket] = asyncio.Lock()
                lock = self._locks[ticket]
            
            # Try to acquire with timeout
            try:
                await asyncio.wait_for(lock.acquire(), timeout=self.lock_timeout)
                lock_acquired = True
            except asyncio.TimeoutError:
                current_holder = self._lock_holders.get(ticket, "unknown")
                raise TimeoutError(
                    f"Failed to acquire async lock for position {ticket} after {self.lock_timeout}s. "
                    f"Currently held by: {current_holder}"
                )
            
            # Record lock holder
            async with self._global_lock:
                self._lock_holders[ticket] = operation
                self._lock_times[ticket] = datetime.now()
            
            logger.debug(f"Async lock acquired for position {ticket} by {operation}")
            
            yield
            
        finally:
            if lock_acquired:
                async with self._global_lock:
                    self._lock_holders.pop(ticket, None)
                    self._lock_times.pop(ticket, None)
                lock.release()
                
                duration = time.time() - start_time
                logger.debug(f"Async lock released for position {ticket} after {duration:.2f}s")


# Global instances
_position_lock_manager = PositionLockManager()
_async_position_lock_manager = AsyncPositionLockManager()


def get_position_lock_manager() -> PositionLockManager:
    """Get the global position lock manager"""
    return _position_lock_manager


def get_async_position_lock_manager() -> AsyncPositionLockManager:
    """Get the global async position lock manager"""
    return _async_position_lock_manager