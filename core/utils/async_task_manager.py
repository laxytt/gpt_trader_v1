"""
Async task manager for tracking and managing background tasks.
Prevents fire-and-forget tasks from failing silently.
"""

import asyncio
import logging
from typing import Set, Dict, Optional, Callable, Any
from datetime import datetime, timezone
import weakref
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class TaskInfo:
    """Information about a managed task"""
    def __init__(self, task: asyncio.Task, name: str, critical: bool = False):
        self.task = task
        self.name = name
        self.critical = critical
        self.created_at = datetime.now(timezone.utc)
        self.error: Optional[Exception] = None
        
    @property
    def is_done(self) -> bool:
        return self.task.done()
    
    @property
    def is_failed(self) -> bool:
        return self.task.done() and not self.task.cancelled() and self.task.exception() is not None


class AsyncTaskManager:
    """
    Manages async tasks to prevent silent failures and ensure proper cleanup.
    
    Features:
    - Tracks all created tasks
    - Logs task failures
    - Ensures cleanup on shutdown
    - Provides task monitoring
    """
    
    _instance: Optional['AsyncTaskManager'] = None
    
    def __new__(cls):
        """Singleton pattern for global task manager"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._tasks: Set[TaskInfo] = set()
            self._task_stats = {
                'created': 0,
                'completed': 0,
                'failed': 0,
                'cancelled': 0
            }
            self._error_handlers: Dict[str, Callable[[Exception], None]] = {}
            self._shutdown_in_progress = False
            self._initialized = True
    
    def create_task(
        self,
        coro,
        name: str,
        critical: bool = False,
        error_handler: Optional[Callable[[Exception], None]] = None
    ) -> asyncio.Task:
        """
        Create and track an async task.
        
        Args:
            coro: Coroutine to run
            name: Descriptive name for the task
            critical: If True, task failure will trigger system alert
            error_handler: Optional custom error handler
            
        Returns:
            The created task
        """
        if self._shutdown_in_progress:
            logger.warning(f"Refusing to create task '{name}' during shutdown")
            return None
        
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running, close the coroutine to prevent warning
            coro.close()
            logger.warning(f"No event loop available for task '{name}', skipping")
            return None
        
        # Create the task
        task = asyncio.create_task(coro, name=name)
        task_info = TaskInfo(task, name, critical)
        
        # Add to tracking
        self._tasks.add(task_info)
        self._task_stats['created'] += 1
        
        # Set up completion callback
        task.add_done_callback(lambda t: self._task_completed(task_info, error_handler))
        
        logger.debug(f"Created task '{name}' (critical={critical})")
        return task
    
    def _task_completed(self, task_info: TaskInfo, error_handler: Optional[Callable]):
        """Handle task completion"""
        try:
            if task_info.task.cancelled():
                self._task_stats['cancelled'] += 1
                logger.debug(f"Task '{task_info.name}' was cancelled")
                
            elif exception := task_info.task.exception():
                self._task_stats['failed'] += 1
                task_info.error = exception
                
                # Log the error
                if task_info.critical:
                    logger.error(
                        f"Critical task '{task_info.name}' failed: {exception}",
                        exc_info=exception
                    )
                else:
                    logger.warning(f"Task '{task_info.name}' failed: {exception}")
                
                # Call error handler if provided
                if error_handler:
                    try:
                        error_handler(exception)
                    except Exception as e:
                        logger.error(f"Error handler for '{task_info.name}' failed: {e}")
                
            else:
                self._task_stats['completed'] += 1
                logger.debug(f"Task '{task_info.name}' completed successfully")
                
        except Exception as e:
            logger.error(f"Error in task completion handler: {e}")
        finally:
            # Remove from tracking (after a delay to allow inspection)
            asyncio.create_task(self._cleanup_task(task_info))
    
    async def _cleanup_task(self, task_info: TaskInfo):
        """Remove task from tracking after a delay"""
        await asyncio.sleep(60)  # Keep for 1 minute for debugging
        self._tasks.discard(task_info)
    
    def get_active_tasks(self) -> list[TaskInfo]:
        """Get list of currently active tasks"""
        return [t for t in self._tasks if not t.is_done]
    
    def get_failed_tasks(self) -> list[TaskInfo]:
        """Get list of failed tasks"""
        return [t for t in self._tasks if t.is_failed]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get task statistics"""
        active_tasks = self.get_active_tasks()
        return {
            **self._task_stats,
            'active': len(active_tasks),
            'active_names': [t.name for t in active_tasks]
        }
    
    async def wait_for_tasks(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all tasks to complete.
        
        Returns:
            True if all tasks completed, False if timeout
        """
        active_tasks = self.get_active_tasks()
        if not active_tasks:
            return True
        
        tasks = [t.task for t in active_tasks]
        logger.info(f"Waiting for {len(tasks)} active tasks to complete...")
        
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for tasks: {[t.name for t in active_tasks]}")
            return False
    
    async def shutdown(self, timeout: float = 30.0):
        """
        Shutdown task manager and cancel remaining tasks.
        
        Args:
            timeout: Maximum time to wait for tasks to complete
        """
        logger.info("Shutting down async task manager...")
        self._shutdown_in_progress = True
        
        # First, wait for tasks to complete naturally
        completed = await self.wait_for_tasks(timeout=timeout/2)
        
        if not completed:
            # Cancel remaining tasks
            active_tasks = self.get_active_tasks()
            logger.warning(f"Cancelling {len(active_tasks)} remaining tasks")
            
            for task_info in active_tasks:
                task_info.task.cancel()
            
            # Wait for cancellation
            await self.wait_for_tasks(timeout=timeout/2)
        
        # Log final stats
        logger.info(f"Task manager shutdown complete. Stats: {self.get_stats()}")
    
    @asynccontextmanager
    async def task_group(self, name: str):
        """Context manager for grouping related tasks"""
        group_tasks = []
        
        def create_group_task(coro, task_name: str, **kwargs):
            full_name = f"{name}.{task_name}"
            task = self.create_task(coro, full_name, **kwargs)
            group_tasks.append(task)
            return task
        
        # Temporarily replace create_task
        original_create = self.create_task
        self.create_task = create_group_task
        
        try:
            yield self
        finally:
            # Restore original method
            self.create_task = original_create
            
            # Wait for group tasks
            if group_tasks:
                await asyncio.gather(*group_tasks, return_exceptions=True)


# Global task manager instance
_task_manager = AsyncTaskManager()


def get_task_manager() -> AsyncTaskManager:
    """Get the global task manager instance"""
    return _task_manager


# Convenience functions
def create_task(coro, name: str, **kwargs) -> asyncio.Task:
    """Create a managed task using the global task manager"""
    return _task_manager.create_task(coro, name, **kwargs)


async def shutdown_tasks(timeout: float = 30.0):
    """Shutdown the global task manager"""
    await _task_manager.shutdown(timeout)


# Decorator for auto-tracked async functions
def tracked_async(name: Optional[str] = None, critical: bool = False):
    """Decorator to automatically track async function calls"""
    def decorator(func):
        task_name = name or func.__name__
        
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        
        def tracked_wrapper(*args, **kwargs):
            return create_task(
                wrapper(*args, **kwargs),
                name=task_name,
                critical=critical
            )
        
        # Preserve original function for direct await
        tracked_wrapper.__wrapped__ = func
        tracked_wrapper.__name__ = func.__name__
        return tracked_wrapper
    
    return decorator