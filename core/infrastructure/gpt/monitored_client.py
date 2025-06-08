"""
Monitored GPT Client
Extension of GPTClient that includes request monitoring for dashboard visualization
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import asyncio
import uuid

from core.infrastructure.gpt.client import GPTClient
from config.settings import GPTSettings

logger = logging.getLogger(__name__)


class MonitoredGPTClient(GPTClient):
    """
    Extended GPT client with request monitoring capabilities.
    Tracks all requests for visualization in the dashboard.
    """
    
    def __init__(self, config: GPTSettings):
        super().__init__(config)
        self.request_monitors: List[Callable] = []
        self.request_history = []
        self.max_history_size = 1000
    
    def add_monitor(self, monitor_func: Callable):
        """Add a monitoring function that will be called for each request"""
        self.request_monitors.append(monitor_func)
    
    def remove_monitor(self, monitor_func: Callable):
        """Remove a monitoring function"""
        if monitor_func in self.request_monitors:
            self.request_monitors.remove(monitor_func)
    
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Enhanced chat completion with monitoring.
        
        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
            metadata: Additional metadata for tracking (agent_type, context, etc.)
            
        Returns:
            Dictionary containing response and metadata
        """
        # Generate request ID
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Extract metadata
        agent_type = metadata.get('agent_type', 'Unknown') if metadata else 'Unknown'
        context = metadata.get('context', {}) if metadata else {}
        
        # Log request start
        request_data = {
            'id': request_id,
            'agent_type': agent_type,
            'timestamp': datetime.now(),
            'messages': messages,
            'temperature': temperature or self.config.temperature,
            'max_tokens': max_tokens or self.config.max_tokens,
            'context': context,
            'status': 'pending'
        }
        
        # Notify monitors of request start
        for monitor in self.request_monitors:
            try:
                monitor('request_start', request_data)
            except Exception as e:
                logger.error(f"Monitor error on request start: {e}")
        
        try:
            # Make the actual request
            result = await super().chat_completion(messages, temperature, max_tokens, timeout)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Update request data with results
            request_data.update({
                'status': 'completed',
                'duration': duration,
                'response': result,
                'token_usage': result.get('token_usage', {}),
                'estimated_cost': result.get('estimated_cost', 0),
                'model': result.get('model', self.config.model)
            })
            
            # Store in history
            self._add_to_history(request_data)
            
            # Notify monitors of request completion
            for monitor in self.request_monitors:
                try:
                    monitor('request_complete', request_data)
                except Exception as e:
                    logger.error(f"Monitor error on request complete: {e}")
            
            return result
            
        except Exception as e:
            # Log error
            duration = time.time() - start_time
            request_data.update({
                'status': 'failed',
                'duration': duration,
                'error': str(e)
            })
            
            # Store in history
            self._add_to_history(request_data)
            
            # Notify monitors of request failure
            for monitor in self.request_monitors:
                try:
                    monitor('request_failed', request_data)
                except Exception as e2:
                    logger.error(f"Monitor error on request failed: {e2}")
            
            # Re-raise original exception
            raise
    
    def analyze_with_response(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        model_override: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Enhanced synchronous analysis method with monitoring.
        
        Args:
            prompt: The analysis prompt
            temperature: Sampling temperature
            model_override: Override the default model for this request
            metadata: Additional metadata for tracking
            
        Returns:
            The GPT response content as string
        """
        # Handle async context properly
        try:
            loop = asyncio.get_running_loop()
            # We're already in an async context, use nest_asyncio
            import nest_asyncio
            nest_asyncio.apply()
        except RuntimeError:
            # No event loop running, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            should_close_loop = True
        else:
            should_close_loop = False
        
        # Temporarily override model if specified
        original_model = self.config.model
        if model_override:
            self.config.model = model_override
            # Update encoding for new model
            self.encoding = self._get_encoding()
        
        try:
            # Run async method synchronously
            messages = [{"role": "user", "content": prompt}]
            if should_close_loop:
                result = loop.run_until_complete(
                    self.chat_completion(messages, temperature=temperature, metadata=metadata)
                )
            else:
                # Use asyncio.run_coroutine_threadsafe for existing loop
                future = asyncio.run_coroutine_threadsafe(
                    self.chat_completion(messages, temperature=temperature, metadata=metadata),
                    loop
                )
                result = future.result(timeout=self.config.timeout_seconds)
            
            return result['content']
        finally:
            # Restore original model
            if model_override:
                self.config.model = original_model
                self.encoding = self._get_encoding()
            
            # Close loop if we created it
            if should_close_loop and not loop.is_closed():
                loop.close()
    
    def _add_to_history(self, request_data: Dict[str, Any]):
        """Add request to history with size limit"""
        self.request_history.append(request_data)
        
        # Trim history if too large
        if len(self.request_history) > self.max_history_size:
            self.request_history = self.request_history[-self.max_history_size:]
    
    def get_request_history(
        self,
        limit: Optional[int] = None,
        agent_type: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get request history with optional filters.
        
        Args:
            limit: Maximum number of requests to return
            agent_type: Filter by agent type
            status: Filter by status (pending, completed, failed)
            
        Returns:
            List of request data dictionaries
        """
        history = self.request_history
        
        # Apply filters
        if agent_type:
            history = [r for r in history if r.get('agent_type') == agent_type]
        
        if status:
            history = [r for r in history if r.get('status') == status]
        
        # Apply limit
        if limit:
            history = history[-limit:]
        
        return history
    
    def get_usage_stats_by_agent(self) -> Dict[str, Dict[str, Any]]:
        """Get usage statistics grouped by agent type"""
        stats = {}
        
        for request in self.request_history:
            agent = request.get('agent_type', 'Unknown')
            
            if agent not in stats:
                stats[agent] = {
                    'total_requests': 0,
                    'completed_requests': 0,
                    'failed_requests': 0,
                    'total_tokens': 0,
                    'total_cost': 0.0,
                    'avg_duration': 0.0,
                    'durations': []
                }
            
            stats[agent]['total_requests'] += 1
            
            if request.get('status') == 'completed':
                stats[agent]['completed_requests'] += 1
                
                # Token usage
                token_usage = request.get('token_usage', {})
                stats[agent]['total_tokens'] += token_usage.get('total_tokens', 0)
                
                # Cost
                stats[agent]['total_cost'] += request.get('estimated_cost', 0)
                
                # Duration
                if request.get('duration'):
                    stats[agent]['durations'].append(request['duration'])
            
            elif request.get('status') == 'failed':
                stats[agent]['failed_requests'] += 1
        
        # Calculate average durations
        for agent, data in stats.items():
            if data['durations']:
                data['avg_duration'] = sum(data['durations']) / len(data['durations'])
            del data['durations']  # Remove raw durations
        
        return stats
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary across all requests"""
        total_cost = 0.0
        cost_by_model = {}
        cost_by_agent = {}
        
        for request in self.request_history:
            if request.get('status') == 'completed':
                cost = request.get('estimated_cost', 0)
                total_cost += cost
                
                # By model
                model = request.get('model', 'unknown')
                cost_by_model[model] = cost_by_model.get(model, 0) + cost
                
                # By agent
                agent = request.get('agent_type', 'Unknown')
                cost_by_agent[agent] = cost_by_agent.get(agent, 0) + cost
        
        return {
            'total_cost': total_cost,
            'cost_by_model': cost_by_model,
            'cost_by_agent': cost_by_agent,
            'request_count': len(self.request_history),
            'avg_cost_per_request': total_cost / len(self.request_history) if self.request_history else 0
        }
    
    def clear_history(self):
        """Clear request history"""
        self.request_history.clear()
        
        # Notify monitors
        for monitor in self.request_monitors:
            try:
                monitor('history_cleared', {})
            except Exception as e:
                logger.error(f"Monitor error on history clear: {e}")


# Export main class
__all__ = ['MonitoredGPTClient']