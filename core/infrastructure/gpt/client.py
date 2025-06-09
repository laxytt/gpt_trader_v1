"""
OpenAI GPT client for API communication and response handling.
Handles API calls, rate limiting, error handling, and response processing.
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import asyncio
import tiktoken
import uuid

import openai
from openai import OpenAI

from config.settings import GPTSettings
from core.domain.exceptions import (
    GPTAPIError, GPTResponseError, TimeoutError,
    ErrorContext, ErrorMessages
)
from core.domain.enums import GPTModels
from core.utils.circuit_breaker import gpt_circuit_breaker
from core.infrastructure.gpt.request_logger import GPTRequest, get_request_logger
from core.infrastructure.gpt.rate_limiter import get_rate_limiter
from core.utils.async_task_manager import get_task_manager


logger = logging.getLogger(__name__)


class GPTClient:
    """
    OpenAI GPT client with robust error handling and rate limiting.
    Provides a clean interface to GPT API with automatic retries and response validation.
    """
    
    def __init__(self, config: GPTSettings):
        self.config = config
        self.client = OpenAI(api_key=config.api_key)
        self.encoding = self._get_encoding()
        
        # Rate limiting and retry configuration
        self.request_count = 0
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum seconds between requests
        
        # Production components
        self.request_logger = get_request_logger()
        self.rate_limiter = get_rate_limiter()
        
        # Metadata for logging
        self.current_agent_type = None
        self.current_symbol = None
        self.current_request_type = None
        
    def _get_encoding(self) -> tiktoken.Encoding:
        """Get tiktoken encoding for token counting"""
        try:
            return tiktoken.encoding_for_model(self.config.model)
        except KeyError:
            logger.warning(f"Model {self.config.model} not found in tiktoken, using fallback")
            return tiktoken.get_encoding("cl100k_base")
    
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        agent_type: Optional[str] = None,
        symbol: Optional[str] = None,
        request_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send chat completion request to GPT API.
        
        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
            agent_type: Agent making the request (for logging)
            symbol: Trading symbol (for logging)
            request_type: Type of request (analyze, debate, synthesize)
            
        Returns:
            Dictionary containing response and metadata
            
        Raises:
            GPTAPIError: If API call fails
            GPTResponseError: If response is invalid
            TimeoutError: If request times out
        """
        # Set defaults
        temperature = temperature or self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens
        timeout = timeout or self.config.timeout_seconds
        
        # Update metadata for logging
        agent_type = agent_type or self.current_agent_type
        symbol = symbol or self.current_symbol
        request_type = request_type or self.current_request_type or 'unknown'
        
        # Create request ID for tracking
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        with ErrorContext("GPT API call") as ctx:
            ctx.add_detail("model", self.config.model)
            ctx.add_detail("message_count", len(messages))
            ctx.add_detail("request_id", request_id)
            ctx.add_detail("agent_type", agent_type)
            ctx.add_detail("symbol", symbol)
            
            # Estimate token usage
            input_tokens = self._estimate_tokens(messages)
            ctx.add_detail("estimated_input_tokens", input_tokens)
            
            # Apply production rate limiting
            priority = 2 if agent_type == "RiskManager" else 1  # Higher priority for risk manager
            acquired = await self.rate_limiter.wait_if_needed(
                model=self.config.model,
                estimated_tokens=input_tokens,
                max_wait=30.0,
                priority=priority
            )
            
            if not acquired:
                raise GPTAPIError("Rate limit timeout - unable to acquire tokens")
            
            # Prepare request parameters
            request_params = {
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature,
                "timeout": timeout
            }
            
            if max_tokens:
                request_params["max_tokens"] = max_tokens
            
            # Create GPT request for logging
            gpt_request = GPTRequest(
                id=request_id,
                timestamp=datetime.now(timezone.utc),
                model=self.config.model,
                agent_type=agent_type,
                symbol=symbol,
                request_type=request_type,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            try:
                # Execute request with retries
                response = await self._execute_with_retries(request_params)
                
                # Process and validate response
                result = self._process_response(response, input_tokens)
                
                # Update request with response data
                gpt_request.response_text = result['content']
                gpt_request.completion_tokens = result['token_usage']['completion_tokens']
                gpt_request.prompt_tokens = result['token_usage']['prompt_tokens']
                gpt_request.total_tokens = result['token_usage']['total_tokens']
                gpt_request.cost = result['estimated_cost']
                gpt_request.duration_ms = int((time.time() - start_time) * 1000)
                
                # Report actual usage to rate limiter
                self.rate_limiter.report_actual_usage(
                    model=self.config.model,
                    actual_tokens=result['token_usage']['total_tokens']
                )
                
                self.request_count += 1
                logger.info(f"GPT request completed - Agent: {agent_type}, "
                           f"Tokens: {result['token_usage']['total_tokens']}, "
                           f"Cost: ${result['estimated_cost']:.4f}, "
                           f"Duration: {gpt_request.duration_ms}ms")
                
                # Log request asynchronously with task tracking
                task_manager = get_task_manager()
                task_manager.create_task(
                    self.request_logger.log_request(gpt_request),
                    name=f"log_gpt_request_{gpt_request.id}",
                    error_handler=lambda e: logger.error(f"Failed to log GPT request: {e}")
                )
                
                return result
                
            except Exception as e:
                # Log error
                gpt_request.error = str(e)
                gpt_request.duration_ms = int((time.time() - start_time) * 1000)
                
                # Log error request with task tracking
                task_manager = get_task_manager()
                task_manager.create_task(
                    self.request_logger.log_request(gpt_request),
                    name=f"log_gpt_error_{gpt_request.id}",
                    error_handler=lambda e: logger.error(f"Failed to log GPT error: {e}")
                )
                raise
    
    async def _apply_rate_limiting(self):
        """Apply rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f} seconds")
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    @gpt_circuit_breaker
    async def _execute_with_retries(self, request_params: Dict[str, Any]) -> Any:
        """Execute API request with retry logic and circuit breaker protection"""
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                logger.debug(f"GPT API attempt {attempt + 1}/{self.config.max_retries}")
                
                # Update retry count in request params for logging
                if hasattr(self, '_current_gpt_request'):
                    self._current_gpt_request.retry_count = attempt
                
                # Make async call (simulate with sync for now since OpenAI doesn't have true async)
                response = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self.client.chat.completions.create(**request_params)
                )
                
                return response
                
            except openai.RateLimitError as e:
                last_exception = e
                wait_time = min(60, (2 ** attempt))  # Exponential backoff, max 60s
                logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}")
                await asyncio.sleep(wait_time)
                
            except openai.APITimeoutError as e:
                last_exception = e
                logger.warning(f"API timeout on attempt {attempt + 1}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    
            except openai.APIConnectionError as e:
                last_exception = e
                logger.warning(f"API connection error on attempt {attempt + 1}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    
            except openai.AuthenticationError as e:
                # Don't retry authentication errors
                raise GPTAPIError(ErrorMessages.GPT_API_KEY_MISSING, status_code=401) from e
                
            except openai.BadRequestError as e:
                # Don't retry bad request errors
                raise GPTAPIError(f"Bad request: {str(e)}", status_code=400) from e
                
            except Exception as e:
                last_exception = e
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        
        # All retries failed
        if isinstance(last_exception, openai.RateLimitError):
            raise GPTAPIError(ErrorMessages.GPT_QUOTA_EXCEEDED, status_code=429) from last_exception
        elif isinstance(last_exception, openai.APITimeoutError):
            raise TimeoutError(ErrorMessages.GPT_TIMEOUT) from last_exception
        else:
            raise GPTAPIError(f"GPT API failed after {self.config.max_retries} attempts") from last_exception
    
    def _process_response(self, response: Any, input_tokens: int) -> Dict[str, Any]:
        """Process and validate GPT response"""
        try:
            # Extract response content
            if not response.choices:
                raise GPTResponseError("No choices in GPT response")
            
            choice = response.choices[0]
            content = choice.message.content
            
            if not content:
                raise GPTResponseError("Empty content in GPT response")
            
            # Extract token usage
            usage = response.usage
            token_usage = {
                'prompt_tokens': usage.prompt_tokens if usage else input_tokens,
                'completion_tokens': usage.completion_tokens if usage else 0,
                'total_tokens': usage.total_tokens if usage else input_tokens
            }
            
            # Calculate estimated cost
            estimated_cost = self._calculate_cost(token_usage)
            
            return {
                'content': content,
                'model': response.model,
                'finish_reason': choice.finish_reason,
                'token_usage': token_usage,
                'estimated_cost': estimated_cost,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing GPT response: {e}")
            raise GPTResponseError(f"Failed to process GPT response: {str(e)}") from e
    
    def _estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate token count for messages"""
        try:
            total_tokens = 0
            
            for message in messages:
                # Count tokens in text content
                if isinstance(message.get('content'), str):
                    total_tokens += len(self.encoding.encode(message['content']))
                elif isinstance(message.get('content'), list):
                    # Handle multimodal content (text + images)
                    for content_item in message['content']:
                        if content_item.get('type') == 'text':
                            total_tokens += len(self.encoding.encode(content_item.get('text', '')))
                        elif content_item.get('type') == 'image_url':
                            # Rough estimate for image tokens (depends on image size)
                            total_tokens += 85  # Base cost for image processing
                
                # Add tokens for role and message structure
                total_tokens += 4  # Approximate overhead per message
            
            # Add overhead for conversation structure
            total_tokens += 3
            
            return total_tokens
            
        except Exception as e:
            logger.warning(f"Token estimation failed: {e}")
            return 1000  # Conservative fallback estimate
    
    def _calculate_cost(self, token_usage: Dict[str, int]) -> float:
        """Calculate estimated cost for API call"""
        model_config = GPTModels.MODELS.get(self.config.model, {})
        
        input_cost_per_1k = model_config.get('cost_per_1k_input', 0.01)
        output_cost_per_1k = model_config.get('cost_per_1k_output', 0.03)
        
        input_cost = (token_usage['prompt_tokens'] / 1000) * input_cost_per_1k
        output_cost = (token_usage['completion_tokens'] / 1000) * output_cost_per_1k
        
        return input_cost + output_cost
    
    def parse_json_response(self, content: str) -> Dict[str, Any]:
        """
        Parse JSON from GPT response content.
        
        Args:
            content: GPT response content
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            GPTResponseError: If JSON parsing fails
        """
        try:
            # Try to find JSON in the response
            content = content.strip()
            
            # Look for JSON block markers
            if '```json' in content:
                start = content.find('```json') + 7
                end = content.find('```', start)
                if end != -1:
                    content = content[start:end].strip()
            
            # Look for JSON object braces
            start = content.find('{')
            end = content.rfind('}')
            
            if start != -1 and end != -1 and end > start:
                json_str = content[start:end + 1]
                return json.loads(json_str)
            
            # Try parsing entire content as JSON
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.debug(f"Content to parse: {content[:500]}...")
            raise GPTResponseError(f"Invalid JSON in GPT response: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error parsing JSON: {e}")
            raise GPTResponseError(f"Failed to parse GPT response: {str(e)}")
    
    def validate_response_schema(
        self, 
        response_data: Dict[str, Any], 
        required_keys: List[str]
    ) -> bool:
        """
        Validate that response contains required keys.
        
        Args:
            response_data: Parsed response data
            required_keys: List of required keys
            
        Returns:
            bool: True if all required keys present
            
        Raises:
            GPTResponseError: If validation fails
        """
        missing_keys = [key for key in required_keys if key not in response_data]
        
        if missing_keys:
            raise GPTResponseError(f"Missing required keys in GPT response: {missing_keys}")
        
        return True
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this client session"""
        return {
            'total_requests': self.request_count,
            'model': self.config.model,
            'session_start': getattr(self, '_session_start', datetime.now(timezone.utc).isoformat())
        }
    
    def reset_stats(self):
        """Reset usage statistics"""
        self.request_count = 0
        self._session_start = datetime.now(timezone.utc).isoformat()
    
    def analyze_with_response(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        model_override: Optional[str] = None,
        agent_type: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> str:
        """
        Simple synchronous method for agent analysis
        
        Args:
            prompt: The analysis prompt
            temperature: Sampling temperature
            model_override: Override the default model for this request
            agent_type: Agent making the request
            symbol: Trading symbol
            
        Returns:
            The GPT response content as string
        """
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, use nest_asyncio to handle it
            import nest_asyncio
            nest_asyncio.apply()
        except RuntimeError:
            # No event loop running, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Temporarily override model if specified
        original_model = self.config.model
        if model_override:
            self.config.model = model_override
            # Update encoding for new model
            self.encoding = self._get_encoding()
        
        try:
            # Run async method synchronously
            messages = [{"role": "user", "content": prompt}]
            result = loop.run_until_complete(
                self.chat_completion(
                    messages, 
                    temperature=temperature,
                    agent_type=agent_type or self.current_agent_type,
                    symbol=symbol or self.current_symbol,
                    request_type='analyze'
                )
            )
            
            return result['content']
        finally:
            # Restore original model
            if model_override:
                self.config.model = original_model
                self.encoding = self._get_encoding()
            
            # Close the loop
            loop.close()
    
    def set_context(self, agent_type: Optional[str] = None, symbol: Optional[str] = None):
        """Set context for request logging"""
        if agent_type:
            self.current_agent_type = agent_type
        if symbol:
            self.current_symbol = symbol


# Export main class
__all__ = ['GPTClient']