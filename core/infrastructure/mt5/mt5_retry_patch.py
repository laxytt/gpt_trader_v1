"""
Patch to add retry logic to MT5Client methods
"""

from core.utils.retry_utils import retry_with_backoff, RetryConfig, MT5_DATA_RETRY_CONFIG

# Add these imports to MT5Client:
# from core.utils.retry_utils import retry_with_backoff, MT5_DATA_RETRY_CONFIG

# Update methods in MT5Client with decorators:

class MT5ClientRetryMethods:
    """Methods to be added/updated in MT5Client for retry logic"""
    
    @retry_with_backoff(config=MT5_DATA_RETRY_CONFIG)
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol information with retry logic"""
        # Existing implementation
        pass
    
    @retry_with_backoff(config=MT5_DATA_RETRY_CONFIG)
    def get_symbol_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current tick data with retry logic"""
        # Existing implementation
        pass
    
    @retry_with_backoff(config=RetryConfig(max_attempts=3, initial_delay=1.0))
    def get_rates(self, symbol: str, timeframe: str, start_pos: int = 0, count: int = 100) -> Optional[List[Dict]]:
        """Get historical rates with retry logic"""
        # Existing implementation
        pass