"""
Centralized configuration management for the GPT Trading System.
Uses Pydantic for validation and environment variable handling.
"""

import os
from typing import List, Dict, Optional, Annotated, Any, Type
from pydantic import Field, field_validator, ConfigDict, BeforeValidator
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource
from pathlib import Path
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Fix for environment variables with inline comments
# Clean all environment variables by removing inline comments
for key, value in list(os.environ.items()):
    if isinstance(value, str) and '#' in value and (
        key.startswith('TRADING_') or 
        key.startswith('OPENAI_') or 
        key.startswith('ML_') or
        key.startswith('RATE_') or
        key.startswith('LOG_') or
        key.startswith('CIRCUIT_')
    ):
        # Remove inline comment
        clean_value = value.split('#')[0].strip()
        os.environ[key] = clean_value

# Import symbol configurations
from .symbols import get_symbols_by_group, get_all_supported_symbols

# Check for production settings file
if os.path.exists('config/production_settings.py'):
    from .production_settings import PRODUCTION_COUNCIL_CONFIG, RATE_LIMITER_CONFIG


class MT5Settings(BaseSettings):
    """MetaTrader 5 specific configuration"""
    model_config = ConfigDict(env_prefix="MT5_", extra="ignore")
    
    files_dir: str = Field(..., description="MT5 MQL5/Files directory path")
    magic_number: int = Field(10032024, description="Magic number for trade identification")
    timeout_seconds: int = Field(10, description="Timeout for operations in seconds")
    max_retries: int = Field(3, description="Maximum retry attempts")


class GPTSettings(BaseSettings):
    """OpenAI GPT configuration"""
    model_config = ConfigDict(env_prefix="OPENAI_", extra="ignore")
    
    api_key: str = Field(..., description="OpenAI API key", repr=False)  # repr=False prevents key from being printed
    model: str = Field("gpt-4o-mini", description="GPT model to use")
    risk_manager_model: str = Field("gpt-4-turbo-preview", description="GPT model for Risk Manager (veto power)")
    max_retries: int = Field(3, description="Maximum retry attempts for API calls")
    timeout_seconds: int = Field(120, description="Request timeout in seconds")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens per request")
    temperature: float = Field(0.1, description="Temperature for GPT responses")
    
    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v):
        """Validate API key format and prevent exposure"""
        if not v or len(v) < 20:
            raise ValueError("Invalid OpenAI API key format")
        # Mask the key for any logging
        masked = f"{v[:8]}...{v[-4:]}" if len(v) > 12 else "***"
        logger.debug(f"OpenAI API key configured: {masked}")
        return v


class TradingSettings(BaseSettings):
    """Trading system configuration"""
    model_config = ConfigDict(
        env_prefix="TRADING_", 
        extra="ignore"
    )
    
    symbols: List[str] = Field(
        default_factory=lambda: get_symbols_by_group("conservative"),
        description="List of symbols to trade"
    )
    
    @field_validator('symbols', mode='before')
    @classmethod
    def parse_symbols_from_string(cls, v):
        """Parse symbols from comma-separated string"""
        # If it's a string, parse it as comma-separated
        if isinstance(v, str) and not v.startswith('['):
            return [s.strip() for s in v.split(',') if s.strip()]
        # If it's already a list, return it
        elif isinstance(v, list):
            return v
        # For any other case, try default
        else:
            return get_symbols_by_group("conservative")
    start_hour: int = Field(7, ge=0, le=23, description="Trading start hour (UTC)")
    end_hour: int = Field(23, ge=0, le=23, description="Trading end hour (UTC)")
    risk_per_trade_percent: float = Field(1.5, gt=0, le=10, description="Risk per trade as percentage")
    max_open_trades: int = Field(3, gt=0, description="Maximum concurrent open trades")
    bars_for_analysis: int = Field(100, gt=0, description="Number of bars for chart analysis")
    bars_for_json: int = Field(20, gt=0, description="Number of bars for JSON data")
    cycle_interval_minutes: int = Field(60, gt=0, description="Minutes between trading cycles")
    offline_validation_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum validation score to proceed with GPT call")
    
    # Trading Council settings
    use_council: bool = Field(True, description="Use multi-agent Trading Council for decisions")
    council_quick_mode: bool = Field(True, description="Use quick mode (no debates, just initial analysis)")
    council_min_confidence: float = Field(75.0, ge=0.0, le=100.0, description="Minimum council confidence to trade")
    council_llm_weight: float = Field(0.7, ge=0.0, le=1.0, description="Weight for LLM confidence in hybrid scoring")
    council_ml_weight: float = Field(0.3, ge=0.0, le=1.0, description="Weight for ML confidence in hybrid scoring")
    council_debate_rounds: int = Field(1, ge=1, le=5, description="Number of debate rounds in council")
    council_agent_delay: float = Field(0.5, ge=0.0, le=10.0, description="Delay in seconds between agent calls")
    symbol_processing_delay: float = Field(2.0, ge=0.0, le=60.0, description="Delay between processing symbols")
    
    # Cache settings
    cache_enabled: bool = Field(True, description="Enable market state caching")
    cache_similarity_threshold: float = Field(0.85, ge=0.0, le=1.0, description="Minimum similarity to use cache")
    cache_ttl_minutes: int = Field(60, gt=0, description="Cache validity period in minutes")
    cache_size_mb: int = Field(500, gt=0, description="Maximum cache size in MB")
    
    @field_validator('end_hour')
    @classmethod
    def validate_trading_hours(cls, v, info):
        if 'start_hour' in info.data and v <= info.data['start_hour']:
            raise ValueError('end_hour must be greater than start_hour')
        return v
    
    @field_validator('symbols')
    @classmethod
    def validate_symbols(cls, v):
        """Validate that symbols are supported"""
        supported_symbols = get_all_supported_symbols()
        invalid_symbols = [s for s in v if s.upper() not in supported_symbols]
        if invalid_symbols:
            raise ValueError(f'Unsupported symbols: {invalid_symbols}. Supported: {supported_symbols[:10]}...')
        return [s.upper() for s in v]  # Normalize to uppercase


class DatabaseSettings(BaseSettings):
    """Database configuration"""
    model_config = ConfigDict(env_prefix="DB_", extra="ignore")
    
    db_path: str = Field("data/trades.db", description="SQLite database path")
    max_memory_cases: int = Field(300, gt=0, description="Maximum cases in RAG memory")
    embedding_model: str = Field("sentence-transformers/all-MiniLM-L6-v2", description="Sentence transformer model")


class NewsSettings(BaseSettings):
    """News filtering configuration"""
    model_config = ConfigDict(env_prefix="NEWS_", extra="ignore")
    
    file_path: str = Field("data/forexfactory_week.json", description="News data file path")
    lookahead_minutes: int = Field(2880, gt=0, description="News lookahead window in minutes")
    restriction_window_before: int = Field(2, ge=0, description="Minutes before news to restrict trading")
    restriction_window_after: int = Field(2, ge=0, description="Minutes after news to restrict trading")


class MarketAuxSettings(BaseSettings):
    """MarketAux API configuration"""
    model_config = ConfigDict(env_prefix="MARKETAUX_", extra="ignore")
    
    api_token: Optional[str] = Field(None, description="MarketAux API token", repr=False)  # Hide token
    enabled: bool = Field(False, description="Enable MarketAux integration")
    daily_limit: int = Field(100, gt=0, description="Daily API request limit")
    requests_per_minute: int = Field(5, gt=0, description="Requests per minute limit")
    cache_ttl_hours: int = Field(24, gt=0, description="Cache TTL in hours")
    min_relevance_score: float = Field(0.3, ge=0.0, le=1.0, description="Minimum relevance score for articles")
    sentiment_weight: float = Field(0.3, ge=0.0, le=1.0, description="Weight of sentiment in trading decisions")
    high_impact_only: bool = Field(False, description="Only use high-impact news")
    free_plan: bool = Field(True, description="Using MarketAux free plan (affects API strategy)")
    
    @field_validator('enabled')
    @classmethod
    def validate_marketaux_config(cls, v, info):
        if v and not info.data.get('api_token'):
            raise ValueError('MarketAux API token required when enabled=True')
        return v
    
    @field_validator('api_token')
    @classmethod
    def validate_token(cls, v):
        """Validate and mask token for logging"""
        if v and len(v) > 8:
            masked = f"{v[:4]}...{v[-4:]}"
            logger.debug(f"MarketAux API token configured: {masked}")
        return v


class TelegramSettings(BaseSettings):
    """Telegram notification configuration"""
    model_config = ConfigDict(env_prefix="TELEGRAM_", extra="ignore")
    
    token: Optional[str] = Field(None, description="Telegram bot token", repr=False)  # Hide token
    chat_id: Optional[str] = Field(None, description="Telegram chat ID")
    enabled: bool = Field(False, description="Enable Telegram notifications")
    
    @field_validator('enabled')
    @classmethod
    def validate_telegram_config(cls, v, info):
        if v and (not info.data.get('token') or not info.data.get('chat_id')):
            raise ValueError('Telegram token and chat_id required when enabled=True')
        return v
    
    @field_validator('token')
    @classmethod
    def validate_token(cls, v):
        """Validate and mask token for logging"""
        if v and len(v) > 8:
            masked = f"{v[:6]}...{v[-4:]}"
            logger.debug(f"Telegram bot token configured: {masked}")
        return v


class MLSettings(BaseSettings):
    """Machine Learning configuration"""
    model_config = ConfigDict(env_prefix="ML_", extra="ignore", protected_namespaces=())
    
    enabled: bool = Field(False, description="Enable ML-based signal generation")
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum confidence for ML signals")
    fallback_to_gpt: bool = Field(True, description="Use GPT when ML confidence is low")
    update_frequency_days: int = Field(30, gt=0, description="Days between model retraining")
    min_training_samples: int = Field(1000, gt=0, description="Minimum samples required for training")
    feature_lookback_periods: List[int] = Field(
        default=[5, 10, 20, 50], 
        description="Lookback periods for feature engineering"
    )
    use_ensemble: bool = Field(False, description="Use ensemble of models instead of single model")


class PathSettings(BaseSettings):
    """File and directory paths"""
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    screenshots_dir: Path = Field(default_factory=lambda: Path("screenshots"))
    logs_dir: Path = Field(default_factory=lambda: Path("logs"))
    
    @field_validator('data_dir', 'screenshots_dir', 'logs_dir')
    @classmethod
    def ensure_absolute_path(cls, v, info):
        if not v.is_absolute() and 'base_dir' in info.data:
            return info.data['base_dir'] / v
        return v


class Settings(BaseSettings):
    """Main settings container"""
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8", 
        case_sensitive=False,
        extra="ignore"
    )
    
    # Sub-configurations
    mt5: MT5Settings = Field(default_factory=MT5Settings)
    gpt: GPTSettings = Field(default_factory=GPTSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    news: NewsSettings = Field(default_factory=NewsSettings)
    marketaux: MarketAuxSettings = Field(default_factory=MarketAuxSettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    
    # Global settings
    debug: bool = Field(False, description="Enable debug mode")
    log_level: str = Field("INFO", description="Logging level")
    
    # Production settings
    openai_tier: str = Field("tier_1", description="OpenAI subscription tier")
    rate_limit_safety_margin: float = Field(0.8, ge=0.0, le=1.0, description="Safety margin for rate limits")
    log_gpt_requests: bool = Field(True, description="Log all GPT requests for dashboard")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories_exist()
    
    def _ensure_directories_exist(self):
        """Create necessary directories if they don't exist"""
        for dir_path in [self.paths.data_dir, self.paths.screenshots_dir, self.paths.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @property
    def is_telegram_enabled(self) -> bool:
        """Check if Telegram notifications are properly configured"""
        return (
            self.telegram.enabled and 
            self.telegram.token is not None and 
            self.telegram.chat_id is not None
        )


# Thread-safe global settings instance
import threading

_settings_lock = threading.Lock()
_settings_instance = None

def get_settings() -> Settings:
    """Get the global settings instance (thread-safe)"""
    global _settings_instance
    
    if _settings_instance is None:
        with _settings_lock:
            # Double-check locking pattern
            if _settings_instance is None:
                _settings_instance = Settings()
    
    return _settings_instance


# Export commonly used settings
__all__ = [
    'Settings',
    'MT5Settings', 
    'GPTSettings',
    'TradingSettings',
    'DatabaseSettings',
    'NewsSettings',
    'MarketAuxSettings',
    'TelegramSettings',
    'MLSettings',
    'PathSettings',
    'get_settings'
]