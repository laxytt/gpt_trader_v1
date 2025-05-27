"""
Centralized configuration management for the GPT Trading System.
Uses Pydantic for validation and environment variable handling.
"""

import os
from typing import List, Dict, Optional
from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings
from pathlib import Path

# Import symbol configurations
from .symbols import get_symbols_by_group, get_all_supported_symbols


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
    
    api_key: str = Field(..., description="OpenAI API key")
    model: str = Field("gpt-4.1-2025-04-14", description="GPT model to use")
    max_retries: int = Field(3, description="Maximum retry attempts for API calls")
    timeout_seconds: int = Field(60, description="Request timeout in seconds")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens per request")
    temperature: float = Field(0.1, description="Temperature for GPT responses")


class TradingSettings(BaseSettings):
    """Trading system configuration"""
    symbols: List[str] = Field(
        default_factory=lambda: get_symbols_by_group("conservative"),
        description="List of symbols to trade"
    )
    start_hour: int = Field(7, ge=0, le=23, description="Trading start hour (UTC)")
    end_hour: int = Field(23, ge=0, le=23, description="Trading end hour (UTC)")
    risk_per_trade_percent: float = Field(1.5, gt=0, le=10, description="Risk per trade as percentage")
    max_open_trades: int = Field(5, gt=0, description="Maximum concurrent open trades")
    bars_for_analysis: int = Field(80, gt=0, description="Number of bars for chart analysis")
    bars_for_json: int = Field(20, gt=0, description="Number of bars for JSON data")
    
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


class TelegramSettings(BaseSettings):
    """Telegram notification configuration"""
    model_config = ConfigDict(env_prefix="TELEGRAM_", extra="ignore")
    
    token: Optional[str] = Field(None, description="Telegram bot token")
    chat_id: Optional[str] = Field(None, description="Telegram chat ID")
    enabled: bool = Field(False, description="Enable Telegram notifications")
    
    @field_validator('enabled')
    @classmethod
    def validate_telegram_config(cls, v, info):
        if v and (not info.data.get('token') or not info.data.get('chat_id')):
            raise ValueError('Telegram token and chat_id required when enabled=True')
        return v


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
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    
    # Global settings
    debug: bool = Field(False, description="Enable debug mode")
    log_level: str = Field("INFO", description="Logging level")
    
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


# Global settings instance
def get_settings() -> Settings:
    """Get the global settings instance"""
    if not hasattr(get_settings, '_instance'):
        get_settings._instance = Settings()
    return get_settings._instance


# Export commonly used settings
__all__ = [
    'Settings',
    'MT5Settings', 
    'GPTSettings',
    'TradingSettings',
    'DatabaseSettings',
    'NewsSettings',
    'TelegramSettings',
    'PathSettings',
    'get_settings'
]