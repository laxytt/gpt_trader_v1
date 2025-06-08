"""
Pytest configuration and fixtures for integration tests.
"""

import pytest
import asyncio
import os
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone
import json

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Settings, get_settings
from core.domain.models import MarketData, Candle, TradingSignal, SignalType, Trade
from core.infrastructure.mt5.client import MT5Client
from core.infrastructure.database.repositories import TradeRepository, SignalRepository


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="gpt_trader_test_")
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_settings(test_data_dir):
    """Create test settings with temporary directories."""
    settings = Settings()
    
    # Override paths to use test directory
    settings.paths.data_dir = test_data_dir / "data"
    settings.paths.logs_dir = test_data_dir / "logs"
    settings.paths.screenshots_dir = test_data_dir / "screenshots"
    settings.database.db_path = str(test_data_dir / "test.db")
    
    # Ensure directories exist
    settings.paths.data_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.screenshots_dir.mkdir(parents=True, exist_ok=True)
    
    # Test configuration
    settings.trading.symbols = ["EURUSD", "GBPUSD"]
    settings.trading.max_open_trades = 2
    settings.trading.risk_per_trade_percent = 1.0
    settings.debug = True
    
    return settings


@pytest.fixture
def mock_mt5_client():
    """Create a mock MT5 client."""
    client = Mock(spec=MT5Client)
    
    # Mock methods
    client.initialize.return_value = True
    client.is_initialized.return_value = True
    client.ensure_connection.return_value = True
    
    # Mock account info
    client.get_account_info.return_value = {
        'balance': 10000.0,
        'equity': 10000.0,
        'margin_level': 0.0,
        'login': 12345
    }
    
    # Mock symbol info
    client.get_symbol_info.return_value = {
        'bid': 1.0800,
        'ask': 1.0801,
        'digits': 5,
        'trade_contract_size': 100000,
        'volume_min': 0.01,
        'volume_max': 100.0,
        'volume_step': 0.01,
        'point': 0.00001
    }
    
    # Mock tick data
    client.get_symbol_tick.return_value = {
        'bid': 1.0800,
        'ask': 1.0801,
        'time': datetime.now(timezone.utc).timestamp()
    }
    
    # Mock spread
    client.get_spread.return_value = 1.0
    
    return client


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    candles = []
    base_price = 1.0800
    
    for i in range(100):
        open_price = base_price + (i % 10) * 0.0001
        high = open_price + 0.0002
        low = open_price - 0.0001
        close = open_price + 0.0001
        
        candle = Candle(
            timestamp=datetime.now(timezone.utc),
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=1000.0 + i * 10,
            spread=1.0,
            atr14=0.0010,
            rsi14=50.0 + (i % 20) - 10,
            ma20=base_price,
            ma50=base_price - 0.0005,
            bb_upper=base_price + 0.0020,
            bb_middle=base_price,
            bb_lower=base_price - 0.0020,
            macd=0.0001,
            macd_signal=0.0000,
            macd_histogram=0.0001
        )
        candles.append(candle)
    
    return MarketData(
        symbol="EURUSD",
        timeframe="H1",
        candles=candles
    )


@pytest.fixture
def mock_gpt_client():
    """Create a mock GPT client."""
    client = AsyncMock()
    
    # Mock signal generation response
    client.generate_signal.return_value = {
        'signal': 'BUY',
        'confidence': 75,
        'reasoning': 'Test reasoning',
        'entry': 1.0800,
        'stop_loss': 1.0750,
        'take_profit': 1.0850
    }
    
    # Mock analysis response
    client.analyze_with_response.return_value = json.dumps({
        'RECOMMENDATION': 'BUY',
        'CONFIDENCE': '75',
        'ENTRY': '1.0800',
        'STOP_LOSS': '1.0750',
        'TAKE_PROFIT': '1.0850',
        'REASONING': 'Test analysis'
    })
    
    return client


@pytest.fixture
async def test_database(test_settings):
    """Create test database with schema."""
    from core.infrastructure.database.migrations import run_migrations
    
    # Run migrations to create schema
    run_migrations(test_settings.database.db_path)
    
    yield test_settings.database.db_path
    
    # Cleanup is handled by test_data_dir fixture


@pytest.fixture
def trade_repository(test_database):
    """Create trade repository with test database."""
    return TradeRepository(test_database)


@pytest.fixture
def signal_repository(test_database):
    """Create signal repository with test database."""
    return SignalRepository(test_database)


@pytest.fixture
def sample_signal():
    """Create a sample trading signal."""
    return TradingSignal(
        symbol="EURUSD",
        signal=SignalType.BUY,
        entry_price=1.0800,
        stop_loss=1.0750,
        take_profit=1.0850,
        confidence=75.0,
        risk_class="B",
        reason="Test signal",
        metadata={'test': True}
    )


@pytest.fixture
def sample_trade():
    """Create a sample trade."""
    return Trade(
        signal_id="test-signal-123",
        symbol="EURUSD",
        side="BUY",
        entry_price=1.0800,
        stop_loss=1.0750,
        take_profit=1.0850,
        position_size=0.1,
        ticket=12345,
        status="OPEN",
        open_time=datetime.now(timezone.utc)
    )


@pytest.fixture
def mock_news_service():
    """Create a mock news service."""
    service = AsyncMock()
    
    service.get_filtered_news.return_value = [
        {
            'title': 'Test News',
            'impact': 'high',
            'currency': 'USD',
            'forecast': '1.5%',
            'previous': '1.2%'
        }
    ]
    
    service.get_data_file_status.return_value = {
        'file_exists': True,
        'file_age_hours': 1.0
    }
    
    return service


@pytest.fixture
def mock_memory_service():
    """Create a mock memory service."""
    service = AsyncMock()
    
    service.search_similar_cases.return_value = []
    service.add_trade_case = AsyncMock()
    service.get_memory_stats.return_value = {
        'total_cases': 0,
        'index_size': 0
    }
    
    return service


@pytest.fixture
def mock_health_monitor():
    """Create a mock health monitor."""
    from core.services.health_monitor import SystemHealth, HealthStatus
    
    monitor = AsyncMock()
    
    monitor.check_health.return_value = SystemHealth(
        status=HealthStatus.HEALTHY,
        metrics={}
    )
    
    return monitor


# Environment setup fixtures

@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Setup test environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    monkeypatch.setenv("MT5_FILES_DIR", "/tmp/mt5_files")
    monkeypatch.setenv("MARKETAUX_TOKEN", "test-marketaux-token")


@pytest.fixture
def disable_external_calls(monkeypatch):
    """Disable all external API calls for testing."""
    # Disable OpenAI calls
    monkeypatch.setattr("openai.OpenAI", Mock)
    
    # Disable MT5
    monkeypatch.setattr("MetaTrader5.initialize", Mock(return_value=True))
    monkeypatch.setattr("MetaTrader5.shutdown", Mock())
    
    # Disable network calls
    monkeypatch.setattr("aiohttp.ClientSession", AsyncMock)
    monkeypatch.setattr("requests.get", Mock)
    monkeypatch.setattr("requests.post", Mock)