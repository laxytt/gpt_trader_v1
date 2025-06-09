"""
Main entry point for the GPT Trading System.
Initializes all components and starts the trading orchestrator.
"""

import asyncio
import logging
import sys
import signal
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings
from core.domain.exceptions import (
    TradingSystemError, ConfigurationError, MT5ConnectionError, MT5InitializationError,
    GPTAPIError, DatabaseError, InitializationError, ServiceNotAvailableError,
    AuthenticationError, NetworkError
)
from core.infrastructure.mt5.client import MT5Client
from core.infrastructure.mt5.data_provider import MT5DataProvider
from core.infrastructure.mt5.order_manager import MT5OrderManager
from core.infrastructure.gpt.client import GPTClient
from core.infrastructure.gpt.signal_generator import GPTSignalGenerator
from core.infrastructure.database.repositories import (
    TradeRepository, SignalRepository, MemoryCaseRepository
)
from core.services.council_signal_service import CouncilSignalService
from core.services.enhanced_council_signal_service import EnhancedCouncilSignalService
from core.services.trade_service import TradeService
from core.services.news_service import NewsService
from core.services.enhanced_news_service import EnhancedNewsService
from core.services.memory_service import MemoryService
from core.services.market_service import MarketService
from core.services.trading_orchestrator import TradingOrchestrator
from core.services.model_management_service import ModelManagementService, ModelRepository
from core.services.portfolio_risk_service import PortfolioRiskManager
from core.infrastructure.gpt.request_logger import get_request_logger
from core.infrastructure.gpt.rate_limiter import get_rate_limiter
from core.utils.chart_utils import ChartGenerator
from core.infrastructure.notifications.telegram import TelegramNotifier
from core.infrastructure.marketaux import MarketAuxClient
from core.services.health_monitor import HealthMonitor
from core.utils.async_task_manager import get_task_manager, shutdown_tasks
from core.utils.structured_logger import setup_logging as setup_structured_logging, get_logger


logger = get_logger(__name__)


class DependencyContainer:
    """
    Dependency injection container for the trading system.
    Manages component creation and dependency resolution.
    """
    
    def __init__(self, settings):
        self.settings = settings
        self._instances = {}
    
    def get_or_create(self, component_name: str, factory_func):
        """Get existing instance or create new one"""
        if component_name not in self._instances:
            try:
                self._instances[component_name] = factory_func()
                logger.info(f"✅ Created component: {component_name}")
            except MT5InitializationError:
                raise  # Re-raise MT5 specific errors
            except ConnectionError as e:
                logger.error(f"❌ Connection error creating {component_name}: {e}")
                raise NetworkError(f"Failed to create {component_name} due to network error") from e
            except (KeyError, ValueError) as e:
                logger.error(f"❌ Configuration error creating {component_name}: {e}")
                raise ConfigurationError(f"Invalid configuration for {component_name}: {str(e)}") from e
            except Exception as e:
                logger.error(f"❌ Failed to create {component_name}: {e}")
                raise InitializationError(f"Failed to initialize {component_name}: {str(e)}") from e
        return self._instances[component_name]
    
    # Infrastructure Components
    
    def mt5_client(self) -> MT5Client:
        """Create MT5 client"""
        return self.get_or_create('mt5_client', 
            lambda: MT5Client(self.settings.mt5))
    
    def gpt_client(self) -> GPTClient:
        """Create GPT client with production enhancements"""
        def create_gpt_client():
            client = GPTClient(self.settings.gpt)
            # Configure rate limiter tier
            rate_limiter = get_rate_limiter(
                tier=self.settings.openai_tier,
                safety_margin=self.settings.rate_limit_safety_margin
            )
            logger.info(f"GPT Client configured with OpenAI tier: {self.settings.openai_tier}")
            return client
            
        return self.get_or_create('gpt_client', create_gpt_client)
    
    def chart_generator(self) -> ChartGenerator:
        """Create chart generator"""
        return self.get_or_create('chart_generator',
            lambda: ChartGenerator())
    
    def telegram_notifier(self) -> TelegramNotifier:
        """Create Telegram notifier"""
        if not self.settings.is_telegram_enabled:
            return None
        
        return self.get_or_create('telegram_notifier',
            lambda: TelegramNotifier(self.settings.telegram))
    
    # Data Layer Components
    
    def mt5_data_provider(self) -> MT5DataProvider:
        """Create MT5 data provider"""
        return self.get_or_create('mt5_data_provider',
            lambda: MT5DataProvider(
                self.mt5_client(),
                self.chart_generator()
            ))
    
    def mt5_order_manager(self) -> MT5OrderManager:
        """Create MT5 order manager"""
        return self.get_or_create('mt5_order_manager',
            lambda: MT5OrderManager(
                self.mt5_client(),
                self.settings.trading
            ))
    
    def trade_repository(self) -> TradeRepository:
        """Create trade repository"""
        return self.get_or_create('trade_repository',
            lambda: TradeRepository(self.settings.database.db_path))
    
    def signal_repository(self) -> SignalRepository:
        """Create signal repository"""
        return self.get_or_create('signal_repository',
            lambda: SignalRepository(self.settings.database.db_path))
    
    def marketaux_client(self) -> MarketAuxClient:
        """Create MarketAux client if enabled"""
        if not self.settings.marketaux.enabled:
            return None
        
        return self.get_or_create('marketaux_client',
            lambda: MarketAuxClient(
                api_token=self.settings.marketaux.api_token,
                cache_db_path=self.settings.paths.data_dir / "marketaux_cache.db",
                daily_limit=self.settings.marketaux.daily_limit,
                requests_per_minute=self.settings.marketaux.requests_per_minute
            ))
    
    def memory_case_repository(self) -> MemoryCaseRepository:
        """Create memory case repository"""
        return self.get_or_create('memory_case_repository',
            lambda: MemoryCaseRepository(self.settings.database.db_path))
    
    def model_repository(self) -> ModelRepository:
        """Create model repository"""
        return self.get_or_create('model_repository',
            lambda: ModelRepository(self.settings.database.db_path))
    
    # GPT Components
    
    def signal_generator(self) -> GPTSignalGenerator:
        """Create GPT signal generator"""
        return self.get_or_create('signal_generator',
            lambda: GPTSignalGenerator(
                self.gpt_client(),
                prompts_dir="config/prompts"
            ))
    
    # Service Layer Components
    
    def news_service(self) -> NewsService:
        """Create news service"""
        # Use enhanced news service if MarketAux is enabled
        if self.settings.marketaux.enabled:
            return self.get_or_create('news_service',
                lambda: EnhancedNewsService(
                    news_config=self.settings.news,
                    marketaux_config=self.settings.marketaux,
                    db_path=self.settings.paths.base_dir / "data"
                ))
        else:
            return self.get_or_create('news_service',
                lambda: NewsService(self.settings.news))
    
    def memory_service(self) -> MemoryService:
        """Create memory service"""
        return self.get_or_create('memory_service',
            lambda: MemoryService(
                self.memory_case_repository(),
                self.settings.database
            ))
    
    def market_service(self) -> MarketService:
        """Create market service"""
        return self.get_or_create('market_service',
            lambda: MarketService(
                self.mt5_data_provider(),
                self.mt5_client()
            ))
    
    def model_management_service(self) -> ModelManagementService:
        """Create model management service"""
        return self.get_or_create('model_management_service',
            lambda: ModelManagementService(
                models_dir=self.settings.paths.base_dir / "models",
                model_repository=self.model_repository()
            ))
    
    def portfolio_risk_manager(self) -> PortfolioRiskManager:
        """Create portfolio risk manager"""
        return self.get_or_create('portfolio_risk_manager',
            lambda: PortfolioRiskManager(
                trade_repository=self.trade_repository(),
                mt5_client=self.mt5_client(),
                trading_config=self.settings.trading
            ))
    
    def signal_service(self) -> CouncilSignalService:
        """Create council signal service"""
        # Use enhanced council signal service if MarketAux is enabled
        if self.settings.marketaux.enabled:
            return self.get_or_create('signal_service',
                lambda: EnhancedCouncilSignalService(
                    data_provider=self.mt5_data_provider(),
                    gpt_client=self.gpt_client(),
                    enhanced_news_service=self.news_service(),  # This will be EnhancedNewsService
                    memory_service=self.memory_service(),
                    signal_repository=self.signal_repository(),
                    trading_config=self.settings.trading,
                    marketaux_config=self.settings.marketaux,
                    chart_generator=self.chart_generator(),
                    screenshots_dir=str(self.settings.paths.screenshots_dir),
                    enable_offline_validation=True,  # Always enabled for council
                    account_balance=getattr(self.settings.mt5, 'initial_balance', 10000),
                    min_confidence_threshold=self.settings.trading.council_min_confidence
                ))
        else:
            return self.get_or_create('signal_service',
                lambda: CouncilSignalService(
                    data_provider=self.mt5_data_provider(),
                    gpt_client=self.gpt_client(),
                    news_service=self.news_service(),
                    memory_service=self.memory_service(),
                    signal_repository=self.signal_repository(),
                    trading_config=self.settings.trading,
                    chart_generator=self.chart_generator(),
                    screenshots_dir=str(self.settings.paths.screenshots_dir),
                    enable_offline_validation=True,  # Always enabled for council
                    account_balance=getattr(self.settings.mt5, 'initial_balance', 10000),
                    min_confidence_threshold=self.settings.trading.council_min_confidence
                ))
    
    def trade_service(self) -> TradeService:
        """Create trade service"""
        return self.get_or_create('trade_service',
            lambda: TradeService(
                order_manager=self.mt5_order_manager(),
                trade_repository=self.trade_repository(),
                news_service=self.news_service(),
                memory_service=self.memory_service(),
                gpt_client=self.gpt_client(),
                trading_config=self.settings.trading,
                portfolio_risk_manager=self.portfolio_risk_manager()
            ))
    
    def health_monitor(self) -> HealthMonitor:
        """Create health monitor"""
        return self.get_or_create('health_monitor',
            lambda: HealthMonitor(
                settings=self.settings,
                mt5_client=self.mt5_client(),
                gpt_client=self.gpt_client(),
                marketaux_client=self.marketaux_client() if self.settings.marketaux.enabled else None
            ))
    
    def trading_orchestrator(self) -> TradingOrchestrator:
        """Create trading orchestrator"""
        return self.get_or_create('trading_orchestrator',
            lambda: TradingOrchestrator(
                signal_service=self.signal_service(),
                trade_service=self.trade_service(),
                news_service=self.news_service(),
                memory_service=self.memory_service(),
                market_service=self.market_service(),
                mt5_client=self.mt5_client(),
                trading_config=self.settings.trading,
                settings=self.settings,
                health_monitor=self.health_monitor()
            ))


class LoggingSetup:
    """Sets up logging configuration for the trading system"""
    
    @staticmethod
    def setup_logging(settings):
        """Configure enhanced structured logging"""
        # Use the new structured logging setup
        setup_structured_logging(
            settings={
                'log_level': settings.log_level.upper(),
                'production_mode': not settings.debug,
                'paths': {
                    'logs_dir': str(settings.paths.logs_dir)
                }
            },
            log_dir=settings.paths.logs_dir,
            structured_logs=True,
            async_handler=True
        )
        
        logger.info(f"Enhanced logging configured - Level: {settings.log_level}")


class TradingSystemBootstrap:
    """
    Main bootstrap class that initializes and starts the trading system.
    """
    
    def __init__(self):
        self.settings = None
        self.container = None
        self.orchestrator = None
        self.notification_service = None
        self._shutdown_requested = False
    
    async def initialize(self):
        """Initialize the trading system"""
        try:
            logger.info("🚀 Initializing GPT Trading System")
            
            # Load configuration
            self.settings = get_settings()
            logger.info(f"📊 Configuration loaded - Debug: {self.settings.debug}")
            
            # Setup logging
            LoggingSetup.setup_logging(self.settings)
            
            # Create dependency container
            self.container = DependencyContainer(self.settings)
            
            # Initialize notification service
            self.notification_service = self.container.telegram_notifier()
            if self.notification_service:
                await self.notification_service.send_message("🚀 GPT Trading System starting up...")
            
            # Validate critical components
            await self._validate_system()
            
            # Create orchestrator
            self.orchestrator = self.container.trading_orchestrator()
            
            # Initialize request logger cleanup
            if self.settings.log_gpt_requests:
                request_logger = get_request_logger()
                request_logger.cleanup_old_requests(days=7)
                logger.info("📝 GPT request logging enabled")
            
            logger.info("✅ System initialization complete")
            
        except ConfigurationError:
            raise  # Re-raise configuration errors as-is
        except MT5ConnectionError as e:
            logger.error(f"❌ MT5 connection failed during initialization: {e}")
            if self.notification_service:
                await self.notification_service.send_message(f"❌ MT5 connection failed: {str(e)}")
            raise
        except DatabaseError as e:
            logger.error(f"❌ Database initialization failed: {e}")
            if self.notification_service:
                await self.notification_service.send_message(f"❌ Database error: {str(e)}")
            raise
        except Exception as e:
            logger.exception(f"❌ Unexpected initialization error: {e}")
            if self.notification_service:
                await self.notification_service.send_message(f"❌ System initialization failed: {str(e)}")
            raise InitializationError(f"System initialization failed: {str(e)}") from e
    
    async def _validate_system(self):
        """Validate critical system components"""
        logger.info("🔍 Validating system components...")
        
        # Test MT5 connection
        mt5_client = self.container.mt5_client()
        if not mt5_client.initialize():
            raise ConfigurationError("MT5 connection failed")
        
        # Test GPT client
        gpt_client = self.container.gpt_client()
        # Could add a simple test call here
        
        # Test database access
        trade_repo = self.container.trade_repository()
        # Database tables are created automatically
        
        # Validate trading symbols
        for symbol in self.settings.trading.symbols:
            if not mt5_client.get_symbol_info(symbol):
                logger.warning(f"⚠️ Symbol {symbol} not available in MT5")
        
        # Check MarketAux if enabled
        if self.settings.marketaux.enabled:
            marketaux_client = self.container.marketaux_client()
            if marketaux_client:
                logger.info("✅ MarketAux client initialized")
            else:
                logger.warning("⚠️ MarketAux enabled but client failed to initialize")
        
        logger.info("✅ System validation complete")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        self._shutdown_attempts = 0
        
        def signal_handler(sig, frame):
            self._shutdown_attempts += 1
            
            if self._shutdown_attempts == 1:
                logger.info(f"🛑 Received signal {sig}, initiating graceful shutdown...")
                self._shutdown_requested = True
                if self.orchestrator:
                    self.orchestrator._shutdown_requested = True
                    # Cancel any running GPT requests
                    if hasattr(self.orchestrator, 'symbol_processor'):
                        self.orchestrator.symbol_processor._shutdown_requested = True
            elif self._shutdown_attempts == 2:
                logger.warning("⚠️ Second shutdown signal received, forcing exit in 5 seconds...")
                # Force exit after delay
                import threading
                threading.Timer(5.0, lambda: os._exit(1)).start()
            else:
                logger.error("💥 Third shutdown signal, forcing immediate exit!")
                os._exit(1)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def run(self):
        """Run the main trading system"""
        try:
            await self.initialize()
            self.setup_signal_handlers()
            
            logger.info("🎯 Starting trading orchestrator...")
            logger.info(f"📈 Symbols: {', '.join(self.settings.trading.symbols)}")
            logger.info(f"⏰ Trading hours: {self.settings.trading.start_hour:02d}:00 - {self.settings.trading.end_hour:02d}:00 UTC")
            logger.info(f"💰 Risk per trade: {self.settings.trading.risk_per_trade_percent}%")
            logger.info(f"🔢 Max open trades: {self.settings.trading.max_open_trades}")
            
            # Log MarketAux status
            if self.settings.marketaux.enabled:
                logger.info("📰 MarketAux integration: ENABLED")
                logger.info(f"   - Sentiment weight: {self.settings.marketaux.sentiment_weight}")
                logger.info(f"   - Daily limit: {self.settings.marketaux.daily_limit} requests")
            else:
                logger.info("📰 MarketAux integration: DISABLED")
            
            # Log production settings
            logger.info("⚙️ Production Configuration:")
            logger.info(f"   - OpenAI Tier: {self.settings.openai_tier}")
            logger.info(f"   - Rate Limit Safety: {self.settings.rate_limit_safety_margin * 100:.0f}%")
            logger.info(f"   - Agent Delay: {self.settings.trading.council_agent_delay}s")
            logger.info(f"   - Symbol Delay: {self.settings.trading.symbol_processing_delay}s")
            logger.info(f"   - Quick Mode: {self.settings.trading.council_quick_mode}")
            logger.info(f"   - GPT Request Logging: {self.settings.log_gpt_requests}")
            
            if self.notification_service:
                await self.notification_service.send_message(
                    f"✅ GPT Trading System started\n"
                    f"📈 Symbols: {', '.join(self.settings.trading.symbols[:3])}{'...' if len(self.settings.trading.symbols) > 3 else ''}\n"
                    f"⏰ Hours: {self.settings.trading.start_hour:02d}-{self.settings.trading.end_hour:02d} UTC"
                )
            
            # Run the orchestrator
            await self.orchestrator.run()
            
        except KeyboardInterrupt:
            logger.info("🛑 Keyboard interrupt received")
        except MT5ConnectionError as e:
            logger.error(f"💥 MT5 connection lost: {e}")
            if self.notification_service:
                await self.notification_service.send_message(f"💥 MT5 connection lost: {str(e)}")
            raise
        except GPTAPIError as e:
            logger.error(f"💥 GPT API error: {e}")
            if self.notification_service:
                await self.notification_service.send_message(f"💥 GPT API error: {str(e)}")
            raise
        except TradingSystemError:
            raise  # Re-raise known trading system errors
        except Exception as e:
            logger.exception(f"💥 Unexpected error in trading system: {e}")
            if self.notification_service:
                await self.notification_service.send_message(f"💥 Fatal error: {str(e)}")
            raise TradingSystemError(f"Unexpected trading system error: {str(e)}") from e
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown of the trading system"""
        logger.info("🔄 Shutting down GPT Trading System...")
        
        try:
            # Stop orchestrator if running
            if self.orchestrator:
                # Set the stop flag first
                self.orchestrator._shutdown_requested = True
                
                # Try to stop gracefully with a timeout
                try:
                    await asyncio.wait_for(self.orchestrator.stop(), timeout=10.0)
                except asyncio.TimeoutError:
                    logger.warning("⚠️ Orchestrator stop timed out, forcing shutdown")
            
            # Shutdown async tasks with a shorter timeout
            logger.info("📌 Waiting for background tasks to complete...")
            try:
                await asyncio.wait_for(shutdown_tasks(timeout=10.0), timeout=15.0)
            except asyncio.TimeoutError:
                logger.warning("⚠️ Background task shutdown timed out")
            
            # Cancel all remaining tasks
            tasks = [t for t in asyncio.all_tasks() if t != asyncio.current_task()]
            if tasks:
                logger.info(f"🛑 Cancelling {len(tasks)} remaining tasks...")
                for task in tasks:
                    task.cancel()
                # Wait briefly for cancellation
                try:
                    await asyncio.gather(*tasks, return_exceptions=True)
                except:
                    pass  # Ignore errors during cancellation
            
            # Send shutdown notification
            if self.notification_service:
                try:
                    await asyncio.wait_for(
                        self.notification_service.send_message("👋 GPT Trading System shutdown complete"),
                        timeout=2.0
                    )
                except asyncio.TimeoutError:
                    pass  # Don't wait for notification
            
            logger.info("👋 Shutdown complete")
            
        except asyncio.CancelledError:
            logger.info("Shutdown tasks cancelled")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            # Don't re-raise during shutdown to ensure cleanup completes


async def main():
    """Main entry point"""
    try:
        # Create and run the trading system
        system = TradingSystemBootstrap()
        await system.run()
        
    except ConfigurationError as e:
        logger.error(f"❌ Configuration error: {e}")
        sys.exit(1)
    except TradingSystemError as e:
        logger.error(f"❌ Trading system error: {e}")
        sys.exit(2)
    except Exception as e:
        logger.exception(f"💥 Unexpected error: {e}")
        sys.exit(3)


if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    # Run the trading system
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Trading system stopped by user")
        sys.exit(0)