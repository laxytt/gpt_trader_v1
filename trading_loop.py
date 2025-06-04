"""
Main entry point for the GPT Trading System.
Initializes all components and starts the trading orchestrator.
"""

import asyncio
import logging
import sys
import signal
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings
from core.domain.exceptions import TradingSystemError, ConfigurationError
from core.infrastructure.mt5.client import MT5Client
from core.infrastructure.mt5.data_provider import MT5DataProvider
from core.infrastructure.mt5.order_manager import MT5OrderManager
from core.infrastructure.gpt.client import GPTClient
from core.infrastructure.gpt.signal_generator import GPTSignalGenerator
from core.infrastructure.database.repositories import (
    TradeRepository, SignalRepository, MemoryCaseRepository
)
from core.services.council_signal_service import CouncilSignalService
from core.services.trade_service import TradeService
from core.services.news_service import NewsService
from core.services.memory_service import MemoryService
from core.services.market_service import MarketService
from core.services.trading_orchestrator import TradingOrchestrator
from core.services.model_management_service import ModelManagementService, ModelRepository
from core.services.portfolio_risk_service import PortfolioRiskManager
from core.utils.chart_utils import ChartGenerator
from core.infrastructure.notifications.telegram import TelegramNotifier


logger = logging.getLogger(__name__)


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
            except Exception as e:
                logger.error(f"❌ Failed to create {component_name}: {e}")
                raise
        return self._instances[component_name]
    
    # Infrastructure Components
    
    def mt5_client(self) -> MT5Client:
        """Create MT5 client"""
        return self.get_or_create('mt5_client', 
            lambda: MT5Client(self.settings.mt5))
    
    def gpt_client(self) -> GPTClient:
        """Create GPT client"""
        return self.get_or_create('gpt_client',
            lambda: GPTClient(self.settings.gpt))
    
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
                settings=self.settings
            ))


class LoggingSetup:
    """Sets up logging configuration for the trading system"""
    
    @staticmethod
    def setup_logging(settings):
        """Configure logging based on settings"""
        log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
        
        # Create logs directory
        settings.paths.logs_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(
                    settings.paths.logs_dir / 'trading_system.log',
                    encoding='utf-8'
                )
            ]
        )
        
        # Set specific logger levels
        if not settings.debug:
            # Reduce noise from external libraries
            logging.getLogger('urllib3').setLevel(logging.WARNING)
            logging.getLogger('requests').setLevel(logging.WARNING)
            logging.getLogger('openai').setLevel(logging.WARNING)
            logging.getLogger('matplotlib').setLevel(logging.WARNING)
            logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
        
        logger.info(f"Logging configured - Level: {settings.log_level}")


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
            
            logger.info("✅ System initialization complete")
            
        except Exception as e:
            logger.exception(f"❌ System initialization failed: {e}")
            if self.notification_service:
                await self.notification_service.send_message(f"❌ System initialization failed: {str(e)}")
            raise
    
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
        
        logger.info("✅ System validation complete")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(sig, frame):
            logger.info(f"🛑 Received signal {sig}, initiating graceful shutdown...")
            self._shutdown_requested = True
            if self.orchestrator:
                self.orchestrator._shutdown_requested = True
                asyncio.create_task(self.orchestrator.stop())
        
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
        except Exception as e:
            logger.exception(f"💥 Fatal error in trading system: {e}")
            if self.notification_service:
                await self.notification_service.send_message(f"💥 Fatal error: {str(e)}")
            raise
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown of the trading system"""
        logger.info("🔄 Shutting down GPT Trading System...")
        
        try:
            # Stop orchestrator if running
            if self.orchestrator:
                await self.orchestrator.stop()
            
            # Send shutdown notification
            if self.notification_service:
                await self.notification_service.send_message("👋 GPT Trading System shutdown complete")
            
            logger.info("👋 Shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


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