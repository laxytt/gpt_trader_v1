import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from config import (
    TRADING_START_HOUR,
    TRADING_END_HOUR,
    SYMBOL_LIST,
    TELEGRAM_TOKEN,
    TELEGRAM_CHAT_ID
)
from core.database import init_db, get_trade_by_symbol
from core.mt5_utils import ensure_mt5_initialized, prefilter_instruments
from core.trade_cycle import trade_cycle
from core.trade_manager import manage_active_trade
from core.news_filter import is_news_restricted_now
from core.telegram_logger import TelegramLogger

# ========== LOGGER SETUP ==========
logger = logging.getLogger(__name__)

class LoggingConfigurator:
    """Configures logging for the trading system"""
    
    @staticmethod
    def setup_basic_logging():
        """Setup basic console logging"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)]
        )
    
    @staticmethod
    def setup_telegram_logging():
        """Setup Telegram logging if credentials available"""
        if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
            logger.info("🔔 Telegram logging enabled")
            sys.stdout = TelegramLogger(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
            return True
        return False
    
    @staticmethod
    def setup_exception_handling():
        """Setup global exception handler"""
        def global_except_hook(exctype, value, tb):
            logger.exception("Unhandled exception occurred", exc_info=(exctype, value, tb))
        
        sys.excepthook = global_except_hook

class TradingScheduler:
    """Manages trading schedule and timing"""
    
    def __init__(self, start_hour: int = TRADING_START_HOUR, end_hour: int = TRADING_END_HOUR):
        self.start_hour = start_hour
        self.end_hour = end_hour
    
    def is_trading_time(self) -> bool:
        """Check if current time is within trading hours"""
        now = datetime.now()
        return self.start_hour <= now.hour < self.end_hour
    
    def seconds_until_trading_start(self) -> float:
        """Calculate seconds until next trading session starts"""
        now = datetime.now()
        start_today = now.replace(
            hour=self.start_hour, 
            minute=0, 
            second=0, 
            microsecond=0
        )
        
        if now < start_today:
            return (start_today - now).total_seconds()
        
        # Next trading session is tomorrow
        start_tomorrow = start_today + timedelta(days=1)
        return (start_tomorrow - now).total_seconds()
    
    def wait_until_next_h1_boundary(self):
        """Wait until the next hour boundary for synchronized execution"""
        now = datetime.now()
        next_boundary = (now + timedelta(hours=1)).replace(
            minute=0, 
            second=0, 
            microsecond=0
        )
        delay = (next_boundary - now).total_seconds()
        
        logger.info(f"⏳ Sleeping {delay:.2f} seconds until {next_boundary.strftime('%H:%M:%S')}")
        time.sleep(delay)

class SymbolFilter:
    """Filters and selects symbols for trading"""
    
    def __init__(self, symbol_list: List[str]):
        self.symbol_list = symbol_list
    
    def get_candidate_symbols(self) -> List[str]:
        """Get symbols that pass the prefiltering criteria"""
        try:
            candidates = prefilter_instruments(self.symbol_list)
            logger.info(f"🔎 Candidates after ATR/vol filter: {candidates}")
            return candidates
        except Exception as e:
            logger.exception("Prefiltering failed, using fallback symbol")
            return ["EURUSD"]  # Safe fallback
    
    def is_symbol_tradeable(self, symbol: str) -> bool:
        """Check if symbol can be traded (not restricted by news)"""
        try:
            return not is_news_restricted_now(symbol, now=datetime.now(timezone.utc))
        except Exception as e:
            logger.warning(f"Error checking news restriction for {symbol}: {e}")
            return True  # Default to tradeable if check fails

class TradeExecutor:
    """Executes trading operations for symbols"""
    
    def __init__(self):
        self.symbol_filter = SymbolFilter(SYMBOL_LIST)
    
    def process_symbol(self, symbol: str) -> bool:
        """
        Process a single symbol for trading opportunities
        
        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            # Check if symbol is restricted by news
            if not self.symbol_filter.is_symbol_tradeable(symbol):
                logger.info(f"🚫 News restriction active - skipping {symbol}")
                return True  # Not an error, just skipped
            
            # Check for existing trade
            existing_trade = get_trade_by_symbol(symbol)
            
            if self._has_active_trade(existing_trade):
                return self._manage_existing_trade(symbol, existing_trade)
            else:
                return self._seek_new_opportunity(symbol)
                
        except Exception as e:
            logger.exception(f"❌ Error processing {symbol}: {e}")
            return False
    
    def _has_active_trade(self, trade: Optional[dict]) -> bool:
        """Check if there's an active trade"""
        return trade is not None and trade.get("status") == "open"
    
    def _manage_existing_trade(self, symbol: str, trade: dict) -> bool:
        """Manage an existing active trade"""
        try:
            ticket = trade.get("ticket")
            logger.info(f"🟡 Managing active trade for {symbol} (ticket={ticket})")
            
            # Pass the full trade object to maintain consistency with trade_cycle
            manage_active_trade(trade)
            return True
            
        except Exception as e:
            logger.error(f"Error managing trade for {symbol}: {e}")
            return False
    
    def _seek_new_opportunity(self, symbol: str) -> bool:
        """Look for new trading opportunities"""
        try:
            logger.info(f"🔍 Looking for new setup on {symbol}")
            result = trade_cycle(symbol=symbol)
            
            # Log the result for debugging
            if result:
                logger.debug(f"Trade cycle result for {symbol}: {result.get('signal', 'unknown')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in trade cycle for {symbol}: {e}")
            return False

class SystemInitializer:
    """Handles system initialization and health checks"""
    
    @staticmethod
    def initialize_system() -> bool:
        """Initialize all system components"""
        try:
            # Initialize database
            logger.info("📊 Initializing database...")
            init_db()
            
            # Initialize MT5 connection
            logger.info("🔌 Initializing MT5 connection...")
            if not ensure_mt5_initialized():
                logger.error("❌ Failed to initialize MT5 connection")
                return False
            
            logger.info("✅ System initialization complete")
            return True
            
        except Exception as e:
            logger.exception(f"❌ System initialization failed: {e}")
            return False

class TradingLoopManager:
    """Main trading loop manager with improved structure"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.scheduler = TradingScheduler()
        self.symbol_filter = SymbolFilter(symbols)
        self.executor = TradeExecutor()
        self.cycle_count = 0
    
    def run(self):
        """Run the main trading loop"""
        logger.info("📆 GPT Trader loop started (multi-symbol, H1 synchronized)")
        
        # Initialize system
        if not SystemInitializer.initialize_system():
            logger.error("❌ System initialization failed. Exiting.")
            return
        
        # Main loop
        try:
            while True:
                if not self._is_ready_to_trade():
                    self._wait_for_trading_hours()
                    continue
                
                self._execute_trading_cycle()
                self._wait_for_next_cycle()
                
        except KeyboardInterrupt:
            logger.info("🛑 Trading loop stopped by user")
        except Exception as e:
            logger.exception(f"💥 Fatal error in trading loop: {e}")
        finally:
            logger.info("👋 Trading loop terminated")
    
    def _is_ready_to_trade(self) -> bool:
        """Check if system is ready for trading"""
        return self.scheduler.is_trading_time()
    
    def _wait_for_trading_hours(self):
        """Wait until trading hours begin"""
        sleep_seconds = self.scheduler.seconds_until_trading_start()
        hours = int(sleep_seconds // 3600)
        minutes = int((sleep_seconds % 3600) // 60)
        
        logger.info(f"⏸️ Outside trading hours. Sleeping {hours}h {minutes}m until next session.")
        time.sleep(sleep_seconds)
    
    def _execute_trading_cycle(self):
        """Execute one complete trading cycle"""
        self.cycle_count += 1
        logger.info(f"🔄 Starting cycle #{self.cycle_count} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get symbols to process
        candidate_symbols = self.symbol_filter.get_candidate_symbols()
        
        if not candidate_symbols:
            logger.warning("⚠️ No candidate symbols after filtering")
            return
        
        # Process each symbol
        success_count = 0
        for symbol in candidate_symbols:
            if self.executor.process_symbol(symbol):
                success_count += 1
        
        logger.info(f"✅ Cycle #{self.cycle_count} complete: {success_count}/{len(candidate_symbols)} symbols processed successfully")
    
    def _wait_for_next_cycle(self):
        """Wait until the next cycle should start"""
        self.scheduler.wait_until_next_h1_boundary()

class TradingSystemBootstrap:
    """Bootstraps and starts the entire trading system"""
    
    @staticmethod
    def start():
        """Start the trading system with proper initialization"""
        # Setup logging
        configurator = LoggingConfigurator()
        configurator.setup_basic_logging()
        configurator.setup_exception_handling()
        
        # Setup Telegram logging if available
        telegram_enabled = configurator.setup_telegram_logging()
        
        logger.info("🚀 Starting GPT Trading System")
        logger.info(f"📈 Symbols: {SYMBOL_LIST}")
        logger.info(f"⏰ Trading hours: {TRADING_START_HOUR}:00 - {TRADING_END_HOUR}:00")
        
        if telegram_enabled:
            logger.info("📱 Telegram notifications: ENABLED")
        else:
            logger.info("📱 Telegram notifications: DISABLED")
        
        # Create and run trading loop
        loop_manager = TradingLoopManager(SYMBOL_LIST)
        loop_manager.run()

# ========== MAIN ENTRY POINT ==========
if __name__ == "__main__":
    TradingSystemBootstrap.start()