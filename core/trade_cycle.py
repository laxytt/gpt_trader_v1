import logging
from datetime import datetime, timezone
import time
from core.gpt_interface import ask_gpt_for_signal
from core.news_utils import NEWS_FILE
from core.resync_logger import log_resync
from core.utils import is_file_fresh
from core.mt5_utils import (
    ensure_mt5_initialized,
    get_current_open_position,
    is_position_opened,
    open_trade_in_mt5
)
from core.trade_manager import manage_active_trade
from core.database import load_trade_state, save_trade_state

logger = logging.getLogger(__name__)

class TradeStateManager:
    """Manages trade state synchronization between MT5 and local database"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
    
    def get_synchronized_state(self) -> dict:
        """Get current trade state, ensuring MT5 and local DB are synchronized"""
        mt5_trade = get_current_open_position(symbol=self.symbol)
        local_trade = load_trade_state(self.symbol)
        
        # Case 1: MT5 has trade, local doesn't know about it
        if mt5_trade and (local_trade.get("status") != "open" or 
                         local_trade.get("ticket") != mt5_trade.get("ticket")):
            return self._sync_mt5_to_local(mt5_trade)
        
        # Case 2: Local thinks trade is open, but MT5 doesn't have it
        elif not mt5_trade and local_trade.get("status") == "open":
            return self._sync_local_to_idle()
        
        # Case 3: States are synchronized
        return local_trade
    
    def _sync_mt5_to_local(self, mt5_trade: dict) -> dict:
        """Sync MT5 trade state to local database"""
        logger.info(f"Resync: MT5 reports open trade for {self.symbol}, updating local state.")
        
        synchronized_trade = {
            **mt5_trade,
            "status": "open",
            "symbol": self.symbol,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        save_trade_state(synchronized_trade)
        log_resync("MT5->Local", f"{self.symbol}, ticket={mt5_trade.get('ticket')}")
        
        return synchronized_trade
    
    def _sync_local_to_idle(self) -> dict:
        """Set local state to idle when no MT5 trade exists"""
        logger.info(f"Resync: No open trade in MT5 for {self.symbol}, updating local state to idle.")
        
        idle_state = {
            "symbol": self.symbol, 
            "status": "idle", 
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        save_trade_state(idle_state)
        log_resync("Local->Idle", f"{self.symbol}")
        
        return idle_state

class TradeCycleExecutor:
    """Executes the main trading cycle for a symbol"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.state_manager = TradeStateManager(symbol)
    
    def execute(self) -> dict:
        """Execute one complete trade cycle"""
        # Pre-flight checks
        if not self._can_trade():
            return {"signal": "WAIT", "reason": "Economic calendar file outdated."}
        
        if not ensure_mt5_initialized():
            return {"signal": "WAIT", "reason": "MT5 initialization failed."}
        
        # Get synchronized trade state
        current_trade = self.state_manager.get_synchronized_state()
        logger.info(f"Current trade state for {self.symbol}: {current_trade.get('status', 'unknown')}")
        
        # Handle existing position
        if current_trade.get("status") == "open":
            return self._manage_existing_trade(current_trade)
        
        # Look for new trading opportunity
        return self._seek_new_opportunity()
    
    def _can_trade(self) -> bool:
        """Check if trading conditions are met"""
        if not is_file_fresh(NEWS_FILE, max_age_sec=7 * 24 * 3600):
            logger.warning("Economic calendar file outdated. Skipping trading cycle.")
            return False
        return True
    
    def _manage_existing_trade(self, trade: dict) -> dict:
        """Manage an existing open trade"""
        logger.info(f"Managing active trade for {self.symbol}")
        
        # Delegate to trade manager
        manage_active_trade(trade)
        
        # Check if trade is still open after management
        if not is_position_opened(self.symbol, trade.get("ticket")):
            logger.info(f"Trade closed for {self.symbol}, updating state to idle.")
            self._update_to_idle_state()
        
        return {"signal": "MANAGING", "reason": "Active trade being managed"}
    
    def _seek_new_opportunity(self) -> dict:
        """Look for new trading opportunities"""
        logger.info(f"Requesting GPT signal for {self.symbol}")
        
        signal = ask_gpt_for_signal(symbol=self.symbol)
        
        if not signal or signal.get("signal") == "WAIT":
            logger.info(f"No actionable signal for {self.symbol}. GPT recommendation: {signal}")
            return signal or {"signal": "WAIT", "reason": "No signal received"}
        
        return self._attempt_trade_execution(signal)
    
    def _attempt_trade_execution(self, signal: dict) -> dict:
        """Attempt to execute a trade based on GPT signal"""
        trade_result = open_trade_in_mt5(signal)
        
        if self._is_trade_successful(trade_result):
            self._save_successful_trade(signal, trade_result)
            logger.info(f"Trade opened successfully for {self.symbol}.")
            return {"signal": "EXECUTED", "reason": "Trade opened successfully"}
        else:
            logger.error(f"Failed to open trade for {self.symbol}. Response: {trade_result}")
            return {"signal": "FAILED", "reason": "Trade execution failed"}
    
    def _is_trade_successful(self, trade_result) -> bool:
        """Check if trade execution was successful"""
        return trade_result and getattr(trade_result, "retcode", None) == 10009
    
    def _save_successful_trade(self, signal: dict, trade_result):
        """Save successful trade state to database"""
        trade_state = {
            "symbol": signal["symbol"],
            "status": "open",
            "side": signal["signal"],
            "entry": signal["entry"],
            "sl": signal["sl"],
            "tp": signal["tp"],
            "ticket": getattr(trade_result, "order"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        save_trade_state(trade_state)
    
    def _update_to_idle_state(self):
        """Update trade state to idle"""
        idle_state = {
            "symbol": self.symbol,
            "status": "idle", 
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        save_trade_state(idle_state)

def trade_cycle(symbol: str = "EURUSD") -> dict:
    """
    Execute a complete trading cycle for the given symbol.
    
    This is the main entry point that:
    1. Synchronizes trade state between MT5 and local DB
    2. Manages existing trades or seeks new opportunities
    3. Returns the cycle result
    
    Args:
        symbol: The trading symbol to process
        
    Returns:
        dict: Result of the trade cycle execution
    """
    executor = TradeCycleExecutor(symbol)
    result = executor.execute()
    
    # Brief cooldown to prevent rapid-fire requests
    time.sleep(2)
    
    return result