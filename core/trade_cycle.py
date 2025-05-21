import logging
from datetime import datetime, timezone
import time
from core.gpt_interface import ask_gpt_for_signal
from core.news_utils import NEWS_FILE
from core.utils import is_file_fresh, log_resync
from core.mt5_utils import (
    ensure_mt5_initialized,
    get_current_open_position,
    is_position_opened,
    open_trade_in_mt5
)
from core.trade_manager import manage_active_trade
from core.database import load_trade_state, save_trade_state

logger = logging.getLogger(__name__)

def trade_cycle(symbol: str = "EURUSD"):
    if not is_file_fresh(NEWS_FILE, max_age_sec=7 * 24 * 3600):
        logger.warning("Economic calendar file outdated. Skipping trading cycle.")
        return {"signal": "WAIT", "reason": "Economic calendar file outdated."}

    ensure_mt5_initialized()

    mt5_trade = get_current_open_position(symbol=symbol)
    local_trade = load_trade_state(symbol)

    # Synchronizacja stanu MT5 z lokalną bazą
    if mt5_trade and (local_trade["status"] != "open" or local_trade.get("ticket") != mt5_trade.get("ticket")):
        logger.info(f"Resync: MT5 reports open trade for {symbol}, updating local state.")
        mt5_trade["status"] = "open"
        save_trade_state(mt5_trade)
        log_resync("MT5->Local", f"{symbol}, ticket={mt5_trade.get('ticket')}")
        local_trade = mt5_trade

    elif not mt5_trade and local_trade["status"] == "open":
        logger.info(f"Resync: No open trade in MT5 for {symbol}, updating local state to idle.")
        idle_state = {"symbol": symbol, "status": "idle", "timestamp": datetime.now(timezone.utc).isoformat()}
        save_trade_state(idle_state)
        log_resync("Local->Idle", f"{symbol}")
        local_trade = idle_state

    logger.info(f"Current trade state for {symbol}: {local_trade}")

    # Zarządzanie istniejącą pozycją
    if local_trade["status"] == "open":
        logger.info(f"Managing active trade
