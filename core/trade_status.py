from datetime import datetime, timezone
import json
import os
import MetaTrader5 as mt5
from core.paths import TRADE_STATUS_FILE

def load_trade_status():
    """
    Loads the current trade status from file, with MT5 sync fallback if missing or invalid.
    """
    # Default state
    state = {"status": "idle", "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")}

    # Try loading from file if present and non-empty
    if os.path.exists(TRADE_STATUS_FILE) and os.path.getsize(TRADE_STATUS_FILE) > 0:
        try:
            with open(TRADE_STATUS_FILE, "r") as f:
                data = json.load(f)
                if "status" in data:
                    return data
        except json.JSONDecodeError:
            print("❌ trade_status.json is corrupted. Resetting to idle.")
            return state

    # Fallback: check MT5 for live positions (if state is idle)
    if state["status"] == "idle":
        if not mt5.initialize():
            print("❌ Could not init MT5 in load_trade_status()")
            return state

        positions = mt5.positions_get()
        print(f"🧪 Live MT5 positions: {positions}")
        if positions:
            pos = positions[0]
            state = {
                "status": "open",
                "symbol": pos.symbol,
                "side": "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                "entry": pos.price_open,
                "sl": pos.sl,
                "tp": pos.tp,
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
                "ticket": pos.ticket
            }
            save_trade_status(state)
        mt5.shutdown()

    return state

def save_trade_status(state):
    """
    Saves the current trade status to file.
    Also saves the latest signal to last_signal.json if 'signal' present.
    """
    with open(TRADE_STATUS_FILE, "w") as f:
        json.dump
