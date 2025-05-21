import os
import json
from datetime import datetime, timezone
import MetaTrader5 as mt5
from core.paths import OPEN_TRADES_FILE

# You can put these in core/paths.py if you prefer centralization
def _state_path(symbol):
    return f"trade_status_{symbol}.json"

def load_trade_status(symbol):
    state = {"status": "idle", "symbol": symbol, "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")}
    path = _state_path(symbol)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            with open(path, "r") as f:
                data = json.load(f)
                if "status" in data:
                    return data
        except json.JSONDecodeError:
            print(f"❌ {path} is corrupted. Resetting to idle for {symbol}.")
    # Fallback: try to sync from MT5 **for this symbol only**
    if state["status"] == "idle":
        if not mt5.initialize():
            print("❌ Could not init MT5 in load_trade_status()")
            return state
        positions = mt5.positions_get(symbol=symbol)
        print(f"🧪 Live MT5 positions for {symbol}: {positions}")
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
            save_trade_status(state, symbol)
    return state

def save_trade_status(state, symbol):
    """
    Saves the current trade status for a given symbol.
    """
    path = _state_path(symbol)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)

# (Optionally, migrate old global file on first run)
def migrate_global_status_file(global_file="trade_status.json", symbol="EURUSD"):
    if os.path.exists(global_file):
        with open(global_file, "r") as f:
            data = json.load(f)
        save_trade_status(data, symbol)
        os.rename(global_file, global_file + ".bak")
        print(f"Migrated {global_file} to per-symbol {symbol} file.")


def load_all_open_trades():
    if os.path.exists(OPEN_TRADES_FILE):
        with open(OPEN_TRADES_FILE, "r") as f:
            return json.load(f)
    return {}

def save_all_open_trades(trades_dict):
    with open(OPEN_TRADES_FILE, "w") as f:
        json.dump(trades_dict, f, indent=2)
