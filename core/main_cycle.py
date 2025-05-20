from datetime import datetime, timezone
import time
from core.gpt_interface import ask_gpt_for_reflection, ask_gpt_for_signal
from core.news_utils import NEWS_FILE
from core.utils import is_file_fresh, log_resync
from mt5_utils import ensure_mt5_initialized, get_current_open_position, get_last_closed_price, get_risk_values, is_position_opened, open_trade_in_mt5
from core.trade_status import load_trade_status, save_trade_status
from core.trade_manager import TIER_RISK_MAPPING, manage_active_trade

def main_cycle():
     # -- Step 1: Always resync state from MT5 at the start of the cycle --
    try:
        mt5_live_trade = get_current_open_position()
    except Exception as e:
        log_resync("Error", f"MT5 API error: {str(e)}")

    if not is_file_fresh(NEWS_FILE, max_age_sec=7*24*3600):
        print(f"⛔ Calendar file {NEWS_FILE} is older than 7 days! Skipping trading this cycle.")
        # Optionally send a Telegram alert here.
        return  {"signal": "WAIT", "reason": "Economic calendar file is outdated."}

    local_state = load_trade_status()

    # CASE 1: MT5 live trade but local state idle OR wrong trade
    if mt5_live_trade and (local_state.get("status") != "open" or local_state.get("ticket") != mt5_live_trade.get("ticket")):
        print("🔄 Resync: MT5 reports open trade, updating local state.")
        mt5_live_trade["status"] = "open"
        save_trade_status(mt5_live_trade)
        log_resync("MT5->Local", f"ticket={mt5_live_trade.get('ticket')}, symbol={mt5_live_trade.get('symbol')}")
        local_state = mt5_live_trade  # Resynced!

    # CASE 2: Local state says open, but MT5 is flat
    if not mt5_live_trade and local_state.get("status") == "open":
        print("🔄 Resync: No open trade in MT5, setting local state to idle.")
        idle_state = {"status": "idle", "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")}
        save_trade_status(idle_state)
        log_resync("Local->Idle", f"ticket={local_state.get('ticket')}, symbol={local_state.get('symbol')}")
        local_state = idle_state
        
    state = load_trade_status()
    print(f"📦 Current trade state: {state}")

    # Validate trade state structure
    if "symbol" not in state:
        print("⚠️ Invalid state file — missing symbol. Skipping cycle.")
        return

    check_for_signal = True
    position_status = is_position_opened(state["symbol"], state.get("ticket"))

    if position_status is None:
        print("⚠️ Could not determine position status due to MT5 error. Skipping cycle.")
        return

    if state["status"] == "open":
        print(f"🟢 Detected open trade on {state['symbol']} (ticket={state.get('ticket')})")
        # Manage the open trade
        manage_active_trade(state)
        check_for_signal = False  # Don't open a new one

        # Optional: update status to idle if position closed
        position_open = is_position_opened(state["symbol"], state.get("ticket"))
        if position_open is False:
            print("✅ Trade is now closed — updating state to idle.")
            state["status"] = "idle"
            save_trade_status(state)

    if check_for_signal:
        print("🧠 Asking GPT for trade signal...")
        signal = ask_gpt_for_signal()
        if not signal:
            print("🛑 No actionable signal.")
            return
        if  signal.get("signal") == "WAIT":
            print(f"📈 GPT suggested signal: {signal}")
            return
        
        print(f"📈 GPT suggested signal: {signal}")
        # Place the trade
        trade_result = open_trade_in_mt5(signal)
        if trade_result and getattr(trade_result, "retcode", None) == 10009:
            print("✅ Trade opened successfully.")
            # Save new trade status
            trade_state = {
                "status": "open",
                "symbol": signal["symbol"],
                "side": signal["signal"],
                "entry": signal["entry"],
                "sl": signal["sl"],
                "tp": signal["tp"],
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
                "ticket": getattr(trade_result, "order", None)
            }
            save_trade_status(trade_state)
        else:
            print("❌ Failed to open trade or unknown error.")

    # Cooldown logic to prevent rapid re-entry (if needed)
    time.sleep(2)
