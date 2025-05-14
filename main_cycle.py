from trade_status import load_trade_status, save_trade_status
from mt5_utils import check_if_position_closed, get_last_closed_price, open_trade_in_mt5
from gpt_interface import ask_gpt_for_signal
from logger import log_closed_trade, log_gpt_signal
from datetime import datetime


def main_cycle():
    print("🔄 Starting main cycle...")
    state = load_trade_status()
    print(f"📦 Current trade state: {state}")

    if state["status"] == "open":
        print("🔍 Checking if position is still open...")
        if check_if_position_closed(state["symbol"], state["side"]):
            print("✅ Position is closed.")
            exit_price = get_last_closed_price(state["symbol"])
            sl, tp = state["sl"], state["tp"]

            if abs(exit_price - tp) < 0.0002:
                result = "TP_hit"
            elif abs(exit_price - sl) < 0.0002:
                result = "SL_hit"
            else:
                result = "manual_or_partial"

            state["exit_price"] = exit_price
            log_closed_trade(state, result)
            save_trade_status({"status": "idle"})
            print(f"📈 Trade closed → Result: {result}, Exit Price: {exit_price}")
        else:
            print("🚫 Position still open. Skipping GPT call.")
    else:
        print("🧠 Asking GPT for trade signal...")
        signal = ask_gpt_for_signal()
        log_gpt_signal(signal)
        print("📩 GPT Response:", signal)

        if signal["signal"] in ["BUY", "SELL"]:
            print(f"🚀 Opening trade: {signal['signal']} @ {signal['entry']}")
            open_trade_in_mt5(signal)
            save_trade_status({
                "status": "open",
                "symbol": signal["symbol"],
                "side": signal["signal"],
                "entry": signal["entry"],
                "sl": signal["sl"],
                "tp": signal["tp"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
            })
        else:
            print("⏸️ GPT recommends: WAIT (no trade conditions met)")

    print("✅ Cycle complete.\n")
