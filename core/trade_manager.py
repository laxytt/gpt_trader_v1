import os
import json
import time
from datetime import datetime, timezone
import MetaTrader5 as mt5
from core.gpt_interface import ask_gpt_for_reflection
from core.gpt_trade_manager import ask_gpt_for_trade_management
from core.logger import append_trade_to_file
from core.paths import COMPLETED_TRADES_FILE, OPEN_TRADE_FILE
from core.utils import format_trade_case
from core.mt5_utils import ensure_mt5_initialized
import openai
from core.database import memory

MAGIC_NUMBER = 10032024
TIER_RISK_MAPPING = {
    "A": 2.0,
    "B": 1.5,
    "C": 1.0
}

client = openai.OpenAI()

def trade_timeout_check(open_time_str, max_candles=15, timeframe_minutes=5):
    """
    Check if a trade has exceeded its maximum allowed open time.
    """
    try:
        open_time = datetime.strptime(open_time_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        elapsed_minutes = (now - open_time).total_seconds() / 60
        return elapsed_minutes > (max_candles * timeframe_minutes)
    except Exception as e:
        return False  # If parsing fails, don't timeout

def load_open_trade():
    if os.path.exists(OPEN_TRADE_FILE):
        with open(OPEN_TRADE_FILE, "r") as f:
            return json.load(f)
    return None

def save_open_trade(trade):
    with open(OPEN_TRADE_FILE, "w") as f:
        json.dump(trade, f, indent=2)

def clear_open_trade():
    if os.path.exists(OPEN_TRADE_FILE):
        os.remove(OPEN_TRADE_FILE)

def log_closed_trade(trade, result):
    trade["closed"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    trade["result"] = result

    trade_with_reflection = ask_gpt_for_reflection(trade)
    append_trade_to_file(trade_with_reflection, COMPLETED_TRADES_FILE)

    # Use the enriched trade for the case
    last_signal = None
    if os.path.exists("last_signal.json"):
        try:
            with open("last_signal.json", "r") as f:
                last_signal = json.load(f)
        except Exception as e:
            print("⚠️ Could not load last_signal.json:", e)

    try:
        # Use trade_with_reflection here, not the original trade!
        case = format_trade_case(trade_with_reflection, result, last_signal)
        memory.add_case(case)
        print("📚 Logged trade case to memory.")
    except Exception as e:
        print("⚠️ Could not log case to memory:", e)


def update_drawdown(trade):
    symbol = trade["symbol"]
    entry_price = trade["entry"]

    if not ensure_mt5_initialized():
        return None

    symbol_data = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if not symbol_data or not tick:
        return

    point = symbol_data.point
    current_price = tick.bid if trade["side"] == "BUY" else tick.ask

    if trade["side"] == "BUY":
        drawdown = entry_price - current_price
    else:
        drawdown = current_price - entry_price

    drawdown_pips = round(drawdown / point, 1)
    prev_max = trade.get("max_drawdown_pips", 0)

    if drawdown_pips > prev_max:
        trade["max_drawdown_pips"] = drawdown_pips
        save_open_trade(trade)
        print(f"📉 Updated max drawdown: {drawdown_pips} pips")

def maybe_breakeven_adjustment(trade):
    if not ensure_mt5_initialized():
        return

    symbol_info = mt5.symbol_info_tick(trade["symbol"])
    if not symbol_info:
        return

    entry = trade["entry"]
    sl = trade["sl"]
    ticket = trade.get("ticket")

    if not ticket:
        print("❌ No ticket to move SL.")
        return

    move_trigger = abs(entry - sl)
    trigger_price = entry + move_trigger if trade["side"] == "BUY" else entry - move_trigger
    current_price = symbol_info.bid if trade["side"] == "BUY" else symbol_info.ask

    should_adjust = (current_price >= trigger_price) if trade["side"] == "BUY" else (current_price <= trigger_price)

    if should_adjust and abs(sl - entry) > 1e-5:
        print(f"🔁 Triggering breakeven SL update for {trade['symbol']} to {entry}")

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": entry,
            "tp": trade.get("tp"),
            "symbol": trade["symbol"],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC
        }

        result = mt5.order_send(request)
        if hasattr(result, "retcode") and result.retcode == mt5.TRADE_RETCODE_DONE:
            print("✅ SL moved to breakeven successfully.")
            trade["sl"] = entry
            save_open_trade(trade)
        else:
            print(f"❌ SL move failed: {getattr(result, 'retcode', None)} — {getattr(result, 'comment', '')}")


def close_position_now(trade):
    symbol = trade["symbol"]
    ticket = trade.get("ticket")

    if not ensure_mt5_initialized():
        return None
    
    position = next((p for p in mt5.positions_get(symbol=symbol) if p.ticket == ticket), None)

    if not position:
        print(f"⚠️ No open position found for ticket {ticket}")
        return False

    close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = mt5.symbol_info_tick(symbol).bid if close_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(symbol).ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": position.volume,
        "type": close_type,
        "position": ticket,
        "price": price,
        "deviation": 10,
        "magic": MAGIC_NUMBER,
        "comment": "GPT Auto-Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }

    result = mt5.order_send(request)
    if hasattr(result, "retcode") and result.retcode == mt5.TRADE_RETCODE_DONE:
        print("✅ Trade closed successfully")
        return True
    else:
        print(f"❌ Failed to close trade. Retcode: {getattr(result, 'retcode', None)}, Comment: {getattr(result, 'comment', '')}")
        return False

def modify_sl_tp_in_mt5(symbol, ticket, new_sl=None, new_tp=None):
    if not ensure_mt5_initialized():
        return None
     
    positions = mt5.positions_get(symbol=symbol)

    if positions is None:
        print("❌ MT5 positions_get() returned None. Possible disconnection or error.")
        mt5.shutdown()
        time.sleep(1)
        mt5.initialize()
        return None
    
    if not positions:
        print(f"⚠️ No open positions found for symbol {symbol}")
        return False

    position = next((p for p in positions if p.ticket == ticket), None)
    if not position:
        print(f"⚠️ No position found for SL/TP update (ticket={ticket})")
        return False

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": symbol,
        "position": ticket,
        "sl": new_sl if new_sl else position.sl,
        "tp": new_tp if new_tp else position.tp,
        "magic": MAGIC_NUMBER,
        "comment": "GPT SL/TP Modify"
    }

    result = mt5.order_send(request)
    if hasattr(result, "retcode") and result.retcode == mt5.TRADE_RETCODE_DONE:
        print("✅ SL/TP updated successfully")
        return True
    else:
        print(f"❌ Failed to modify SL/TP: {getattr(result, 'retcode', None)}, {getattr(result, 'comment', '')}")
        return False

def manage_active_trade(trade):
    # Check for timeout (max open candles)
    if trade_timeout_check(trade.get("timestamp", ""), max_candles=15):
        print("⏱️ Trade exceeded max open time. Closing now.")
        close_position_now(trade)
        log_closed_trade(trade, "timeout_close")
        clear_open_trade()
        return

    update_drawdown(trade)

    suggestion = ask_gpt_for_trade_management(trade)
    decision = suggestion["decision"]

    if decision == "HOLD":
        print("🟢 GPT: HOLD — keeping position open.")
        maybe_breakeven_adjustment(trade)
        return

    elif decision in ("MOVE_SL", "MOVE_TP"):
        new_sl = suggestion.get("new_sl")
        new_tp = suggestion.get("new_tp")
        print(f"🔁 GPT: Adjusting SL/TP — SL: {new_sl}, TP: {new_tp}")
        success = modify_sl_tp_in_mt5(trade["symbol"], trade.get("ticket"), new_sl, new_tp)
        if success:
            if new_sl: trade["sl"] = new_sl
            if new_tp: trade["tp"] = new_tp
            save_open_trade(trade)

    elif decision == "CLOSE_NOW":
        print("❌ GPT: Closing trade immediately.")
        close_position_now(trade)
        log_closed_trade(trade, "GPT_close")
        
        clear_open_trade()

    elif decision == "SCALE_IN":
        print("📈 GPT: SCALE-IN requested — not implemented yet.")
    else:
        print(f"⚠️ Unknown GPT decision: {decision}")
