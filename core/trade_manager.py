import os
import json
import time
from datetime import datetime, timezone
import MetaTrader5 as mt5
from core.gpt_interface import ask_gpt_for_reflection
from core.logger import append_trade_to_file
from core.paths import COMPLETED_TRADES_FILE, OPEN_TRADE_FILE
from core.utils import format_trade_case, get_market_session, get_volatility_context, get_win_loss_streak
from mt5_data import get_recent_candle_history_and_chart
from mt5_utils import ensure_mt5_initialized
from core.news_utils import get_upcoming_news
import openai
from core.rag_memory import TradeMemoryRAG
import pandas as pd

MAGIC_NUMBER = 10032024

client = openai.OpenAI()
memory = TradeMemoryRAG()

SYSTEM_PROMPT_MANAGE = """
You are an expert algorithmic trade manager operating a VSA-based intraday strategy under strict FTMO risk management rules.

**You are always given:**
- Trade info (side, entry, SL, TP, RR, PnL, drawdown, time open).
- Last 20 M5 candles (with indicators: EMA50, EMA200, RSI14, ATR14, rsi_slope).
- Last 20 H1 candles (same indicators).
- Latest market context (session, volatility, win/loss streak, news).
- Upcoming macro news events (within next 30 minutes).

**Your job:**
- Analyze *both* H1 (for trend/background/major S/R) and M5 (for recent structure and trade progress).
- Confirm the trade remains valid in context of VSA and current price/volume/indicator behavior.
- Recommend **one** management action:  
    - `"HOLD"` (keep position open, trend/logic intact)
    - `"MOVE_SL"` (move stop-loss, e.g. to breakeven/ATR, only if justified)
    - `"CLOSE_NOW"` (close immediately: trend reverses, adverse volume, or high-impact news imminent)

**Decision logic:**
- **HOLD**: Only if trade rationale and context remain valid per multi-timeframe VSA rules. Be conservative in “chop”/low ATR or after losing streaks.
- **MOVE_SL**: If position is in profit and structure supports risk reduction (e.g., trail to breakeven after +1R, or ATR swing).
- **CLOSE_NOW**: If trend is reversing, sudden volume spike against, major news within 2 minutes, or trade has timed out (open > 15 candles).

**Always explain your rationale:**  
State what you see in both H1 and M5, and why your action fits strict VSA/FTMO rules.

**Output strict JSON only:**  
{
  "decision": "HOLD" | "MOVE_SL" | "CLOSE_NOW",
  "reason": "Short explanation of rationale. Mention H1/M5 context, volume, structure, news if relevant.",
  "risk_class": "A" | "B" | "C"
}

**Examples:**

HOLD:
{
  "decision": "HOLD",
  "reason": "M5 trend and VSA context unchanged; H1 uptrend intact; volume steady, no news risk.",
  "risk_class": "A"
}

MOVE_SL:
{
  "decision": "MOVE_SL",
  "reason": "Trade reached +1R, price above entry; M5 shows consolidation, H1 still in trend. Moving SL to breakeven.",
  "risk_class": "B"
}

CLOSE_NOW:
{
  "decision": "CLOSE_NOW",
  "reason": "High-impact news (NFP) in 1 minute; H1 and M5 both show mixed signals, closing to avoid event risk.",
  "risk_class": "A"
}
"""


TIER_RISK_MAPPING = {
    "A": 2.0,
    "B": 1.5,
    "C": 1.0
}

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

    # 1. Always log to your completed trades file
    append_trade_to_file(trade, COMPLETED_TRADES_FILE)

    # 2. Try to enrich with last_signal.json if available
    last_signal = None
    if os.path.exists("last_signal.json"):
        try:
            with open("last_signal.json", "r") as f:
                last_signal = json.load(f)
        except Exception as e:
            print("⚠️ Could not load last_signal.json:", e)

    # 3. Always log to RAG/case memory—enriched if possible, fallback to trade info
    try:
        case = format_trade_case(trade, result, last_signal)
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

def ask_gpt_for_trade_management(trade):
    """
    Uses both M5 and H1 data to generate robust management logic for open trades.
    """
    # Fetch both M5 and H1 candle histories
    recent = get_recent_candle_history_and_chart(symbol=trade["symbol"])
    
    if recent.get("error"):
        print(f"⛔ {recent['error']}. Unable to manage trade reliably, returning HOLD.")
        return {
            "decision": "HOLD",
            "reason": f"Skipped management due to: {recent['error']} (stale/missing data/screenshots)",
            "risk_class": "C"
        }
    
    history_m5 = recent.get("history_m5", [])
    history_h1 = recent.get("history_h1", [])
    if not isinstance(history_m5, list) or len(history_m5) == 0:
        print("⛔ No M5 candle history available. Returning HOLD.")
        return {
            "decision": "HOLD",
            "reason": "No M5 candle history (data error)",
            "risk_class": "C"
        }
    if not isinstance(history_h1, list) or len(history_h1) == 0:
        print("⚠️ Warning: H1 candle history missing. Management may be less reliable.")

    df_m5 = pd.DataFrame(history_m5)
    df_h1 = pd.DataFrame(history_h1)
    last_bar = df_m5.iloc[-1] if not df_m5.empty else None
    now = datetime.now(timezone.utc)
    session = get_market_session(now)
    volatility = get_volatility_context(df_m5)
    streak_info = get_win_loss_streak()

    # Prepare scenario summaries
    if last_bar is not None:
        hold_summary = (
            f"Price {last_bar['close']:.5f} near EMA50 ({last_bar['ema50']:.5f}), "
            f"trend continues, no reversal signal."
        )
        move_sl_summary = (
            f"Market structure has improved, potential to trail SL to breakeven or above last swing (ATR: {last_bar['atr14']:.5f}), RSI: {last_bar['rsi14']:.1f}."
        )
        close_now_summary = (
            f"Sudden volume spike against position or major news approaching, "
            f"risk of adverse move (Vol: {last_bar['volume']}), "
            f"floating P/L: {trade.get('floating', 0.0):.1f}."
        )
    else:
        hold_summary = move_sl_summary = close_now_summary = "Not enough data"

    scenario_string = (
        f"\nManagement context:\n"
        f"- Time: {now.isoformat(timespec='seconds')}\n"
        f"- Session: {session}\n"
        f"- Volatility: {volatility}\n"
        f"- Recent win/loss streak: {streak_info['streak_type']} ({streak_info['streak_length']})\n"
        f"- Win rate (last {streak_info['sample_size']}): {streak_info['win_rate']:.1%}\n\n"
        f"Available scenarios (based on M5):\n"
        f"A) HOLD: {hold_summary}\n"
        f"B) MOVE_SL: {move_sl_summary}\n"
        f"C) CLOSE_NOW: {close_now_summary}\n\n"
        f"Higher timeframe (H1) context is available for trend/background logic.\n"
        f"Please choose the best scenario, explain your rationale in 1-2 sentences, "
        f"and output your decision in JSON format as:\n"
        '{\n  "decision": "HOLD"|"MOVE_SL"|"CLOSE_NOW",\n  "reason": "...",\n  "risk_class": "A"|"B"|"C"\n}\n'
    )

    market_context = {
        "timestamp": last_bar["timestamp"] if last_bar is not None else None,
        "price": last_bar["close"] if last_bar is not None else None,
        "ema50": last_bar["ema50"] if last_bar is not None else None,
        "ema200": last_bar["ema200"] if last_bar is not None else None,
        "rsi14": last_bar["rsi14"] if last_bar is not None else None,
        "atr14": last_bar["atr14"] if last_bar is not None else None,
        "volume": last_bar["volume"] if last_bar is not None else None
    }

    payload = {
        "trade": trade,
        "market_m5": market_context,
        "history_m5": history_m5,
        "history_h1": history_h1,
        "news": get_upcoming_news(within_minutes=30),
        "feedback": {
            "floating_profit": trade.get("floating", 0.0),
            "drawdown": trade.get("max_drawdown_pips", 0.0),
            "time_open_minutes": trade.get("minutes_open", 0)
        }
    }

    prompt_json = json.dumps(payload, indent=2)

    response = client.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_MANAGE},
            {"role": "user", "content": scenario_string},
            {"role": "user", "content": prompt_json}
        ]
    )

    result = response.choices[0].message.content
    print("🧠 GPT Management Suggestion:", result)
    return json.loads(result)


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
        ask_gpt_for_reflection(trade)
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
        ask_gpt_for_reflection(trade)
        clear_open_trade()

    elif decision == "SCALE_IN":
        print("📈 GPT: SCALE-IN requested — not implemented yet.")
    else:
        print(f"⚠️ Unknown GPT decision: {decision}")
