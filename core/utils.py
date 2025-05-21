import base64
from datetime import datetime, timezone
import os
import time
import numpy as np
from core.rag_memory import TradeMemoryRAG

from core.paths import COMPLETED_TRADES_FILE

def log_resync(event, details, file_path="resync_log.txt"):
    """
    Logs a resynchronization event with timestamp, event type, and details.
    """
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} | {event} | {details}\n")

def is_file_fresh(path, max_age_sec=120):
    """
    Checks if a file exists, is non-empty, and was modified within the last max_age_sec seconds.
    """
    if not path or not os.path.exists(path):
        return False
    if os.path.getsize(path) == 0:
        return False
    mtime = os.path.getmtime(path)
    age = time.time() - mtime
    return age < max_age_sec      

def get_market_session(timestamp_utc):
    """
    Returns 'Asia', 'Europe', or 'NY' depending on UTC time.
    """
    hour = timestamp_utc.hour
    if 0 <= hour < 7:
        return "Asia"
    elif 7 <= hour < 13:
        return "Europe"
    else:
        return "New York"

def get_volatility_context(df):
    """
    Returns 'low', 'medium', or 'high' volatility based on ATR vs. historical mean.
    """
    recent_atr = df["atr14"].iloc[-1]
    mean_atr = df["atr14"].mean()
    if recent_atr < 0.7 * mean_atr:
        return "low"
    elif recent_atr > 1.3 * mean_atr:
        return "high"
    else:
        return "medium"

import json

def get_win_loss_streak(symbol="EURUSD", sample_size=10):
    """
    Returns the win/loss streak and win rate for the specified symbol,
    optionally over the last `sample_size` trades.
    """
    try:
        # Update your trade memory/query logic to support per-symbol querying!
        memory = TradeMemoryRAG()
        cases = memory.query_cases(symbol=symbol, limit=sample_size)
        # This part below is pseudocode—use your actual logic to process cases:
        win_count = 0
        streak_type = "N/A"
        streak_length = 0
        results = [case.get("result", "").lower() for case in cases]
        # Compute streak and win rate logic here...
        # (Assume 'WIN' if "result" field matches win criteria.)
        if results:
            win_count = sum("win" in res for res in results)
            win_rate = win_count / len(results)
            # Calculate streak_type and streak_length (custom logic)
            last_type = None
            current_streak = 0
            for res in reversed(results):
                this_type = "win" if "win" in res else "loss"
                if last_type is None:
                    last_type = this_type
                if this_type == last_type:
                    current_streak += 1
                else:
                    break
            streak_type = last_type
            streak_length = current_streak
        else:
            win_rate = 0.0
        return {
            "streak_type": streak_type,
            "streak_length": streak_length,
            "win_rate": win_rate,
            "sample_size": sample_size
        }
    except Exception as e:
        print(f"⚠️ get_win_loss_streak() error for {symbol}: {e}")
        return {
            "streak_type": "N/A",
            "streak_length": 0,
            "win_rate": 0.0,
            "sample_size": sample_size
        }


def encode_image_as_b64(path):
    if path and os.path.exists(path):
        with open(path, "rb") as img_file:
            return {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64.b64encode(img_file.read()).decode('utf-8')}"}
            }
    return None

def is_weekend():
    return datetime.now(timezone.utc).weekday() in [5, 6]  # Saturday=5, Sunday=6

def format_trade_case(trade, result, last_signal=None):
    if last_signal:
        # Best: Use enriched context from last_signal
        context = (
            f"EMA50={last_signal.get('ema50', '?')}, EMA200={last_signal.get('ema200', '?')}, "
            f"RSI={last_signal.get('rsi14', '?')}, Volume={last_signal.get('volume', '?')}, ATR={last_signal.get('atr14', '?')}"
        )
        return {
            "context": context,
            "signal": last_signal.get("signal"),
            "rr": last_signal.get("rr"),
            "reason": last_signal.get("reason", ""),
            "result": result,
            "timestamp": trade.get("closed", trade.get("timestamp", "")),
            "id": f"{trade.get('timestamp','')}_{trade.get('entry','')}"
        }
    else:
        # Fallback: Use plain trade info (not as rich, but better than nothing)
        return {
            "context": f"Trade: {trade.get('side','?')} at {trade.get('entry','?')}, SL={trade.get('sl','?')}, TP={trade.get('tp','?')}",
            "signal": trade.get("side"),
            "rr": trade.get("rr", "?"),
            "reason": trade.get("reason", ""),
            "result": result,
            "timestamp": trade.get("closed", trade.get("timestamp", "")),
            "id": f"{trade.get('timestamp','')}_{trade.get('entry','')}"
        }


def np_encoder(obj):
    if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8,
                        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)