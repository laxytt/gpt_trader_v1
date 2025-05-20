import base64
from datetime import datetime, timezone
import os
import time

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

def get_win_loss_streak(logfile=COMPLETED_TRADES_FILE, n=10):
    """
    Returns win/loss streak and basic stats from the last n trades.
    """
    try:
        with open(logfile, "r", encoding="utf-8") as f:
            lines = f.readlines()[-n:]
        results = [json.loads(line).get("result") for line in lines]
        streak = []
        for res in reversed(results):
            if not streak or res == streak[-1]:
                streak.append(res)
            else:
                break
        win_rate = results.count("TP_hit") / len(results) if results else 0
        return {
            "streak_type": streak[0] if streak else None,
            "streak_length": len(streak),
            "win_rate": win_rate,
            "sample_size": len(results)
        }
    except Exception:
        return {
            "streak_type": None, "streak_length": 0, "win_rate": None, "sample_size": 0
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
