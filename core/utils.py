import os
import time
import base64
from datetime import datetime, timezone
import numpy as np

def is_file_fresh(path: str, max_age_sec: int = 120) -> bool:
    return (
        path and os.path.exists(path) and
        os.path.getsize(path) > 0 and
        (time.time() - os.path.getmtime(path)) < max_age_sec
    )

def get_market_session(timestamp_utc: datetime) -> str:
    hour = timestamp_utc.hour
    if 0 <= hour < 7:
        return "Asia"
    elif 7 <= hour < 13:
        return "Europe"
    else:
        return "New York"

def get_volatility_context(df) -> str:
    recent_atr = df["atr14"].iloc[-1]
    mean_atr = df["atr14"].mean()
    if recent_atr < 0.7 * mean_atr:
        return "low"
    elif recent_atr > 1.3 * mean_atr:
        return "high"
    else:
        return "medium"

def encode_image_as_b64(path: str) -> dict | None:
    if path and os.path.exists(path):
        with open(path, "rb") as img_file:
            b64 = base64.b64encode(img_file.read()).decode("utf-8")
            return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
    return None

def is_weekend() -> bool:
    return datetime.now(timezone.utc).weekday() in [5, 6]

def format_trade_case(trade: dict, result: str, last_signal: dict = None) -> dict:
    if last_signal:
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
