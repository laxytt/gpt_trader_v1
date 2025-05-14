import MetaTrader5 as mt5
import pandas as pd
import talib
import os
import shutil
import glob
import time
from datetime import datetime, timedelta
from pathlib import Path
from datetime import datetime

def find_mt5_screenshot(screenshot_name):
    base_path = os.path.join(os.getenv("APPDATA"), "MetaQuotes", "Terminal")
    terminals = glob.glob(os.path.join(base_path, "*"))

    for term_path in terminals:
        screenshot_path = os.path.join(term_path, "MQL5", "Files", screenshot_name)
        if os.path.isfile(screenshot_path):
            return screenshot_path
    return None

def get_recent_candle_history_and_chart(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M5, count=300, screenshot_name="chart_EURUSD_M5.png"):
   

    # Update trigger file to request screenshot
    MT5_FILES_DIR = "C:/Users/laxyt/AppData/Roaming/MetaQuotes/Terminal/D0E8209F77C8CF37AD8BF550E51FF075/MQL5/Files"
    trigger_path = f"{MT5_FILES_DIR}/trigger.txt"

    with open(trigger_path, "w") as f:
        f.write(f"triggered at {datetime.now().isoformat()}")
        f.flush()
        os.fsync(f.fileno())

    time.sleep(5)

    # Initialize MT5
    if not mt5.initialize():
        raise RuntimeError("MT5 initialization failed")

    # Fetch candle data
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    df["ema50"] = talib.EMA(df["close"], timeperiod=50)
    df["ema200"] = talib.EMA(df["close"], timeperiod=200)
    df["rsi14"] = talib.RSI(df["close"], timeperiod=14)
    df["atr14"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)

    df.dropna(inplace=True)

    last_bars = df.tail(20)
    candles = []
    for _, row in last_bars.iterrows():
        candles.append({
            "timestamp": row["time"].strftime("%Y-%m-%d %H:%M"),
            "open": round(row["open"], 5),
            "high": round(row["high"], 5),
            "low": round(row["low"], 5),
            "close": round(row["close"], 5),
            "volume": int(row["tick_volume"]),
            "ema50": round(row["ema50"], 5),
            "ema200": round(row["ema200"], 5),
            "rsi14": round(row["rsi14"], 2),
            "atr14": round(row["atr14"], 5)
        })

    # Find and copy screenshot
    full_path = find_mt5_screenshot(screenshot_name)
    if full_path:
        # Check file timestamp (must be within last 2 minutes)
        modified_time = datetime.fromtimestamp(os.path.getmtime(full_path))
        print(f"📸 Screenshot last modified at: {modified_time}")

        if datetime.now() - modified_time > timedelta(minutes=1):
            raise RuntimeError(f"Screenshot found but is too old (last modified: {modified_time}).")

        shutil.copy(full_path, ".")
    else:
        raise FileNotFoundError(f"Screenshot '{screenshot_name}' not found in any MT5 terminal directory. Make sure Screenshot.mq5 is running in MT5.")

    mt5.shutdown()

    return {
        "symbol": symbol,
        "timeframe": "M5",
        "history": candles,
        "screenshot": screenshot_name
    }
