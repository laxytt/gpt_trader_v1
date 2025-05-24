from datetime import datetime
import MetaTrader5 as mt5
import pandas as pd
from core.chart_utils import generate_chart_with_rsi_and_volume
import os
import time
import talib

DEFAULT_SYMBOL = "EURUSD"
DEFAULT_TIMEFRAME = mt5.TIMEFRAME_H1
DEFAULT_HISTORY_BARS = 20
INDICATOR_WARMUP_BARS = 50

# For screenshots, you may want to configure these
SCREENSHOT_DIR = "D:/gpt_trader_v1/screenshots"
MT5_FILES_DIR = r"C:/Users/laxyt/AppData/Roaming/MetaQuotes/Terminal/D0E8209F77C8CF37AD8BF550E51FF075/MQL5/Files"

def ensure_mt5_initialized():
    """Ensure MetaTrader 5 is initialized before API calls."""
    if not mt5.initialize():
        print("❌ Could not initialize MT5.")
        return False
    return True

def get_candles(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME, bars=DEFAULT_HISTORY_BARS):
    fetch_bars = max(250, bars)
    if not ensure_mt5_initialized():
        return pd.DataFrame()
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, fetch_bars)
    
    if rates is None or len(rates) == 0:
        print(f"❌ Could not fetch rates for {symbol} ({fetch_bars} bars)")
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['timestamp'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.rename(columns={'tick_volume': 'volume'})
    return df

def calculate_indicators(df):
    if df.empty:
        return df
    df["ema50"] = talib.EMA(df["close"], timeperiod=50)
    df["ema200"] = talib.EMA(df["close"], timeperiod=200)
    df["rsi14"] = talib.RSI(df["close"], timeperiod=14)
    df["atr14"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)
    df["rsi_slope"] = df["rsi14"].diff()
    return df

def get_candle_history_with_indicators(symbol, timeframe, bars):
    """
    Fetches candles for a symbol and timeframe, computes indicators,
    and returns a list of dicts (cleaned for JSON/GPT).
    """
    df = get_candles(symbol=symbol, timeframe=timeframe, bars=bars)
    if df is None or df.empty:
        return []
    df = calculate_indicators(df)
    df = df.dropna(subset=["ema200", "ema50", "rsi14", "atr14", "rsi_slope"]).reset_index(drop=True)
    candles = df.tail(bars).to_dict(orient="records")
    for candle in candles:
        if isinstance(candle.get("timestamp"), pd.Timestamp):
            candle["timestamp"] = candle["timestamp"].isoformat()
    return candles

def trigger_and_get_screenshot(symbol, tf, screenshots_dir=SCREENSHOT_DIR):
    TRIGGER_PATH = os.path.join(MT5_FILES_DIR, "trigger.txt")
    filename = f"chart_{symbol}_{tf}.png"
    with open(TRIGGER_PATH, "w") as f:
        f.write(filename)
        f.flush()
        os.fsync(f.fileno())
    print(f"🟢 Requested MT5 screenshot for {symbol} {tf} ({filename}) at {datetime.now()}")
    screenshot_path = os.path.join(screenshots_dir, filename)
    waited = 0
    while waited < 10:
        if os.path.exists(screenshot_path):
            mtime = os.path.getmtime(screenshot_path)
            age = time.time() - mtime
            if age < 30:
                print(f"✅ Screenshot updated: {screenshot_path}")
                return screenshot_path
        time.sleep(1)
        waited += 1
    print(f"⚠️ Screenshot for {symbol} {tf} did not update in 10s (check MT5 and file paths!)")
    return None

def get_recent_candle_history_and_chart(symbol="EURUSD", bars_json=20, bars_chart=80):
    """
    Returns both a wide screenshot for VSA context and trimmed JSON for GPT.
    bars_chart: bars for chart/screenshot (visual context, e.g. 80)
    bars_json: bars for GPT JSON payload (API cost control, e.g. 20)
    """
    # Fetch enough bars for charting
    candles_h1_full = get_candle_history_with_indicators(symbol, mt5.TIMEFRAME_H1, bars=bars_chart)
    candles_h4_full = get_candle_history_with_indicators(symbol, mt5.TIMEFRAME_H4, bars=bars_chart)

    if not candles_h1_full or not candles_h4_full:
        print("❌ Candle data is empty after fetch! Check MT5 connection or symbol/timeframe.")
        return {
            "symbol": symbol,
            "history_h1": [],
            "history_h4": [],
            "screenshot_h1": None,
            "screenshot_h4": None,
            "error": "Missing or empty candle data"
        }

    # Only last N for GPT JSON
    candles_h1_json = candles_h1_full[-bars_json:] if len(candles_h1_full) >= bars_json else candles_h1_full
    candles_h4_json = candles_h4_full[-bars_json:] if len(candles_h4_full) >= bars_json else candles_h4_full

    # Generate screenshots using all bars fetched (bars_chart)
    df_h1 = pd.DataFrame(candles_h1_full)
    df_h4 = pd.DataFrame(candles_h4_full)

    # Set index for mplfinance if needed
    if "timestamp" in df_h1.columns and not isinstance(df_h1.index, pd.DatetimeIndex):
        df_h1.index = pd.to_datetime(df_h1["timestamp"])
    if "timestamp" in df_h4.columns and not isinstance(df_h4.index, pd.DatetimeIndex):
        df_h4.index = pd.to_datetime(df_h4["timestamp"])

    chart_path_h1 = os.path.join(SCREENSHOT_DIR, f"auto_chart_{symbol}_H1.png")
    chart_path_h4 = os.path.join(SCREENSHOT_DIR, f"auto_chart_{symbol}_H4.png")
    generate_chart_with_rsi_and_volume(df_h1, chart_path_h1, title=f"{symbol} H1")
    generate_chart_with_rsi_and_volume(df_h4, chart_path_h4, title=f"{symbol} H4")

    return {
        "symbol": symbol,
        "history_h1": candles_h1_json,
        "history_h4": candles_h4_json,
        "screenshot_h1": chart_path_h1,
        "screenshot_h4": chart_path_h4,
    }
