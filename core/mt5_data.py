from datetime import datetime
import MetaTrader5 as mt5
import pandas as pd
from core.chart_utils import generate_chart_with_rsi_and_volume
from core.paths import SCREENSHOT_PATH_H1, SCREENSHOT_PATH_M5, SCREENSHOT_TRIGGER_FILE, TRIGGER_PATH
from core.utils import is_file_fresh, is_weekend
import os
import time
import talib

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DATA_DIR)

DEFAULT_SYMBOL = "EURUSD"
DEFAULT_TIMEFRAME = mt5.TIMEFRAME_M5
DEFAULT_HISTORY_BARS = 20
INDICATOR_WARMUP_BARS = 50

TEST_MODE = False

def ensure_mt5_initialized():
    if not mt5.initialize():
        print("❌ Could not initialize MT5.")
        return False
    return True

def get_candle_history_with_indicators(symbol, timeframe, bars):
    """
    Helper to fetch candles from MT5 for a given symbol and timeframe,
    calculate VSA-relevant indicators, and output a clean list of dicts with ISO timestamps.

    Args:
        symbol (str): Trading symbol, e.g. "EURUSD"
        timeframe (int): MT5 timeframe constant, e.g. mt5.TIMEFRAME_M5
        bars (int): Number of bars to fetch and return

    Returns:
        List[Dict]: Each dict contains open, close, high, low, spread, volume, EMA50, EMA200, RSI14, ATR14, rsi_slope, timestamp (ISO string)
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


def get_candles(symbol=DEFAULT_SYMBOL, timeframe=DEFAULT_TIMEFRAME, bars=DEFAULT_HISTORY_BARS):
    fetch_bars = max(250, bars)
    if not ensure_mt5_initialized():
        return []
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, fetch_bars)
    mt5.shutdown()
    if rates is None or len(rates) == 0:
        print(f"❌ Could not fetch rates for {symbol} ({fetch_bars} bars)")
        return []
    df = pd.DataFrame(rates)
    df['timestamp'] = pd.to_datetime(df['time'], unit='s', utc=True)
    # Rename tick_volume to volume (compat with rest of code)
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

def trigger_and_get_screenshot(symbol, tf, screenshots_dir="D:/gpt_trader_v1/screenshots"):
    MT5_FILES_DIR = r"C:/Users/laxyt/AppData/Roaming/MetaQuotes/Terminal/D0E8209F77C8CF37AD8BF550E51FF075/MQL5/Files"
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
    Returns both a wide screenshot for VSA context and lean JSON for GPT input.
    - bars_chart: bars for chart/screenshot (visual context)
    - bars_json: bars for GPT JSON payload (API cost control)
    """
    # Fetch enough bars for charting (use bars_chart)
    candles_m5_full = get_candle_history_with_indicators(symbol, mt5.TIMEFRAME_M5, bars=bars_chart)
    candles_h1_full = get_candle_history_with_indicators(symbol, mt5.TIMEFRAME_H1, bars=bars_chart)

    # Defensive: Check for data presence BEFORE charting!
    if not candles_m5_full or not candles_h1_full:
        print("❌ Candle data is empty after fetch! Check MT5 connection or symbol/timeframe.")
        return {
            "symbol": symbol,
            "history_m5": [],
            "history_h1": [],
            "screenshot_m5": None,
            "screenshot_h1": None,
            "error": "Missing or empty candle data"
        }

    # Slice just the last bars_json for GPT (if too short, use as many as available)
    candles_m5_json = candles_m5_full[-bars_json:] if len(candles_m5_full) >= bars_json else candles_m5_full
    candles_h1_json = candles_h1_full[-bars_json:] if len(candles_h1_full) >= bars_json else candles_h1_full

    # For chart: use all bars fetched (bars_chart)
    df_m5 = pd.DataFrame(candles_m5_full)
    df_h1 = pd.DataFrame(candles_h1_full)

    # Set index for mplfinance if needed
    if "timestamp" in df_m5.columns and not isinstance(df_m5.index, pd.DatetimeIndex):
        df_m5.index = pd.to_datetime(df_m5["timestamp"])
    if "timestamp" in df_h1.columns and not isinstance(df_h1.index, pd.DatetimeIndex):
        df_h1.index = pd.to_datetime(df_h1["timestamp"])

    chart_path_m5 = "D:/gpt_trader_v1/screenshots/auto_chart_EURUSD_M5.png"
    chart_path_h1 = "D:/gpt_trader_v1/screenshots/auto_chart_EURUSD_H1.png"
    generate_chart_with_rsi_and_volume(df_m5, chart_path_m5, title="EURUSD M5")
    generate_chart_with_rsi_and_volume(df_h1, chart_path_h1, title="EURUSD H1")

    return {
        "symbol": symbol,
        "history_m5": candles_m5_json,  # Only last N for GPT
        "history_h1": candles_h1_json,
        "screenshot_m5": chart_path_m5,
        "screenshot_h1": chart_path_h1,
    }

