import logging
import os
import time
import MetaTrader5 as mt5
import pandas as pd
import talib
from core.chart_utils import generate_chart_with_rsi_and_volume
from config import MT5_FILES_DIR, SCREENSHOT_DIR

logger = logging.getLogger(__name__)

def ensure_mt5_initialized() -> bool:
    if not mt5.initialize():
        logger.error("MT5 initialization failed.")
        return False
    return True

def fetch_candles(symbol: str, timeframe, bars: int) -> pd.DataFrame:
    if not ensure_mt5_initialized():
        return pd.DataFrame()

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, max(250, bars))
    if rates is None or len(rates) == 0:
        logger.error(f"No rates fetched for {symbol} ({bars} bars)")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df['timestamp'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    return df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ema50"] = talib.EMA(df["close"], 50)
    df["ema200"] = talib.EMA(df["close"], 200)
    df["rsi14"] = talib.RSI(df["close"], 14)
    df["atr14"] = talib.ATR(df["high"], df["low"], df["close"], 14)
    df["rsi_slope"] = df["rsi14"].diff()
    return df.dropna()

def get_candle_history(symbol: str, timeframe, bars: int) -> list[dict]:
    df = fetch_candles(symbol, timeframe, bars)
    if df.empty:
        return []
    df = calculate_indicators(df)
    return df.tail(bars).to_dict(orient='records')

def request_screenshot_mt5(symbol: str, timeframe: str, timeout_sec: int = 10) -> str | None:
    trigger_file = os.path.join(MT5_FILES_DIR, "trigger.txt")
    screenshot_file = f"{symbol}_{timeframe}.png"
    screenshot_path = os.path.join(SCREENSHOT_DIR, screenshot_file)

    with open(trigger_file, "w") as f:
        f.write(screenshot_file)
        f.flush()
        os.fsync(f.fileno())
    
    logger.info(f"Requested MT5 screenshot for {symbol} {timeframe}")

    start_time = time.time()
    while (time.time() - start_time) < timeout_sec:
        if os.path.exists(screenshot_path):
            file_age = time.time() - os.path.getmtime(screenshot_path)
            if file_age < 30:
                logger.info(f"Screenshot updated: {screenshot_path}")
                return screenshot_path
        time.sleep(1)
    
    logger.warning(f"Screenshot for {symbol} {timeframe} not received within {timeout_sec} seconds")
    return None

def create_manual_chart(df: pd.DataFrame, symbol: str, timeframe: str) -> str:
    if "timestamp" in df.columns:
        df.set_index(pd.to_datetime(df["timestamp"]), inplace=True)

    filename = f"{symbol}_{timeframe}_manual.png"
    chart_path = os.path.join(SCREENSHOT_DIR, filename)
    generate_chart_with_rsi_and_volume(df, chart_path, title=f"{symbol} {timeframe}")
    return chart_path

def get_recent_data_and_charts(symbol: str, bars_json: int = 20, bars_chart: int = 80) -> dict:
    candles_h1_full = get_candle_history(symbol, mt5.TIMEFRAME_H1, bars_chart)
    candles_h4_full = get_candle_history(symbol, mt5.TIMEFRAME_H4, bars_chart)

    if not candles_h1_full or not candles_h4_full:
        logger.error(f"No candle data for {symbol}")
        return {"symbol": symbol, "error": "Missing candle data"}

    df_h1 = pd.DataFrame(candles_h1_full)
    df_h4 = pd.DataFrame(candles_h4_full)

    chart_h1 = create_manual_chart(df_h1, symbol, "H1")
    chart_h4 = create_manual_chart(df_h4, symbol, "H4")

    return {
        "symbol": symbol,
        "history_h1": candles_h1_full[-bars_json:],
        "history_h4": candles_h4_full[-bars_json:],
        "screenshot_h1": chart_h1,
        "screenshot_h4": chart_h4,
    }
