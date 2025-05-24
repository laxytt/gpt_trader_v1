import logging
import mplfinance as mpf
import pandas as pd
import ta
logger = logging.getLogger(__name__)

def generate_chart_with_rsi_and_volume(df: pd.DataFrame, out_path: str, title: str = "") -> str:
    """
    Generates a candlestick chart with EMA50/EMA200, volume bars, and RSI14 subplot.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df.index = pd.to_datetime(df["timestamp"])
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or 'timestamp' column.")

    required_cols = {"open", "high", "low", "close", "volume"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required OHLCV columns: {required_cols - set(df.columns)}")

    if "rsi14" not in df.columns:
        try:
            df['rsi14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        except ImportError as e:
            raise ImportError("Install 'ta' package for RSI calculation.") from e

    apds = [
        mpf.make_addplot(df["ema50"], color='blue', width=1, label="EMA50"),
        mpf.make_addplot(df["ema200"], color='orange', width=1, label="EMA200"),
        mpf.make_addplot(df["rsi14"], panel=2, color='purple', secondary_y=False, ylabel='RSI 14')
    ]

    mpf.plot(
        df,
        type='candle',
        volume=True,
        style='yahoo',
        title=title,
        addplot=apds,
        panel_ratios=(6, 2, 2),
        figratio=(12, 8),
        figscale=1.3,
        savefig=str(out_path),
        ylabel='Price',
        ylabel_lower='Volume'
    )

    logger.info(f"✅ Chart saved: {out_path}")
    return str(out_path)
