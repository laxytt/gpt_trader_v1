import mplfinance as mpf
import pandas as pd

def generate_chart_with_rsi_and_volume(df, out_path, title=""):
    """
    Generates a candlestick chart with EMA50/EMA200, volume bars, and RSI14 subplot.
    Saves to out_path.
    """
    # Ensure index is datetime (mplfinance requires it)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df.index = pd.to_datetime(df["timestamp"])
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or 'timestamp' column.")

    # Prepare data: need columns open, high, low, close, volume
    required_cols = {"open", "high", "low", "close", "volume"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame missing required columns: {required_cols - set(df.columns)}")

    # RSI (use your calculated one, or fallback to ta-lib/ta if missing)
    if "rsi14" not in df.columns:
        try:
            import ta
            df['rsi14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        except ImportError:
            raise ImportError("Install the 'ta' package for RSI calculation.")

    # Create the RSI subplot
    apds = [
        mpf.make_addplot(df['ema50'], color='blue', width=1, label="EMA50"),
        mpf.make_addplot(df['ema200'], color='orange', width=1, label="EMA200"),
        mpf.make_addplot(df['rsi14'], panel=2, color='purple', secondary_y=False, ylabel='RSI 14')
    ]

    # Plot chart
    mpf.plot(
        df,
        type='candle',
        volume=True,
        style='yahoo',
        title=title,
        addplot=apds,
        panel_ratios=(6,2,2),
        figratio=(12,8),
        figscale=1.3,
        savefig=out_path,
        ylabel='Price',
        ylabel_lower='Volume'
    )
    print(f"✅ Generated chart: {out_path}")
    return out_path
