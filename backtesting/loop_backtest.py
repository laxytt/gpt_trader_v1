import hashlib
import os
import time
import traceback
import pandas as pd
import numpy as np
import talib
import openai
from datetime import timedelta
import json
import matplotlib.pyplot as plt
import io
import base64
import pickle
import mplfinance as mpf

# ==== CONFIGURATION ====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKTESTING_DIR = os.path.join(BASE_DIR, "backtesting")
BACKTESTING_DATA_DIR = os.path.join(BACKTESTING_DIR, "data")
M5_PATH = os.path.join(BACKTESTING_DATA_DIR, "EURUSD5.csv")
H1_PATH = os.path.join(BACKTESTING_DATA_DIR, "EURUSD60.csv")
LOOKBACK_M5 = 20
LOOKBACK_H1 = 20
MAX_TRADE_BARS = 25
COLNAMES = ["timestamp", "open", "high", "low", "close", "volume"]
INITIAL_BALANCE = 100000  # or whatever you want
lot_size = 0.1           # standard lots (adjust for your risk)
pip_value = 10           # $10 per pip per 1 lot on EURUSD

# ==== SYSTEM PROMPTS ====
SYSTEM_PROMPT = """
You are an expert Volume Spread Analysis (VSA) trader and coach. Generate strictly rule-based signals for intraday trading on EURUSD, FTMO-style risk.

You are given:
1. EURUSD H1 and M5 chart screenshots.
2. Last 20 H1 candles (with open, close, high, low, spread, volume, EMA50, EMA200, RSI14, ATR14, rsi_slope).
3. Last 20 M5 candles (same fields).
4. Upcoming macroeconomic events (timestamp, impact).
5. Up to 3 similar historical trade cases.
6. Market context (session, volatility, win/loss streak).

**RULES:**
- Always analyze H1 history for background/context: trend, accumulation/distribution.
- Only look for entries on M5 **if the H1 background agrees**.
- Confirm any trade with at least two types of evidence (e.g., volume + candle, volume + trend).
- Never act on a single-bar signal without confirmation.
- WAIT if context is ambiguous, ATR is low, or signals conflict.
- Never trade if high-impact macro news is due within 2 minutes (WAIT).
- BUY only after strong bullish VSA (Stopping Volume, Backholding, Selling Climax, confirmed Test/Spring) **plus confirmation**.
- SELL only after strong bearish VSA (No Demand, UpThrust, Supply Coming In, Buying Climax, Trap Up Move, failed Test) **plus confirmation**.
- Risk/Reward must be >= 2.0. SL just beyond confirming bar.
- WAIT if signal/context is unclear or conflicting.

**Always explain your rationale. Use the format:**
"Detected [VSA pattern] at [bar/timeframe]. Confirmation: [X]. Background: [Y]."
Do not invent signals.

**Output JSON only:**

{
  "symbol": "EURUSD",
  "signal": "BUY" | "SELL" | "WAIT",
  "entry": float or null,
  "sl": float or null,
  "tp": float or null,
  "rr": float or null,
  "risk_class": "A" | "B" | "C",
  "reason": "Short rationale, e.g., 'Stopping Volume on M5 at support, confirmed by up bar and rising volume; H1 trend up.'"
}

**Examples:**

BUY:
{
  "symbol": "EURUSD",
  "signal": "BUY",
  "entry": 1.0780,
  "sl": 1.0765,
  "tp": 1.0810,
  "rr": 2.0,
  "risk_class": "A",
  "reason": "Stopping Volume detected on M5 after downtrend, confirmed by up bar with increased volume; H1 trend up, no high-impact news."
}

SELL:
{
  "symbol": "EURUSD",
  "signal": "SELL",
  "entry": 1.0825,
  "sl": 1.0840,
  "tp": 1.0790,
  "rr": 2.33,
  "risk_class": "A",
  "reason": "UpThrust on M5 after uptrend, confirmed by high volume and down bar; H1 trend down, no high-impact news."
}

WAIT:
{
  "symbol": "EURUSD",
  "signal": "WAIT",
  "entry": null,
  "sl": null,
  "tp": null,
  "rr": null,
  "risk_class": "C",
  "reason": "Volume low, context mixed, or high-impact macro news in 1 minute."
}
"""

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

# ==== SETUP OPENAI ====
client = openai.OpenAI()  # <-- FILL IN

gpt_cache_file = "gpt_signal_cache.pkl"
if os.path.exists(gpt_cache_file):
    with open(gpt_cache_file, "rb") as f:
        gpt_cache = pickle.load(f)
else:
    gpt_cache = {}

def get_signal_with_cache(m5_slice, h1_slice, system_prompt, screenshots=None):
    ctx_hash = context_hash(m5_slice, h1_slice)
    if ctx_hash in gpt_cache:
        print(f"    [CACHE] Using cached GPT result.")
        return gpt_cache[ctx_hash]
    else:
        signal = ask_gpt_for_signal_offline(
            m5_slice=m5_slice,
            h1_slice=h1_slice,
            system_prompt=system_prompt,
            screenshots=screenshots,
        )
        gpt_cache[ctx_hash] = signal
        with open(gpt_cache_file, "wb") as f:
            pickle.dump(gpt_cache, f)
        return signal

def plot_and_encode_candles(df, filename=None, save_to_disk=False, label="M5"):
    if df is None or len(df) == 0:
        print(f"[WARN] Tried to plot empty DataFrame for {label}. Skipping image.")
        return None, None

    dff = df.copy()
    dff = dff.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    if not pd.api.types.is_datetime64_any_dtype(dff["timestamp"]):
        dff["timestamp"] = pd.to_datetime(dff["timestamp"])
    dff = dff.set_index("timestamp")
    if dff.shape[0] == 0 or dff.isnull().all().any():
        print(f"[WARN] DataFrame for {label} is empty after cleaning. Skipping plot.")
        return None, None

    apds = []
    if "ema50" in dff.columns:
        apds.append(mpf.make_addplot(dff["ema50"], color="orange", width=1.0))
    if "ema200" in dff.columns:
        apds.append(mpf.make_addplot(dff["ema200"], color="green", width=1.0))
    if "rsi14" in dff.columns:
        apds.append(mpf.make_addplot(dff["rsi14"], panel=2, color="purple", ylabel="RSI14"))

    num_panels = 2 + (1 if "rsi14" in dff.columns else 0)
    panel_ratios = (3, 1, 1) if num_panels == 3 else (3, 1)

    try:
        fig, axes = mpf.plot(
            dff,
            type='candle',
            addplot=apds,
            volume=True,
            style='yahoo',
            title=f"{label} candles",
            returnfig=True,
            figratio=(12,6),
            figscale=1.2,
            panel_ratios=panel_ratios
        )
    except Exception as e:
        print(f"[WARN] mplfinance failed to plot {label}: {e}")
        return None, None

    if save_to_disk and filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches="tight")
    buf.seek(0)
    img_bytes = buf.read()
    encoded = base64.b64encode(img_bytes).decode('utf-8')
    plt.close(fig)
    return encoded, filename if save_to_disk and filename else None


def convert_timestamps_to_iso(dict_list):
    for d in dict_list:
        if "timestamp" in d and not isinstance(d["timestamp"], str):
            d["timestamp"] = d["timestamp"].isoformat()
    return dict_list


def safe_parse_json(text):
    text = text.strip().strip("`")
    if text.lower().startswith("json"):
        text = text[4:].strip()
    return json.loads(text)

def load_candles(path, colnames):
    df = pd.read_csv(path, sep="\t", names=colnames, header=None)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    for c in colnames[1:]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.sort_values("timestamp").reset_index(drop=True)

def calculate_indicators(df):
    df["ema50"] = talib.EMA(df["close"], timeperiod=50)
    df["ema200"] = talib.EMA(df["close"], timeperiod=200)
    df["rsi14"] = talib.RSI(df["close"], timeperiod=14)
    df["atr14"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)
    df["rsi_slope"] = df["rsi14"].diff()
    return df

def ask_gpt_for_signal_offline(
    m5_slice, h1_slice, 
    system_prompt, 
    screenshots=None,
    session='Europe', volatility='medium', streak_info=None,
    upcoming_news=None
):
    if isinstance(m5_slice, pd.DataFrame):
        m5_slice = m5_slice.to_dict(orient="records")
    if isinstance(h1_slice, pd.DataFrame):
        h1_slice = h1_slice.to_dict(orient="records")
    prompt_data = {
        "symbol": "EURUSD",
        "history_m5": convert_timestamps_to_iso(m5_slice),
        "history_h1": convert_timestamps_to_iso(h1_slice),
        "market_session": session,
        "volatility": volatility,
        "upcoming_news": upcoming_news if upcoming_news else []
    }
    if streak_info:
        prompt_data.update(streak_info)
    valid_screenshots = [img for img in screenshots or [] if img]
    if valid_screenshots:
        prompt_data["screenshots"] = valid_screenshots

    prompt_json = json.dumps(prompt_data, indent=2)
    message_content = [
        {"type": "text", "text": "Here's historical EURUSD M5+H1 candle and indicator data, macro news, and context."},
    ]
    if valid_screenshots:
        for img in valid_screenshots:
            message_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})
    message_content += [
        {"type": "text", "text": "Market context provided. Output decision in strict JSON as before."},
        {"type": "text", "text": prompt_json}
    ]
    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message_content}
                ]
            )
            break
        except openai.RateLimitError as e:
            print(f"[RateLimit] Waiting 15s due to: {e}")
            time.sleep(15)
        except Exception as e:
            print(f"[GPT ERROR]: {e}")
            traceback.print_exc()
            print(f"Prompt data: {prompt_data}")
            print(f"Message content: {message_content}")
            return {"signal": "WAIT", "reason": "GPT API error."}


    msg = response.choices[0].message.content
    return safe_parse_json(msg)

def ask_gpt_for_trade_management_offline(
    trade,
    m5_slice,
    h1_slice,
    management_prompt,
    session='Europe',
    volatility='medium',
    streak_info=None,
    upcoming_news=None
):
    if isinstance(m5_slice, pd.DataFrame):
        m5_slice = m5_slice.to_dict(orient="records")
    if isinstance(h1_slice, pd.DataFrame):
        h1_slice = h1_slice.to_dict(orient="records")
    last_bar = m5_slice[-1] if len(m5_slice) > 0 else {}
    payload = {
        "trade": trade,
        "market_m5": last_bar,
        "history_m5": m5_slice,
        "history_h1": h1_slice,
        "news": upcoming_news if upcoming_news else [],
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
            {"role": "system", "content": management_prompt},
            {"role": "user", "content": "Management context for open trade. Output your management decision in strict JSON."},
            {"role": "user", "content": prompt_json}
        ]
    )
    result = response.choices[0].message.content
    return safe_parse_json(result)

def simulate_trade_result(open_trade, future_m5):
    entry = open_trade['entry']
    sl = open_trade['sl']
    tp = open_trade['tp']
    side = open_trade['side']
    for j, row in enumerate(future_m5.itertuples(index=False), 1):
        low, high = row.low, row.high
        ts = row.timestamp
        if side == "BUY":
            if low <= sl:
                return sl, ts, 'SL'
            if high >= tp:
                return tp, ts, 'TP'
        else:  # SELL
            if high >= sl:
                return sl, ts, 'SL'
            if low <= tp:
                return tp, ts, 'TP'
        if j >= MAX_TRADE_BARS:
            return row.close, ts, 'timeout'
    last = future_m5.iloc[-1]
    return last.close, last.timestamp, 'timeout'

def context_hash(m5_slice, h1_slice):
    m5_str = str([(d['open'], d['high'], d['low'], d['close'], d['volume']) for d in m5_slice])
    h1_str = str([(d['open'], d['high'], d['low'], d['close'], d['volume']) for d in h1_slice])
    return hashlib.sha256((m5_str + "|" + h1_str).encode('utf-8')).hexdigest()

def backtest_gpt_vsa(df_m5, df_h1):
    trades = []
    open_trade = None
    last_ctx_hash = None
    last_gpt_result = None
    screenshots_dir_m5 = "screenshots/m5"
    screenshots_dir_h1 = "screenshots/h1"
    os.makedirs(screenshots_dir_m5, exist_ok=True)
    os.makedirs(screenshots_dir_h1, exist_ok=True)
    balance = INITIAL_BALANCE

    for i in range(max(LOOKBACK_M5, 100), min(1100, len(df_m5) - MAX_TRADE_BARS)):
        m5_slice = df_m5.iloc[i-LOOKBACK_M5:i].copy().to_dict(orient="records")
        last_m5 = m5_slice[-1]
        h1_slice = df_h1[df_h1["timestamp"] <= df_m5.iloc[i-1]["timestamp"]].tail(LOOKBACK_H1).copy().to_dict(orient="records")

        ctx_hash = context_hash(m5_slice, h1_slice)

        if ctx_hash == last_ctx_hash and last_gpt_result is not None:
            signal = last_gpt_result
            print(f"    [CACHE HIT] Skipping GPT call at {df_m5.iloc[i-1]['timestamp']}, reusing previous signal.")
        else:
            signal = ask_gpt_for_signal_offline(
                m5_slice=m5_slice,
                h1_slice=h1_slice,
                system_prompt=SYSTEM_PROMPT,
                # screenshots=screenshots  # (if used)
            )
            last_ctx_hash = ctx_hash
            last_gpt_result = signal
            print(f"    [GPT CALLED] Signal: {signal}")

        ts = last_m5['timestamp']
        if isinstance(ts, str):
            ts = pd.to_datetime(ts)
        m5_img_path = f"{screenshots_dir_m5}/{ts.strftime('%Y%m%d_%H%M')}.png"
        h1_img_path = f"{screenshots_dir_h1}/{ts.strftime('%Y%m%d_%H%M')}.png"
        save_img = (i % 50 == 0)
        m5_slice_df = pd.DataFrame(m5_slice)
        h1_slice_df = pd.DataFrame(h1_slice)
        m5_image, h1_image = None, None
        if len(m5_slice_df) == 0:
            print(f"[WARN] Skipping M5 plot at bar {i} due to empty slice.")
        else:
            m5_image, _ = plot_and_encode_candles(m5_slice_df, filename=m5_img_path, save_to_disk=save_img, label="m5")
        if len(h1_slice_df) == 0:
            print(f"[WARN] Skipping H1 plot at bar {i} due to empty slice.")
        else:
            h1_image, _ = plot_and_encode_candles(h1_slice_df, filename=h1_img_path, save_to_disk=save_img, label="h1")
        screenshots = []
        if m5_image:
            screenshots.append(m5_image)
        if h1_image:
            screenshots.append(h1_image)

        if i % 50 == 0 or open_trade:
            print(f"\n[{i}/{len(df_m5)}] {last_m5['timestamp']} | Price: {last_m5['close']:.5f} | {'TRADE OPEN' if open_trade else 'No trade open'}")

        if open_trade:
            print(f"  [MANAGE] Trade open: {open_trade['side']} | Entry {open_trade['entry']:.5f} SL {open_trade['sl']:.5f} TP {open_trade['tp']:.5f}")
            mgmt = ask_gpt_for_trade_management_offline(
                trade=open_trade,
                m5_slice=m5_slice,
                h1_slice=h1_slice,
                management_prompt=SYSTEM_PROMPT_MANAGE,
            )
            print(f"    [GPT Management Decision] {mgmt}")
            if mgmt["decision"] == "CLOSE_NOW":
                print(f"    [CLOSE_NOW] Closed trade at {last_m5['close']:.5f} | {last_m5['timestamp']}")
                pl_pips = (last_m5['close'] - open_trade["entry"]) * (10000 if open_trade["side"] == "BUY" else -10000)
                money_pl = pl_pips * pip_value * lot_size
                balance += money_pl
                open_trade.update({
                    "exit_price": last_m5["close"],
                    "exit_time": last_m5["timestamp"],
                    "result": "GPT_close",
                    "pl_pips": round(pl_pips, 1),
                    "money_pl": round(money_pl, 2),
                    "balance": round(balance, 2)
                })
                print(f"    [P/L] {pl_pips:.1f} pips, Money P/L: ${money_pl:.2f}, Balance: ${balance:.2f}")
                trades.append(open_trade)
                open_trade = None
                continue
            elif mgmt["decision"] == "MOVE_SL" and "new_sl" in mgmt:
                old_sl = open_trade["sl"]
                open_trade["sl"] = mgmt["new_sl"]
                print(f"    [MOVE_SL] SL moved from {old_sl:.5f} to {mgmt['new_sl']:.5f}")
            elif mgmt["decision"] == "HOLD":
                print(f"    [HOLD] Trade unchanged.")

        # --- Entry logic if no trade is open ---
        if open_trade is None:
            print(f"  [SIGNAL] Getting signal (cache-aware)...")
            signal = get_signal_with_cache(
                m5_slice=m5_slice,
                h1_slice=h1_slice,
                system_prompt=SYSTEM_PROMPT,
                screenshots=screenshots
            )
            print(f"    [GPT/CACHE Signal] {signal}")

            if signal.get("signal") in ("BUY", "SELL"):
                print(f"    [TRADE OPEN] {signal['signal']} @ {signal['entry']:.5f} SL={signal['sl']:.5f} TP={signal['tp']:.5f} | Reason: {signal.get('reason')}")
                open_trade = {
                    "side": signal["signal"],
                    "entry": signal["entry"],
                    "sl": signal["sl"],
                    "tp": signal["tp"],
                    "open_time": last_m5['timestamp'],
                    "reason": signal.get("reason", ""),
                    "rr": signal.get("rr"),
                    "risk_class": signal.get("risk_class"),
                }
                # Simulate trade out to future bars
                future_m5 = df_m5.iloc[i+1:i+1+MAX_TRADE_BARS].copy()
                exit_price, exit_time, result = simulate_trade_result(open_trade, future_m5)
                pl_pips = (exit_price - open_trade["entry"]) * (10000 if open_trade["side"] == "BUY" else -10000)
                money_pl = pl_pips * pip_value * lot_size
                balance += money_pl
                open_trade.update({
                    "exit_price": exit_price,
                    "exit_time": exit_time,
                    "result": result,
                    "pl_pips": round(pl_pips, 1),
                    "money_pl": round(money_pl, 2),
                    "balance": round(balance, 2)
                })
                print(f"    [EXIT] Trade result: {result} @ {exit_price:.5f} | P/L: {pl_pips:.1f} pips (${money_pl:.2f}) | Balance: ${balance:.2f}")
                trades.append(open_trade)
                open_trade = None
            else:
                print(f"    [WAIT] No trade opened. Reason: {signal.get('reason')}")

            print(f"\n[Backtest completed. Total trades: {len(trades)}]")
            if trades:
                n = len(trades)
                win = sum(1 for t in trades if t["result"] == "TP")
                loss = sum(1 for t in trades if t["result"] == "SL")
                timeout = sum(1 for t in trades if t["result"] == "timeout")
                gpt_close = sum(1 for t in trades if t["result"] == "GPT_close")
                total_pips = sum(t["pl_pips"] for t in trades)
                print(f"\nWin rate: {win}/{n} = {win/n:.1%}")
                print(f"Loss rate: {loss}/{n} = {loss/n:.1%}")
                print(f"Timeouts: {timeout}, GPT closes: {gpt_close}")
                print(f"Total P/L: {total_pips:.1f} pips")
                print(f"Avg P/L per trade: {total_pips/n:.2f} pips")
                # Save all trades for analysis
                pd.DataFrame(trades).to_csv("backtest_trades_results.csv", index=False)
            return trades

if __name__ == "__main__":
    print("Loading data...")
    df_m5 = load_candles(M5_PATH, COLNAMES)
    df_h1 = load_candles(H1_PATH, COLNAMES)
    df_m5 = calculate_indicators(df_m5)
    df_h1 = calculate_indicators(df_h1)

    print("Running backtest...")
    trades = backtest_gpt_vsa(df_m5, df_h1)

    if trades:
        n = len(trades)
        win = sum(1 for t in trades if t["result"] == "TP")
        loss = sum(1 for t in trades if t["result"] == "SL")
        print(f"\nWin rate: {win}/{n} = {win/n:.1%}")
        print(f"Loss rate: {loss}/{n} = {loss/n:.1%}")