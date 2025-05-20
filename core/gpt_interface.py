import json
import os
from datetime import datetime, timezone

import openai
import pandas as pd
import tiktoken
from mt5_data import get_recent_candle_history_and_chart

from core.news_utils import get_upcoming_news
from core.rag_memory import TradeMemoryRAG
from core.utils import (encode_image_as_b64, get_market_session,
                        get_volatility_context, get_win_loss_streak, np_encoder)

client = openai.OpenAI()  
memory = TradeMemoryRAG()

SYSTEM_PROMPT = """
You are an expert Volume Spread Analysis (VSA) trader and coach. Generate strictly rule-based signals for intraday trading on EURUSD, FTMO-style risk.

You are given:
1. EURUSD H1 and M5 chart screenshots.
2. Last 20 H1 candles (with open, close, high, low, spread, volume, EMA50, EMA200, RSI14, ATR14, rsi_slope).
3. Last 20 M5 candles (same fields).
4. Upcoming macroeconomic events (timestamp, impact).
5. Up to 3 similar historical trade cases.
6. Market context (session, volatility, win/loss streak).

# VSA Pattern Characteristics and Signal Examples

- **Cause & Effect:** Every move has a cause; be objective, not emotional. Observe which side (demand or supply) is driving the market.
- **Effort & Result:** Price action must be analyzed in the context of volume (effort) and resulting price change (result).
- **Volume as Market Energy:** Volume is the market's "energy". Colors: Green (bullish/up bar), Red (bearish/down bar), Pink (volume less than prior two bars).
- **Bar/Candle Types:** UpBar = close > previous, DownBar = close < previous. Analyze wide/narrow spreads in context.
- **Volume & Spread Analysis:** Wide or narrow spread, compared to previous candles, is critical for reading market intent.
- **Signal Sequences:** Never trade on a single bar or unconfirmed signal. Always look for a *sequence* of signals (e.g., test + confirmation).
- **Confirmation:** Strong signals require confirmation—by volume, by subsequent bars, or by background context.
- **Background (Tło):** Always start from higher timeframes (H1). Assess trend, accumulation/distribution, and context before entering.
- **Smart Money & Market Cycles:** Watch for actions of large players (Smart Money), typically seen in accumulation/distribution phases. Higher timeframe context is always dominant.

## Key VSA Signal Types (Pattern Triggers)

- **Stopping Volume:** After a decline, very high volume down bar, then up bar or two-bar reversal. Confirm with volume and next bars.
- **Backholding:** Narrow spread after a sharp decline, with exceptionally high volume—may indicate trend reversal.
- **Selling Climax:** Wide spread down bar, huge volume, close mid-bar, signals possible end of sell-off.
- **No Supply:** DownBar with very low/pink volume, confirmed by UpBar and volume rise. Context: after prior demand signals.
- **No Demand:** UpBar with low volume, less than two prior bars; confirmation by DownBar. Indicates weak buying interest.
- **Upthrust:** Candle with upper shadow, close in lower third, often high volume. Indicates supply overcoming demand, especially after up move.
- **Supply Coming In:** High volume, long upper wick, close mid-bar—signals possible top, especially after uptrend.
- **Trap Up Move:** UpBar with high volume, closes near low, long upper wick, often after false breakout. Confirmed by subsequent weakness.
- **Test:** Narrow/medium bar, low volume, lower wick; must be confirmed by UpBar with higher volume. Only valid after prior accumulation or demand signals.
- **TwoBar Reversal:** First bar (down), second bar (up), second closes above open of first, with higher green volume.
- **Shakeout:** Sudden drop, wide spread, closes in upper third, high volume—often signals washout before reversal.

## VSA Entry Logic

- Always analyze higher timeframe first (H1): Is market trending, accumulating, or distributing?
- Entry only if M5 and H1 context agree (trend, structure).
- Confirm any trade with at least two types of evidence (e.g., volume + candle, volume + trend).
- Never act on single-bar or single-timeframe signals.
- Entry triggers:
    - After a clear sequence: e.g., stopping volume + up bar, no supply + confirmation, two-bar reversal with correct volume context, etc.
    - Volume must confirm price action.
    - Stop loss below/above confirming bar or setup.
    - Prefer entry immediately after confirmation bar; skip if context changes.
- Never trade against the trend; do not try to catch falling knives or tops in strong trends.
- Risk/Reward must be >= 2.0. Always include stop loss.

## Risk & FTMO Account Protection

- WAIT if context is ambiguous, ATR is low, or signals conflict.
- Never trade if high-impact macro news is due within 2 minutes (WAIT).
- Do not enter if worst-case loss could breach $90,000 account limit. If open trades risk breaching, WAIT or CLOSE NOW.
- If in doubt, WAIT. No position is also a position.

## Output Instructions

- Always explain rationale, referencing detected VSA pattern, confirmation, and background.
- Do NOT invent signals.

**IMPORTANT:**  
- Always return ALL of these keys in your SINGLE JSON: `symbol`, `signal`, `entry`, `sl`, `tp`, `rr`, `risk_class`, `reason`—even if some values are null.
- `symbol` must be `"EURUSD"`.
- If you do not detect a valid setup, return `"signal": "WAIT"` and set all other keys except `reason` and `risk_class` to null.

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

**Example BUY:**
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

**Example SELL:**
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

**Example WAIT:**
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






def safe_parse_json(text):
    try:
        # 🩹 Strip Markdown formatting (triple backticks)
        if text.strip().startswith("```"):
            text = text.strip().strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()  # remove "json" after ```
        return json.loads(text)
    except json.JSONDecodeError as e:
        print("❌ Failed to parse GPT response as JSON:")
        print(text)
        with open("last_failed_response.txt", "w", encoding="utf-8") as f:
            f.write(text)
        raise e

def estimate_tokens_and_cost(prompt, completion, model="gpt-4.1-2025-04-14"):
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    prompt_tokens = len(enc.encode(prompt))
    completion_tokens = len(enc.encode(completion)) if completion else 0
    total_cost = (prompt_tokens / 1000 * 0.005) + (completion_tokens / 1000 * 0.015)
    return {
        "tokens_prompt": prompt_tokens,
        "tokens_completion": completion_tokens,
        "cost_usd": round(total_cost, 4)
    }

def ask_gpt_for_signal():
    # Load chart data + screenshots for both M5 and H1
    prompt_data = get_recent_candle_history_and_chart()

    if prompt_data.get("error"):
        print(f"⛔ {prompt_data['error']}. Skipping trading this cycle.")
        # Optionally send Telegram alert here.
        return {"signal": "WAIT", "reason": prompt_data['error']}

    # 🔁 Inject previous signal decision
    if os.path.exists("last_signal.json"):
        with open("last_signal.json", "r") as f:
            prompt_data["previous_gpt_decision"] = json.load(f)

    # Pop screenshots and encode for multimodal prompt
    screenshot_m5 = prompt_data.pop("screenshot_m5", None)
    screenshot_h1 = prompt_data.pop("screenshot_h1", None)
    images = []
    for path in [screenshot_m5, screenshot_h1]:
        encoded = encode_image_as_b64(path)
        if encoded:
            images.append(encoded)

    # Defensive check: both M5 and H1 must be present
    history_m5 = prompt_data.get("history_m5", [])
    history_h1 = prompt_data.get("history_h1", [])
    if not isinstance(history_m5, list) or len(history_m5) == 0:
        print("❌ Invalid or empty M5 history list.")
        return {"signal": "WAIT", "reason": "Invalid or missing M5 price history"}
    if not isinstance(history_h1, list) or len(history_h1) == 0:
        print("⚠️ Warning: H1 history is missing. GPT may have less reliable background/context.")

    df_m5 = pd.DataFrame(history_m5)
    df_h1 = pd.DataFrame(history_h1)
    now = datetime.now(timezone.utc)
    session = get_market_session(now)
    volatility = get_volatility_context(df_m5)
    streak_info = get_win_loss_streak()

    # Prepare scenario summaries from last candle/indicators (main entry: M5)
    if not df_m5.empty:
        last = df_m5.iloc[-1]
        long_summary = (
            f"Price {last['close']:.5f} above EMA50 ({last['ema50']:.5f}), RSI rising ({last['rsi14']:.1f}), "
            f"bullish candle, ATR {last['atr14']:.5f}"
        )
        short_summary = (
            f"Price {last['close']:.5f} below EMA200 ({last['ema200']:.5f}), RSI falling ({last['rsi14']:.1f}), "
            f"bearish candle, ATR {last['atr14']:.5f}"
        )
        skip_summary = (
            f"No clear volume/momentum; RSI ({last['rsi14']:.1f}), ATR low ({last['atr14']:.5f}), small range"
        )
    else:
        long_summary = short_summary = skip_summary = "Not enough data"

    # Add context fields to prompt_data for transparency/debugging
    prompt_data["market_session"] = session
    prompt_data["volatility"] = volatility
    prompt_data["streak_type"] = streak_info["streak_type"]
    prompt_data["streak_length"] = streak_info["streak_length"]
    prompt_data["win_rate"] = streak_info["win_rate"]
    prompt_data["streak_sample_size"] = streak_info["sample_size"]

    # Build scenario string for GPT (including both M5 and H1 info)
    scenario_string = (
        f"\nMarket context:\n"
        f"- Time: {now.isoformat(timespec='seconds')}\n"
        f"- Session: {session}\n"
        f"- Volatility: {volatility}\n"
        f"- Recent win/loss streak: {streak_info['streak_type']} ({streak_info['streak_length']})\n"
        f"- Win rate (last {streak_info['sample_size']}): {streak_info['win_rate']:.1%}\n\n"
        f"Available scenarios (based on M5):\n"
        f"A) Long: {long_summary}\n"
        f"B) Short: {short_summary}\n"
        f"C) Skip: {skip_summary}\n\n"
        f"Higher timeframe (H1) context is available for trend/background logic.\n"
        f"Please choose the best scenario (A, B, or C), explain your rationale in 1-2 sentences, "
        f"and output your decision in JSON format as:\n"
        '{\n  "signal": "BUY"|"SELL"|"WAIT",\n  "reason": "..."\n}\n'
    )

    # Prepare trade context for memory query (M5 context is primary)
    if not df_m5.empty:
        details = df_m5.iloc[-1]
        context_text = (
            f"EMA50={details['ema50']}, EMA200={details['ema200']}, RSI={details['rsi14']}, "
            f"Volume={details['volume']}, ATR={details['atr14']}"
        )
    else:
        context_text = "No recent candle data"

    # Query similar historical cases (RAG memory)
    retrieved_cases = memory.query(context_text)
    if isinstance(retrieved_cases, list) and len(retrieved_cases) > 0:
        cleaned = [case for case in retrieved_cases if all(
            k in case for k in ["context", "signal", "rr", "result", "reason"]
        )]
        if cleaned:
            print(f"📚 Retrieved {len(cleaned)} valid historical case(s)")
            prompt_data["historical_cases"] = cleaned
        else:
            print("⚠️ Retrieved cases but incomplete. Using fallback.")
            prompt_data["historical_cases"] = [{
                "context": "No similar past case found.",
                "signal": "WAIT",
                "rr": 0,
                "result": "N/A",
                "reason": "Fallback due to malformed case"
            }]
    else:
        print("📭 No similar historical cases found.")
        prompt_data["historical_cases"] = [{
            "context": "No similar past case found.",
            "signal": "WAIT",
            "rr": 0,
            "result": "N/A",
            "reason": "No matching case in memory"
        }]

    # Add upcoming macro news
    prompt_data["upcoming_news"] = get_upcoming_news(within_minutes=2880)

    # Prepare JSON structure for GPT input (includes both histories)
    prompt_json = json.dumps(prompt_data, indent=2, default=np_encoder)

    # print("📤 Sending visual + indicator data + news to GPT...", prompt_json)
    print("📤 Sending visual + indicator data + news to GPT...", prompt_json)

    # Build multimodal GPT input
    message_content = [
        {"type": "text", "text": "Here's the EURUSD M5 + H1 chart, indicator data (both timeframes), macro news, similar past trades, and enhanced context:"},
        *images,
        {"type": "text", "text": scenario_string},
        {"type": "text", "text": prompt_json}
    ]

    response = client.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message_content}
        ]
    )

    msg = response.choices[0].message.content
    token_info = estimate_tokens_and_cost(prompt_json, msg)
    print("🧾 Token usage and cost:", token_info)

    return safe_parse_json(msg)


def ask_gpt_for_reflection(trade):
    """
    Requests a GPT review of a completed trade, with both M5 and H1 histories for richer context.
    Returns the trade dict with an added "reflection" field.
    """
    # Load both histories for context
    recent = get_recent_candle_history_and_chart(symbol=trade.get("symbol", "EURUSD"), bars_json=20, bars_chart=80)
    history_m5 = recent.get("history_m5", [])
    history_h1 = recent.get("history_h1", [])

    prompt = f"""
You are a trading assistant reviewing a completed trade. You are given:
- Trade summary (side, entry, exit, SL, TP, RR, result).
- Last 20 M5 candles and last 20 H1 candles (indicators included).

Trade:
Side: {trade.get('side')}
Entry: {trade.get('entry')}
Exit: {trade.get('exit_price')}
Stop-loss: {trade.get('sl')}
Take-profit: {trade.get('tp')}
RR: {trade.get('rr')}
Outcome: {trade.get('result')}

M5 context (last candle): {history_m5[-1] if history_m5 else 'N/A'}
H1 context (last candle): {history_h1[-1] if history_h1 else 'N/A'}

Was the signal valid according to strict VSA and multi-timeframe logic? How could it be improved?
Respond in one paragraph.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=[
            {"role": "system", "content": "You are an expert trading coach evaluating closed trades using VSA, multi-timeframe, and strict rules."},
            {"role": "user", "content": prompt}
        ]
    )

    msg = response.choices[0].message.content.strip()
    trade["reflection"] = msg
    print("🪞 Reflection:", msg)
    
    return trade