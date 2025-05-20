import json
import os
from datetime import datetime, timezone

import openai
import pandas as pd
import tiktoken
from mt5_data import get_recent_candle_history_and_chart

from core.news_utils import get_upcoming_news
from core.paths import COMPLETED_TRADES_FILE
from core.rag_memory import TradeMemoryRAG
from core.utils import (encode_image_as_b64, get_market_session,
                        get_volatility_context, get_win_loss_streak)
from core.telegram_logger import TelegramLogger

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

- Always start from higher timeframe (H1) to analyze trend, accumulation, or distribution.
- Never trade against the H1 trend or without clear context.
- Analyze volume in relation to price: strong volume with price movement indicates strength; falling volume in trend may signal weakness or reversal.
- Look for confirmation: Never act on a single signal, wait for supporting evidence (e.g., candle shape, sequence, background).
- Bar types:
    - UpBar: Close above prior bar.
    - DownBar: Close below prior bar.
    - Wide/Narrow spread: Compare with prior bars.
- Candle + Volume Examples:
    - Stopping Volume: Large volume after downtrend, followed by UpBar or two-bar reversal.
    - Backholding: Narrow spread, very high volume after clear decline, potential reversal.
    - Selling Climax: Wide spread down bar, huge volume, close mid-bar.
    - No Supply: DownBar, very low volume (pink), confirmation by UpBar and volume rise.
    - No Demand: UpBar, low volume, lack of buying, confirmation by DownBar.
    - UpThrust: UpBar, high volume, long upper wick, close at/near low.
    - Supply Coming In: High volume, long upper wick, close mid-bar, context of resistance.
    - Trap Up Move: UpBar, very high volume, close near low, long upper wick, after a false breakout.
    - Test: Narrow/medium bar, low volume, lower wick, confirmation by UpBar with higher volume.
- Sequences: Always favor signal sequences (test + confirmation), not isolated bars.
- Formations: Use additional candlestick formations (Hammer, Engulfing, etc.) only as confirmation if supported by volume/context.
- NEVER open a trade based on one bar, one timeframe, or without volume confirmation.
- WAIT if context, volume, or confirmation are missing or ambiguous.

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

# Account Protection Rule (FTMO Challenge)
- NEVER open a new trade if the account balance or equity could fall below $90,000 due to trade risk.
- If currently open trades are at risk of breaching the max loss, CLOSE NOW or WAIT.
- Your #1 priority is always to keep the account above $90,000. If in doubt, WAIT.
- Only output BUY/SELL if the worst-case loss (SL) does NOT risk breaching this threshold.

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
    prompt_json = json.dumps(prompt_data, indent=2)

    # print("📤 Sending visual + indicator data + news to GPT...", prompt_json)
    print("📤 Sending visual + indicator data + news to GPT...")

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
    """
    # Load both histories for context
    recent = get_recent_candle_history_and_chart(symbol=trade.get("symbol", "EURUSD"))
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

    # Optionally re-log the trade with reflection
    with open(COMPLETED_TRADES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(trade) + "\n")
