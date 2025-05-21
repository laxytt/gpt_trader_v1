import json
import os
import re
from datetime import datetime, timezone

import openai
import pandas as pd
import tiktoken

from mt5_data import get_recent_candle_history_and_chart
from core.news_utils import get_upcoming_news
from core.rag_memory import TradeMemoryRAG
from core.utils import (
    encode_image_as_b64, get_market_session,
    get_volatility_context, get_win_loss_streak, np_encoder, pretty_print_json_box, print_section
)

client = openai.OpenAI()
memory = TradeMemoryRAG()
SYSTEM_PROMPT = """
You are an expert momentum and Volume Spread Analysis (VSA) trader and coach. Generate strictly rule-based signals for trading {symbol} on H1 (entry) with H4 (background/trend), using both momentum and VSA confirmation. FTMO-style risk.

You are given:
1. {symbol} H1 and H4 chart screenshots.
2. Last 20 H1 candles (open, close, high, low, spread, volume, EMA50, EMA200, RSI14, ATR14, rsi_slope).
3. Last 20 H4 candles (same fields).
4. Upcoming macroeconomic events (timestamp, impact).
5. Up to 3 similar historical trade cases.
6. Market context (session, volatility, win/loss streak).

## Momentum Pattern Characteristics (Primary Triggers)
- **Breakout:** H1 closes at a new 10-bar high (BUY) or 10-bar low (SELL), with ATR above median of last 20 H1 bars.
- **Trend Strength:** Price above both EMA50 and EMA200 (BUY) or below both EMAs (SELL). RSI > 60 for BUY, RSI < 40 for SELL.
- **Impulse Candle:** H1 closes with body size >1.2× ATR and closes near high (BUY) or near low (SELL).
- **H4 background must agree:** H4 trend by EMA50/200 or clear structure (trend up for BUY, down for SELL).

## VSA Confirmation (Secondary Requirement)
- Prefer entry after a sequence of bullish or bearish VSA patterns (e.g., Stopping Volume + up bar, Two-Bar Reversal, No Supply + confirmation, Shakeout for BUY; Upthrust, No Demand, Trap Up Move for SELL).
- Volume must confirm price action on the breakout bar (elevated, not below median).
- Never trade solely on momentum—VSA confirmation or absence of supply/demand required.

## Entry & Filter Logic
- Only act if both momentum and VSA criteria are satisfied, and H4 context agrees.
- If in strong trend (per H4), require less strict VSA confirmation, but never ignore volume context.
- Do NOT act if macro news risk in next 2 minutes (WAIT).
- ATR and volume must be above minimum thresholds.
- Never catch tops/bottoms against the H4 trend.

## Stop, Target, Risk
- SL: Just beyond recent swing or confirming candle.
- TP: At least 2× ATR from entry, or next resistance/support.
- RR must be >= 2.0
- Always include stop loss and take profit.

## Output Instructions
- Always explain rationale, referencing both momentum (breakout/impulse) and VSA pattern/confirmation, plus background.
- If no valid setup, return "signal": "WAIT" and set all other keys except reason and risk_class to null.

## Risk & Account Protection
- WAIT if context is ambiguous, ATR is low, or signals conflict.
- Never open a trade if high-impact news is due within 2 minutes.
- Never risk breaching account limits; WAIT or CLOSE if at risk.
- If in doubt, WAIT.

**IMPORTANT:**  
Always return ALL of these keys in your SINGLE JSON: symbol, signal, entry, sl, tp, rr, risk_class, reason—even if some values are null.

{{
  "symbol": "{symbol}",
  "signal": "BUY" | "SELL" | "WAIT",
  "entry": float or null,
  "sl": float or null,
  "tp": float or null,
  "rr": float or null,
  "risk_class": "A" | "B" | "C",
  "reason": "Short rationale, e.g., 'Momentum breakout on H1 to 10-bar high, ATR and volume elevated, VSA test confirmed by up bar and background H4 trend up.'"
}}

**Example WAIT:**
{{
  "symbol": "{symbol}",
  "signal": "WAIT",
  "entry": null,
  "sl": null,
  "tp": null,
  "rr": null,
  "risk_class": "C",
  "reason": "No momentum breakout, volume low, or context/VSA not confirmed. Or macro news in 1 minute."
}}
"""

def print_gpt_tokens_cost(token_info):
    print(f"🧾 [Tokens] prompt: {token_info['tokens_prompt']}, completion: {token_info['tokens_completion']}, cost: ${token_info['cost_usd']}")

def estimate_tokens_and_cost(prompt: str, completion: str, model: str = "gpt-4.1-2025-04-14") -> dict:
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


# ========== SAFE PARSE JSON ==========
def safe_parse_json(text: str) -> dict:
    """
    Parse a single JSON object from a GPT reply, even if extra text follows.
    """
    text = text.strip()
    # Extract first {...} JSON block even if extra text follows
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print("❌ Still failed to parse as JSON:", e)
            print(json_str)
            raise e
    else:
        print("❌ No JSON found in GPT response!")
        print(text)
        raise ValueError("No JSON found")

# ========== GPT SIGNAL ==========
def ask_gpt_for_signal(symbol: str = "EURUSD", bars_json: int = 20, bars_chart: int = 80) -> dict:
    print_section(f"GPT SIGNAL REQUEST — {symbol}")
    prompt_data = get_recent_candle_history_and_chart(symbol=symbol, bars_json=bars_json, bars_chart=bars_chart)

    if prompt_data.get("error"):
        print(f"⛔ {prompt_data['error']}. Skipping trading this cycle.")
        return {"signal": "WAIT", "symbol": symbol, "reason": prompt_data['error'], "entry": None, "sl": None, "tp": None, "rr": None, "risk_class": "C"}

    # Attach previous signal (per symbol)
    last_signal_path = f"last_signal_{symbol}.json"
    if os.path.exists(last_signal_path):
        with open(last_signal_path, "r") as f:
            prompt_data["previous_gpt_decision"] = json.load(f)

    # Screenshots for the GPT vision model
    screenshot_h1 = prompt_data.pop("screenshot_h1", None)
    screenshot_h4 = prompt_data.pop("screenshot_h4", None)
    images = [encode_image_as_b64(path) for path in [screenshot_h1, screenshot_h4] if path]

    # Defensive check
    history_h1 = prompt_data.get("history_h1", [])
    history_h4 = prompt_data.get("history_h4", [])
    if not isinstance(history_h1, list) or len(history_h1) == 0:
        print(f"❌ Invalid or empty H1 history list for {symbol}.")
        return {"signal": "WAIT", "symbol": symbol, "reason": "Invalid or missing H1 price history", "entry": None, "sl": None, "tp": None, "rr": None, "risk_class": "C"}

    df_h1 = pd.DataFrame(history_h1)
    df_h4 = pd.DataFrame(history_h4) if isinstance(history_h4, list) and len(history_h4) > 0 else pd.DataFrame()
    now = datetime.now(timezone.utc)
    session = get_market_session(now)
    volatility = get_volatility_context(df_h1)
    streak_info = get_win_loss_streak(symbol=symbol)

    # Optionally print H4 context summary
    if not df_h4.empty:
        last_h4 = df_h4.iloc[-1]
        print(f"H4 Context — Close: {last_h4['close']:.5f}, EMA50: {last_h4['ema50']:.5f}, EMA200: {last_h4['ema200']:.5f}, RSI: {last_h4['rsi14']:.2f}")

    prompt_data.update({
        "market_session": session,
        "volatility": volatility,
        "streak_type": streak_info["streak_type"],
        "streak_length": streak_info["streak_length"],
        "win_rate": streak_info["win_rate"],
        "streak_sample_size": streak_info["sample_size"]
    })

    scenario_string = (
        f"\nMarket context:\n"
        f"- Time: {now.isoformat(timespec='seconds')}\n"
        f"- Session: {session}\n"
        f"- Volatility: {volatility}\n"
        f"- Recent win/loss streak: {streak_info['streak_type']} ({streak_info['streak_length']})\n"
        f"- Win rate (last {streak_info['sample_size']}): {streak_info['win_rate']:.1%}\n"
        f"Higher timeframe (h4) context is available for trend/background logic.\n"
        f"Please choose the best scenario, explain your rationale, and output your decision as a strict JSON dict.\n"
    )

    # Query RAG memory using symbol for context separation
    context_text = "No recent candle data"
    if not df_h1.empty:
        details = df_h1.iloc[-1]
        context_text = (
            f"EMA50={details['ema50']}, EMA200={details['ema200']}, RSI={details['rsi14']}, "
            f"Volume={details['volume']}, ATR={details['atr14']}"
        )
    retrieved_cases = memory.query(context_text, symbol=symbol)
    prompt_data["historical_cases"] = [
        case for case in (retrieved_cases if retrieved_cases else [])
        if all(k in case for k in ["context", "signal", "rr", "result", "reason"])
    ]

    prompt_data["upcoming_news"] = get_upcoming_news(symbol=symbol, within_minutes=2880)
    prompt_json = json.dumps(prompt_data, indent=2, default=np_encoder)
    print("📤 Sending {} data to GPT...".format(symbol))

    # Build multimodal GPT input
    message_content = [
        {"type": "text", "text": f"Here's the {symbol} H1 + h4 chart, indicator data, macro news, and context:"},
        *images,
        {"type": "text", "text": scenario_string},
        {"type": "text", "text": prompt_json}
    ]

    response = client.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(symbol=symbol)},
            {"role": "user", "content": message_content}
        ]
    )

    msg = response.choices[0].message.content
    token_info = estimate_tokens_and_cost(prompt_json, msg)
    print_gpt_tokens_cost(token_info)

    parsed = safe_parse_json(msg)
    pretty_print_json_box(parsed, title="GPT SIGNAL OUTPUT")

    required_keys = ["symbol", "signal", "entry", "sl", "tp", "rr", "risk_class", "reason"]
    for k in required_keys:
        if k not in parsed:
            parsed[k] = None
    parsed["symbol"] = symbol
    if parsed.get("signal") not in ("BUY", "SELL", "WAIT"):
        parsed["signal"] = "WAIT"
        parsed["reason"] = f"Invalid or unrecognized signal type: {parsed.get('signal')}"
    return parsed

# ========== GPT REFLECTION ==========
def ask_gpt_for_reflection(trade: dict, bars_json: int = 20, bars_chart: int = 80) -> dict:
    symbol = trade.get("symbol", "EURUSD")
    print_section(f"GPT REFLECTION — {symbol}")
    recent = get_recent_candle_history_and_chart(symbol=symbol, bars_json=bars_json, bars_chart=bars_chart)
    history_h1 = recent.get("history_h1", [])
    history_h4 = recent.get("history_h4", [])

    prompt = f"""
You are a trading assistant reviewing a completed trade. You are given:
- Trade summary (side, entry, exit, SL, TP, RR, result).
- Last {bars_json} H1 candles and last {bars_json} h4 candles (indicators included).

Trade:
Side: {trade.get('side')}
Entry: {trade.get('entry')}
Exit: {trade.get('exit_price')}
Stop-loss: {trade.get('sl')}
Take-profit: {trade.get('tp')}
RR: {trade.get('rr')}
Outcome: {trade.get('result')}

H1 context (last candle): {history_h1[-1] if history_h1 else 'N/A'}
h4 context (last candle): {history_h4[-1] if history_h4 else 'N/A'}

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
    print_section("GPT REFLECTION OUTPUT")
    print(msg)
    trade["reflection"] = msg
    return trade
