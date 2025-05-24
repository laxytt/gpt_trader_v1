import json
import logging
import re
from datetime import datetime, timezone
from core.debug_utils import pretty_print_json_box, print_section
from core.statistics import get_win_loss_streak
import openai
import pandas as pd
import tiktoken
from core.mt5_data import get_recent_candle_history_and_chart
from core.news_utils import get_upcoming_news
from core.utils import (
    encode_image_as_b64, get_market_session,
    get_volatility_context,
    np_encoder
)
from core.database import save_gpt_signal_decision, memory

logger = logging.getLogger(__name__)
client = openai.OpenAI()

with open('core/prompts/system_prompt.txt', 'r') as f:
    SYSTEM_PROMPT_TEMPLATE = f.read()

def estimate_tokens_and_cost(prompt, completion, model="gpt-4-1106-preview"):
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback: Use cl100k_base for unrecognized models
        enc = tiktoken.get_encoding("cl100k_base")
    prompt_tokens = len(enc.encode(prompt))
    completion_tokens = len(enc.encode(completion))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }

def safe_parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise e
    else:
        logger.error("No JSON found in GPT response!")
        raise ValueError("No JSON found")

def ask_gpt_for_signal(symbol: str = "EURUSD", bars_json: int = 20, bars_chart: int = 80) -> dict:
    logger.info(f"GPT signal request initiated for {symbol}")
    prompt_data = get_recent_candle_history_and_chart(symbol, bars_json, bars_chart)

    if prompt_data.get("error"):
        reason = prompt_data["error"]
        logger.warning(f"Data error for {symbol}: {reason}")
        return {"signal": "WAIT", "symbol": symbol, "reason": reason}

    last_signal = memory.get_previous_signal(symbol)
    if last_signal:
        prompt_data["previous_gpt_decision"] = last_signal

    images = [encode_image_as_b64(path) for path in [
        prompt_data.pop("screenshot_h1", None), prompt_data.pop("screenshot_h4", None)
    ] if path]

    df_h1 = pd.DataFrame(prompt_data["history_h1"])
    df_h4 = pd.DataFrame(prompt_data.get("history_h4", []))

    now = datetime.now(timezone.utc)
    session = get_market_session(now)
    volatility = get_volatility_context(df_h1)
    streak_info = get_win_loss_streak(symbol)

    scenario_string = (
        f"Market session: {session}\n"
        f"Volatility: {volatility}\n"
        f"Win/loss streak: {streak_info['streak_type']} ({streak_info['streak_length']}), "
        f"Win rate: {streak_info['win_rate']:.1%}\n"
    )

    context_text = df_h1.iloc[-1].to_dict() if not df_h1.empty else {}
    retrieved_cases = memory.query(context_text, symbol)
    prompt_data["historical_cases"] = retrieved_cases or []
    prompt_data["upcoming_news"] = get_upcoming_news(symbol=symbol, within_minutes=2880)

    prompt_json = json.dumps(prompt_data, default=np_encoder)

    message_content = [
        {"type": "text", "text": f"{symbol} H1/H4 data and context:"},
        *images,
        {"type": "text", "text": scenario_string},
        {"type": "text", "text": prompt_json}
    ]

    response = client.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(symbol=symbol)},
            {"role": "user", "content": message_content}
        ]
    )

    reply = response.choices[0].message.content
    token_info = estimate_tokens_and_cost(prompt_json, reply)
    logger.info(f"Tokens & cost: {token_info}")

    signal_decision = safe_parse_json(reply)
    signal_decision["symbol"] = symbol
    required_keys = ["signal", "entry", "sl", "tp", "rr", "risk_class", "reason"]
    for key in required_keys:
        signal_decision.setdefault(key, None)
    if signal_decision["signal"] not in ("BUY", "SELL", "WAIT"):
        signal_decision["signal"] = "WAIT"
        signal_decision["reason"] = "Invalid GPT signal."

    pretty_print_json_box(signal_decision, title=f"GPT Signal Output — {symbol}")
    save_gpt_signal_decision(signal_decision)

    return signal_decision

def ask_gpt_for_reflection(trade: dict, bars_json: int = 20, bars_chart: int = 80) -> dict:
    symbol = trade.get("symbol", "EURUSD")
    print_section(f"GPT REFLECTION — {symbol}")
    recent = get_recent_candle_history_and_chart(symbol=symbol, bars_json=bars_json, bars_chart=bars_chart)
    history_h1 = recent.get("history_h1", [])
    history_h4 = recent.get("history_h4", [])

    prompt = f"""
You are a trading assistant reviewing a completed trade. You are given:
- Trade summary (side, entry, exit, SL, TP, RR, result).
- Last {bars_json} H1 candles and last {bars_json} H4 candles (indicators included).

Trade:
Side: {trade.get('side')}
Entry: {trade.get('entry')}
Exit: {trade.get('exit_price')}
Stop-loss: {trade.get('sl')}
Take-profit: {trade.get('tp')}
RR: {trade.get('rr')}
Outcome: {trade.get('result')}

H1 context (last candle): {history_h1[-1] if history_h1 else 'N/A'}
H4 context (last candle): {history_h4[-1] if history_h4 else 'N/A'}

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
