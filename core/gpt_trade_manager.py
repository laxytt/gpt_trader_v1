import json
import logging
from datetime import datetime, timezone
import pandas as pd
from core.market_data import get_recent_data_and_charts
from core.statistics import get_win_loss_streak
from core.utils import (
     get_market_session, get_volatility_context, np_encoder
)
from core.news_utils import get_upcoming_news
from core.gpt_interface import safe_parse_json
import openai

logger = logging.getLogger(__name__)
client = openai.OpenAI()


def ask_gpt_for_trade_management(trade: dict) -> dict:
    recent = get_recent_data_and_charts(symbol=trade["symbol"])
    if recent.get("error"):
        logger.warning(f"[{trade['symbol']}] GPT trade management fallback: {recent['error']}")
        return {
            "decision": "HOLD",
            "reason": f"Management failed: {recent['error']}",
            "risk_class": "C"
        }

    df_h1 = pd.DataFrame(recent["history_h1"])
    df_h4 = pd.DataFrame(recent.get("history_h4", []))
    last_bar = df_h1.iloc[-1] if not df_h1.empty else None
    now = datetime.now(timezone.utc)
    session = get_market_session(now)
    volatility = get_volatility_context(df_h1)
    streak = get_win_loss_streak()

    # Scenariusze decyzyjne
    hold_summary = f"Price near EMA, trend steady, no volume concern." if last_bar is not None else "N/A"
    close_summary = f"Upcoming news or trend break detected." if last_bar is not None else "N/A"

    scenario_string = (
        f"\nManagement Context:\n"
        f"- Time: {now.isoformat()}\n"
        f"- Session: {session}\n"
        f"- Volatility: {volatility}\n"
        f"- Streak: {streak['streak_type']} ({streak['streak_length']})\n"
        f"- Win rate: {streak['win_rate']:.1%}\n\n"
        f"Options:\nA) HOLD — {hold_summary}\nB) CLOSE — {close_summary}\n"
    )

    payload = {
        "trade": trade,
        "history_h1": recent["history_h1"],
        "history_h4": recent["history_h4"],
        "news": get_upcoming_news(symbol=trade["symbol"], within_minutes=30),
        "floating": trade.get("floating", 0),
        "drawdown": trade.get("max_drawdown_pips", 0),
        "minutes_open": trade.get("minutes_open", 0)
    }

    with open("core/prompts/system_prompt_manage.txt", "r", encoding="utf-8") as f:
        SYSTEM_PROMPT_MANAGE = f.read()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_MANAGE},
        {"role": "user", "content": scenario_string},
        {"role": "user", "content": json.dumps(payload, indent=2, default=np_encoder)}
    ]

    logger.info(f"Sending GPT trade management request for {trade['symbol']}")
    response = client.chat.completions.create(model="gpt-4.1-2025-04-14", messages=messages)

    reply = response.choices[0].message.content
    try:
        result = safe_parse_json(reply)
        return result
    except Exception as e:
        logger.error(f"GPT trade management JSON parse failed: {e}")
        return {
            "decision": "HOLD",
            "reason": "Failed to parse GPT decision.",
            "risk_class": "C"
        }
