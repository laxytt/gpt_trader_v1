from core.gpt_interface import safe_parse_json
import openai
import json
import pandas as pd
from datetime import datetime, timezone

client = openai.OpenAI()

def ask_gpt_for_signal_offline(
    m5_slice, h1_slice, 
    system_prompt, 
    rag_memory=None, 
    session='Europe', volatility='medium', streak_info=None,
    screenshots=None, upcoming_news=None
):
    """
    Like ask_gpt_for_signal, but for backtest/offline mode. 
    Accepts explicit M5 and H1 historical slices (list of dicts).
    """
    import json
    import pandas as pd
    from datetime import datetime, timezone

    # Defensive check
    if not isinstance(m5_slice, (list, pd.DataFrame)) or len(m5_slice) == 0:
        print("❌ Invalid or empty M5 history slice.")
        return {"signal": "WAIT", "reason": "Invalid or missing M5 price history"}
    if not isinstance(h1_slice, (list, pd.DataFrame)) or len(h1_slice) == 0:
        print("⚠️ Warning: H1 history is missing. GPT may have less reliable background/context."}

    # Accept DataFrame or list of dicts
    if isinstance(m5_slice, pd.DataFrame):
        m5_slice = m5_slice.to_dict(orient="records")
    if isinstance(h1_slice, pd.DataFrame):
        h1_slice = h1_slice.to_dict(orient="records")
    
    # Prepare prompt_data like in live version
    prompt_data = {
        "symbol": "EURUSD",
        "history_m5": m5_slice,
        "history_h1": h1_slice,
        "market_session": session,
        "volatility": volatility,
    }
    if streak_info:
        prompt_data.update(streak_info)
    if upcoming_news:
        prompt_data["upcoming_news"] = upcoming_news
    if screenshots:
        prompt_data["screenshots"] = screenshots

    # Insert RAG historical cases if available
    if rag_memory is not None:
        # Example: context_text = ...build context from last bar...
        # retrieved_cases = rag_memory.query(context_text)
        # prompt_data["historical_cases"] = retrieved_cases
        pass  # Optional, add as needed

    prompt_json = json.dumps(prompt_data, indent=2)

    # Compose scenario string (minimal for now)
    scenario_string = (
        f"Market context provided. See prompt JSON for details.\n"
        f"Output decision in strict JSON as before."
    )

    # Build GPT input (adapt as in your real system)
    message_content = [
        {"type": "text", "text": "Here's historical EURUSD M5+H1 candle and indicator data, macro news, and context."},
        {"type": "text", "text": scenario_string},
        {"type": "text", "text": prompt_json}
    ]

    # Assume OpenAI client is initialized as `client`
    response = client.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message_content}
        ]
    )
    msg = response.choices[0].message.content
    return safe_parse_json(msg)  # as before

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

    scenario_string = (
        f"Management context for open trade. See prompt JSON for details. Output your management decision in strict JSON."
    )

    response = client.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=[
            {"role": "system", "content": management_prompt},
            {"role": "user", "content": scenario_string},
            {"role": "user", "content": prompt_json}
        ]
    )

    result = response.choices[0].message.content
    return safe_parse_json(result)
