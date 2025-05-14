import json
from mt5_data import get_recent_candle_history_and_chart
import tiktoken
from datetime import datetime
import openai
import subprocess
import base64
import os




client = openai.OpenAI()  # klient dla nowszego API

SYSTEM_PROMPT = """
You are a professional algorithmic trader following a strict intraday momentum strategy.

You will be given:
1. A screenshot of the EURUSD M5 chart showing the most recent price action.
2. A sequence of candlestick bars including technical indicators such as EMA50, EMA200, RSI14, ATR14, and volume.

Your task is to analyze both the visual chart and the technical data and decide whether a trade signal should be issued.

Use the following logic:
- Only consider a BUY signal if:
  - EMA50 > EMA200 (bullish trend)
  - RSI between 50–70
  - Volume is stable or increasing
  - Risk-to-Reward Ratio (R:R) is at least 2.0

- Only consider a SELL signal if:
  - EMA50 < EMA200 (bearish trend)
  - RSI between 30–50
  - Volume is stable or increasing
  - Risk-to-Reward Ratio (R:R) is at least 2.0

- Respond with "WAIT" if conditions are not clearly met
- Do NOT generate any signal if the data is incomplete or the chart is outdated

Return your answer strictly in this JSON format:

{
  "symbol": "EURUSD",
  "signal": "BUY",  // or "SELL" or "WAIT"
  "entry": 1.0785,
  "sl": 1.0772,
  "tp": 1.0815,
  "rr": 2.25,
  "reason": "Brief explanation here"
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
        print(f"⚠️ Unknown model for tokenizer: {model}, using cl100k_base fallback.")
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
    # Load data and screenshot
    prompt_data = get_recent_candle_history_and_chart()
    screenshot_path = prompt_data.pop("screenshot")

    with open(screenshot_path, "rb") as img_file:
        image_b64 = base64.b64encode(img_file.read()).decode("utf-8")
    
    prompt_json = json.dumps(prompt_data, indent=2)

    print("📤 Sending visual + indicator data to GPT...", prompt_json)

    response = client.chat.completions.create(
        model="gpt-4.1-2025-04-14", 
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Here's the EURUSD M5 chart screenshot and the last 20-bar indicator history."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                {"type": "text", "text": prompt_json}
            ]}
        ]
    )

    # content = response.choices[0].message.content
    msg = response.choices[0].message.content
    token_info = estimate_tokens_and_cost(prompt_json, msg)
    print("🧾 Token usage and cost:", token_info)

    return safe_parse_json(msg)