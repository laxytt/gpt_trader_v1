import os
from core.database import memory  # ✅ to jest prawidłowa instancja

import json
import sys

def main():

    if not os.path.exists("closed_trades.json"):
        print("❌ closed_trades.json not found.")
        sys.exit(1)

    with open("closed_trades.json", "r", encoding="utf-8") as f:
        try:
            trades = json.load(f)
        except json.JSONDecodeError as e:
            print("❌ Failed to parse closed_trades.json:", e)
            sys.exit(1)

    count_added = 0
    for trade in trades:
        if not all(k in trade for k in ["symbol", "entry", "sl", "tp", "timestamp", "side"]):
            print(f"⚠️ Skipping malformed trade: {trade}")
            continue

        if "reason" not in trade:
            trade["reason"] = "No rationale recorded (imported legacy trade)."

        memory.add_case(trade)
        count_added += 1

    print(f"✅ Imported {count_added} trades into semantic memory.")

if __name__ == "__main__":
    main()
