import json
import os

def load_trade_status():
    if not os.path.exists("trade_status.json"):
        return {"status": "idle"}
    with open("trade_status.json", "r") as f:
        return json.load(f)

def save_trade_status(data):
    with open("trade_status.json", "w") as f:
        json.dump(data, f, indent=2)
