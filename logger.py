import json
from datetime import datetime

def log_closed_trade(trade_data, result_type):
    rr = abs((trade_data["tp"] - trade_data["entry"]) / (trade_data["entry"] - trade_data["sl"]))
    log_entry = {
        "symbol": trade_data["symbol"],
        "side": trade_data["side"],
        "entry": trade_data["entry"],
        "sl": trade_data["sl"],
        "tp": trade_data["tp"],
        "exit_price": trade_data["exit_price"],
        "result": result_type,
        "rr": round(rr, 2),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open("closed_trades.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def log_gpt_signal(signal_data):
    if signal_data.get("signal") not in ("BUY", "SELL"):
        return  # ❌ Skip "WAIT" signals
    
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": signal_data.get("symbol", "EURUSD"),
        "signal": signal_data.get("signal"),
        "entry": signal_data.get("entry"),
        "sl": signal_data.get("sl"),
        "tp": signal_data.get("tp"),
        "rr": signal_data.get("rr"),
        "reason": signal_data.get("reason")
    }
    with open("gpt_signals_log.json", "a") as f:
        f.write(json.dumps(log_entry) + "\n")