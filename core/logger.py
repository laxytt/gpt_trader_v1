import json
from datetime import datetime, timezone
import os

from core.paths import COMPLETED_TRADES_FILE

def append_trade_to_file(trade, file_path):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(trade) + "\n")
