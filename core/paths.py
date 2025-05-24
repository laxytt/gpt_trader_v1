from pathlib import Path
import os
from config import MT5_FILES_DIR as MT5_FILES_DIR_FROM_CONFIG

BASE_DIR = Path(__file__).resolve().parent.parent

# Directories
DATA_DIR = BASE_DIR / "data"
CORE_DIR = BASE_DIR / "core"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"
LOGS_DIR = BASE_DIR / "logs"
SCRIPTS_DIR = BASE_DIR / "scripts"

# Files
SCREENSHOT_TRIGGER_FILE = BASE_DIR / "trigger.txt"
COMPLETED_TRADES_FILE = DATA_DIR / "completed_trades.jsonl"
TRADE_STATUS_FILE = DATA_DIR / "trade_status.json"
OPEN_TRADE_FILE = DATA_DIR / "open_trade.json"
OPEN_TRADES_FILE = DATA_DIR / "open_trades.json"

# MT5 trigger path (shared file with MQL5 script)
MT5_FILES_DIR = Path(MT5_FILES_DIR_FROM_CONFIG)
TRIGGER_PATH = MT5_FILES_DIR / "trigger.txt"
