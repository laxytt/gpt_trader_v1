import os

# Get the absolute path to the project root (gpt_trader_v1), regardless of current working directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Subdirectories and files
DATA_DIR = os.path.join(BASE_DIR, "data")
CORE_DIR = os.path.join(BASE_DIR, "core")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")

# Key files
SCREENSHOT_TRIGGER_FILE = os.path.join(BASE_DIR, "trigger.txt")
COMPLETED_TRADES_FILE = os.path.join(DATA_DIR, "completed_trades.jsonl")
TRADE_STATUS_FILE = os.path.join(DATA_DIR, "trade_status.json")
OPEN_TRADE_FILE = os.path.join(DATA_DIR, "open_trade.json")

# Example screenshot files
SCREENSHOT_PATH_M5 = os.path.join(SCREENSHOTS_DIR, "chart_EURUSD_M5.png")
SCREENSHOT_PATH_H1 = os.path.join(SCREENSHOTS_DIR, "chart_EURUSD_H1.png")

MT5_FILES_DIR = r"C:/Users/laxyt/AppData/Roaming/MetaQuotes/Terminal/D0E8209F77C8CF37AD8BF550E51FF075/MQL5/Files"
TRIGGER_PATH = os.path.join(MT5_FILES_DIR, "trigger.txt")