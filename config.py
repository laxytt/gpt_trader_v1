import os
from dotenv import load_dotenv

load_dotenv()

TRADING_START_HOUR = int(os.getenv("TRADING_START_HOUR", "7"))
TRADING_END_HOUR = int(os.getenv("TRADING_END_HOUR", "23"))
EVENT_LOOKAHEAD_MINUTES = 2880  # 2 days

EVENT_BLACKLIST_BY_SYMBOL = {
    "EURISD": ["Federal Funds Rate", "FOMC Statement", "Non-Farm Employment Change",
    "Unemployment Rate", "Average Hourly Earnings", "Advance GDP q/q",
    "FOMC Meeting Minutes", "CPI y/y", "Main Refinancing Rate"],
    "XAUUSD": ["CPI y/y", "FOMC Statement"],
}

SYMBOL_LIST = [
    "EURUSD", "GBPUSD", "USDJPY",
    "US30.cash", "US100.cash", "GER40.cash", "XAUUSD",
]

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

MT5_FILES_DIR = os.getenv("MT5_FILES_DIR")
SCREENSHOT_DIR = os.getenv("SCREENSHOT_DIR")
NEWS_FILE = os.getenv("NEWS_FILE_DIR")



MT5_FILES_DIR = os.getenv("MT5_FILES_DIR", r"C:/Users/YOUR_USER/AppData/Roaming/MetaQuotes/.../MQL5/Files")
