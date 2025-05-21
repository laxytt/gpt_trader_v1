import os
from dotenv import load_dotenv

load_dotenv()

TRADING_START_HOUR = int(os.getenv("TRADING_START_HOUR", "7"))
TRADING_END_HOUR = int(os.getenv("TRADING_END_HOUR", "23"))

SYMBOL_LIST = [
    "EURUSD", "GBPUSD", "USDJPY",
    "US30.cash", "US100.cash", "GER40.cash", "XAUUSD",
]

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
