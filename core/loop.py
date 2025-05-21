import time
from datetime import datetime, timedelta, timezone
from core.main_cycle import main_cycle
from core.news_filter import is_news_restricted_now
from core.mt5_utils import prefilter_instruments, ensure_mt5_initialized
from core.trade_status import load_all_open_trades, save_all_open_trades
from core.trade_manager import manage_active_trade
import MetaTrader5 as mt5
import sys
import traceback
from core.telegram_logger import TelegramLogger
import logging

from core.utils import print_section, sync_open_trades_from_mt5

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("pytorch").setLevel(logging.WARNING)
logging.getLogger("MetaTrader5").setLevel(logging.WARNING)

def global_except_hook(exctype, value, tb):
    if exctype is TypeError and "offset-naive and offset-aware" in str(value):
        print("\n\n=== DATETIME NAIVE/AWARE BUG DETECTED ===")
        traceback.print_tb(tb)
        print(f"\nTypeError: {value}")
        print("=========================================\n\n")
    sys.__excepthook__(exctype, value, tb)

sys.excepthook = global_except_hook

TRADING_START_HOUR = 7
TRADING_END_HOUR = 23

SYMBOL_LIST = [
    "EURUSD", "GBPUSD", "USDJPY", "US30.cash", "US100.cash", "GER40.cash", "XAUUSD",
]

def is_trading_time():
    now = datetime.now()
    return TRADING_START_HOUR <= now.hour < TRADING_END_HOUR

def seconds_until_trading_start(start_hour=7):
    now = datetime.now()
    start_today = now.replace(hour=start_hour, minute=0, second=0, microsecond=0)
    if now < start_today:
        delta = start_today - now
    else:
        start_tomorrow = start_today + timedelta(days=1)
        delta = start_tomorrow - now
    return delta.total_seconds()

# Optional: Use dotenv or config for secrets, not in code!
try:
    TELEGRAM_TOKEN = "YOUR_TOKEN"
    TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        log = TelegramLogger(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
        sys.stdout = log
except ImportError:
    pass

def wait_until_next_h1_boundary():
    now = datetime.now()
    if now.minute == 0 and now.second == 0:
        delay = 0
        next_boundary = now
    else:
        next_boundary = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        delay = (next_boundary - now).total_seconds()
    print(f"⏳ Sleeping {delay:.2f} seconds until {next_boundary.strftime('%Y-%m-%d %H:%M:%S')} (next H1 boundary)...")
    time.sleep(delay)

def run_loop():
    print("📆 GPT Trader loop started (multi-symbol, synced to H1 chart close)")
    if not ensure_mt5_initialized():
        print("❌ Could not initialize MT5 for instrument scanning.")
        return

    while True:
        if not is_trading_time():
            sleep_seconds = seconds_until_trading_start(TRADING_START_HOUR)
            next_start = (datetime.now() + timedelta(seconds=sleep_seconds)).strftime('%H:%M:%S')
            print(f"⏸️ Outside trading hours ({TRADING_START_HOUR}:00–{TRADING_END_HOUR}:00). Sleeping {int(sleep_seconds)}s until {next_start}...")
            time.sleep(sleep_seconds)
            continue

        ensure_mt5_initialized()
        print(f"🔄 Starting multi-symbol main cycle at {datetime.now().strftime('%H:%M:%S')}")

        try:
            candidate_symbols = prefilter_instruments(SYMBOL_LIST)
        except Exception as e:
            print(f"⚠️ Error during prefiltering: {e}")
            candidate_symbols = ["EURUSD"]  # fallback

        print(f"🔎 Candidates passing ATR/volume filter: {candidate_symbols}")

        # --- Load all open trades from file/dict
        open_trades = load_all_open_trades()  # Dict: {symbol: trade_dict}

        for symbol in candidate_symbols:
            try:
                if is_news_restricted_now(symbol, datetime.now(timezone.utc)):
                    print_section(f"NEWS BLOCKED: Skipping {symbol} due to restricted macro event.")
                    continue

                # --- Trade management step (per symbol) ---
                trade = open_trades.get(symbol)
                if trade:
                    print(f"🟡 Managing active trade for {symbol}")
                    manage_active_trade(trade)
                    # Save any updates (e.g., SL/TP moved)
                    open_trades[symbol] = trade
                    # If trade is now closed, remove it
                    if trade.get("status") == "closed":
                        open_trades.pop(symbol)
                        print_section(f"Trade for {symbol} is now closed/removed.")
                    continue

                # --- Entry logic if no open trade ---
                print_section(f"Looking for new trade signal for {symbol}")
                main_cycle(symbol=symbol)

            except Exception as e:
                print_section(f"❌ Exception in main_cycle({symbol})")
                traceback.print_exc()

        # Save the state of open trades after each loop
        save_all_open_trades(open_trades)
        wait_until_next_h1_boundary()

if __name__ == "__main__":
    sync_open_trades_from_mt5()
    run_loop()
