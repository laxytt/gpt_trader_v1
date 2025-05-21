import time
from datetime import datetime, timedelta, timezone
from core.main_cycle import main_cycle
from core.news_filter import is_news_restricted_now
from core.mt5_utils import prefilter_instruments, ensure_mt5_initialized  # <-- new import
import MetaTrader5 as mt5

import sys
import traceback
from core.telegram_logger import TelegramLogger

import logging

logger = logging.getLogger(__name__)
# Set this only once, possibly in your main entrypoint if you want to configure format/output
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

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

# Define your universe here (choose wisely for cost)
SYMBOL_LIST = [
    "EURUSD", "GBPUSD", "USDJPY", "US30.cash", "US100.cash", "GER40.cash","XAUUSD",
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
    TELEGRAM_TOKEN = "7758402434:AAFcBDSOYGIN2_xBLSQmlipp1VfrXHly9u8"
    TELEGRAM_CHAT_ID = "7079805622"
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        log = TelegramLogger(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
        sys.stdout = log
except ImportError:
    pass

def wait_until_next_h1_boundary():
    from datetime import datetime, timedelta
    import time

    now = datetime.now()
    # If it's already right at the top of the hour, don't sleep
    if now.minute == 0 and now.second == 0:
        delay = 0
        next_boundary = now
    else:
        # Next hour at :00:00
        next_boundary = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        delay = (next_boundary - now).total_seconds()
    print(f"⏳ Sleeping {delay:.2f} seconds until {next_boundary.strftime('%Y-%m-%d %H:%M:%S')} (next H1 boundary)...")
    time.sleep(delay)


def run_loop():
    print("📆 GPT Trader loop started (multi-symbol, synced to 5m chart close)")
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

        # --- Prefilter (ATR/volume) ---
        try:
            candidate_symbols = prefilter_instruments(SYMBOL_LIST)
        except Exception as e:
            print(f"⚠️ Error during prefiltering: {e}")
            candidate_symbols = ["EURUSD"]  # fallback

        print(f"🔎 Candidates passing ATR/volume filter: {candidate_symbols}")

        # --- Per-symbol GPT cycle ---
        for symbol in candidate_symbols:
            try:
                # Optionally, skip if news-restricted for this symbol
                if is_news_restricted_now(symbol, datetime.now(timezone.utc)):
                    print(f"⚠️ Skipping {symbol} due to restricted macro news event.")
                    continue
                main_cycle(symbol=symbol)   # <--- pass symbol to your main_cycle!
            except Exception as e:
                print(f"❌ Exception in main_cycle({symbol}): {e}")
                traceback.print_exc()
                # continue to next symbol
        wait_until_next_h1_boundary()
if __name__ == "__main__":
    run_loop()
