import time
from datetime import datetime, timedelta, timezone
from core.main_cycle import main_cycle
from core.news_filter import is_news_restricted_now

import sys
import traceback
from core.telegram_logger import TelegramLogger

def global_except_hook(exctype, value, tb):
    if exctype is TypeError and "offset-naive and offset-aware" in str(value):
        print("\n\n=== DATETIME NAIVE/AWARE BUG DETECTED ===")
        traceback.print_tb(tb)
        print(f"\nTypeError: {value}")
        print("=========================================\n\n")
    # Still call the default handler
    sys.__excepthook__(exctype, value, tb)

sys.excepthook = global_except_hook

TRADING_START_HOUR = 7
TRADING_END_HOUR = 23

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

def wait_until_next_5m_boundary():
    now = datetime.now()
    # Calculate time until next 5-minute boundary
    next_boundary = (now + timedelta(minutes=5 - now.minute % 5)).replace(second=0, microsecond=0)
    delay = (next_boundary - now).total_seconds()
    print(f"⏳ Sleeping {delay:.2f} seconds until {next_boundary.strftime('%H:%M:%S')} (next M5 boundary)...")
    time.sleep(delay)

def run_loop():
    print("📆 GPT Trader loop started (synced to 5m chart close)")
    while True:
        if not is_trading_time():
            # Calculate time until next start
            sleep_seconds = seconds_until_trading_start(TRADING_START_HOUR)
            next_start = (datetime.now() + timedelta(seconds=sleep_seconds)).strftime('%H:%M:%S')
            print(f"⏸️ Outside trading hours ({TRADING_START_HOUR}:00–{TRADING_END_HOUR}:00). Sleeping {int(sleep_seconds)}s until {next_start}...")
            time.sleep(sleep_seconds)
            continue

        wait_until_next_5m_boundary()  # <-- this is the change!
        print(f"🔄 Starting main cycle at {datetime.now().strftime('%H:%M:%S')}")
        try:
            if is_news_restricted_now("EURUSD", datetime.now(timezone.utc)):
                print("⚠️ Skipping main_cycle due to restricted macro news event.")
                continue
            main_cycle()
        except Exception as e:
            print(f"❌ Exception in main_cycle: {e}")
            import traceback
            print(f"❌ Exception in main_cycle: {e}")
            traceback.print_exc()
            raise e
    # Remove time.sleep(1.5) — not needed anymore!

       
if __name__ == "__main__":
    run_loop()
