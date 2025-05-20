import time
from datetime import datetime, timedelta, timezone
from core.main_cycle import main_cycle
from core.news_filter import is_news_restricted_now

import requests
import sys
import traceback


def global_except_hook(exctype, value, tb):
    if exctype is TypeError and "offset-naive and offset-aware" in str(value):
        print("\n\n=== DATETIME NAIVE/AWARE BUG DETECTED ===")
        traceback.print_tb(tb)
        print(f"\nTypeError: {value}")
        print("=========================================\n\n")
    # Still call the default handler
    sys.__excepthook__(exctype, value, tb)

sys.excepthook = global_except_hook



# Optional: Use dotenv or config for secrets, not in code!
try:
    from core.telegram_logger import TelegramLogger
    import os
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
        wait_until_next_5m_boundary()
        print(f"🔄 Starting main cycle at {datetime.now().strftime('%H:%M:%S')}")
        try:
            # If you want to skip main_cycle during restricted news, you can check here:
            if is_news_restricted_now("EURUSD", datetime.now(timezone.utc)):
                print("⚠️ Skipping main_cycle due to restricted macro news event.")
                continue

            main_cycle()
        except Exception as e:
            print(f"❌ Exception in main_cycle: {e}")
            import traceback
            print(f"❌ Exception in main_cycle: {e}")
            traceback.print_exc()
            raise  e

        # Short pause to avoid CPU spin if main_cycle is very fast
        time.sleep(1.5)

if __name__ == "__main__":
    run_loop()
