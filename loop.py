import time
from main_cycle import main_cycle
from sync import wait_until_new_candle
from datetime import datetime, timedelta

def sleep_until_next_5m():
    now = datetime.now()
    next_minute = (now.minute // 5 + 1) * 5
    if next_minute == 60:
        next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        next_time = now.replace(minute=next_minute, second=0, microsecond=0)

    wait_seconds = (next_time - now).total_seconds()
    print(f"⏳ Sleeping {wait_seconds:.2f} seconds until {next_time.strftime('%H:%M:%S')} (next M5 boundary)...")
    time.sleep(wait_seconds)

def run_gpt_trader_loop():
    print("📆 GPT Trader loop started (synced to 5m chart close)")

    while True:
        # Sleep until precise M5 interval (e.g., 14:55:00, 15:00:00)
        sleep_until_next_5m()

        # Optional: small buffer for candle close to finalize
        time.sleep(1.5)

        print(f"🔄 Starting main cycle at {datetime.now().strftime('%H:%M:%S')}")
        try:
            main_cycle()
        except Exception as e:
            print("❌ Error during cycle:", e)

if __name__ == "__main__":
    run_gpt_trader_loop()
