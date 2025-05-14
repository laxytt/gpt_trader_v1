from datetime import datetime
import time

def wait_until_new_candle(timeframe_minutes=5):
    while True:
        now = datetime.now()
        # Check if we are within the first 10 seconds of a new candle
        if now.minute % timeframe_minutes == 0 and now.second < 10:
            print(f"🕒 Synced: new {timeframe_minutes}m candle at {now}")
            break
        time.sleep(1)
