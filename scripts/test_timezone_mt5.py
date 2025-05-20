import MetaTrader5 as mt5
from datetime import datetime, timezone

if not mt5.initialize():
    print("❌ MT5 initialization failed")
    quit()

# Use tick time as proxy for server time
tick = mt5.symbol_info_tick("EURUSD")
if tick is None:
    print("❌ Couldn't retrieve EURUSD tick")
    mt5.shutdown()
    quit()

server_time = datetime.fromtimestamp(tick.time, tz=timezone.utc)
utc_now = datetime.now(timezone.utc)
offset = (server_time - utc_now).total_seconds() / 60

print(f"📉 MT5 server time: {server_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print(f"🌍 UTC now        : {utc_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print(f"⏱️ Offset from UTC: {offset:.1f} minutes")

mt5.shutdown()
