import MetaTrader5 as mt5

def robust_initialize():
    if not mt5.initialize():
        print("❌ Failed to initialize MT5:", mt5.last_error())
        exit()
    print("✅ MT5 initialized.")

robust_initialize()

symbols = [s.name for s in mt5.symbols_get()]
print("Available symbols:", symbols)
symbol = "EURUSD"

info = mt5.symbol_info(symbol)
print("EURUSD info:", info)

tick = mt5.symbol_info_tick(symbol)
print("EURUSD tick:", tick)

if tick is None:
    print(f"❌ symbol_info_tick({symbol}) returned None.")
    exit()

print("Ask price:", tick.ask)

# --- Order Parameters ---
lot = 0.01  # Minimum lot size for most brokers
price = tick.ask
sl = price - 0.0020  # 20 pips SL
tp = price + 0.0040  # 40 pips TP

request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": lot,
    "type": mt5.ORDER_TYPE_BUY,
    "price": price,
    "sl": sl,
    "tp": tp,
    "deviation": 10,
    "magic": 123456,
    "comment": "Test order via Python",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_IOC,
}

print("Sending order:", request)
result = mt5.order_send(request)
print("Order send result:", result)

if result is not None and hasattr(result, "retcode"):
    print("retcode:", result.retcode)
    print("comment:", getattr(result, 'comment', ''))

    # Helpful error message decoding
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print("❌ Trade failed. MT5 error code:", result.retcode)
        # Optionally print error list here or see: https://www.mql5.com/en/docs/constants/errorswarnings/enum_trade_return_codes
    else:
        print("✅ Trade successfully placed!")
else:
    print("❌ order_send() returned None or incomplete result.")

# Optional: shutdown MT5 after script (optional in a quick script)
mt5.shutdown()
