import MetaTrader5 as mt5

def check_if_position_closed(symbol, side):
    mt5.initialize()
    pos = mt5.positions_get(symbol=symbol)
    mt5.shutdown()
    return pos is None or len(pos) == 0

def get_last_closed_price(symbol):
    mt5.initialize()
    deals = mt5.history_deals_get(position=0, from_date=None, to_date=None)
    mt5.shutdown()
    if deals and len(deals) > 0:
        return deals[-1].price
    return None

def open_trade_in_mt5(signal, risk_usd=300.0):
    symbol = signal["symbol"]
    sl_price = signal["sl"]
    entry_price = signal["entry"]

    # ✅ Ensure the symbol is activated in Market Watch
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"❌ Failed to select symbol {symbol}")

    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        raise RuntimeError(f"❌ Could not retrieve symbol info for {symbol}")

    point = symbol_info.point
    sl_distance = abs(entry_price - sl_price)
    sl_points = sl_distance / point

    lot = round(risk_usd / (sl_points * symbol_info.trade_tick_value), 2)
    print(f"💰 Account balance: {mt5.account_info().balance:.2f}, Risk: {risk_usd}, SL points: {sl_points}, Lot: {lot}")

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"❌ Failed to get tick data for {symbol}")
    price = tick.ask if signal["signal"] == "BUY" else tick.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY if signal["signal"] == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl_price,
        "tp": signal["tp"],
        "deviation": 10,
        "magic": 10032024,
        "comment": "GPT Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    print("🚀 Opening trade:", signal)
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Order failed! Retcode: {result.retcode}, Message: {result.comment}")
    else:
        print("✅ Order placed successfully!")
        print("🧾 Order result:", result._asdict())
    return result
