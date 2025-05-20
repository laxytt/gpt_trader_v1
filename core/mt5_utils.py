import MetaTrader5 as mt5
import time
import logging

# Configure logging (recommended for all modules)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def ensure_mt5_initialized(retry=2, delay=1):
    """
    Ensure MT5 is initialized before API calls. Tries to reconnect on failure.
    """
    if mt5.initialize():
        return True
    logging.warning("MT5 initialization failed, retrying...")
    for _ in range(retry):
        time.sleep(delay)
        if mt5.initialize():
            logging.info("MT5 initialized on retry.")
            return True
    logging.error("MT5 initialization failed after retries.")
    return False

def get_last_closed_price(symbol):
    """
    Returns the last closed price for a given symbol.
    """
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 2)
    if rates is not None and len(rates) >= 2:
        return rates[-2]['close']
    return None

def get_risk_values(account_balance, risk_percent=1.5):
    """
    Calculates risk per trade in USD and the lot size (requires pip value, SL, etc.).
    """
    risk_usd = account_balance * risk_percent / 100
    return risk_usd

def is_position_opened(symbol, ticket):
    """
    Returns True if the given ticket is still open for symbol; False if closed; None on error.
    """
    if not ensure_mt5_initialized():
        return None
    try:
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            logging.warning("MT5 positions_get returned None.")
            return None
        if not positions:
            return False
        if ticket is not None:
            for pos in positions:
                if pos.ticket == ticket:
                    return True
            return False
        return bool(positions)
    except Exception as e:
        logging.error(f"Error in is_position_opened: {e}")
        return None

def get_symbol_spread(symbol="EURUSD"):
    if not mt5.initialize():
        print("❌ Could not initialize MT5.")
        return 1.0
    tick = mt5.symbol_info_tick(symbol)
    mt5.shutdown()
    if not tick:
        print(f"❌ Could not get symbol info for {symbol}.")
        return 1.0
    spread = abs(tick.ask - tick.bid) * 10000  # pips for most majors
    return spread

def calculate_lot_size(entry, sl, risk_usd, pip_value=10, spread_pips=1, commission_per_lot=7):
    """
    Returns lot size so that (SL distance + spread + commission) risk <= risk_usd.
    """
    sl_pips = abs(entry - sl) * 10000
    if sl_pips == 0:
        return 0
    total_pip_risk = sl_pips + spread_pips
    per_lot_risk = pip_value * total_pip_risk + commission_per_lot
    lot_size = risk_usd / per_lot_risk
    return round(max(lot_size, 0.01), 2)  # don't allow negative/zero, min 0.01 lot


def open_trade_in_mt5(signal, risk_usd=20):
    """
    Opens a trade based on the GPT signal. Risk per trade is capped at $20 including spread+commission.
    Returns the MT5 order_send result or None on failure.
    """
    if not ensure_mt5_initialized():
        return None

    symbol = signal["symbol"]
    entry = signal["entry"]
    sl = signal["sl"]
    tp = signal["tp"]
    side = signal["signal"]

    pip_value = 10  # For EURUSD, GBPUSD etc. ($10 per pip per lot)
    spread_pips = get_symbol_spread(symbol) or 1.0
    commission_per_lot = 7.0

    lot_size = calculate_lot_size(entry, sl, risk_usd, pip_value, spread_pips, commission_per_lot)

    if lot_size <= 0:
        print("❌ Lot size is zero or negative—trade skipped!")
        return None

    # Direction: 0=buy, 1=sell
    if side == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": 10032024,
        "comment": "GPT Signal",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }

    result = mt5.order_send(request)
    if hasattr(result, "retcode") and result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"✅ Trade opened for {symbol}: {side} {lot_size} lots @ {price}")
        return result
    else:
        print(f"❌ Failed to open trade: {getattr(result, 'retcode', None)} - {getattr(result, 'comment', '')}")
        return result

def get_current_open_position():
    """
    Returns a dict with the current open position's details, or None if none.
    """
    if not ensure_mt5_initialized():
        return None
    positions = mt5.positions_get()
    if not positions:
        return None
    pos = positions[0]
    return {
        "symbol": pos.symbol,
        "side": "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
        "entry": pos.price_open,
        "sl": pos.sl,
        "tp": pos.tp,
        "ticket": pos.ticket
    }

def prefilter_instruments(symbols, min_atr=0.0003, min_volume=200):
    filtered = []
    for symbol in symbols:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 20)
        if rates is None or len(rates) < 15:
            continue
        df = pd.DataFrame(rates)
        atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
        last_vol = df['tick_volume'].iloc[-1]
        # Add more filters if needed (spread, price change, etc.)
        if atr >= min_atr and last_vol >= min_volume:
            filtered.append(symbol)
    return filtered