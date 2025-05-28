# core/domain/enums/mt5_enums.py
"""MetaTrader 5 specific enumerations"""

from enum import Enum, IntEnum

class TimeFrame(Enum):
    """MetaTrader 5 timeframe constants"""
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"
    MN1 = "MN1"

class MT5TimeFrame(IntEnum):
    """MT5 native timeframe values"""
    M1 = 1
    M5 = 5
    M15 = 15
    M30 = 30
    H1 = 16385
    H4 = 16388
    D1 = 16408
    W1 = 32769
    MN1 = 49153

class OrderType(IntEnum):
    """MT5 order types - only used types"""
    BUY = 0
    SELL = 1

class TradeAction(IntEnum):
    """MT5 trade actions - only used types"""
    DEAL = 1
    SLTP = 6

class ReturnCode(IntEnum):
    """MT5 return codes"""
    DONE = 10009
    COMMON_ERROR = 10004
    INVALID_REQUEST = 10013
    INVALID_VOLUME = 10014
    INVALID_PRICE = 10015
    INVALID_STOPS = 10016
    TRADE_DISABLED = 10017
    MARKET_CLOSED = 10018
    NO_MONEY = 10019
    PRICE_CHANGED = 10020
    OFF_QUOTES = 10021
    NO_CHANGES = 10025
    LOCKED = 10028
    FROZEN = 10029

# Utility mappings
TIMEFRAME_TO_MT5 = {
    TimeFrame.M1: MT5TimeFrame.M1,
    TimeFrame.M5: MT5TimeFrame.M5,
    TimeFrame.M15: MT5TimeFrame.M15,
    TimeFrame.M30: MT5TimeFrame.M30,
    TimeFrame.H1: MT5TimeFrame.H1,
    TimeFrame.H4: MT5TimeFrame.H4,
    TimeFrame.D1: MT5TimeFrame.D1,
    TimeFrame.W1: MT5TimeFrame.W1,
    TimeFrame.MN1: MT5TimeFrame.MN1
}

SIGNAL_TO_ORDER_TYPE = {
    "BUY": OrderType.BUY,
    "SELL": OrderType.SELL
}