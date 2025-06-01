# core/domain/enums/trading_enums.py
"""Trading domain enumerations"""

from enum import Enum

class NewsImpact(Enum):
    """News impact levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Currency(Enum):
    """Major currencies"""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    AUD = "AUD"
    CAD = "CAD"
    CHF = "CHF"
    NZD = "NZD"

class TrendDirection(Enum):
    """Market trend directions"""
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"
    UNDEFINED = "undefined"