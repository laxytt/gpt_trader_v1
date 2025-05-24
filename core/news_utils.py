import json
import logging
import os
from datetime import datetime, timedelta, timezone
from config import NEWS_FILE

logger = logging.getLogger(__name__)

SYMBOL_CURRENCY_MAP = {
    "EURUSD": ("EUR", "USD"),
    "GBPUSD": ("GBP", "USD"),
    "USDJPY": ("USD", "JPY"),
    "USDCAD": ("USD", "CAD"),
    "AUDUSD": ("AUD", "USD"),
    "XAUUSD": ("USD",),  # gold typically responds to USD news
    "GER40.cash": ("EUR",),
    "US30.cash": ("USD",),
    # Add more as needed
}

NEWS_FILE = os.getenv("NEWS_FILE_DIR", "data/forexfactory_week.json")

def load_news_data(news_file: str = NEWS_FILE) -> list[dict]:
    """
    Loads news events from a JSON file.
    """
    if not os.path.exists(news_file):
        logger.warning(f"⚠️ News file {news_file} missing.")
        return []
    try:
        with open(news_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Error loading news: {e}")
        return []

def parse_event_datetime(event: dict) -> datetime | None:
    """
    Parses an event's datetime string to a timezone-aware UTC datetime.
    """
    date_str = event.get("date")
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        try:
            return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        except Exception:
            logger.warning(f"Could not parse date: {date_str}")
            return None

def get_upcoming_news(
    within_minutes: int = 15,
    now: datetime = None,
    currencies: tuple[str] = None,
    symbol: str = None
) -> list[dict]:
    """
    Returns upcoming macroeconomic events within the given time window.
    If symbol is specified, derives currencies from it.
    """
    within_minutes = int(within_minutes)

    if now is None:
        now = datetime.now(timezone.utc)
    if currencies is None and symbol:
        currencies = SYMBOL_CURRENCY_MAP.get(symbol.upper(), ("USD", "EUR"))
    elif currencies is None:
        currencies = ("USD", "EUR")

    news = load_news_data()
    filtered = []

    window_start = now
    window_end = now + timedelta(minutes=within_minutes)

    for event in news:
        event_time = parse_event_datetime(event)
        if not event_time:
            continue
        if window_start <= event_time <= window_end:
            if event.get("country") in currencies:
                filtered.append(event)

    return filtered

def find_next_high_impact_news(
    now: datetime = None,
    currencies: tuple[str] = None,
    symbol: str = None
) -> dict | None:
    """
    Returns the next high-impact macroeconomic event after 'now'.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if currencies is None and symbol:
        currencies = SYMBOL_CURRENCY_MAP.get(symbol.upper(), ("USD", "EUR"))
    elif currencies is None:
        currencies = ("USD", "EUR")

    news = load_news_data()
    high_impact_events = []

    for event in news:
        if event.get("impact", "").lower() != "high":
            continue
        event_time = parse_event_datetime(event)
        if not event_time or event_time < now:
            continue
        if event.get("country") in currencies:
            high_impact_events.append((event_time, event))

    high_impact_events.sort(key=lambda x: x[0])
    return high_impact_events[0][1] if high_impact_events else None
