import json
import os
from datetime import datetime, timedelta, timezone

NEWS_FILE = "data/forexfactory_week.json"

# Optional: for per-symbol currency mapping in multi-asset scans
SYMBOL_CURRENCY_MAP = {
    "EURUSD": ("EUR", "USD"),
    "GBPUSD": ("GBP", "USD"),
    "USDJPY": ("USD", "JPY"),
    "USDCAD": ("USD", "CAD"),
    "AUDUSD": ("AUD", "USD"),
    # Add more as needed
}

def load_news_data(news_file=NEWS_FILE):
    """
    Loads news events from a JSON file.
    """
    if not os.path.exists(news_file):
        print(f"⚠️ News file {news_file} missing.")
        return []
    try:
        with open(news_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading news: {e}")
        return []

def parse_event_datetime(event):
    """
    Parses an event's date string to a timezone-aware UTC datetime.
    """
    date_str = event.get("date")
    if not date_str:
        return None
    try:
        # Try parsing with timezone (e.g., 2025-05-11T14:00:00-04:00)
        dt = datetime.fromisoformat(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        try:
            # Fallback: naive parse, assume UTC
            return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        except Exception:
            return None

def get_upcoming_news(within_minutes=15, now=None, currencies=None, symbol=None):
    """
    Returns news events for given currencies within 'within_minutes' from 'now'.
    If 'symbol' is given, uses SYMBOL_CURRENCY_MAP for currency selection.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if currencies is None and symbol:
        currencies = SYMBOL_CURRENCY_MAP.get(symbol, ("USD", "EUR"))
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
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)
        if window_start <= event_time <= window_end:
            if event.get("country") in currencies:
                filtered.append(event)
    return filtered

def find_next_high_impact_news(now=None, currencies=None, symbol=None):
    """
    Returns the next high-impact news event for the given currencies after 'now'.
    If 'symbol' is provided, uses currency map.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if currencies is None and symbol:
        currencies = SYMBOL_CURRENCY_MAP.get(symbol, ("USD", "EUR"))
    elif currencies is None:
        currencies = ("USD", "EUR")
    news = load_news_data()
    high_impact = []
    for event in news:
        event_time = parse_event_datetime(event)
        if not event_time or event.get("impact", "").lower() != "high":
            continue
        if event.get("country") in currencies and event_time >= now:
            high_impact.append((event_time, event))
    high_impact.sort()
    return high_impact[0][1] if high_impact else None
