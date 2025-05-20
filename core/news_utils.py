import json
from datetime import datetime, timedelta, timezone
import os

NEWS_FILE = "data/forexfactory_week.json"

def load_news_data():
    """
    Loads news events from the JSON file.
    """
    if not os.path.exists(NEWS_FILE):
        print(f"⚠️ News file {NEWS_FILE} missing.")
        return []
    try:
        with open(NEWS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading news: {e}")
        return []

def parse_event_datetime(event):
    """
    Parses the event's date string into a timezone-aware UTC datetime.
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

def get_upcoming_news(within_minutes=15, now=None, currencies=("USD", "EUR")):
    """
    Returns list of news events within 'within_minutes' from now for given currencies.
    """
    if now is None:
        now = datetime.now(timezone.utc)
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

def find_next_high_impact_news(now=None, currencies=("USD", "EUR")):
    """
    Returns the next high-impact news event for the specified currencies after 'now'.
    """
    if now is None:
        now = datetime.now(timezone.utc)
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
