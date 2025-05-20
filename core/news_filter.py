from datetime import datetime, timedelta, timezone
from core.news_utils import get_upcoming_news

# For production, consider loading this from config or file!
RESTRICTED_EVENTS = [
    "Federal Funds Rate", "FOMC Statement", "Non-Farm Employment Change",
    "Unemployment Rate", "Average Hourly Earnings", "Advance GDP q/q",
    "FOMC Meeting Minutes", "CPI y/y", "Main Refinancing Rate"
]

def is_event_restricted(event, restricted_events=RESTRICTED_EVENTS):
    """
    Checks if the event title matches any in the restricted event list.
    """
    title = event.get("title", "").lower()
    return any(restricted.lower() in title for restricted in restricted_events)

def is_news_restricted_now(symbol, now=None, minutes_before=2, minutes_after=2, currencies=("USD", "EUR")):
    """
    Returns True if a restricted macro event is within the danger window.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    news = get_upcoming_news(within_minutes=2880, now=now, currencies=currencies)  # large window
    for event in news:
        event_time = None
        try:
            # Expecting UTC-aware datetimes from news_utils
            from core.news_utils import parse_event_datetime
            event_time = parse_event_datetime(event)
        except Exception:
            continue
        if not event_time:
            continue

        # Check time window
        delta = (event_time - now).total_seconds() / 60
        if -minutes_before <= delta <= minutes_after:
            if is_event_restricted(event):
                print(f"🚨 Restricted event '{event['title']}' within danger window ({delta:.1f} min).")
                return True
    return False
