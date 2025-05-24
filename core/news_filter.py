import logging
from datetime import datetime, timedelta, timezone
from core.news_utils import get_upcoming_news, parse_event_datetime
from config import EVENT_BLACKLIST_BY_SYMBOL, EVENT_LOOKAHEAD_MINUTES

logger = logging.getLogger(__name__)

def is_event_restricted(event: dict, symbol: str) -> bool:
    """
    Check if the event's title matches a restricted pattern.
    """
    blacklist = EVENT_BLACKLIST_BY_SYMBOL.get(symbol.upper(), [])
    title = event.get("title", "").lower()
    return any(restricted.lower() in title for restricted in blacklist)

def is_news_restricted_now(
    symbol: str,
    now: datetime = None,
    minutes_before: int = 2,
    minutes_after: int = 2,
    currencies: tuple[str] = ("USD", "EUR")
) -> bool:
    """
    Returns True if a restricted macro event is within the danger window.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    news = get_upcoming_news(
        within_minutes=EVENT_LOOKAHEAD_MINUTES,
        now=now,
        symbol=symbol,
        currencies=currencies
    )

    for event in news:
        try:
            event_time = parse_event_datetime(event)
        except Exception as e:
            logger.warning(f"Could not parse event time: {e}")
            continue

        if not event_time:
            continue

        delta = (event_time - now).total_seconds() / 60
        if -minutes_before <= delta <= minutes_after and is_event_restricted(event, symbol):
            logger.info(f"🚨 Restricted event '{event['title']}' within {delta:.1f} minutes of now.")
            return True
    return False
