"""
News service for managing economic news events and trading restrictions.
Handles news data loading, filtering, and trading restriction logic.
"""

import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone, timedelta
from pathlib import Path

from core.domain.models import NewsEvent
from core.domain.exceptions import (
    NewsError, NewsDataError, NewsRestrictionError, ErrorContext
)
from core.domain.enums import NewsBlacklists, Currency
from config.settings import NewsSettings


logger = logging.getLogger(__name__)


class NewsDataLoader:
    """Handles loading and parsing of news data from various sources"""
    
    def __init__(self, news_config: NewsSettings):
        self.news_config = news_config
        self.news_file_path = Path(news_config.file_path)
        self._cached_news = None
        self._cache_timestamp = None
        self._cache_ttl = timedelta(minutes=30)  # Cache for 30 minutes
    
    def load_news_data(self, force_reload: bool = False) -> List[Dict[str, Any]]:
        """
        Load news data from file with caching.
        
        Args:
            force_reload: Force reload even if cached data exists
            
        Returns:
            List of news event dictionaries
            
        Raises:
            NewsDataError: If loading fails
        """
        # Check cache validity
        if not force_reload and self._is_cache_valid():
            logger.debug("Using cached news data")
            return self._cached_news
        
        with ErrorContext("News data loading"):
            try:
                if not self.news_file_path.exists():
                    logger.warning(f"News file not found: {self.news_file_path}")
                    return []
                
                # Check file age
                file_age = datetime.now() - datetime.fromtimestamp(
                    self.news_file_path.stat().st_mtime
                )
                
                if file_age > timedelta(days=7):
                    logger.warning(f"News file is {file_age.days} days old")
                
                # Load and parse JSON
                with open(self.news_file_path, 'r', encoding='utf-8') as f:
                    news_data = json.load(f)
                
                if not isinstance(news_data, list):
                    raise NewsDataError("News data must be a list")
                
                # Cache the data
                self._cached_news = news_data
                self._cache_timestamp = datetime.now()
                
                logger.info(f"Loaded {len(news_data)} news events from {self.news_file_path}")
                return news_data
                
            except json.JSONDecodeError as e:
                raise NewsDataError(f"Invalid JSON in news file: {str(e)}")
            except Exception as e:
                raise NewsDataError(f"Failed to load news data: {str(e)}")
    
    def _is_cache_valid(self) -> bool:
        """Check if cached data is still valid"""
        if self._cached_news is None or self._cache_timestamp is None:
            return False
        
        age = datetime.now() - self._cache_timestamp
        return age < self._cache_ttl
    
    def get_file_age(self) -> Optional[timedelta]:
        """Get age of the news data file"""
        try:
            if not self.news_file_path.exists():
                return None
            
            mtime = datetime.fromtimestamp(self.news_file_path.stat().st_mtime)
            return datetime.now() - mtime
        except Exception:
            return None


class NewsEventParser:
    """Parses and validates news event data"""
    
    @staticmethod
    def parse_event_datetime(event: Dict[str, Any]) -> Optional[datetime]:
        """
        Parse event datetime from various formats.
        
        Args:
            event: Event dictionary
            
        Returns:
            Parsed datetime or None if parsing fails
        """
        date_str = event.get("date") or event.get("timestamp")
        if not date_str:
            return None
        
        try:
            # Try ISO format first
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            
            # Ensure UTC timezone
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            
            return dt
            
        except ValueError:
            # Try other common formats
            formats = [
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%d/%m/%Y %H:%M",
                "%m/%d/%Y %H:%M"
            ]
            
            for fmt in formats:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            
            logger.warning(f"Could not parse date: {date_str}")
            return None
    
    @staticmethod
    def create_news_event(event_data: Dict[str, Any]) -> Optional[NewsEvent]:
        """
        Create NewsEvent from raw data.
        
        Args:
            event_data: Raw event dictionary
            
        Returns:
            NewsEvent object or None if creation fails
        """
        try:
            timestamp = NewsEventParser.parse_event_datetime(event_data)
            if not timestamp:
                return None
            
            return NewsEvent(
                timestamp=timestamp,
                country=event_data.get("country", "").upper(),
                title=event_data.get("title", ""),
                impact=event_data.get("impact", "").lower(),
                actual=event_data.get("actual"),
                forecast=event_data.get("forecast"),
                previous=event_data.get("previous")
            )
            
        except Exception as e:
            logger.warning(f"Failed to create NewsEvent: {e}")
            return None


class NewsFilter:
    """Filters news events based on symbols and time windows"""
    
    def __init__(self):
        self.symbol_currency_map = {
            "EURUSD": ("EUR", "USD"),
            "GBPUSD": ("GBP", "USD"),
            "USDJPY": ("USD", "JPY"),
            "USDCAD": ("USD", "CAD"),
            "AUDUSD": ("AUD", "USD"),
            "NZDUSD": ("NZD", "USD"),
            "USDCHF": ("USD", "CHF"),
            "XAUUSD": ("USD",),  # Gold typically responds to USD news
            "GER40.cash": ("EUR",),
            "US30.cash": ("USD",),
            "US100.cash": ("USD",),
            "UK100.cash": ("GBP",)
        }
        
        self.blacklists = NewsBlacklists.EVENT_BLACKLIST_BY_SYMBOL
    
    def get_symbol_currencies(self, symbol: str) -> Tuple[str, ...]:
        """
        Get relevant currencies for a trading symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Tuple of currency codes
        """
        return self.symbol_currency_map.get(symbol.upper(), ("USD", "EUR"))
    
    def filter_events_by_time(
        self,
        events: List[NewsEvent],
        start_time: datetime,
        end_time: datetime
    ) -> List[NewsEvent]:
        """
        Filter events by time window.
        
        Args:
            events: List of news events
            start_time: Window start time
            end_time: Window end time
            
        Returns:
            Filtered list of events
        """
        return [
            event for event in events
            if start_time <= event.timestamp <= end_time
        ]
    
    def filter_events_by_symbol(
        self,
        events: List[NewsEvent],
        symbol: str
    ) -> List[NewsEvent]:
        """
        Filter events relevant to a trading symbol.
        
        Args:
            events: List of news events
            symbol: Trading symbol
            
        Returns:
            Filtered list of events
        """
        relevant_currencies = self.get_symbol_currencies(symbol)
        
        return [
            event for event in events
            if event.country in relevant_currencies
        ]
    
    def filter_high_impact_events(self, events: List[NewsEvent]) -> List[NewsEvent]:
        """
        Filter only high-impact events.
        
        Args:
            events: List of news events
            
        Returns:
            List of high-impact events
        """
        return [event for event in events if event.is_high_impact]
    
    def is_event_blacklisted(self, event: NewsEvent, symbol: str) -> bool:
        """
        Check if event is blacklisted for a symbol.
        
        Args:
            event: News event to check
            symbol: Trading symbol
            
        Returns:
            True if event is blacklisted
        """
        blacklist = self.blacklists.get(symbol.upper(), [])
        
        event_title_lower = event.title.lower()
        return any(
            blacklisted_term.lower() in event_title_lower
            for blacklisted_term in blacklist
        )


class NewsService:
    """
    Main service for managing news events and trading restrictions.
    """
    
    def __init__(self, news_config: NewsSettings):
        self.news_config = news_config
        self.data_loader = NewsDataLoader(news_config)
        self.event_parser = NewsEventParser()
        self.news_filter = NewsFilter()
    
    async def get_upcoming_news(
        self,
        symbol: str,
        within_minutes: Optional[int] = None,
        now: Optional[datetime] = None
    ) -> List[NewsEvent]:
        """
        Get upcoming news events for a symbol within time window.
        
        Args:
            symbol: Trading symbol
            within_minutes: Time window in minutes (uses config default if None)
            now: Reference time (uses current time if None)
            
        Returns:
            List of upcoming NewsEvent objects
        """
        if within_minutes is None:
            within_minutes = self.news_config.lookahead_minutes
        
        if now is None:
            now = datetime.now(timezone.utc)
        
        with ErrorContext("Get upcoming news", symbol=symbol) as ctx:
            ctx.add_detail("within_minutes", within_minutes)
            
            try:
                # Load raw news data
                raw_events = self.data_loader.load_news_data()
                
                # Parse events
                news_events = []
                for event_data in raw_events:
                    event = self.event_parser.create_news_event(event_data)
                    if event:
                        news_events.append(event)
                
                # Filter by time window
                end_time = now + timedelta(minutes=within_minutes)
                upcoming_events = self.news_filter.filter_events_by_time(
                    news_events, now, end_time
                )
                
                # Filter by symbol relevance
                relevant_events = self.news_filter.filter_events_by_symbol(
                    upcoming_events, symbol
                )
                
                # Sort by timestamp
                relevant_events.sort(key=lambda x: x.timestamp)
                
                logger.debug(f"Found {len(relevant_events)} upcoming news events for {symbol}")
                return relevant_events
                
            except Exception as e:
                logger.error(f"Failed to get upcoming news for {symbol}: {e}")
                return []
    
    async def is_trading_restricted(
        self,
        symbol: str,
        now: Optional[datetime] = None,
        window_before_minutes: Optional[int] = None,
        window_after_minutes: Optional[int] = None
    ) -> bool:
        """
        Check if trading is restricted due to high-impact news.
        
        Args:
            symbol: Trading symbol
            now: Reference time
            window_before_minutes: Minutes before event to restrict
            window_after_minutes: Minutes after event to restrict
            
        Returns:
            True if trading should be restricted
        """
        if now is None:
            now = datetime.now(timezone.utc)
        
        if window_before_minutes is None:
            window_before_minutes = self.news_config.restriction_window_before
        
        if window_after_minutes is None:
            window_after_minutes = self.news_config.restriction_window_after
        
        with ErrorContext("Check trading restriction", symbol=symbol):
            try:
                # Get news events in restriction window
                total_window = window_before_minutes + window_after_minutes + 5  # 5 min buffer
                upcoming_events = await self.get_upcoming_news(symbol, total_window, now)
                
                # Check each event
                for event in upcoming_events:
                    # Only check high-impact events
                    if not event.is_high_impact:
                        continue
                    
                    # Check if event is blacklisted for this symbol
                    if not self.news_filter.is_event_blacklisted(event, symbol):
                        continue
                    
                    # Calculate time difference
                    time_diff = (event.timestamp - now).total_seconds() / 60
                    
                    # Check if within restriction window
                    if -window_after_minutes <= time_diff <= window_before_minutes:
                        logger.info(
                            f"Trading restricted for {symbol} due to {event.title} "
                            f"in {time_diff:.1f} minutes"
                        )
                        return True
                
                return False
                
            except Exception as e:
                logger.error(f"Error checking news restrictions for {symbol}: {e}")
                # Return True (restrict) on error to be safe
                return True
    
    async def get_next_high_impact_event(
        self,
        symbol: str,
        now: Optional[datetime] = None
    ) -> Optional[NewsEvent]:
        """
        Get the next high-impact event for a symbol.
        
        Args:
            symbol: Trading symbol
            now: Reference time
            
        Returns:
            Next high-impact NewsEvent or None
        """
        if now is None:
            now = datetime.now(timezone.utc)
        
        try:
            # Get events for next 7 days
            upcoming_events = await self.get_upcoming_news(
                symbol, within_minutes=7 * 24 * 60, now=now
            )
            
            # Filter high-impact events
            high_impact_events = self.news_filter.filter_high_impact_events(upcoming_events)
            
            # Filter blacklisted events
            blacklisted_events = [
                event for event in high_impact_events
                if self.news_filter.is_event_blacklisted(event, symbol)
            ]
            
            return blacklisted_events[0] if blacklisted_events else None
            
        except Exception as e:
            logger.error(f"Error getting next high-impact event for {symbol}: {e}")
            return None
    
    async def get_news_summary(self, symbol: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get news summary for a symbol within specified hours.
        
        Args:
            symbol: Trading symbol
            hours: Time window in hours
            
        Returns:
            Dictionary with news summary
        """
        try:
            now = datetime.now(timezone.utc)
            events = await self.get_upcoming_news(symbol, hours * 60, now)
            
            high_impact_count = len(self.news_filter.filter_high_impact_events(events))
            blacklisted_count = len([
                event for event in events
                if self.news_filter.is_event_blacklisted(event, symbol)
            ])
            
            next_high_impact = await self.get_next_high_impact_event(symbol, now)
            is_restricted = await self.is_trading_restricted(symbol, now)
            
            return {
                'symbol': symbol,
                'total_events': len(events),
                'high_impact_events': high_impact_count,
                'blacklisted_events': blacklisted_count,
                'next_high_impact': {
                    'title': next_high_impact.title,
                    'timestamp': next_high_impact.timestamp.isoformat(),
                    'country': next_high_impact.country
                } if next_high_impact else None,
                'trading_restricted': is_restricted,
                'currencies': self.news_filter.get_symbol_currencies(symbol)
            }
            
        except Exception as e:
            logger.error(f"Error generating news summary for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'trading_restricted': True  # Safe default
            }
    
    def get_data_file_status(self) -> Dict[str, Any]:
        """
        Get status of the news data file.
        
        Returns:
            Dictionary with file status information
        """
        file_age = self.data_loader.get_file_age()
        
        return {
            'file_path': str(self.data_loader.news_file_path),
            'file_exists': self.data_loader.news_file_path.exists(),
            'file_age_hours': file_age.total_seconds() / 3600 if file_age else None,
            'cache_valid': self.data_loader._is_cache_valid(),
            'cached_events_count': len(self.data_loader._cached_news) if self.data_loader._cached_news else 0
        }


# Export main service and components
__all__ = [
    'NewsService', 'NewsDataLoader', 'NewsEventParser', 'NewsFilter'
]