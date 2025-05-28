# core/services/news_updater.py
"""Service for updating economic news data from external sources"""

import json
import logging
import aiohttp
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from pathlib import Path

from core.domain.exceptions import NewsDataError, ErrorContext
from config.settings import NewsSettings

logger = logging.getLogger(__name__)


class NewsDataUpdater:
    """Updates economic news data from various sources"""
    
    def __init__(self, news_config: NewsSettings):
        self.news_config = news_config
        self.news_file_path = Path(news_config.file_path)
        self.update_interval_hours = 24  # Update daily
        
        # ForexFactory scraper endpoint (you'll need to implement or use an API)
        self.ff_api_url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        
    async def update_news_data(self) -> bool:
        """
        Update news data from external source.
        
        Returns:
            True if update successful
        """
        try:
            # Check if update is needed
            if not self._should_update():
                logger.info("News data is up to date")
                return True
            
            logger.info("Updating news data...")
            
            # Fetch new data
            news_data = await self._fetch_news_data()
            
            if not news_data:
                logger.error("Failed to fetch news data")
                return False
            
            # Save to file
            self._save_news_data(news_data)
            
            logger.info(f"News data updated successfully - {len(news_data)} events")
            return True
            
        except Exception as e:
            logger.error(f"News update failed: {e}")
            return False
    
    def _should_update(self) -> bool:
        """Check if news data needs updating"""
        if not self.news_file_path.exists():
            return True
        
        file_age = datetime.now() - datetime.fromtimestamp(
            self.news_file_path.stat().st_mtime
        )
        
        return file_age.total_seconds() > (self.update_interval_hours * 3600)
    
    async def _fetch_news_data(self) -> Optional[List[Dict[str, Any]]]:
        """Fetch news data from ForexFactory or alternative source"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.ff_api_url, timeout=30) as response:
                    if response.status == 200:
                        raw_data = await response.json()
                        return self._transform_news_data(raw_data)
                    else:
                        logger.error(f"News API returned status {response.status}")
                        return None
                        
        except asyncio.TimeoutError:
            logger.error("News data fetch timed out")
            return None
        except Exception as e:
            logger.error(f"Failed to fetch news data: {e}")
            return None
    
    def _transform_news_data(self, raw_data: Any) -> List[Dict[str, Any]]:
        """Transform raw news data to our format"""
        transformed = []
        
        # This depends on the actual API response format
        # Here's an example transformation:
        for event in raw_data:
            try:
                transformed_event = {
                    "date": event.get("date"),
                    "country": event.get("country", "").upper(),
                    "title": event.get("title", ""),
                    "impact": event.get("impact", "").lower(),
                    "forecast": event.get("forecast"),
                    "previous": event.get("previous"),
                    "actual": event.get("actual")
                }
                transformed.append(transformed_event)
            except Exception as e:
                logger.warning(f"Failed to transform event: {e}")
                
        return transformed
    
    def _save_news_data(self, news_data: List[Dict[str, Any]]):
        """Save news data to file"""
        # Create backup of old data
        if self.news_file_path.exists():
            backup_path = self.news_file_path.with_suffix('.json.backup')
            self.news_file_path.rename(backup_path)
        
        # Save new data
        with open(self.news_file_path, 'w', encoding='utf-8') as f:
            json.dump(news_data, f, indent=2, ensure_ascii=False)
    
    async def create_mock_news_data(self):
        """Create mock news data for testing when API is unavailable"""
        logger.info("Creating mock news data for testing...")
        
        now = datetime.now(timezone.utc)
        mock_events = []
        
        # Create events for the next 7 days
        for day in range(7):
            date = now + timedelta(days=day)
            
            # Add some mock events
            events_per_day = [
                {
                    "date": date.replace(hour=13, minute=30).isoformat(),
                    "country": "USD",
                    "title": "Unemployment Claims",
                    "impact": "medium",
                    "forecast": "220K",
                    "previous": "218K"
                },
                {
                    "date": date.replace(hour=15, minute=0).isoformat(),
                    "country": "EUR",
                    "title": "ECB Interest Rate Decision",
                    "impact": "high",
                    "forecast": "4.50%",
                    "previous": "4.50%"
                }
            ]
            
            mock_events.extend(events_per_day)
        
        self._save_news_data(mock_events)
        logger.info(f"Created {len(mock_events)} mock news events")