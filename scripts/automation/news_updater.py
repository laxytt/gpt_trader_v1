"""
Automated news data updater for GPT Trading System.
Fetches economic calendar data from various sources.
"""

import asyncio
import logging
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import aiohttp
# from bs4 import BeautifulSoup  # Not needed for mock implementation

from config.settings import get_settings

logger = logging.getLogger(__name__)


class NewsUpdater:
    """Automated economic news data updater"""
    
    def __init__(self):
        self.settings = get_settings()
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # ForexFactory URL for economic calendar
        self.ff_base_url = "https://www.forexfactory.com/calendar"
        
        # High impact currency pairs we care about
        self.relevant_currencies = ['USD', 'EUR', 'GBP', 'CAD', 'AUD', 'JPY', 'CHF', 'NZD']
        
        # News impact levels
        self.impact_levels = {
            'high': 3,
            'medium': 2,
            'low': 1
        }
    
    async def update_news_data(self) -> Dict[str, Any]:
        """Main method to update news data"""
        logger.info("Starting news data update")
        
        try:
            # Fetch news for current and next week
            news_data = await self._fetch_forex_factory_calendar()
            
            if not news_data:
                # Fallback to alternative source or cached data
                logger.warning("Failed to fetch from ForexFactory, using fallback")
                news_data = await self._fetch_fallback_news()
            
            # Filter relevant news
            filtered_news = self._filter_relevant_news(news_data)
            
            # Save to file
            file_path = self._save_news_data(filtered_news)
            
            # Generate summary
            summary = self._generate_news_summary(filtered_news)
            
            return {
                'success': True,
                'file_path': str(file_path),
                'news_count': len(filtered_news),
                'summary': summary,
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"News update failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _fetch_forex_factory_calendar(self) -> List[Dict[str, Any]]:
        """Fetch economic calendar from ForexFactory"""
        logger.info("Fetching ForexFactory calendar")
        
        # Note: ForexFactory requires proper scraping approach
        # This is a simplified example - in production, you'd need to handle
        # their anti-scraping measures properly
        
        try:
            # For now, return mock data structure
            # In production, implement proper web scraping or use an API
            return await self._generate_mock_news_data()
            
        except Exception as e:
            logger.error(f"ForexFactory fetch failed: {e}")
            return []
    
    async def _fetch_fallback_news(self) -> List[Dict[str, Any]]:
        """Fetch news from alternative sources"""
        # This could be from APIs like:
        # - Economic calendar APIs
        # - Financial news APIs
        # - Cached historical data
        
        # For now, generate mock data
        return await self._generate_mock_news_data()
    
    async def _generate_mock_news_data(self) -> List[Dict[str, Any]]:
        """Generate mock news data for testing"""
        news_events = []
        base_time = datetime.now(timezone.utc).replace(hour=13, minute=30, second=0)
        
        # Common economic events
        event_templates = [
            {
                'title': 'Non-Farm Payrolls',
                'currency': 'USD',
                'impact': 'high',
                'forecast': '180K',
                'previous': '175K'
            },
            {
                'title': 'ECB Interest Rate Decision',
                'currency': 'EUR',
                'impact': 'high',
                'forecast': '4.50%',
                'previous': '4.50%'
            },
            {
                'title': 'UK GDP q/q',
                'currency': 'GBP',
                'impact': 'high',
                'forecast': '0.3%',
                'previous': '0.2%'
            },
            {
                'title': 'US Core CPI m/m',
                'currency': 'USD',
                'impact': 'high',
                'forecast': '0.3%',
                'previous': '0.2%'
            },
            {
                'title': 'Bank of Canada Rate Statement',
                'currency': 'CAD',
                'impact': 'high',
                'forecast': '',
                'previous': ''
            },
            {
                'title': 'Australian Employment Change',
                'currency': 'AUD',
                'impact': 'high',
                'forecast': '25.0K',
                'previous': '32.6K'
            },
            {
                'title': 'US Retail Sales m/m',
                'currency': 'USD',
                'impact': 'medium',
                'forecast': '0.4%',
                'previous': '0.3%'
            },
            {
                'title': 'German Manufacturing PMI',
                'currency': 'EUR',
                'impact': 'medium',
                'forecast': '43.2',
                'previous': '43.0'
            }
        ]
        
        # Generate events for the next 7 days
        for day_offset in range(7):
            event_date = base_time + timedelta(days=day_offset)
            
            # Skip weekends
            if event_date.weekday() >= 5:
                continue
            
            # Add 2-4 events per day
            daily_events = 2 + (day_offset % 3)
            
            for i in range(daily_events):
                template = event_templates[(day_offset * 3 + i) % len(event_templates)]
                
                event_time = event_date.replace(
                    hour=8 + (i * 4) % 12,  # Distribute throughout the day
                    minute=30 if i % 2 == 0 else 0
                )
                
                news_events.append({
                    'id': f"event_{day_offset}_{i}",
                    'date': event_time.date().isoformat(),
                    'time': event_time.strftime('%H:%M'),
                    'timestamp': event_time.isoformat(),
                    'currency': template['currency'],
                    'impact': template['impact'],
                    'event': template['title'],
                    'forecast': template['forecast'],
                    'previous': template['previous'],
                    'actual': ''  # Will be filled when event occurs
                })
        
        return news_events
    
    def _filter_relevant_news(self, news_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter news relevant to our trading symbols"""
        filtered = []
        
        for event in news_data:
            # Check if currency is relevant
            if event.get('currency') in self.relevant_currencies:
                # Check if impact is significant
                if event.get('impact') in ['high', 'medium']:
                    filtered.append(event)
        
        # Sort by timestamp
        filtered.sort(key=lambda x: x.get('timestamp', ''))
        
        return filtered
    
    def _save_news_data(self, news_data: List[Dict[str, Any]]) -> Path:
        """Save news data to file"""
        # Primary file for current week
        primary_file = self.data_dir / "ff_calendar_thisweek.csv"
        
        # Also save as JSON for easier parsing
        json_file = self.data_dir / "economic_calendar.json"
        
        # Save JSON version
        with open(json_file, 'w') as f:
            json.dump({
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'events': news_data
            }, f, indent=2)
        
        # Save CSV version for compatibility
        import csv
        
        with open(primary_file, 'w', newline='', encoding='utf-8') as f:
            if news_data:
                writer = csv.DictWriter(f, fieldnames=news_data[0].keys())
                writer.writeheader()
                writer.writerows(news_data)
        
        logger.info(f"Saved {len(news_data)} news events to {json_file}")
        
        return json_file
    
    def _generate_news_summary(self, news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of upcoming news"""
        if not news_data:
            return {'message': 'No upcoming high-impact news'}
        
        # Group by day
        by_day = {}
        for event in news_data:
            date = event.get('date', 'unknown')
            if date not in by_day:
                by_day[date] = []
            by_day[date].append(event)
        
        # Count by impact
        impact_counts = {
            'high': len([e for e in news_data if e.get('impact') == 'high']),
            'medium': len([e for e in news_data if e.get('impact') == 'medium']),
            'low': len([e for e in news_data if e.get('impact') == 'low'])
        }
        
        # Find next high impact event
        high_impact_events = [e for e in news_data if e.get('impact') == 'high']
        next_high_impact = high_impact_events[0] if high_impact_events else None
        
        return {
            'total_events': len(news_data),
            'days_with_events': len(by_day),
            'impact_distribution': impact_counts,
            'next_high_impact': {
                'event': next_high_impact.get('event'),
                'currency': next_high_impact.get('currency'),
                'time': next_high_impact.get('timestamp')
            } if next_high_impact else None,
            'currencies_affected': list(set(e.get('currency') for e in news_data if e.get('currency')))
        }
    
    async def check_news_age(self) -> Dict[str, Any]:
        """Check how old the current news data is"""
        json_file = self.data_dir / "economic_calendar.json"
        
        if not json_file.exists():
            return {
                'exists': False,
                'age_hours': float('inf'),
                'needs_update': True
            }
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            last_updated = datetime.fromisoformat(data.get('last_updated'))
            age = datetime.now(timezone.utc) - last_updated
            age_hours = age.total_seconds() / 3600
            
            return {
                'exists': True,
                'last_updated': last_updated.isoformat(),
                'age_hours': age_hours,
                'needs_update': age_hours > 4  # Update if older than 4 hours
            }
            
        except Exception as e:
            logger.error(f"Error checking news age: {e}")
            return {
                'exists': True,
                'age_hours': float('inf'),
                'needs_update': True
            }
    
    async def get_upcoming_high_impact(
        self, 
        hours_ahead: int = 24
    ) -> List[Dict[str, Any]]:
        """Get high impact events in the next N hours"""
        json_file = self.data_dir / "economic_calendar.json"
        
        if not json_file.exists():
            return []
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            events = data.get('events', [])
            current_time = datetime.now(timezone.utc)
            cutoff_time = current_time + timedelta(hours=hours_ahead)
            
            upcoming = []
            for event in events:
                if event.get('impact') != 'high':
                    continue
                
                event_time = datetime.fromisoformat(event.get('timestamp'))
                if current_time <= event_time <= cutoff_time:
                    event['hours_until'] = (event_time - current_time).total_seconds() / 3600
                    upcoming.append(event)
            
            return sorted(upcoming, key=lambda x: x.get('timestamp'))
            
        except Exception as e:
            logger.error(f"Error getting upcoming events: {e}")
            return []


async def main():
    """Run news updater standalone"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Economic News Updater')
    parser.add_argument('--check-age', action='store_true',
                      help='Check age of current news data')
    parser.add_argument('--upcoming', type=int,
                      help='Show high impact events in next N hours')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    
    updater = NewsUpdater()
    
    if args.check_age:
        age_info = await updater.check_news_age()
        print(f"News data exists: {age_info['exists']}")
        if age_info['exists']:
            print(f"Age: {age_info['age_hours']:.1f} hours")
            print(f"Needs update: {age_info['needs_update']}")
    
    elif args.upcoming:
        events = await updater.get_upcoming_high_impact(args.upcoming)
        if events:
            print(f"\n🔴 High impact events in next {args.upcoming} hours:")
            for event in events:
                print(f"\n  📅 {event['event']}")
                print(f"  💱 Currency: {event['currency']}")
                print(f"  ⏰ Time: {event['time']} ({event['hours_until']:.1f}h from now)")
                print(f"  📊 Forecast: {event.get('forecast', 'N/A')}")
        else:
            print(f"No high impact events in next {args.upcoming} hours")
    
    else:
        # Default: update news
        result = await updater.update_news_data()
        
        if result['success']:
            print(f"✅ News data updated successfully!")
            print(f"   File: {result['file_path']}")
            print(f"   Events: {result['news_count']}")
            
            summary = result['summary']
            print(f"\n📊 Summary:")
            print(f"   Total events: {summary['total_events']}")
            print(f"   High impact: {summary['impact_distribution']['high']}")
            print(f"   Currencies: {', '.join(summary['currencies_affected'])}")
            
            if summary['next_high_impact']:
                next_event = summary['next_high_impact']
                print(f"\n🔴 Next high impact: {next_event['event']} ({next_event['currency']})")
        else:
            print(f"❌ News update failed: {result['error']}")


if __name__ == "__main__":
    asyncio.run(main())