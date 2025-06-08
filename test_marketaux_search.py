#!/usr/bin/env python3
"""Test different MarketAux search parameters to find what works"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_settings
from core.infrastructure.marketaux.client import MarketAuxClient

async def test_searches():
    """Test various search parameters"""
    settings = get_settings()
    
    if not settings.marketaux.enabled:
        print("❌ MarketAux is disabled")
        return
    
    # Initialize client
    cache_path = settings.paths.data_dir / "marketaux_cache.db"
    client = MarketAuxClient(
        api_token=settings.marketaux.api_token,
        cache_db_path=cache_path,
        daily_limit=settings.marketaux.daily_limit,
        requests_per_minute=settings.marketaux.requests_per_minute
    )
    
    # Different search configurations to test
    test_configs = [
        {
            "name": "Simple US news",
            "params": {
                "countries": ['us'],
                "published_after": datetime.now(timezone.utc) - timedelta(hours=24),
                "limit": 5
            }
        },
        {
            "name": "UK + US with forex search",
            "params": {
                "countries": ['gb', 'us'],
                "search": 'forex',
                "published_after": datetime.now(timezone.utc) - timedelta(hours=24),
                "limit": 5
            }
        },
        {
            "name": "Currency news",
            "params": {
                "search": 'currency',
                "published_after": datetime.now(timezone.utc) - timedelta(hours=24),
                "limit": 5
            }
        },
        {
            "name": "No search, just countries",
            "params": {
                "countries": ['us'],
                "published_after": datetime.now(timezone.utc) - timedelta(hours=48),
                "limit": 10
            }
        },
        {
            "name": "Finance filter",
            "params": {
                "countries": ['us'],
                "industries": ['Finance'],
                "published_after": datetime.now(timezone.utc) - timedelta(hours=24),
                "limit": 5
            }
        }
    ]
    
    for config in test_configs:
        print(f"\n🔍 Testing: {config['name']}")
        print(f"   Params: {config['params']}")
        
        try:
            response = await client.get_news(
                **config['params'],
                use_cache=False  # Force fresh requests
            )
            
            print(f"✅ Found {len(response.data)} articles")
            if response.data:
                print("   Sample titles:")
                for article in response.data[:3]:
                    print(f"   - {article.title[:80]}...")
            
            # Small delay between requests
            await asyncio.sleep(2)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            break
    
    # Check final usage
    print("\n📊 Final API usage:")
    analytics = client.get_analytics()
    usage = analytics.get('usage_stats', {})
    print(f"  - Requests today: {usage.get('requests_today', 0)}/{usage.get('daily_limit', 100)}")
    
    # Close aiohttp session properly
    await client.close()

if __name__ == "__main__":
    asyncio.run(test_searches())