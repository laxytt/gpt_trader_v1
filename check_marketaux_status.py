#!/usr/bin/env python3
"""Check MarketAux API status and usage"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_settings
from core.infrastructure.marketaux.client import MarketAuxClient

async def check_marketaux():
    """Check MarketAux status and test API"""
    settings = get_settings()
    
    if not settings.marketaux.enabled:
        print("❌ MarketAux is disabled in settings")
        return
    
    # Initialize client
    cache_path = settings.paths.data_dir / "marketaux_cache.db"
    client = MarketAuxClient(
        api_token=settings.marketaux.api_token,
        cache_db_path=cache_path,
        daily_limit=settings.marketaux.daily_limit,
        requests_per_minute=settings.marketaux.requests_per_minute
    )
    
    # Get analytics
    print("📊 MarketAux Analytics:")
    analytics = client.get_analytics()
    usage = analytics.get('usage_stats', {})
    print(f"  - Requests today: {usage.get('requests_today', 0)}/{usage.get('daily_limit', 100)}")
    print(f"  - Remaining today: {usage.get('remaining_today', 0)}")
    print(f"  - Can make request: {usage.get('can_make_request', False)}")
    
    # Show recent usage
    if 'daily_breakdown' in analytics:
        print("\n📈 Recent usage:")
        for day in analytics['daily_breakdown'][:3]:
            print(f"  - {day['date']}: {day['requests']} requests, {day['articles']} articles")
    
    # Test API with a simple query
    print("\n🔍 Testing API with basic query...")
    try:
        response = await client.get_news(
            countries=['us'],
            search='economy',
            published_after=datetime.now(timezone.utc) - timedelta(hours=24),
            limit=5,
            use_cache=False  # Force fresh request
        )
        
        print(f"✅ API Response: Found {len(response.data)} articles")
        if response.data:
            print("\nSample articles:")
            for i, article in enumerate(response.data[:3]):
                print(f"{i+1}. {article.title}")
                print(f"   Published: {article.published_at}")
                print(f"   Source: {article.source}")
                print(f"   Relevance: {article.relevance_score}")
        else:
            print("⚠️ No articles found - this might indicate:")
            print("  - API limit reached")
            print("  - No matching news in timeframe")
            print("  - API key issues")
            
            # Check if we have cached articles
            print("\n📦 Checking cache...")
            cache_stats = client.cache.get_cache_stats()
            print(f"  - Total cached articles: {cache_stats['total_articles']}")
            print(f"  - Unique symbols: {cache_stats['unique_symbols']}")
            
    except Exception as e:
        print(f"❌ API Error: {e}")
        print("\nPossible issues:")
        print("  1. Invalid API key")
        print("  2. Daily limit exceeded")
        print("  3. Network issues")
    
    finally:
        # Close the client properly
        await client.close()

if __name__ == "__main__":
    asyncio.run(check_marketaux())