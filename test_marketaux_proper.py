#!/usr/bin/env python3
"""Test MarketAux API with proper parameters based on documentation"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_settings
from core.infrastructure.marketaux.client import MarketAuxClient

async def test_marketaux_proper():
    """Test MarketAux API using proper approach from documentation"""
    settings = get_settings()
    
    # Initialize client
    cache_path = settings.paths.data_dir / "marketaux_cache.db"
    client = MarketAuxClient(
        api_token=settings.marketaux.api_token,
        cache_db_path=cache_path,
        daily_limit=settings.marketaux.daily_limit,
        requests_per_minute=settings.marketaux.requests_per_minute
    )
    
    print("🔍 Testing MarketAux API with proper parameters...\n")
    
    # Test 1: Direct forex symbols with entity_types
    print("1️⃣ Forex symbols with currency entity type:")
    try:
        response = await client.get_news(
            symbols=['EURUSD', 'GBPUSD'],
            entity_types=['currency'],
            published_after=datetime.now(timezone.utc) - timedelta(days=7),
            limit=10,
            language='en',
            use_cache=False
        )
        print(f"   Found {len(response.data)} articles")
        for i, article in enumerate(response.data[:3], 1):
            print(f"   {i}. {article.title[:70]}...")
    except Exception as e:
        print(f"   Error: {e}")
    
    await asyncio.sleep(2)
    
    # Test 2: Search with proper syntax
    print("\n2️⃣ Advanced search syntax (forex OR currency):")
    try:
        response = await client.get_news(
            search='forex | currency | "interest rate" | inflation',
            entity_types=['currency'],
            published_after=datetime.now(timezone.utc) - timedelta(days=3),
            limit=10,
            language='en',
            use_cache=False
        )
        print(f"   Found {len(response.data)} articles")
        for i, article in enumerate(response.data[:3], 1):
            print(f"   {i}. {article.title[:70]}...")
            if hasattr(article, 'entities') and article.entities:
                currencies = [e for e in article.entities if e.get('type') == 'currency']
                if currencies:
                    print(f"      Currencies: {[c.get('name') for c in currencies[:3]]}")
    except Exception as e:
        print(f"   Error: {e}")
    
    await asyncio.sleep(2)
    
    # Test 3: By countries with financial filter
    print("\n3️⃣ Countries with financial industry:")
    try:
        response = await client.get_news(
            countries=['us', 'gb'],
            industries=['Finance', 'Financial Markets'],
            published_after=datetime.now(timezone.utc) - timedelta(days=1),
            limit=10,
            language='en',
            use_cache=False
        )
        print(f"   Found {len(response.data)} articles")
        for i, article in enumerate(response.data[:3], 1):
            print(f"   {i}. {article.title[:70]}...")
    except Exception as e:
        print(f"   Error: {e}")
    
    await asyncio.sleep(2)
    
    # Test 4: Combined approach
    print("\n4️⃣ Combined: USD/EUR with economic search:")
    try:
        response = await client.get_news(
            symbols=['USD', 'EUR', 'GBP'],  # Individual currencies
            search='"central bank" | "federal reserve" | ECB | inflation',
            published_after=datetime.now(timezone.utc) - timedelta(days=2),
            limit=10,
            language='en',
            use_cache=False
        )
        print(f"   Found {len(response.data)} articles")
        for i, article in enumerate(response.data[:3], 1):
            print(f"   {i}. {article.title[:70]}...")
            # Check sentiment
            if hasattr(article, 'sentiment'):
                print(f"      Sentiment: {article.sentiment}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Check final usage
    print("\n📊 API usage:")
    analytics = client.get_analytics()
    usage = analytics.get('usage_stats', {})
    print(f"   Requests today: {usage.get('requests_today', 0)}/{usage.get('daily_limit', 100)}")
    
    # Ensure session is closed properly
    await client.close()

if __name__ == "__main__":
    asyncio.run(test_marketaux_proper())