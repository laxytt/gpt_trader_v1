#!/usr/bin/env python3
"""Get any forex-related news with minimal filtering"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_settings
from core.infrastructure.marketaux.client import MarketAuxClient

async def get_forex_news():
    """Get forex news with different approaches"""
    settings = get_settings()
    
    # Initialize client
    cache_path = settings.paths.data_dir / "marketaux_cache.db"
    client = MarketAuxClient(
        api_token=settings.marketaux.api_token,
        cache_db_path=cache_path,
        daily_limit=settings.marketaux.daily_limit,
        requests_per_minute=settings.marketaux.requests_per_minute
    )
    
    print("🔍 Testing different approaches to get forex news...\n")
    
    # Approach 1: Just search for forex/currency terms
    print("1️⃣ Search-based approach (forex/currency):")
    try:
        response = await client.get_news(
            search='forex OR currency OR "exchange rate" OR dollar OR euro',
            published_after=datetime.now(timezone.utc) - timedelta(hours=48),
            limit=10,
            use_cache=False
        )
        print(f"   Found {len(response.data)} articles")
        for i, article in enumerate(response.data[:3], 1):
            print(f"   {i}. {article.title[:60]}...")
    except Exception as e:
        print(f"   Error: {e}")
    
    await asyncio.sleep(2)
    
    # Approach 2: Financial/economic news
    print("\n2️⃣ Economic news approach:")
    try:
        response = await client.get_news(
            search='economy OR inflation OR "interest rate" OR "central bank"',
            published_after=datetime.now(timezone.utc) - timedelta(hours=48),
            limit=10,
            use_cache=False
        )
        print(f"   Found {len(response.data)} articles")
        for i, article in enumerate(response.data[:3], 1):
            print(f"   {i}. {article.title[:60]}...")
            # Check content
            text = f"{article.title} {article.description}".lower()
            if any(term in text for term in ['rate', 'inflation', 'bank', 'economy']):
                print(f"      ✓ Contains economic terms")
    except Exception as e:
        print(f"   Error: {e}")
    
    await asyncio.sleep(2)
    
    # Approach 3: No filters at all - just get latest news
    print("\n3️⃣ Unfiltered approach (latest news):")
    try:
        response = await client.get_news(
            published_after=datetime.now(timezone.utc) - timedelta(hours=12),
            limit=20,
            use_cache=False
        )
        print(f"   Found {len(response.data)} articles")
        
        # Filter for financial content
        financial_articles = []
        for article in response.data:
            text = f"{article.title} {article.description}".lower()
            financial_terms = ['market', 'stock', 'economy', 'rate', 'bank', 'finance', 'trading', 'investment']
            if any(term in text for term in financial_terms):
                financial_articles.append(article)
        
        print(f"   Financial articles: {len(financial_articles)}")
        for i, article in enumerate(financial_articles[:3], 1):
            print(f"   {i}. {article.title[:60]}...")
            
    except Exception as e:
        print(f"   Error: {e}")
    
    # Check usage
    print("\n📊 API usage:")
    analytics = client.get_analytics()
    usage = analytics.get('usage_stats', {})
    print(f"   Requests today: {usage.get('requests_today', 0)}/{usage.get('daily_limit', 100)}")
    
    await client.close()

if __name__ == "__main__":
    asyncio.run(get_forex_news())