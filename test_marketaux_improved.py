#!/usr/bin/env python3
"""Improved MarketAux API test for forex news fetching"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_settings
from core.infrastructure.marketaux.client import MarketAuxClient

async def test_marketaux_improved():
    """Test MarketAux API using improved approach based on findings"""
    settings = get_settings()
    
    # Initialize client
    cache_path = settings.paths.data_dir / "marketaux_cache.db"
    client = MarketAuxClient(
        api_token=settings.marketaux.api_token,
        cache_db_path=cache_path,
        daily_limit=settings.marketaux.daily_limit,
        requests_per_minute=settings.marketaux.requests_per_minute
    )
    
    try:
        print("🔍 Testing MarketAux API with improved parameters...\n")
        
        # Test 1: Broad forex search without entity_types filter
        print("1️⃣ Forex search without entity_types filter:")
        try:
            response = await client.get_news(
                search='forex | currency | "exchange rate" | "central bank" | EUR | USD | GBP | JPY',
                published_after=datetime.now(timezone.utc) - timedelta(days=3),
                limit=20,
                language='en',
                use_cache=False
            )
            print(f"   Found {len(response.data)} articles")
            for i, article in enumerate(response.data[:5], 1):
                print(f"   {i}. {article.title[:70]}...")
                # Check for forex-related content
                text = f"{article.title} {article.description or ''} {article.snippet or ''}".lower()
                forex_keywords = ['forex', 'currency', 'dollar', 'euro', 'pound', 'yen', 'exchange', 'fx', 
                                'usd', 'eur', 'gbp', 'jpy', 'central bank', 'fed', 'ecb', 'boe']
                matches = [kw for kw in forex_keywords if kw in text]
                if matches:
                    print(f"      Relevance: {matches[:5]}")
        except Exception as e:
            print(f"   Error: {e}")
        
        await asyncio.sleep(2)
        
        # Test 2: Individual currency symbols
        print("\n2️⃣ Individual currency symbols (USD, EUR, GBP):")
        try:
            response = await client.get_news(
                symbols=['USD', 'EUR', 'GBP', 'JPY'],
                published_after=datetime.now(timezone.utc) - timedelta(days=2),
                limit=20,
                language='en',
                use_cache=False
            )
            print(f"   Found {len(response.data)} articles")
            for i, article in enumerate(response.data[:5], 1):
                print(f"   {i}. {article.title[:70]}...")
                if hasattr(article, 'symbols') and article.symbols:
                    print(f"      Symbols: {article.symbols}")
        except Exception as e:
            print(f"   Error: {e}")
        
        await asyncio.sleep(2)
        
        # Test 3: Country-based with forex search terms
        print("\n3️⃣ Major economies with forex search:")
        try:
            response = await client.get_news(
                countries=['us', 'gb', 'de', 'jp'],  # US, UK, Germany (for EUR), Japan
                search='currency | "exchange rate" | forex | "monetary policy"',
                published_after=datetime.now(timezone.utc) - timedelta(days=1),
                limit=20,
                language='en',
                use_cache=False
            )
            print(f"   Found {len(response.data)} articles")
            for i, article in enumerate(response.data[:5], 1):
                print(f"   {i}. {article.title[:70]}...")
                if hasattr(article, 'countries') and article.countries:
                    print(f"      Countries: {article.countries}")
        except Exception as e:
            print(f"   Error: {e}")
        
        await asyncio.sleep(2)
        
        # Test 4: Test the raw API response structure
        print("\n4️⃣ Testing raw API response structure:")
        try:
            analysis = await client.test_api_response(limit=3)
            print(f"   Found {analysis.get('data_count', 0)} articles")
            print(f"   Article fields: {analysis.get('article_fields', [])[:10]}")
            print(f"   Entity types found: {analysis.get('entity_types', set())}")
            if analysis.get('sample_article'):
                article = analysis['sample_article']
                print(f"   Sample title: {article.get('title', '')[:70]}...")
                if 'entities' in article and article['entities']:
                    print(f"   Entity example: {article['entities'][0]}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Check final usage
        print("\n📊 API usage:")
        analytics = client.get_analytics()
        usage = analytics.get('usage_stats', {})
        print(f"   Requests today: {usage.get('requests_today', 0)}/{usage.get('daily_limit', 100)}")
        
    finally:
        # Ensure we close the session properly
        await client.close()
        print("\n✅ Session closed properly")

if __name__ == "__main__":
    asyncio.run(test_marketaux_improved())