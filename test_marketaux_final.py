#!/usr/bin/env python3
"""Final test of MarketAux forex news fetching with free plan optimizations"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_settings
from core.infrastructure.marketaux.client import MarketAuxClient
from core.infrastructure.marketaux.forex_utils import ForexSymbolConverter

async def test_marketaux_final():
    """Test final implementation with free plan optimizations"""
    settings = get_settings()
    
    # Initialize client with free plan mode
    cache_path = settings.paths.data_dir / "marketaux_cache.db"
    client = MarketAuxClient(
        api_token=settings.marketaux.api_token,
        cache_db_path=cache_path,
        daily_limit=settings.marketaux.daily_limit,
        requests_per_minute=settings.marketaux.requests_per_minute,
        free_plan=True  # Explicitly set free plan
    )
    
    try:
        print("🔍 Testing MarketAux with free plan optimizations...\n")
        
        # Test 1: Get forex news for EURUSD and GBPUSD
        print("1️⃣ Testing forex pairs (EURUSD, GBPUSD):")
        symbols = ['EURUSD', 'GBPUSD']
        
        # Show how symbols are converted
        params = ForexSymbolConverter.get_marketaux_params_for_symbols(symbols, free_plan=True)
        print(f"   Converted parameters:")
        print(f"   - Countries: {params.get('countries', [])}")
        print(f"   - Search query: {params.get('search', '')[:100]}...")
        print()
        
        try:
            response = await client.get_news(
                symbols=symbols,
                published_after=datetime.now(timezone.utc) - timedelta(days=2),
                limit=10,  # Free plan allows max 3 per request, but we can paginate
                language='en',
                use_cache=False
            )
            
            print(f"   ✅ Found {len(response.data)} articles")
            
            # Analyze results for forex relevance
            forex_relevant = 0
            for i, article in enumerate(response.data, 1):
                print(f"\n   Article {i}: {article.title[:70]}...")
                print(f"   - Published: {article.published_at.strftime('%Y-%m-%d %H:%M UTC')}")
                print(f"   - Source: {article.source}")
                print(f"   - High Impact: {article.is_high_impact}")
                
                # Check if article mentions our target symbols
                if article.symbols:
                    print(f"   - Detected symbols: {article.symbols}")
                    forex_relevant += 1
                
                # Show relevance analysis
                text = f"{article.title} {article.description or ''}".lower()
                forex_terms = ['forex', 'currency', 'dollar', 'euro', 'pound', 'exchange', 
                              'eur/usd', 'gbp/usd', 'eurusd', 'gbpusd']
                found_terms = [term for term in forex_terms if term in text]
                if found_terms:
                    print(f"   - Forex terms found: {found_terms}")
                    forex_relevant += 1
                
                # Show sentiment if available
                if hasattr(article, 'sentiment') and article.sentiment:
                    print(f"   - Sentiment: {article.sentiment.overall.value}")
            
            print(f"\n   📊 Forex relevance: {forex_relevant}/{len(response.data)} articles")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 2: Test batch symbol summary
        print("\n2️⃣ Testing batch symbol summaries:")
        try:
            summaries = await client.get_batch_symbol_summaries(
                symbols=['EURUSD', 'GBPUSD', 'USDJPY'],
                hours=24,
                use_cache=True
            )
            
            for symbol, summary in summaries.items():
                print(f"\n   {symbol}:")
                print(f"   - Articles: {summary['article_count']}")
                print(f"   - High impact: {summary['high_impact_count']}")
                print(f"   - Sentiment: {summary['sentiment_label']} ({summary['average_sentiment']:.2f})")
                
                if summary['latest_articles']:
                    print(f"   - Latest: {summary['latest_articles'][0]['title'][:60]}...")
                    
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Check API usage
        print("\n📊 API usage summary:")
        analytics = client.get_analytics()
        usage = analytics.get('usage_stats', {})
        print(f"   - Requests today: {usage.get('requests_today', 0)}/{usage.get('daily_limit', 100)}")
        print(f"   - Can make request: {usage.get('can_make_request', False)}")
        
        # Show configuration
        print("\n⚙️ Configuration:")
        print(f"   - Free plan mode: {client.free_plan}")
        print(f"   - Daily limit: {client.daily_limit}")
        print(f"   - Cache enabled: True")
        
    finally:
        # Ensure session is closed
        await client.close()
        print("\n✅ Session closed properly")

if __name__ == "__main__":
    asyncio.run(test_marketaux_final())