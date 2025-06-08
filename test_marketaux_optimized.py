#!/usr/bin/env python3
"""Test optimized MarketAux forex news fetching"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_settings
from core.infrastructure.marketaux.client import MarketAuxClient
from core.infrastructure.marketaux.forex_utils import ForexSymbolConverter

async def test_optimized_marketaux():
    """Test optimized implementation with better forex targeting"""
    settings = get_settings()
    
    # Initialize client with free plan mode
    cache_path = settings.paths.data_dir / "marketaux_cache.db"
    client = MarketAuxClient(
        api_token=settings.marketaux.api_token,
        cache_db_path=cache_path,
        daily_limit=settings.marketaux.daily_limit,
        requests_per_minute=settings.marketaux.requests_per_minute,
        free_plan=True
    )
    
    try:
        print("🔍 Testing Optimized MarketAux Forex News Fetching\n")
        
        # Test 1: Show optimized search query
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
        print("1️⃣ Optimized Search Query Generation:")
        
        params = ForexSymbolConverter.get_marketaux_params_for_symbols(symbols, free_plan=True)
        print(f"Input symbols: {symbols}")
        print(f"Countries: {params.get('countries', [])}")
        print(f"Search query preview: {params.get('search', '')[:100]}...")
        print(f"Full query length: {len(params.get('search', ''))} characters\n")
        
        # Test 2: Fetch news with optimized parameters
        print("2️⃣ Fetching Forex News:")
        
        try:
            response = await client.get_news(
                symbols=symbols,
                published_after=datetime.now(timezone.utc) - timedelta(hours=12),  # Last 12 hours
                limit=10,
                language='en',
                use_cache=False
            )
            
            print(f"✅ Found {len(response.data)} articles\n")
            
            # Analyze each article
            print("Article Analysis:")
            print("-" * 80)
            
            for i, article in enumerate(response.data, 1):
                # Calculate relevance for each symbol
                relevances = {}
                for symbol in symbols:
                    relevances[symbol] = article.get_symbol_relevance(symbol)
                
                # Best relevance score
                best_relevance = max(relevances.values()) if relevances else 0
                best_symbol = max(relevances, key=relevances.get) if relevances else "N/A"
                
                print(f"\n{i}. {article.title[:60]}...")
                print(f"   Published: {article.published_at.strftime('%Y-%m-%d %H:%M UTC')}")
                print(f"   Relevance: {best_relevance:.2f} (best match: {best_symbol})")
                print(f"   Detected symbols: {article.symbols if article.symbols else 'None'}")
                print(f"   High impact: {'YES' if article.is_high_impact else 'No'}")
                
                # Show why it's relevant
                if best_relevance > 0:
                    text_sample = f"{article.title} {article.description or ''}"[:150]
                    print(f"   Text preview: {text_sample}...")
            
            # Test 3: Show relevance analysis for most relevant article
            if response.data:
                # Find most relevant article
                best_article = None
                best_score = 0
                for article in response.data:
                    max_rel = max(article.get_symbol_relevance(s) for s in symbols)
                    if max_rel > best_score:
                        best_score = max_rel
                        best_article = article
                
                if best_article and best_score > 0:
                    print("\n" + "=" * 80)
                    print("3️⃣ Most Relevant Article Analysis:")
                    print(f"Title: {best_article.title}")
                    print(f"Published: {best_article.published_at.strftime('%Y-%m-%d %H:%M UTC')}")
                    print("\nRelevance Breakdown:")
                    
                    for symbol in symbols:
                        score = best_article.get_symbol_relevance(symbol)
                        print(f"   {symbol}: {score:.2f}")
                        
                        # Explain why it's relevant
                        text_content = f"{best_article.title} {best_article.description or ''}".upper()
                        reasons = []
                        
                        if symbol in text_content:
                            reasons.append("direct mention")
                        elif len(symbol) == 6:
                            base, quote = symbol[:3], symbol[3:]
                            if f"{base}/{quote}" in text_content:
                                reasons.append("pair format found")
                            elif base in text_content and quote in text_content:
                                reasons.append("both currencies mentioned")
                            elif base in text_content or quote in text_content:
                                reasons.append("one currency mentioned")
                        
                        if best_article.is_high_impact:
                            reasons.append("high impact news")
                        
                        if reasons:
                            print(f"      Reasons: {', '.join(reasons)}")
                
        except Exception as e:
            print(f"❌ Error fetching news: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 4: Batch symbol summaries
        print("\n" + "=" * 80)
        print("4️⃣ 24-Hour Symbol Summaries:")
        try:
            summaries = await client.get_batch_symbol_summaries(
                symbols=symbols,
                hours=24,
                min_relevance=0.3,  # Lower threshold to catch more articles
                use_cache=True
            )
            
            for symbol, summary in summaries.items():
                print(f"\n{symbol}:")
                print(f"   Total articles: {summary['article_count']}")
                print(f"   High impact: {summary['high_impact_count']}")
                print(f"   Sentiment: {summary['sentiment_label']} ({summary['average_sentiment']:.2f})")
                
                if summary['latest_articles']:
                    print("   Latest articles:")
                    for j, article in enumerate(summary['latest_articles'][:3], 1):
                        print(f"      {j}. {article['title'][:50]}...")
                        print(f"         Relevance: {article['relevance']:.2f}")
                else:
                    print("   No relevant articles found")
            
        except Exception as e:
            print(f"❌ Error in summaries: {e}")
        
        # Show usage
        print("\n" + "=" * 80)
        print("📊 API Usage:")
        analytics = client.get_analytics()
        usage = analytics.get('usage_stats', {})
        print(f"   Requests today: {usage.get('requests_today', 0)}/{usage.get('daily_limit', 100)}")
        print(f"   Remaining: {usage.get('remaining_today', 0)}")
        
    finally:
        await client.close()
        print("\n✅ Session closed properly")

if __name__ == "__main__":
    asyncio.run(test_optimized_marketaux())