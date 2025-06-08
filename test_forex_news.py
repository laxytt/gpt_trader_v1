#!/usr/bin/env python3
"""Test forex news fetching specifically"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_settings
from core.services.enhanced_news_service import EnhancedNewsService

async def test_forex_news():
    """Test the enhanced news service for forex"""
    settings = get_settings()
    
    # Initialize enhanced news service
    news_service = EnhancedNewsService(
        news_config=settings.news,
        marketaux_config=settings.marketaux,
        db_path=settings.paths.data_dir / "trades.db"
    )
    
    print("🔍 Testing forex news fetching...")
    
    # Test fetching for forex symbols
    symbols = ['EURUSD', 'GBPUSD']
    
    print(f"\n📰 Fetching news for: {symbols}")
    news_by_symbol = await news_service.fetch_realtime_market_news(
        symbols=symbols,
        hours=48,  # Look back 48 hours
        force_refresh=True
    )
    
    # Display results
    for symbol, articles in news_by_symbol.items():
        print(f"\n🔹 {symbol}: {len(articles)} articles")
        
        if articles:
            # Show top 5 articles
            for i, article in enumerate(articles[:5], 1):
                print(f"\n{i}. {article['title']}")
                print(f"   Impact: {article.get('trading_impact', 'unknown')}")
                print(f"   Keywords: {', '.join(article.get('keywords', []))}")
                print(f"   Importance: {article.get('importance_score', 0):.2f}")
                print(f"   Published: {article['published_at']}")
    
    # Test hybrid data
    print("\n\n🔍 Testing hybrid news data (calendar + news)...")
    hybrid_data = await news_service.get_hybrid_news_data(
        symbol='EURUSD',
        calendar_hours=48,
        news_hours=24
    )
    
    print(f"\n📅 Calendar Events: {len(hybrid_data['calendar_events']['upcoming'])}")
    if hybrid_data['calendar_events']['upcoming']:
        for event in hybrid_data['calendar_events']['upcoming'][:3]:
            print(f"  - {event.title} ({event.country}) - {event.timestamp}")
    
    print(f"\n📰 Market News: {hybrid_data['market_news']['count']} articles")
    print(f"   Average Sentiment: {hybrid_data['market_news']['sentiment']:.3f}")
    
    print(f"\n📊 Combined Summary: {hybrid_data['combined_summary']}")
    
    # Close properly
    if hasattr(news_service, 'marketaux_client'):
        await news_service.marketaux_client.close()

if __name__ == "__main__":
    asyncio.run(test_forex_news())