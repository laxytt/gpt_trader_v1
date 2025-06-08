#!/usr/bin/env python3
"""Debug MarketAux forex news fetching"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_settings
from core.infrastructure.marketaux.client import MarketAuxClient
from core.infrastructure.marketaux.forex_utils import ForexSymbolConverter

async def debug_forex():
    """Debug forex news fetching step by step"""
    settings = get_settings()
    
    # Initialize client
    cache_path = settings.paths.data_dir / "marketaux_cache.db"
    client = MarketAuxClient(
        api_token=settings.marketaux.api_token,
        cache_db_path=cache_path,
        daily_limit=settings.marketaux.daily_limit,
        requests_per_minute=settings.marketaux.requests_per_minute
    )
    
    # Test symbol conversion
    symbols = ['EURUSD', 'GBPUSD']
    print("🔍 Testing symbol conversion...")
    
    params = ForexSymbolConverter.get_marketaux_params_for_symbols(symbols)
    print(f"Symbols: {symbols}")
    print(f"Converted params:")
    print(f"  - Countries: {params.get('countries', [])}")
    print(f"  - Search: {params.get('search', '')}")
    print(f"  - Symbols (currencies): {params.get('symbols', [])}")
    
    # Test direct API call
    print("\n📰 Testing direct API call...")
    try:
        response = await client.get_news(
            countries=params.get('countries', []),
            published_after=datetime.now(timezone.utc) - timedelta(hours=24),
            limit=10,
            use_cache=False  # Force fresh
        )
        
        print(f"✅ API returned {len(response.data)} articles")
        
        if response.data:
            print("\nRaw articles:")
            for i, article in enumerate(response.data[:5], 1):
                print(f"\n{i}. {article.title}")
                print(f"   Countries: {getattr(article, 'countries', [])}")
                print(f"   Entities: {len(getattr(article, 'entities', []))}")
                print(f"   Keywords: {getattr(article, 'keywords', [])}")
                
                # Check if this would match forex
                text = f"{article.title} {article.description}".lower()
                forex_keywords = ['forex', 'currency', 'dollar', 'euro', 'pound', 'exchange', 'fx']
                matches = [kw for kw in forex_keywords if kw in text]
                print(f"   Forex keywords found: {matches}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(debug_forex())