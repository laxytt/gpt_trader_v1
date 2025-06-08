# MarketAux Integration Guide

## Overview

This guide explains how to integrate MarketAux news and sentiment analysis into the GPT Trading System for production use.

## Features

### 1. Real-time News Feed
- Access to global financial news from 75,000+ sources
- Filtered by trading symbols and relevance
- High-impact event detection

### 2. Sentiment Analysis
- Article-level sentiment scoring
- Currency-specific sentiment tracking
- Market mood indicators

### 3. Smart Caching
- SQLite-based cache with TTL
- Reduces API calls and costs
- Offline availability

### 4. Request Management
- Rate limiting (100/day, 5/min)
- Request prioritization by symbol importance
- Usage analytics and tracking

## Setup

### 1. Get MarketAux API Token

1. Sign up at https://www.marketaux.com/
2. Get your API token from the dashboard
3. Free tier includes 100 requests/day

### 2. Configure Environment

Update your `.env` file:

```bash
# MarketAux Configuration
MARKETAUX_API_TOKEN=your_actual_api_token_here
MARKETAUX_ENABLED=true
MARKETAUX_DAILY_LIMIT=100
MARKETAUX_REQUESTS_PER_MINUTE=5
MARKETAUX_CACHE_TTL_HOURS=24
MARKETAUX_MIN_RELEVANCE_SCORE=0.3
MARKETAUX_SENTIMENT_WEIGHT=0.3
MARKETAUX_HIGH_IMPACT_ONLY=false
```

### 3. Run Migration

```bash
python scripts/migrate_to_marketaux.py
```

This will:
- Create MarketAux database tables
- Verify configuration
- Provide setup instructions

### 4. Test Integration

```bash
python scripts/test_marketaux.py
```

## Usage in Trading System

### Enhanced News Service

The system automatically uses `EnhancedNewsService` when MarketAux is enabled:

```python
from core.services.enhanced_news_service import EnhancedNewsService

# Initialize with both ForexFactory and MarketAux
async with EnhancedNewsService(
    news_config=settings.news,
    marketaux_config=settings.marketaux,
    db_path=settings.paths.data_dir / "trades.db"
) as news_service:
    
    # Get enhanced context with sentiment
    context = await news_service.get_enhanced_news_context(
        symbol="EURUSD",
        lookahead_hours=48,  # Calendar events
        lookback_hours=24    # News articles
    )
    
    # Check trading restrictions
    is_restricted, reason = await news_service.is_trading_restricted(
        symbol="EURUSD",
        consider_sentiment=True
    )
```

### Trading Council Integration

The Fundamental Analyst automatically uses sentiment data:

```python
# In your trading loop, news context now includes:
news_context = [
    "Market Sentiment: positive (0.65)",
    "Upcoming Economic Events:",
    "- US CPI (USD) in 45 minutes",
    "Recent Market News:",
    "- Fed signals steady rates [positive] (2.3h ago)",
    "- EUR weakens on growth concerns [negative] (5.1h ago)"
]
```

### Sentiment-Adjusted Confidence

The council adjusts trading confidence based on sentiment:

```python
# Original confidence: 85%
# Sentiment score: -0.5 (bearish)
# Sentiment weight: 0.3
# For a SELL signal: confidence increases to 88%
# For a BUY signal: confidence decreases to 82%
```

## Configuration Options

### Essential Settings

- `MARKETAUX_ENABLED`: Enable/disable the integration
- `MARKETAUX_API_TOKEN`: Your API authentication token
- `MARKETAUX_DAILY_LIMIT`: Maximum requests per day (respect API limits)

### Performance Tuning

- `MARKETAUX_CACHE_TTL_HOURS`: How long to cache articles (24-48 recommended)
- `MARKETAUX_REQUESTS_PER_MINUTE`: Rate limiting (5 recommended)
- `MARKETAUX_MIN_RELEVANCE_SCORE`: Filter threshold (0.3-0.5 recommended)

### Trading Impact

- `MARKETAUX_SENTIMENT_WEIGHT`: How much sentiment affects decisions (0.2-0.4)
- `MARKETAUX_HIGH_IMPACT_ONLY`: Only consider high-impact news

## Best Practices

### 1. API Usage Optimization

```python
# Prioritize major pairs
priority_symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']

# Use batch requests
contexts = await news_service.get_multi_symbol_context(
    symbols=symbols,
    max_requests=10  # Limit daily usage
)
```

### 2. Cache Management

```python
# Clean expired entries periodically
news_service.clean_cache()

# Check cache status
analytics = news_service.get_analytics()
print(f"Cache size: {analytics['marketaux']['cache_size']}")
```

### 3. Error Handling

The system gracefully handles:
- API failures (falls back to ForexFactory)
- Rate limiting (uses cache)
- Network issues (continues with calendar data)

### 4. Monitoring

Monitor these metrics:
- Daily API usage vs limit
- Cache hit rate
- Sentiment impact on trades
- API error rate

## Advanced Features

### Custom Sentiment Analysis

```python
# Override simple sentiment with advanced models
class AdvancedSentimentAnalyzer:
    def analyze(self, article: MarketAuxArticle) -> MarketAuxSentiment:
        # Use FinBERT or custom model
        sentiment = self.finbert_model.predict(article.text)
        return MarketAuxSentiment(...)
```

### Symbol-Specific Configuration

```python
# Different settings per symbol
symbol_configs = {
    'XAUUSD': {
        'sentiment_weight': 0.4,  # Gold more news-sensitive
        'min_relevance': 0.5
    },
    'EURUSD': {
        'sentiment_weight': 0.3,
        'min_relevance': 0.3
    }
}
```

### News Trading Strategies

1. **Sentiment Momentum**: Trade in direction of strong sentiment
2. **Sentiment Reversal**: Fade extreme sentiment
3. **News Catalyst**: Enter on high-impact news with technical confirmation

## Troubleshooting

### Common Issues

1. **"Rate limit exceeded"**
   - Check daily usage: `SELECT COUNT(*) FROM marketaux_api_usage WHERE date(request_time) = date('now')`
   - Increase cache TTL
   - Reduce request frequency

2. **"No sentiment data"**
   - Verify API token is valid
   - Check if MarketAux is enabled
   - Review logs for API errors

3. **"Cache database locked"**
   - Ensure single instance running
   - Check file permissions

### Debug Mode

Enable detailed logging:

```python
import logging
logging.getLogger('core.infrastructure.marketaux').setLevel(logging.DEBUG)
logging.getLogger('core.services.enhanced_news_service').setLevel(logging.DEBUG)
```

## Performance Impact

- API latency: 200-500ms per request
- Cache queries: <10ms
- Sentiment analysis: ~50ms per article
- Total overhead: <1s per symbol with cache

## Cost Considerations

- Free tier: 100 requests/day
- Pro tier: 3,000 requests/day ($99/month)
- Enterprise: Custom limits

### Optimization Tips

1. Cache everything (24-48 hour TTL)
2. Prioritize major pairs
3. Batch requests when possible
4. Use high relevance threshold
5. Monitor usage daily

## Integration Checklist

- [ ] Obtain MarketAux API token
- [ ] Update .env configuration
- [ ] Run database migration
- [ ] Test API connectivity
- [ ] Verify sentiment analysis
- [ ] Monitor first day usage
- [ ] Adjust weights based on results
- [ ] Set up usage alerts
- [ ] Document any custom changes

## Support

- MarketAux API docs: https://docs.marketaux.com/
- Rate limits: https://www.marketaux.com/pricing
- Support: support@marketaux.com

Remember: Start with conservative settings and adjust based on performance!