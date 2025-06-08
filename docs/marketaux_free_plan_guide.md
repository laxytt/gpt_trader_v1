# MarketAux Free Plan Optimization Guide

## Overview
This guide documents the optimizations made to the MarketAux integration to work effectively within free plan limitations.

## Free Plan Limitations
- **3 articles per request** (vs 50+ for paid plans)
- **100 requests per day**
- **No entity data** (entities array is empty)
- **No market stats API access**
- **Limited metadata**

## Optimizations Implemented

### 1. Search Query Optimization
Instead of relying on symbol parameters (which return 0 results), we use highly targeted search queries:

```python
# Core forex terms
"forex" | "fx market" | "currency" | "exchange rate"

# Specific pair mentions
"EUR/USD" | "GBP/USD"

# Currency-specific terms
("euro currency" | "eurozone" | "EUR USD") | ("pound sterling" | "cable" | "GBP USD")

# Central banks
(fed | ecb | boe)

# Economic indicators
("interest rate" | inflation | nfp | "non-farm payroll" | gdp | cpi)
```

### 2. Symbol Detection Without Entities
Since the free plan doesn't provide entity data, we extract symbols from article text:

- Direct mentions: EURUSD, EUR/USD, EUR-USD
- Currency name recognition: dollar, euro, pound, yen
- Both currencies mentioned = likely pair coverage

### 3. Relevance Scoring System
Articles are scored based on multiple factors:

- **0.9**: Direct pair mention (EUR/USD in text)
- **0.7**: Both currencies of pair mentioned
- **0.4**: Single currency mentioned
- **+0.2**: Forex-specific terms found
- **+0.15**: Economic indicators mentioned
- **+0.1**: High impact classification

### 4. Configuration
Set in `.env` file:
```
MARKETAUX_FREE_PLAN=true
MARKETAUX_ENABLED=true
MARKETAUX_API_TOKEN=your_token
MARKETAUX_DAILY_LIMIT=100
```

## Usage Best Practices

### 1. Request Prioritization
With only 100 requests/day, prioritize:
- Major pairs: EURUSD, GBPUSD, USDJPY
- Peak news times: London/NY session overlaps
- High-volatility periods

### 2. Caching Strategy
- Enable caching (24-hour TTL by default)
- Batch requests when possible
- Use `get_batch_symbol_summaries()` for multiple symbols

### 3. Filtering
- Set `min_relevance=0.3` to catch more articles
- Focus on high-impact news during limited quota
- Use time windows (e.g., last 12 hours) to get recent news

## API Usage Example

```python
# Optimized for free plan
response = await client.get_news(
    symbols=['EURUSD', 'GBPUSD'],  # Will be converted to search terms
    published_after=datetime.now(timezone.utc) - timedelta(hours=12),
    limit=10,  # Will return max 3 due to free plan
    language='en',
    use_cache=True  # Important for quota management
)

# Filter by relevance
relevant_articles = [
    article for article in response.data
    if article.get_symbol_relevance('EURUSD') >= 0.4
]
```

## Monitoring Usage

Check remaining quota:
```python
analytics = client.get_analytics()
usage = analytics.get('usage_stats', {})
print(f"Used: {usage['requests_today']}/{usage['daily_limit']}")
print(f"Remaining: {usage['remaining_today']}")
```

## Limitations & Workarounds

### Limited Article Count
- **Issue**: Only 3 articles per request
- **Workaround**: Make multiple targeted requests with different search terms

### No Entity Data
- **Issue**: Can't filter by entity type
- **Workaround**: Text analysis and pattern matching

### General Content
- **Issue**: May get non-forex financial news
- **Workaround**: Strict relevance filtering and forex-specific search terms

## Future Improvements

1. **Paid Plan Features** (when upgraded):
   - Use entity_types=['currency'] filter
   - Access to 50+ articles per request
   - Market sentiment data
   - Historical data access

2. **Alternative Data Sources**:
   - Combine with ForexFactory calendar
   - Add RSS feeds from forex sites
   - Integrate free forex APIs

## Testing

Run the test scripts to verify optimization:
```bash
python test_marketaux_optimized.py
```

This will show:
- Search query generation
- Article relevance scoring
- Symbol detection accuracy
- API usage statistics