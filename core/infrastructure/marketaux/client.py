"""MarketAux API client with rate limiting and error handling"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import aiohttp
from aiohttp import ClientTimeout
import backoff

from .models import MarketAuxArticle, MarketAuxResponse, MarketAuxSentiment, SentimentScore, ApiUsageStats
from .cache import MarketAuxCache
from .forex_utils import ForexSymbolConverter
from core.domain.exceptions import ExternalAPIError, ErrorContext
from core.utils.circuit_breaker import circuit_breaker

logger = logging.getLogger(__name__)


class MarketAuxClient:
    """Async client for MarketAux API with production-ready features"""
    
    BASE_URL = "https://api.marketaux.com/v1"
    
    def __init__(
        self,
        api_token: str,
        cache_db_path: Path,
        daily_limit: int = 100,
        requests_per_minute: int = 5,
        timeout_seconds: int = 30,
        max_retries: int = 3,
        free_plan: bool = True
    ):
        self.api_token = api_token
        self.cache = MarketAuxCache(cache_db_path, daily_limit=daily_limit)
        self.daily_limit = daily_limit
        self.requests_per_minute = requests_per_minute
        self.timeout = ClientTimeout(total=timeout_seconds)
        self.max_retries = max_retries
        self.free_plan = free_plan
        
        # Rate limiting
        self._request_times: List[datetime] = []
        self._lock = asyncio.Lock()
        
        # Session management
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._session:
            await self._session.close()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create session"""
        if not self._session:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session
    
    async def _check_rate_limits(self) -> bool:
        """Check if we can make a request"""
        async with self._lock:
            # Check daily limit
            if not self.cache.can_make_request():
                logger.warning("MarketAux daily request limit reached")
                return False
            
            # Check per-minute limit
            now = datetime.now()
            minute_ago = now - timedelta(minutes=1)
            
            # Clean old request times
            self._request_times = [t for t in self._request_times if t > minute_ago]
            
            if len(self._request_times) >= self.requests_per_minute:
                wait_time = (self._request_times[0] + timedelta(minutes=1) - now).total_seconds()
                if wait_time > 0:
                    logger.info(f"Rate limit: waiting {wait_time:.1f}s before next request")
                    await asyncio.sleep(wait_time)
                    return await self._check_rate_limits()  # Recursive check
            
            # Record this request time
            self._request_times.append(now)
            return True
    
    @circuit_breaker(
        name="marketaux_api",
        failure_threshold=5,
        recovery_timeout=120,  # 2 minutes recovery
        expected_exception=ExternalAPIError,
        success_threshold=2
    )
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        max_time=60
    )
    async def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make API request with retries"""
        if not await self._check_rate_limits():
            # Get current usage stats for more detailed error message
            stats = self.get_usage_stats()
            raise ExternalAPIError(
                f"MarketAux daily rate limit exceeded ({stats.requests_today}/{stats.daily_limit} requests used)", 
                api_name="MarketAux"
            )
        
        url = f"{self.BASE_URL}/{endpoint}"
        params['api_token'] = self.api_token
        
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as response:
                response_data = await response.json()
                
                # Log request
                self.cache.log_api_request(
                    endpoint=endpoint,
                    params={k: v for k, v in params.items() if k != 'api_token'},
                    response_code=response.status,
                    articles_returned=len(response_data.get('data', [])),
                    error_message=response_data.get('error', {}).get('message') if response.status >= 400 else None
                )
                
                # Update rate limits
                if response.status < 400:
                    self.cache.update_rate_limit()
                
                # Handle errors
                if response.status >= 400:
                    error_msg = response_data.get('error', {}).get('message', 'Unknown error')
                    raise ExternalAPIError(f"MarketAux API error ({response.status}): {error_msg}", api_name="MarketAux")
                
                return response_data
                
        except aiohttp.ClientError as e:
            logger.error(f"MarketAux request failed: {e}")
            self.cache.log_api_request(
                endpoint=endpoint,
                params={k: v for k, v in params.items() if k != 'api_token'},
                error_message=str(e)
            )
            raise ExternalAPIError(f"MarketAux connection error: {e}", api_name="MarketAux")
    
    async def get_news(
        self,
        symbols: Optional[List[str]] = None,
        countries: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
        industries: Optional[List[str]] = None,
        entity_types: Optional[List[str]] = None,
        search: Optional[str] = None,
        filter_entities: bool = True,
        must_have_entities: bool = False,
        published_after: Optional[datetime] = None,
        published_before: Optional[datetime] = None,
        sort: str = "published_at",
        sort_order: str = "desc",
        limit: int = 50,
        page: int = 1,
        language: str = "en",
        use_cache: bool = True,
        cache_ttl_hours: int = 24
    ) -> MarketAuxResponse:
        """
        Get news articles from MarketAux
        
        Args:
            symbols: Stock/forex symbols to filter by
            countries: Country codes to filter by
            entities: Specific entities to search for
            industries: Industries to filter by
            search: Search query string
            filter_entities: Whether to filter by entities
            must_have_entities: Require articles to have entities
            published_after: Get articles published after this date
            published_before: Get articles published before this date
            sort: Field to sort by
            sort_order: Sort order (asc/desc)
            limit: Number of results per page
            page: Page number
            use_cache: Whether to use cached results
            cache_ttl_hours: Cache TTL in hours
            
        Returns:
            MarketAuxResponse with articles
        """
        with ErrorContext("MarketAux get_news") as ctx:
            ctx.add_detail("symbols", symbols)
            ctx.add_detail("use_cache", use_cache)
            
            # Convert forex symbols to currencies if needed
            original_symbols = symbols
            if symbols:
                # Check if we have forex pairs
                forex_params = ForexSymbolConverter.get_marketaux_params_for_symbols(symbols, free_plan=self.free_plan)
                
                if self.free_plan:
                    # For free plan, don't use symbols parameter as it returns 0 results
                    symbols = None  # Clear symbols for free plan
                else:
                    # For paid plan, use converted symbols
                    if 'symbols' in forex_params:
                        symbols = forex_params['symbols']
                
                # Add countries if not already specified
                if not countries and 'countries' in forex_params:
                    countries = forex_params['countries']
                # Add search query if not already specified
                if not search and 'search' in forex_params:
                    search = forex_params['search']
                    
                logger.debug(f"Converted forex symbols {original_symbols} to params: countries={countries}, search={search}")
            
            # Check cache first if enabled
            if use_cache and original_symbols and published_after:
                cached_articles = self.cache.get_cached_articles(
                    symbol=original_symbols[0] if original_symbols else None,
                    since=published_after,
                    limit=limit
                )
                
                if cached_articles:
                    logger.info(f"Using {len(cached_articles)} cached MarketAux articles")
                    return MarketAuxResponse(
                        data=cached_articles,
                        meta={'found': len(cached_articles), 'page': 1, 'limit': limit}
                    )
            
            # Build request parameters
            params = {
                'limit': min(limit, 100),  # API max is 100
                'page': page,
                'sort': sort,
                'sort_order': sort_order,
                'filter_entities': str(filter_entities).lower(),
                'must_have_entities': str(must_have_entities).lower(),
                'language': language
            }
            
            if symbols:
                params['symbols'] = ','.join(symbols)
            if countries:
                params['countries'] = ','.join(countries).lower()
            if entities:
                params['entities'] = ','.join(entities)
            if industries:
                params['industries'] = ','.join(industries)
            if entity_types:
                params['entity_types'] = ','.join(entity_types)
            if published_after:
                params['published_after'] = published_after.strftime('%Y-%m-%dT%H:%M')
            if published_before:
                params['published_before'] = published_before.strftime('%Y-%m-%dT%H:%M')
            if search:
                params['search'] = search
            
            # Make API request
            response_data = await self._make_request('news/all', params)
            
            # Parse response
            articles = []
            for article_data in response_data.get('data', []):
                try:
                    # Extract trading symbols from entities (free plan has no entities)
                    trading_symbols = []
                    article_entities = article_data.get('entities', [])
                    
                    # For free plan, try to extract symbols from title/description
                    if not article_entities and original_symbols:
                        # Check if any of our target symbols are mentioned in the text
                        text_content = f"{article_data.get('title', '')} {article_data.get('description', '')} {article_data.get('snippet', '')}".upper()
                        
                        for symbol in original_symbols:
                            symbol_found = False
                            
                            # Check direct symbol mention (EURUSD)
                            if symbol in text_content:
                                symbol_found = True
                            
                            # Check formatted versions (EUR/USD, EUR-USD)
                            elif len(symbol) == 6:
                                formatted1 = f"{symbol[:3]}/{symbol[3:]}"
                                formatted2 = f"{symbol[:3]}-{symbol[3:]}"
                                formatted3 = f"{symbol[:3]} {symbol[3:]}"
                                if any(fmt in text_content for fmt in [formatted1, formatted2, formatted3]):
                                    symbol_found = True
                            
                            # Check if both currencies are mentioned
                            elif len(symbol) == 6:
                                base = symbol[:3]
                                quote = symbol[3:]
                                # Look for currency names too
                                currency_names = {
                                    'EUR': ['EURO', 'EUROZONE'],
                                    'USD': ['DOLLAR', 'GREENBACK', 'BUCK'],
                                    'GBP': ['POUND', 'STERLING', 'CABLE'],
                                    'JPY': ['YEN', 'JAPANESE'],
                                    'AUD': ['AUSSIE', 'AUSTRALIAN'],
                                    'CAD': ['LOONIE', 'CANADIAN'],
                                    'CHF': ['SWISS', 'SWISSIE'],
                                    'NZD': ['KIWI', 'NEW ZEALAND']
                                }
                                
                                base_found = base in text_content or any(name in text_content for name in currency_names.get(base, []))
                                quote_found = quote in text_content or any(name in text_content for name in currency_names.get(quote, []))
                                
                                if base_found and quote_found:
                                    symbol_found = True
                            
                            if symbol_found and symbol not in trading_symbols:
                                trading_symbols.append(symbol)
                    
                    if article_entities:
                        for entity in article_entities:
                            # Check for symbol in entity
                            symbol = entity.get('symbol')
                            if symbol:
                                trading_symbols.append(symbol)
                    
                    # Determine if high impact based on various factors
                    is_high_impact = False
                    
                    # Check entities for high-impact sentiment scores (if available)
                    if article_entities:
                        for entity in article_entities:
                            sentiment_score = entity.get('sentiment_score', 0)
                            # High positive or negative sentiment indicates high impact
                            if sentiment_score is not None and abs(sentiment_score) > 0.7:
                                is_high_impact = True
                                break
                    
                    # Check relevance score
                    relevance = article_data.get('relevance_score')
                    if relevance is None:
                        relevance = 0.5
                    else:
                        relevance = float(relevance)
                    if relevance > 0.8:
                        is_high_impact = True
                    
                    # Check for breaking news keywords in title
                    title_lower = article_data.get('title', '').lower()
                    description_lower = article_data.get('description', '').lower()
                    full_text_lower = f"{title_lower} {description_lower}"
                    
                    # Extended keywords for forex impact detection
                    breaking_keywords = ['breaking', 'alert', 'urgent', 'flash', 'major', 'crisis', 'emergency']
                    forex_impact_keywords = ['rate cut', 'rate hike', 'intervention', 'devaluation', 'inflation surge',
                                           'central bank', 'fomc', 'ecb decision', 'boe announcement', 'monetary policy',
                                           'nfp', 'non-farm', 'gdp', 'cpi', 'interest rate']
                    
                    if any(keyword in title_lower for keyword in breaking_keywords):
                        is_high_impact = True
                    elif any(keyword in full_text_lower for keyword in forex_impact_keywords):
                        is_high_impact = True
                    
                    # Handle keywords - API might return string or list
                    keywords_raw = article_data.get('keywords', [])
                    if isinstance(keywords_raw, str):
                        # Convert comma-separated string to list
                        if keywords_raw.strip():  # Non-empty string
                            keywords = [k.strip() for k in keywords_raw.split(',') if k.strip()]
                        else:
                            keywords = []
                    elif isinstance(keywords_raw, list):
                        keywords = keywords_raw
                    else:
                        keywords = []
                    
                    # Create article object
                    article = MarketAuxArticle(
                        uuid=article_data['uuid'],
                        title=article_data['title'],
                        description=article_data.get('description'),
                        snippet=article_data.get('snippet'),
                        url=article_data['url'],
                        image_url=article_data.get('image_url'),
                        published_at=article_data['published_at'],
                        source=article_data.get('source', 'unknown'),
                        relevance_score=relevance,
                        countries=article_data.get('countries', []),
                        entities=article_entities,
                        highlights=article_data.get('similar', []),  # Use 'similar' field if no highlights
                        symbols=trading_symbols,
                        keywords=keywords,
                        is_high_impact=is_high_impact
                    )
                    
                    articles.append(article)
                    
                except Exception as e:
                    logger.error(f"Error parsing article {article_data.get('uuid')}: {e}")
                    logger.debug(f"Article data structure: {article_data}")
                    logger.debug(f"Article keys: {list(article_data.keys())}")
                    if 'entities' in article_data:
                        logger.debug(f"First entity structure: {article_data['entities'][0] if article_data['entities'] else 'No entities'}")
                    continue
            
            # Cache articles if requested
            if use_cache and articles:
                self.cache.cache_articles(articles, ttl_hours=cache_ttl_hours)
            
            return MarketAuxResponse(
                data=articles,
                meta=response_data.get('meta', {}),
                error=response_data.get('error')
            )
    
    async def get_news_with_sentiment(
        self,
        symbols: Optional[List[str]] = None,
        **kwargs
    ) -> List[MarketAuxArticle]:
        """
        Get news with sentiment analysis
        
        This method fetches news and enriches it with sentiment analysis.
        Note: In production, sentiment might come from MarketAux premium tier
        or a separate sentiment analysis service.
        """
        # Get news
        response = await self.get_news(symbols=symbols, **kwargs)
        
        # Add sentiment analysis (simplified for now)
        for article in response.data:
            article.sentiment = self._analyze_sentiment(article)
        
        return response.data
    
    def _analyze_sentiment(self, article: MarketAuxArticle) -> MarketAuxSentiment:
        """
        Analyze sentiment from article content and entity sentiment scores
        
        This uses both:
        1. Entity-specific sentiment scores from MarketAux API
        2. Keyword-based analysis as fallback
        """
        # First, check if we have entity sentiment scores from API
        entity_sentiments = {}
        has_api_sentiment = False
        aggregate_scores = []
        
        for entity in article.entities:
            # Extract sentiment score from entity if available
            if 'sentiment_score' in entity:
                has_api_sentiment = True
                sentiment_score = entity.get('sentiment_score', 0)
                aggregate_scores.append(sentiment_score)
                
                # Store entity-specific sentiment
                symbol = entity.get('symbol', '')
                if symbol:
                    # Convert numeric score to sentiment category
                    entity_sentiments[symbol.upper()] = SentimentScore.from_numeric(sentiment_score)
        
        # If we have API sentiment scores, use them
        if has_api_sentiment and aggregate_scores:
            avg_sentiment_score = sum(aggregate_scores) / len(aggregate_scores)
            
            # Convert to sentiment scores
            if avg_sentiment_score > 0:
                positive_score = min(1.0, abs(avg_sentiment_score))
                negative_score = 0.0
            else:
                positive_score = 0.0
                negative_score = min(1.0, abs(avg_sentiment_score))
            
            neutral_score = 1.0 - positive_score - negative_score
            overall = SentimentScore.from_numeric(avg_sentiment_score)
            confidence = 0.9  # High confidence for API-provided sentiment
            
        else:
            # Fallback to keyword-based sentiment analysis
            text = f"{article.title} {article.description or ''} {article.snippet or ''}"
            text_lower = text.lower()
            
            # Sentiment keywords (simplified)
            very_positive_words = ['surge', 'soar', 'rally', 'breakthrough', 'excellent', 'strong']
            positive_words = ['gain', 'rise', 'up', 'growth', 'improve', 'better', 'advance']
            negative_words = ['fall', 'drop', 'down', 'decline', 'worse', 'weak', 'concern']
            very_negative_words = ['crash', 'plunge', 'collapse', 'crisis', 'disaster', 'slump']
            
            # Count occurrences
            very_pos_count = sum(word in text_lower for word in very_positive_words)
            pos_count = sum(word in text_lower for word in positive_words)
            neg_count = sum(word in text_lower for word in negative_words)
            very_neg_count = sum(word in text_lower for word in very_negative_words)
            
            # Calculate scores
            total_words = max(1, len(text.split()))
            
            positive_score = min(1.0, (very_pos_count * 2 + pos_count) / total_words * 10)
            negative_score = min(1.0, (very_neg_count * 2 + neg_count) / total_words * 10)
            neutral_score = max(0.0, 1.0 - positive_score - negative_score)
            
            # Normalize scores
            total_score = positive_score + negative_score + neutral_score
            if total_score > 0:
                positive_score /= total_score
                negative_score /= total_score
                neutral_score /= total_score
            
            # Determine overall sentiment
            net_sentiment = positive_score - negative_score
            overall = SentimentScore.from_numeric(net_sentiment)
            confidence = 0.6  # Lower confidence for keyword-based analysis
            
            # For keyword-based analysis, apply same sentiment to all entities
            for entity in article.entities:
                symbol = entity.get('symbol', '')
                if symbol:
                    entity_sentiments[symbol.upper()] = overall
        
        return MarketAuxSentiment(
            overall=overall,
            confidence=confidence,
            positive_score=positive_score,
            negative_score=negative_score,
            neutral_score=neutral_score,
            entity_sentiments=entity_sentiments
        )
    
    async def get_batch_symbol_summaries(
        self,
        symbols: List[str],
        hours: int = 24,
        min_relevance: float = 0.3,
        use_cache: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get news summaries for multiple symbols in a single API request
        
        This is much more efficient than calling get_symbol_news_summary
        for each symbol individually.
        
        Args:
            symbols: List of trading symbols
            hours: Hours to look back
            min_relevance: Minimum relevance score
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary mapping symbols to their summaries
        """
        if not symbols:
            return {}
            
        # Get news for all symbols in one request
        published_after = datetime.now() - timedelta(hours=hours)
        articles = await self.get_news_with_sentiment(
            symbols=symbols,
            published_after=published_after,
            limit=100,  # Increase limit for multiple symbols
            use_cache=use_cache
        )
        
        # Group articles by symbol and calculate summaries
        symbol_summaries = {}
        for symbol in symbols:
            # Filter articles relevant to this symbol
            relevant_articles = []
            for article in articles:
                relevance = article.get_symbol_relevance(symbol)
                if relevance >= min_relevance:
                    # Store relevance for sorting
                    article._temp_relevance = relevance
                    relevant_articles.append(article)
            
            # Sort by relevance (highest first)
            relevant_articles.sort(key=lambda a: getattr(a, '_temp_relevance', 0), reverse=True)
            
            # Calculate aggregate sentiment
            if relevant_articles:
                sentiments = [a.sentiment for a in relevant_articles if a.sentiment]
                avg_sentiment = sum(s.numeric_score for s in sentiments) / len(sentiments) if sentiments else 0
                
                # High impact articles
                high_impact = [a for a in relevant_articles if a.is_high_impact]
                
                symbol_summaries[symbol] = {
                    'symbol': symbol,
                    'article_count': len(relevant_articles),
                    'high_impact_count': len(high_impact),
                    'average_sentiment': avg_sentiment,
                    'sentiment_label': SentimentScore.from_numeric(avg_sentiment).value,
                    'latest_articles': [
                        {
                            'title': a.title,
                            'published_at': a.published_at.isoformat(),
                            'sentiment': a.sentiment.overall.value if a.sentiment else None,
                            'relevance': a.get_symbol_relevance(symbol),
                            'is_high_impact': a.is_high_impact
                        }
                        for a in relevant_articles[:5]
                    ],
                    'time_range_hours': hours
                }
            else:
                symbol_summaries[symbol] = {
                    'symbol': symbol,
                    'article_count': 0,
                    'high_impact_count': 0,
                    'average_sentiment': 0,
                    'sentiment_label': 'neutral',
                    'latest_articles': [],
                    'time_range_hours': hours
                }
        
        return symbol_summaries
    
    async def get_symbol_news_summary(
        self,
        symbol: str,
        hours: int = 24,
        min_relevance: float = 0.3
    ) -> Dict[str, Any]:
        """Get news summary for a specific trading symbol"""
        published_after = datetime.now() - timedelta(hours=hours)
        
        # Get news
        articles = await self.get_news_with_sentiment(
            symbols=[symbol],
            published_after=published_after,
            limit=100
        )
        
        # Filter by relevance
        relevant_articles = [
            a for a in articles
            if a.get_symbol_relevance(symbol) >= min_relevance
        ]
        
        # Calculate aggregate sentiment
        if relevant_articles:
            sentiments = [a.sentiment for a in relevant_articles if a.sentiment]
            avg_sentiment = sum(s.numeric_score for s in sentiments) / len(sentiments) if sentiments else 0
            
            # High impact articles
            high_impact = [a for a in relevant_articles if a.is_high_impact]
            
            return {
                'symbol': symbol,
                'article_count': len(relevant_articles),
                'high_impact_count': len(high_impact),
                'average_sentiment': avg_sentiment,
                'sentiment_label': SentimentScore.from_numeric(avg_sentiment).value,
                'latest_articles': [
                    {
                        'title': a.title,
                        'published_at': a.published_at.isoformat(),
                        'sentiment': a.sentiment.overall.value if a.sentiment else None,
                        'relevance': a.get_symbol_relevance(symbol),
                        'is_high_impact': a.is_high_impact
                    }
                    for a in relevant_articles[:5]
                ],
                'time_range_hours': hours
            }
        else:
            return {
                'symbol': symbol,
                'article_count': 0,
                'high_impact_count': 0,
                'average_sentiment': 0,
                'sentiment_label': 'neutral',
                'latest_articles': [],
                'time_range_hours': hours
            }
    
    async def prioritize_requests(
        self,
        symbols: List[str],
        max_requests: int = 10
    ) -> List[str]:
        """
        Prioritize which symbols to request news for based on:
        1. Time since last update
        2. Trading volume/importance
        3. Available API quota
        """
        stats = self.cache.get_usage_stats()
        remaining = min(stats.remaining_today, max_requests)
        
        if remaining <= 0:
            logger.warning("No remaining MarketAux API requests for today")
            return []
        
        # For now, simple round-robin with most important symbols
        # In production, you'd track last update time per symbol
        priority_symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
        
        prioritized = []
        for symbol in priority_symbols:
            if symbol in symbols and len(prioritized) < remaining:
                prioritized.append(symbol)
        
        # Add remaining symbols
        for symbol in symbols:
            if symbol not in prioritized and len(prioritized) < remaining:
                prioritized.append(symbol)
        
        return prioritized[:remaining]
    
    def get_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get API usage analytics"""
        report = self.cache.get_api_usage_report(days=days)
        
        # Add usage stats
        usage_stats = self.get_usage_stats()
        report['usage_stats'] = {
            'requests_today': usage_stats.requests_today,
            'remaining_today': usage_stats.remaining_today,
            'daily_limit': usage_stats.daily_limit,
            'can_make_request': usage_stats.can_make_request
        }
        
        # Format API logs from daily breakdown
        api_logs = []
        for day_stat in report.get('daily_breakdown', []):
            api_logs.append({
                'timestamp': day_stat['date'],
                'endpoint': 'news/all',
                'response_code': 200,  # Assumed successful for daily stats
                'articles': day_stat.get('articles', 0)
            })
        report['api_logs'] = api_logs
        
        return report
    
    def clean_cache(self):
        """Clean expired cache entries"""
        self.cache.clean_expired_cache()
    
    def get_usage_stats(self) -> 'ApiUsageStats':
        """Get current API usage statistics"""
        return self.cache.get_usage_stats()
    
    async def test_api_response(self, limit: int = 1) -> Dict[str, Any]:
        """
        Test the API response structure to understand the actual fields returned
        
        Returns:
            Dictionary with sample response and field analysis
        """
        try:
            # Make a minimal request
            response_data = await self._make_request('news/all', {'limit': limit})
            
            # Analyze response structure
            analysis = {
                'meta': response_data.get('meta', {}),
                'data_count': len(response_data.get('data', [])),
                'error': response_data.get('error'),
                'sample_article': None,
                'article_fields': [],
                'entity_fields': [],
                'entity_types': set()
            }
            
            if response_data.get('data'):
                sample = response_data['data'][0]
                analysis['sample_article'] = sample
                analysis['article_fields'] = list(sample.keys())
                
                # Analyze entities
                if 'entities' in sample and sample['entities']:
                    analysis['entity_fields'] = list(sample['entities'][0].keys())
                    analysis['entity_types'] = {e.get('type') for e in sample['entities'] if 'type' in e}
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error testing API response: {e}")
            return {'error': str(e)}
    
    async def close(self):
        """Close the aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None