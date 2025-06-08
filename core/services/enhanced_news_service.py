"""
Enhanced News Service combining ForexFactory calendar and MarketAux news
with smart request prioritization and sentiment analysis
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone, timedelta
from pathlib import Path
import asyncio

from core.domain.models import NewsEvent
from core.domain.exceptions import NewsError, ErrorContext
from core.services.news_service import NewsService
from core.infrastructure.marketaux import MarketAuxClient, MarketAuxArticle
from config.settings import NewsSettings, MarketAuxSettings

logger = logging.getLogger(__name__)


class EnhancedNewsContext:
    """Enhanced news context with sentiment analysis"""
    
    def __init__(
        self,
        calendar_events: List[NewsEvent],
        news_articles: List[MarketAuxArticle],
        sentiment_summary: Dict[str, Any]
    ):
        self.calendar_events = calendar_events
        self.news_articles = news_articles
        self.sentiment_summary = sentiment_summary
        self.generated_at = datetime.now(timezone.utc)
    
    def to_text_context(self, max_items: int = 10) -> List[str]:
        """Convert to text format for agents, prioritizing high-impact news"""
        context_lines = []
        
        # Add sentiment summary
        if self.sentiment_summary:
            sentiment_text = f"Market Sentiment: {self.sentiment_summary['sentiment_label']} ({self.sentiment_summary['average_sentiment']:.2f})"
            context_lines.append(sentiment_text)
        
        # Add high-impact calendar events (ForexFactory)
        high_impact_events = [e for e in self.calendar_events if e.is_high_impact][:5]
        if high_impact_events:
            context_lines.append("\n📅 Upcoming Economic Events:")
            for event in high_impact_events:
                time_until = (event.timestamp - datetime.now(timezone.utc)).total_seconds() / 60
                if time_until < 0:
                    context_lines.append(f"- ⚠️ {event.title} ({event.country}) - IN PROGRESS")
                else:
                    context_lines.append(f"- {event.title} ({event.country}) in {time_until:.0f} minutes")
        
        # Add real-time market news (MarketAux) with importance weighting
        if self.news_articles:
            # Separate by impact level
            high_impact_news = []
            medium_impact_news = []
            
            for article in self.news_articles:
                # Check if this is from MarketAux with importance scoring
                if hasattr(article, 'trading_impact') or 'trading_impact' in getattr(article, '__dict__', {}):
                    impact = getattr(article, 'trading_impact', 'low')
                    if impact == 'high':
                        high_impact_news.append(article)
                    elif impact == 'medium':
                        medium_impact_news.append(article)
                else:
                    # Legacy news without impact scoring
                    medium_impact_news.append(article)
            
            # Add high-impact news first
            if high_impact_news:
                context_lines.append("\n🔴 High-Impact Market News:")
                for article in high_impact_news[:5]:
                    context_lines.append(self._format_news_item(article))
            
            # Add medium-impact news if space remains
            remaining_slots = max_items - len(high_impact_events) - len(high_impact_news)
            if medium_impact_news and remaining_slots > 0:
                context_lines.append("\n📰 Market News:")
                for article in medium_impact_news[:remaining_slots]:
                    context_lines.append(self._format_news_item(article))
        
        return context_lines
    
    def _format_news_item(self, article) -> str:
        """Format a news article for agent consumption"""
        # Extract key information
        hours_ago = (datetime.now(timezone.utc) - article.published_at).total_seconds() / 3600
        
        # Build components
        components = []
        
        # Add keywords if available
        keywords = getattr(article, 'keywords', getattr(article, 'important_keywords', []))
        if keywords and isinstance(keywords, list):
            keywords_str = f"[{', '.join(keywords[:3])}]"
            components.append(keywords_str)
        
        # Add sentiment
        sentiment = getattr(article, 'sentiment', None)
        if sentiment and hasattr(sentiment, 'numeric_score'):
            # Handle MarketAuxSentiment object
            sentiment_score = sentiment.numeric_score
            if sentiment_score != 0:
                sentiment_emoji = "📈" if sentiment_score > 0 else "📉"
                components.append(f"{sentiment_emoji}{sentiment_score:+.2f}")
        elif isinstance(sentiment, (int, float)) and sentiment != 0:
            # Handle numeric sentiment
            sentiment_emoji = "📈" if sentiment > 0 else "📉"
            components.append(f"{sentiment_emoji}{sentiment:+.2f}")
        
        # Build final string
        prefix = " ".join(components)
        if prefix:
            return f"- {article.title} {prefix} ({hours_ago:.1f}h ago)"
        else:
            return f"- {article.title} ({hours_ago:.1f}h ago)"
    
    def get_trading_bias(self) -> Dict[str, Any]:
        """Get overall trading bias from news context"""
        # Calendar event impact
        restriction_score = 0
        for event in self.calendar_events[:3]:  # Next 3 events
            if event.is_high_impact:
                time_until = (event.timestamp - datetime.now(timezone.utc)).total_seconds() / 60
                if time_until < 30:  # Within 30 minutes
                    restriction_score += 1
        
        # Sentiment impact
        sentiment_score = self.sentiment_summary.get('average_sentiment', 0)
        
        # Combined bias
        if restriction_score >= 2:
            bias = "CAUTION"
            confidence = 0.8
        elif abs(sentiment_score) > 0.5:
            bias = "BULLISH" if sentiment_score > 0 else "BEARISH"
            confidence = min(abs(sentiment_score), 0.9)
        else:
            bias = "NEUTRAL"
            confidence = 0.5
        
        return {
            'bias': bias,
            'confidence': confidence,
            'restriction_score': restriction_score,
            'sentiment_score': sentiment_score
        }


class EnhancedNewsService:
    """
    Enhanced news service combining multiple data sources with intelligent caching
    and request prioritization for production use
    """
    
    def __init__(
        self,
        news_config: NewsSettings,
        marketaux_config: MarketAuxSettings,
        db_path: Path
    ):
        self.news_config = news_config
        self.marketaux_config = marketaux_config
        
        # Initialize base news service (ForexFactory)
        self.forex_factory_service = NewsService(news_config)
        
        # Initialize MarketAux client if enabled
        self.marketaux_client = None
        if marketaux_config.enabled and marketaux_config.api_token:
            cache_path = db_path.parent / "marketaux_cache.db"
            self.marketaux_client = MarketAuxClient(
                api_token=marketaux_config.api_token,
                cache_db_path=cache_path,
                daily_limit=marketaux_config.daily_limit,
                requests_per_minute=marketaux_config.requests_per_minute,
                free_plan=marketaux_config.free_plan
            )
            plan_type = "free" if marketaux_config.free_plan else "paid"
            logger.info(f"MarketAux integration enabled ({plan_type} plan)")
        else:
            logger.info("MarketAux integration disabled")
        
        # Request prioritization
        self._last_symbol_update: Dict[str, datetime] = {}
        self._symbol_importance: Dict[str, float] = {
            'EURUSD': 1.0,
            'GBPUSD': 0.9,
            'USDJPY': 0.8,
            'XAUUSD': 0.8,
            'AUDUSD': 0.7,
            'USDCAD': 0.6,
            'NZDUSD': 0.5,
            'USDCHF': 0.5
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        if self.marketaux_client:
            await self.marketaux_client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.marketaux_client:
            await self.marketaux_client.__aexit__(exc_type, exc_val, exc_tb)
    
    async def get_enhanced_news_context(
        self,
        symbol: str,
        lookahead_hours: int = 48,
        lookback_hours: int = 24,
        use_marketaux: bool = True
    ) -> EnhancedNewsContext:
        """
        Get enhanced news context combining calendar events and news articles
        
        Args:
            symbol: Trading symbol
            lookahead_hours: Hours to look ahead for calendar events
            lookback_hours: Hours to look back for news articles
            use_marketaux: Whether to use MarketAux data
            
        Returns:
            EnhancedNewsContext with combined data and sentiment
        """
        with ErrorContext("Get enhanced news context", symbol=symbol) as ctx:
            # Get ForexFactory calendar events
            calendar_events = await self.forex_factory_service.get_upcoming_news(
                symbol=symbol,
                within_minutes=lookahead_hours * 60
            )
            
            # Get MarketAux news if enabled
            news_articles = []
            sentiment_summary = {'sentiment_label': 'neutral', 'average_sentiment': 0}
            
            if use_marketaux and self.marketaux_client and self._should_update_symbol(symbol):
                try:
                    # Get news with sentiment
                    news_articles = await self.marketaux_client.get_news_with_sentiment(
                        symbols=[symbol],
                        published_after=datetime.now(timezone.utc) - timedelta(hours=lookback_hours),
                        limit=50,
                        use_cache=True,
                        cache_ttl_hours=self.marketaux_config.cache_ttl_hours
                    )
                    
                    # Get sentiment summary
                    sentiment_summary = await self.marketaux_client.get_symbol_news_summary(
                        symbol=symbol,
                        hours=lookback_hours,
                        min_relevance=self.marketaux_config.min_relevance_score
                    )
                    
                    # Update last request time
                    self._last_symbol_update[symbol] = datetime.now(timezone.utc)
                    
                except Exception as e:
                    logger.error(f"MarketAux error for {symbol}: {e}")
                    # Continue with just calendar data
            
            return EnhancedNewsContext(
                calendar_events=calendar_events,
                news_articles=news_articles,
                sentiment_summary=sentiment_summary
            )
    
    async def is_trading_restricted(
        self,
        symbol: str,
        now: Optional[datetime] = None,
        consider_sentiment: bool = True
    ) -> Tuple[bool, str]:
        """
        Enhanced trading restriction check considering both calendar and sentiment
        
        Returns:
            Tuple of (is_restricted, reason)
        """
        # First check calendar-based restrictions
        calendar_restricted = await self.forex_factory_service.is_trading_restricted(
            symbol=symbol,
            now=now
        )
        
        if calendar_restricted:
            return True, "High-impact economic event nearby"
        
        # Check sentiment-based restrictions if enabled
        if consider_sentiment and self.marketaux_client and self.marketaux_config.enabled:
            try:
                # Get recent sentiment
                sentiment_data = await self.marketaux_client.get_symbol_news_summary(
                    symbol=symbol,
                    hours=1,  # Last hour
                    min_relevance=0.5  # Higher threshold for restrictions
                )
                
                # Check for extreme sentiment with high-impact news
                if sentiment_data['high_impact_count'] > 0:
                    avg_sentiment = sentiment_data['average_sentiment']
                    if abs(avg_sentiment) > 0.8:  # Very strong sentiment
                        return True, f"Extreme market sentiment detected ({sentiment_data['sentiment_label']})"
                
            except Exception as e:
                logger.error(f"Error checking sentiment restrictions: {e}")
        
        return False, ""
    
    async def get_multi_symbol_context(
        self,
        symbols: List[str],
        max_requests: int = 10
    ) -> Dict[str, EnhancedNewsContext]:
        """
        Get news context for multiple symbols with smart prioritization
        
        Args:
            symbols: List of trading symbols
            max_requests: Maximum MarketAux requests to make
            
        Returns:
            Dictionary mapping symbols to their news contexts
        """
        result = {}
        
        # Get calendar events for all symbols (no API limit)
        calendar_tasks = []
        for symbol in symbols:
            task = self.forex_factory_service.get_upcoming_news(
                symbol=symbol,
                within_minutes=48 * 60  # 48 hours
            )
            calendar_tasks.append(task)
        
        calendar_results = await asyncio.gather(*calendar_tasks, return_exceptions=True)
        calendar_by_symbol = {}
        for symbol, events in zip(symbols, calendar_results):
            if isinstance(events, Exception):
                logger.error(f"Error getting calendar for {symbol}: {events}")
                calendar_by_symbol[symbol] = []
            else:
                calendar_by_symbol[symbol] = events
        
        # Check if we can use MarketAux
        if self.marketaux_client and self.marketaux_config.enabled:
            # Prioritize symbols for MarketAux
            prioritized_symbols = await self._prioritize_symbols(symbols, max_requests)
            
            if prioritized_symbols:
                try:
                    # Get news for prioritized symbols in a SINGLE batch request
                    summaries = await self.marketaux_client.get_batch_symbol_summaries(
                        symbols=prioritized_symbols,
                        hours=24,
                        min_relevance=self.marketaux_config.min_relevance_score,
                        use_cache=True
                    )
                    
                    # Update last request times
                    now = datetime.now(timezone.utc)
                    for symbol in prioritized_symbols:
                        self._last_symbol_update[symbol] = now
                    
                except Exception as e:
                    logger.error(f"MarketAux batch error: {e}")
                    summaries = {}
            else:
                summaries = {}
        else:
            summaries = {}
        
        # Build contexts for each symbol
        for symbol in symbols:
            calendar_events = calendar_by_symbol.get(symbol, [])
            
            # Use MarketAux data if available, otherwise empty
            if symbol in summaries:
                # We have MarketAux data
                sentiment_summary = summaries[symbol]
                # Note: We don't have the actual articles in batch summary,
                # but we have the summary data which is what's most important
                news_articles = []
            else:
                # No MarketAux data for this symbol
                sentiment_summary = {
                    'sentiment_label': 'neutral',
                    'average_sentiment': 0,
                    'article_count': 0,
                    'high_impact_count': 0
                }
                news_articles = []
            
            result[symbol] = EnhancedNewsContext(
                calendar_events=calendar_events,
                news_articles=news_articles,
                sentiment_summary=sentiment_summary
            )
        
        return result
    
    async def _prioritize_symbols(
        self,
        symbols: List[str],
        max_requests: int
    ) -> List[str]:
        """
        Prioritize which symbols should get MarketAux requests
        
        Prioritization based on:
        1. Time since last update
        2. Symbol importance/volume
        3. Available API quota
        """
        if not self.marketaux_client:
            return []
        
        # Check available quota
        can_request = self.marketaux_client.cache.can_make_request()
        if not can_request:
            logger.warning("MarketAux daily limit reached")
            return []
        
        # Score symbols
        symbol_scores = []
        now = datetime.now(timezone.utc)
        
        for symbol in symbols:
            # Time since last update (higher score for older)
            last_update = self._last_symbol_update.get(symbol)
            if last_update:
                hours_since = (now - last_update).total_seconds() / 3600
                time_score = min(hours_since / 24, 1.0)  # Max 1.0 after 24 hours
            else:
                time_score = 1.0  # Never updated
            
            # Symbol importance
            importance_score = self._symbol_importance.get(symbol, 0.3)
            
            # Combined score
            total_score = time_score * 0.7 + importance_score * 0.3
            symbol_scores.append((symbol, total_score))
        
        # Sort by score
        symbol_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top symbols within limit
        return [s[0] for s in symbol_scores[:max_requests]]
    
    def _should_update_symbol(self, symbol: str) -> bool:
        """Check if symbol should be updated based on cache age"""
        last_update = self._last_symbol_update.get(symbol)
        if not last_update:
            return True
        
        age_hours = (datetime.now(timezone.utc) - last_update).total_seconds() / 3600
        return age_hours >= self.marketaux_config.cache_ttl_hours
    
    async def get_news_summary_for_agents(
        self,
        symbol: str,
        max_items: int = 10
    ) -> List[str]:
        """
        Get formatted news summary for trading agents
        
        Returns list of news context strings suitable for GPT prompts
        """
        context = await self.get_enhanced_news_context(symbol)
        return context.to_text_context(max_items)
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get combined analytics from all news sources"""
        analytics = {
            'forex_factory': self.forex_factory_service.get_data_file_status(),
            'last_symbol_updates': {
                symbol: last_update.isoformat() if last_update else None
                for symbol, last_update in self._last_symbol_update.items()
            }
        }
        
        if self.marketaux_client:
            analytics['marketaux'] = self.marketaux_client.get_analytics()
        
        return analytics
    
    def get_data_file_status(self) -> Dict[str, Any]:
        """
        Get status of the news data file.
        This method ensures compatibility with the regular NewsService.
        
        Returns:
            Dictionary with file status information
        """
        # Delegate to the forex_factory_service which is the regular NewsService
        return self.forex_factory_service.get_data_file_status()
    
    async def get_next_high_impact_event(
        self,
        symbol: str,
        now: Optional[datetime] = None
    ) -> Optional[NewsEvent]:
        """
        Get the next high-impact event for a symbol.
        This method ensures compatibility with the regular NewsService.
        
        Args:
            symbol: Trading symbol
            now: Reference time
            
        Returns:
            Next high-impact NewsEvent or None
        """
        # Delegate to the forex_factory_service for calendar events
        return await self.forex_factory_service.get_next_high_impact_event(symbol, now)
    
    async def get_news_summary(self, symbol: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get enhanced news summary including both calendar and market news.
        
        Args:
            symbol: Trading symbol
            hours: Time window in hours
            
        Returns:
            Dictionary with enhanced news summary
        """
        # Use hybrid approach - get both calendar and real-time news
        hybrid_data = await self.get_hybrid_news_data(
            symbol=symbol,
            calendar_hours=48,  # Look ahead for calendar
            news_hours=hours   # Look back for news
        )
        
        # Build enhanced summary
        summary = {
            'symbol': symbol,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'calendar': hybrid_data['calendar_events'],
            'market_news': hybrid_data['market_news'],
            'combined_summary': hybrid_data['combined_summary']
        }
        
        # Add news-based trading recommendations
        if hybrid_data['market_news']['sentiment'] < -0.2:
            summary['news_bias'] = 'bearish'
        elif hybrid_data['market_news']['sentiment'] > 0.2:
            summary['news_bias'] = 'bullish'
        else:
            summary['news_bias'] = 'neutral'
        
        return summary
    
    async def get_upcoming_news(
        self,
        symbol: str,
        within_minutes: Optional[int] = None,
        now: Optional[datetime] = None
    ) -> List[NewsEvent]:
        """
        Get upcoming news events for a symbol within time window.
        This method ensures compatibility with the regular NewsService.
        
        Args:
            symbol: Trading symbol
            within_minutes: Time window in minutes (uses config default if None)
            now: Reference time (uses current time if None)
            
        Returns:
            List of upcoming NewsEvent objects
        """
        # Delegate to the forex_factory_service
        return await self.forex_factory_service.get_upcoming_news(symbol, within_minutes, now)
    
    def clean_cache(self):
        """Clean expired cache entries"""
        if self.marketaux_client:
            self.marketaux_client.clean_cache()
    
    async def fetch_realtime_market_news(
        self,
        symbols: List[str],
        hours: int = 24,
        force_refresh: bool = False
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch real-time market news from MarketAux for multiple symbols efficiently.
        
        Args:
            symbols: List of trading symbols
            hours: Hours to look back for news
            force_refresh: Force API call even if cache exists
            
        Returns:
            Dictionary mapping symbols to their news articles
        """
        if not self.marketaux_client or not self.marketaux_config.enabled:
            logger.warning("MarketAux not enabled, returning empty news")
            return {symbol: [] for symbol in symbols}
        
        try:
            # Calculate time window
            published_after = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            # Convert forex symbols to countries using ForexSymbolConverter
            from core.infrastructure.marketaux.forex_utils import ForexSymbolConverter
            marketaux_params = ForexSymbolConverter.get_marketaux_params_for_symbols(symbols)
            
            logger.info(f"Fetching real-time news for symbols: {symbols}")
            logger.info(f"MarketAux params - Countries: {marketaux_params.get('countries', [])}, Search: {marketaux_params.get('search', '')}")
            
            # Fetch news using what actually works with MarketAux
            # Don't pass symbols - use search instead
            response = await self.marketaux_client.get_news(
                search='forex | currency | "exchange rate" | inflation | "interest rate" | "central bank"',
                entity_types=['currency'],  # Filter for currency entities
                published_after=published_after,
                limit=100,  # Max allowed
                language='en',
                use_cache=not force_refresh,
                cache_ttl_hours=self.marketaux_config.cache_ttl_hours,
                filter_entities=True,
                must_have_entities=False
            )
            
            # Group and weight articles by relevance to each symbol
            news_by_symbol = {symbol: [] for symbol in symbols}
            
            logger.info(f"Processing {len(response.data)} articles from MarketAux")
            
            for article in response.data:
                # Calculate article importance based on multiple factors
                article_data = self._process_article_for_trading(article, symbols)
                
                # Debug log
                if article_data['symbol_relevance']:
                    logger.debug(f"Article: {article.title[:50]}... Relevance: {article_data['symbol_relevance']}")
                
                # Add to relevant symbols with calculated importance
                for symbol, importance_score in article_data['symbol_relevance'].items():
                    if importance_score > 0.05:  # Very low threshold to get more articles
                        news_by_symbol[symbol].append({
                            'title': article.title,
                            'description': article.description,
                            'url': article.url,
                            'published_at': article.published_at,
                            'source': article.source,
                            'sentiment': getattr(article, 'sentiment', 0),
                            'relevance_score': article.relevance_score,
                            'importance_score': importance_score,
                            'entities': article_data['key_entities'],
                            'trading_impact': article_data['trading_impact'],
                            'keywords': article_data['important_keywords']
                        })
            
            # Sort articles by importance for each symbol
            for symbol in news_by_symbol:
                news_by_symbol[symbol].sort(
                    key=lambda x: x['importance_score'] * x['relevance_score'], 
                    reverse=True
                )
                # Keep top 20 most important articles per symbol
                news_by_symbol[symbol] = news_by_symbol[symbol][:20]
                logger.info(f"{symbol}: Found {len(news_by_symbol[symbol])} important news articles")
            
            return news_by_symbol
            
        except Exception as e:
            logger.error(f"Error fetching real-time news: {e}")
            return {symbol: [] for symbol in symbols}
    
    def _process_article_for_trading(
        self, 
        article: Any, 
        symbols: List[str]
    ) -> Dict[str, Any]:
        """
        Process article to determine trading relevance and importance.
        
        Returns:
            Dict with symbol_relevance, trading_impact, key_entities, etc.
        """
        # Keywords that indicate high-impact news
        HIGH_IMPACT_KEYWORDS = {
            'rate': 3.0, 'rates': 3.0, 'interest': 3.0,
            'inflation': 3.0, 'cpi': 3.0, 'gdp': 3.0,
            'employment': 2.5, 'unemployment': 2.5, 'jobs': 2.5, 'payroll': 2.5,
            'fed': 3.0, 'ecb': 3.0, 'boe': 3.0, 'boj': 3.0, 'fomc': 3.0,
            'central bank': 3.0, 'monetary': 2.5, 'policy': 2.0,
            'stimulus': 2.5, 'tapering': 2.5, 'hawkish': 2.5, 'dovish': 2.5,
            'recession': 3.0, 'growth': 2.0, 'crisis': 3.0,
            'trade': 2.0, 'tariff': 2.0, 'sanctions': 2.5,
            'oil': 2.0, 'energy': 2.0, 'commodity': 1.5,
            'forex': 2.0, 'fx': 2.0, 'currency': 2.0, 'exchange': 1.5,
            'dollar': 1.5, 'euro': 1.5, 'pound': 1.5, 'yen': 1.5
        }
        
        # Calculate base importance from keywords
        text_lower = f"{article.title} {article.description}".lower()
        
        # Also check article's own keywords if available
        article_keywords = getattr(article, 'keywords', [])
        if article_keywords and isinstance(article_keywords, list):
            text_lower += ' ' + ' '.join(article_keywords).lower()
        
        keyword_score = 0
        important_keywords = []
        
        for keyword, weight in HIGH_IMPACT_KEYWORDS.items():
            if keyword in text_lower:
                keyword_score += weight
                important_keywords.append(keyword)
        
        # Normalize keyword score (0-1)
        keyword_importance = min(keyword_score / 10.0, 1.0)
        
        # Determine which symbols this affects
        from core.infrastructure.marketaux.forex_utils import ForexSymbolConverter
        symbol_relevance = {}
        
        for symbol in symbols:
            currencies = ForexSymbolConverter.get_currencies_for_symbol(symbol)
            countries = ForexSymbolConverter.get_countries_for_symbol(symbol)
            
            relevance_score = 0
            
            # Check if article countries match symbol countries
            article_countries = getattr(article, 'countries', [])
            if article_countries and countries:
                # Any country match is relevant
                if any(country in countries for country in article_countries):
                    relevance_score += 0.5
            else:
                # If no country data, check for country mentions in text
                country_keywords = {
                    'us': ['us ', 'u.s.', 'united states', 'america', 'fed ', 'federal reserve', 'dollar'],
                    'gb': ['uk ', 'u.k.', 'britain', 'british', 'england', 'boe ', 'bank of england', 'pound'],
                    'de': ['german', 'germany', 'ecb ', 'european central bank', 'eurozone', 'euro '],
                    'jp': ['japan', 'japanese', 'boj ', 'bank of japan', 'yen']
                }
                
                for country_code, keywords in country_keywords.items():
                    if country_code in countries:
                        if any(kw in text_lower for kw in keywords):
                            relevance_score += 0.4
                            break
            
            # Check currency mentions
            for currency in currencies:
                if currency.lower() in text_lower:
                    relevance_score += 0.3
                # Check currency names
                currency_names = {
                    'USD': 'dollar', 'EUR': 'euro', 'GBP': 'pound',
                    'JPY': 'yen', 'CHF': 'franc', 'CAD': 'canadian',
                    'AUD': 'australian', 'NZD': 'zealand'
                }
                if currency in currency_names and currency_names[currency] in text_lower:
                    relevance_score += 0.2
            
            # Check currency entities in the article
            entities = getattr(article, 'entities', [])
            if entities:
                for entity in entities:
                    if entity.get('type') == 'currency' and entity.get('name'):
                        entity_name = entity['name'].upper()
                        # Check if this currency pair matches our symbol
                        if symbol in entity_name or any(curr in entity_name for curr in currencies):
                            relevance_score += 0.5
                            break
            
            # If article has any forex/economic keywords, boost relevance
            if keyword_importance > 0:
                relevance_score = max(relevance_score, 0.3)
            
            # Combine with keyword importance
            symbol_relevance[symbol] = min(relevance_score + (keyword_importance * 0.5), 1.0)
        
        # Extract key entities
        entities = getattr(article, 'entities', [])
        key_entities = [
            entity for entity in entities 
            if entity.get('type') in ['ORG', 'PERSON', 'GPE'] and 
            entity.get('relevance_score', 0) > 0.7
        ][:5]  # Top 5 entities
        
        # Determine trading impact
        if keyword_importance > 0.7:
            trading_impact = 'high'
        elif keyword_importance > 0.4:
            trading_impact = 'medium'
        else:
            trading_impact = 'low'
        
        return {
            'symbol_relevance': symbol_relevance,
            'trading_impact': trading_impact,
            'key_entities': key_entities,
            'important_keywords': important_keywords[:5],
            'keyword_importance': keyword_importance
        }
    
    async def get_hybrid_news_data(
        self,
        symbol: str,
        calendar_hours: int = 48,
        news_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get hybrid news data combining ForexFactory calendar and MarketAux news.
        
        Args:
            symbol: Trading symbol
            calendar_hours: Hours to look ahead for economic events
            news_hours: Hours to look back for news articles
            
        Returns:
            Combined news data with both calendar events and market news
        """
        # Get calendar events from ForexFactory
        calendar_context = await self.forex_factory_service.get_news_context(
            symbol=symbol,
            hours=calendar_hours,
            check_blacklist=True
        )
        
        # Get real-time news from MarketAux
        realtime_news = await self.fetch_realtime_market_news(
            symbols=[symbol],
            hours=news_hours
        )
        
        # Combine data
        return {
            'calendar_events': {
                'upcoming': calendar_context.upcoming_events,
                'high_impact': calendar_context.high_impact_events,
                'restrictions': calendar_context.restrictions,
                'summary': calendar_context.get_summary()
            },
            'market_news': {
                'articles': realtime_news.get(symbol, []),
                'count': len(realtime_news.get(symbol, [])),
                'sentiment': self._calculate_average_sentiment(realtime_news.get(symbol, []))
            },
            'combined_summary': self._create_combined_summary(
                calendar_context,
                realtime_news.get(symbol, [])
            )
        }
    
    def _calculate_average_sentiment(self, articles: List[Dict[str, Any]]) -> float:
        """Calculate average sentiment from articles"""
        if not articles:
            return 0.0
        
        sentiments = [a.get('sentiment', 0) for a in articles]
        return sum(sentiments) / len(sentiments) if sentiments else 0.0
    
    def _create_combined_summary(
        self,
        calendar_context: Any,
        news_articles: List[Dict[str, Any]]
    ) -> str:
        """Create a combined summary of calendar and news data"""
        summary_parts = []
        
        # Calendar summary
        if calendar_context.high_impact_events:
            summary_parts.append(
                f"📅 {len(calendar_context.high_impact_events)} high-impact events upcoming"
            )
        
        # News summary
        if news_articles:
            avg_sentiment = self._calculate_average_sentiment(news_articles)
            sentiment_label = "positive" if avg_sentiment > 0.1 else "negative" if avg_sentiment < -0.1 else "neutral"
            summary_parts.append(
                f"📰 {len(news_articles)} recent articles ({sentiment_label} sentiment)"
            )
        
        return " | ".join(summary_parts) if summary_parts else "No significant news"


# Export main service
__all__ = ['EnhancedNewsService', 'EnhancedNewsContext']