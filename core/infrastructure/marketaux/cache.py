"""MarketAux cache management with SQLite backend"""

import json
import logging
import sqlite3
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager

from .models import MarketAuxArticle, ApiUsageStats

logger = logging.getLogger(__name__)


class MarketAuxCache:
    """Manages caching of MarketAux data and API usage tracking"""
    
    def __init__(self, db_path: Path, daily_limit: int = 100):
        self.db_path = db_path
        self.daily_limit = daily_limit
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        with self._get_connection() as conn:
            # Articles cache table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS marketaux_articles (
                    uuid TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    snippet TEXT,
                    url TEXT NOT NULL,
                    image_url TEXT,
                    published_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    relevance_score REAL,
                    countries TEXT,  -- JSON array
                    entities TEXT,   -- JSON array
                    highlights TEXT, -- JSON array
                    sentiment_data TEXT,  -- JSON object
                    symbols TEXT,    -- JSON array
                    keywords TEXT,   -- JSON array
                    is_high_impact INTEGER DEFAULT 0,
                    cached_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
            """)
            
            # API usage tracking table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS marketaux_api_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_time TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    params TEXT,  -- JSON
                    response_code INTEGER,
                    articles_returned INTEGER,
                    error_message TEXT
                )
            """)
            
            # Request tracking for rate limiting
            conn.execute("""
                CREATE TABLE IF NOT EXISTS marketaux_rate_limits (
                    date TEXT PRIMARY KEY,
                    requests_today INTEGER DEFAULT 0,
                    last_request_time TEXT,
                    daily_limit INTEGER DEFAULT 100
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_published ON marketaux_articles(published_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_expires ON marketaux_articles(expires_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_api_usage_time ON marketaux_api_usage(request_time)")
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection context manager"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def cache_articles(self, articles: List[MarketAuxArticle], ttl_hours: int = 24):
        """Cache articles with TTL"""
        now = datetime.now()
        expires_at = now + timedelta(hours=ttl_hours)
        
        with self._get_connection() as conn:
            for article in articles:
                # Prepare data
                sentiment_data = None
                if article.sentiment:
                    sentiment_data = json.dumps({
                        'overall': article.sentiment.overall.value,
                        'confidence': article.sentiment.confidence,
                        'positive_score': article.sentiment.positive_score,
                        'negative_score': article.sentiment.negative_score,
                        'neutral_score': article.sentiment.neutral_score,
                        'entity_sentiments': {k: v.value for k, v in article.sentiment.entity_sentiments.items()}
                    })
                
                # Insert or replace
                conn.execute("""
                    INSERT OR REPLACE INTO marketaux_articles (
                        uuid, title, description, snippet, url, image_url,
                        published_at, source, relevance_score, countries,
                        entities, highlights, sentiment_data, symbols,
                        keywords, is_high_impact, cached_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    article.uuid,
                    article.title,
                    article.description,
                    article.snippet,
                    article.url,
                    article.image_url,
                    article.published_at.isoformat(),
                    article.source,
                    article.relevance_score,
                    json.dumps(article.countries),
                    json.dumps(article.entities),
                    json.dumps(article.highlights),
                    sentiment_data,
                    json.dumps(article.symbols),
                    json.dumps(article.keywords),
                    int(article.is_high_impact),
                    now.isoformat(),
                    expires_at.isoformat()
                ))
            
            conn.commit()
            logger.info(f"Cached {len(articles)} MarketAux articles")
    
    def get_cached_articles(
        self,
        symbol: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[MarketAuxArticle]:
        """Get cached articles"""
        now = datetime.now()
        
        query = """
            SELECT * FROM marketaux_articles
            WHERE expires_at > ?
        """
        params = [now.isoformat()]
        
        if since:
            query += " AND published_at >= ?"
            params.append(since.isoformat())
        
        query += " ORDER BY published_at DESC LIMIT ?"
        params.append(limit)
        
        articles = []
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            
            for row in cursor:
                try:
                    # Reconstruct article
                    article_data = {
                        'uuid': row['uuid'],
                        'title': row['title'],
                        'description': row['description'],
                        'snippet': row['snippet'],
                        'url': row['url'],
                        'image_url': row['image_url'],
                        'published_at': row['published_at'],
                        'source': row['source'],
                        'relevance_score': row['relevance_score'],
                        'countries': json.loads(row['countries']),
                        'entities': json.loads(row['entities']),
                        'highlights': json.loads(row['highlights']),
                        'symbols': json.loads(row['symbols']),
                        'keywords': json.loads(row['keywords']),
                        'is_high_impact': bool(row['is_high_impact'])
                    }
                    
                    # Reconstruct sentiment if available
                    if row['sentiment_data']:
                        sentiment_data = json.loads(row['sentiment_data'])
                        from .models import MarketAuxSentiment, SentimentScore
                        
                        article_data['sentiment'] = MarketAuxSentiment(
                            overall=SentimentScore(sentiment_data['overall']),
                            confidence=sentiment_data['confidence'],
                            positive_score=sentiment_data['positive_score'],
                            negative_score=sentiment_data['negative_score'],
                            neutral_score=sentiment_data['neutral_score'],
                            entity_sentiments={
                                k: SentimentScore(v) 
                                for k, v in sentiment_data.get('entity_sentiments', {}).items()
                            }
                        )
                    
                    article = MarketAuxArticle(**article_data)
                    
                    # Filter by symbol relevance if specified
                    if symbol:
                        if article.get_symbol_relevance(symbol) >= 0.3:
                            articles.append(article)
                    else:
                        articles.append(article)
                        
                except Exception as e:
                    logger.error(f"Error reconstructing article {row['uuid']}: {e}")
                    continue
        
        return articles
    
    def log_api_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
        response_code: Optional[int] = None,
        articles_returned: int = 0,
        error_message: Optional[str] = None
    ):
        """Log API request for analytics"""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO marketaux_api_usage (
                    request_time, endpoint, params, response_code,
                    articles_returned, error_message
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                endpoint,
                json.dumps(params),
                response_code,
                articles_returned,
                error_message
            ))
            conn.commit()
    
    def get_usage_stats(self) -> ApiUsageStats:
        """Get current API usage statistics"""
        today = datetime.now().date()
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        with self._get_connection() as conn:
            # Get today's stats
            cursor = conn.execute("""
                SELECT COUNT(*) as count FROM marketaux_api_usage
                WHERE date(request_time) = date(?)
                AND response_code < 400
            """, (today.isoformat(),))
            
            requests_today = cursor.fetchone()['count']
            
            # Get this hour's stats
            cursor = conn.execute("""
                SELECT COUNT(*) as count FROM marketaux_api_usage
                WHERE request_time >= ?
                AND response_code < 400
            """, (current_hour.isoformat(),))
            
            requests_this_hour = cursor.fetchone()['count']
            
            # Get last request time
            cursor = conn.execute("""
                SELECT MAX(request_time) as last_time FROM marketaux_api_usage
                WHERE response_code < 400
            """)
            
            row = cursor.fetchone()
            last_request_time = None
            if row['last_time']:
                last_request_time = datetime.fromisoformat(row['last_time'])
            
            return ApiUsageStats(
                date=datetime.now(),
                requests_today=requests_today,
                requests_this_hour=requests_this_hour,
                last_request_time=last_request_time,
                daily_limit=self.daily_limit,
                hourly_limit=60  # Increase from default 10 to 60 per hour
            )
    
    def update_rate_limit(self):
        """Update rate limit tracking after request"""
        stats = self.get_usage_stats()
        stats.update_for_request()
        
        # Store in database for persistence
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO marketaux_rate_limits (
                    date, requests_today, last_request_time, daily_limit
                ) VALUES (?, ?, ?, ?)
            """, (
                stats.date.date().isoformat(),
                stats.requests_today,
                stats.last_request_time.isoformat() if stats.last_request_time else None,
                stats.daily_limit
            ))
            conn.commit()
    
    def can_make_request(self) -> bool:
        """Check if we can make another API request"""
        stats = self.get_usage_stats()
        return stats.can_make_request
    
    def clean_expired_cache(self):
        """Remove expired cache entries"""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                DELETE FROM marketaux_articles
                WHERE expires_at < ?
            """, (datetime.now().isoformat(),))
            
            deleted = cursor.rowcount
            conn.commit()
            
            if deleted > 0:
                logger.info(f"Cleaned {deleted} expired MarketAux cache entries")
    
    def get_api_usage_report(self, days: int = 7) -> Dict[str, Any]:
        """Get API usage report for analytics"""
        since = datetime.now() - timedelta(days=days)
        
        with self._get_connection() as conn:
            # Total requests
            cursor = conn.execute("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN response_code < 400 THEN 1 ELSE 0 END) as successful,
                       SUM(CASE WHEN response_code >= 400 THEN 1 ELSE 0 END) as failed,
                       SUM(articles_returned) as total_articles
                FROM marketaux_api_usage
                WHERE request_time >= ?
            """, (since.isoformat(),))
            
            stats = cursor.fetchone()
            
            # Daily breakdown
            cursor = conn.execute("""
                SELECT date(request_time) as day,
                       COUNT(*) as requests,
                       SUM(articles_returned) as articles
                FROM marketaux_api_usage
                WHERE request_time >= ?
                GROUP BY date(request_time)
                ORDER BY day DESC
            """, (since.isoformat(),))
            
            daily_stats = []
            for row in cursor:
                daily_stats.append({
                    'date': row['day'],
                    'requests': row['requests'],
                    'articles': row['articles'] or 0
                })
            
            # Error analysis
            cursor = conn.execute("""
                SELECT error_message, COUNT(*) as count
                FROM marketaux_api_usage
                WHERE request_time >= ? AND error_message IS NOT NULL
                GROUP BY error_message
                ORDER BY count DESC
                LIMIT 10
            """, (since.isoformat(),))
            
            errors = []
            for row in cursor:
                errors.append({
                    'message': row['error_message'],
                    'count': row['count']
                })
            
            return {
                'period_days': days,
                'total_requests': stats['total'] or 0,
                'successful_requests': stats['successful'] or 0,
                'failed_requests': stats['failed'] or 0,
                'total_articles': stats['total_articles'] or 0,
                'daily_breakdown': daily_stats,
                'top_errors': errors,
                'cache_size': self._get_cache_size()
            }
    
    def _get_cache_size(self) -> Dict[str, int]:
        """Get current cache size"""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) as total,
                       COUNT(CASE WHEN expires_at > ? THEN 1 END) as valid
                FROM marketaux_articles
            """, (datetime.now().isoformat(),))
            
            row = cursor.fetchone()
            return {
                'total_articles': row['total'],
                'valid_articles': row['valid']
            }