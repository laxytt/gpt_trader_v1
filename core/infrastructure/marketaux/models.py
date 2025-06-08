"""MarketAux data models"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class SentimentScore(str, Enum):
    """Sentiment score categories"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"
    
    @property
    def numeric_score(self) -> float:
        """Convert sentiment to numeric score between -1 and 1"""
        scores = {
            self.VERY_POSITIVE: 1.0,
            self.POSITIVE: 0.5,
            self.NEUTRAL: 0.0,
            self.NEGATIVE: -0.5,
            self.VERY_NEGATIVE: -1.0
        }
        return scores.get(self, 0.0)
    
    @classmethod
    def from_numeric(cls, score: float) -> 'SentimentScore':
        """Convert numeric score to sentiment category"""
        if score >= 0.7:
            return cls.VERY_POSITIVE
        elif score >= 0.3:
            return cls.POSITIVE
        elif score <= -0.7:
            return cls.VERY_NEGATIVE
        elif score <= -0.3:
            return cls.NEGATIVE
        else:
            return cls.NEUTRAL


class MarketAuxSentiment(BaseModel):
    """Sentiment analysis result"""
    overall: SentimentScore
    confidence: float = Field(ge=0.0, le=1.0)
    
    # Detailed scores
    positive_score: float = Field(ge=0.0, le=1.0)
    negative_score: float = Field(ge=0.0, le=1.0)
    neutral_score: float = Field(ge=0.0, le=1.0)
    
    # Entity-specific sentiments (currencies, companies, etc.)
    entity_sentiments: Dict[str, SentimentScore] = Field(default_factory=dict)
    
    @property
    def numeric_score(self) -> float:
        """Get overall numeric sentiment score"""
        return self.overall.numeric_score


class MarketAuxArticle(BaseModel):
    """News article from MarketAux"""
    uuid: str
    title: str
    description: Optional[str] = None
    snippet: Optional[str] = None
    url: str
    image_url: Optional[str] = None
    published_at: datetime
    source: str
    
    # Relevance and categorization
    relevance_score: float = Field(ge=0.0, le=1.0)
    countries: List[str] = Field(default_factory=list)
    entities: List[Dict[str, Any]] = Field(default_factory=list)  # Companies, currencies, etc.
    highlights: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Sentiment analysis
    sentiment: Optional[MarketAuxSentiment] = None
    
    # Trading-specific fields
    symbols: List[str] = Field(default_factory=list)  # Extracted trading symbols
    keywords: List[str] = Field(default_factory=list)
    is_high_impact: bool = Field(default=False)
    
    @field_validator('published_at', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        if isinstance(v, str):
            # Handle ISO format with Z suffix
            if v.endswith('Z'):
                v = v[:-1] + '+00:00'
            return datetime.fromisoformat(v)
        return v
    
    @field_validator('keywords', mode='before')
    @classmethod
    def parse_keywords(cls, v):
        """Parse keywords from string or list format"""
        if isinstance(v, str):
            # Convert comma-separated string to list
            if v.strip():
                return [k.strip() for k in v.split(',') if k.strip()]
            else:
                return []
        elif isinstance(v, list):
            return v
        else:
            return []
    
    def get_symbol_relevance(self, symbol: str) -> float:
        """Get relevance score for a specific trading symbol (optimized for free plan)"""
        symbol_upper = symbol.upper()
        relevance_score = 0.0
        
        # Direct symbol match in our extracted symbols
        if symbol_upper in [s.upper() for s in self.symbols]:
            return 1.0
        
        # For free plan: Check article text content
        text_content = f"{self.title} {self.description or ''} {self.snippet or ''}".upper()
        
        # Currency pair matching for forex
        if len(symbol_upper) == 6:  # Forex pair like EURUSD
            base = symbol_upper[:3]
            quote = symbol_upper[3:]
            
            # Check for direct pair mentions
            pair_formats = [
                symbol_upper,  # EURUSD
                f"{base}/{quote}",  # EUR/USD
                f"{base}-{quote}",  # EUR-USD
                f"{base} {quote}"   # EUR USD (with space)
            ]
            
            # Direct pair mention is highest relevance
            if any(fmt in text_content for fmt in pair_formats):
                relevance_score = 0.9
            else:
                # Check if both currencies are mentioned
                currency_terms = {
                    'EUR': ['EUR', 'EURO', 'EUROZONE', 'EUROPEAN CURRENCY'],
                    'USD': ['USD', 'DOLLAR', 'US DOLLAR', 'GREENBACK', 'DXY'],
                    'GBP': ['GBP', 'POUND', 'STERLING', 'BRITISH POUND', 'CABLE'],
                    'JPY': ['JPY', 'YEN', 'JAPANESE YEN'],
                    'AUD': ['AUD', 'AUSSIE', 'AUSTRALIAN DOLLAR'],
                    'CAD': ['CAD', 'LOONIE', 'CANADIAN DOLLAR'],
                    'CHF': ['CHF', 'SWISS', 'SWISS FRANC', 'SWISSIE'],
                    'NZD': ['NZD', 'KIWI', 'NEW ZEALAND DOLLAR']
                }
                
                base_found = any(term in text_content for term in currency_terms.get(base, [base]))
                quote_found = any(term in text_content for term in currency_terms.get(quote, [quote]))
                
                if base_found and quote_found:
                    relevance_score = 0.7
                elif base_found or quote_found:
                    relevance_score = 0.4
            
            # Check entities if available (paid plan)
            if self.entities:
                currency_count = 0
                for entity in self.entities:
                    if entity.get('type') == 'currency':
                        if entity.get('symbol', '').upper() in [base, quote]:
                            currency_count += 1
                
                if currency_count == 2:
                    relevance_score = max(relevance_score, 0.8)
                elif currency_count == 1:
                    relevance_score = max(relevance_score, 0.5)
        
        # Boost relevance for forex-specific content
        forex_indicators = [
            'FOREX', 'FX MARKET', 'CURRENCY PAIR', 'EXCHANGE RATE',
            'PIP', 'SPREAD', 'CURRENCY TRADING', 'FX TRADING'
        ]
        
        forex_count = sum(1 for term in forex_indicators if term in text_content)
        if forex_count > 0:
            relevance_score += min(0.2, forex_count * 0.05)
        
        # Check central bank and economic indicators
        if any(symbol_upper[:3] in text_content or symbol_upper[3:] in text_content for _ in [1]):
            econ_terms = [
                'CENTRAL BANK', 'INTEREST RATE', 'MONETARY POLICY',
                'INFLATION', 'GDP', 'NFP', 'NON-FARM', 'CPI',
                'FOMC', 'ECB', 'BOE', 'BOJ', 'RBA', 'BOC'
            ]
            
            econ_count = sum(1 for term in econ_terms if term in text_content)
            if econ_count > 0:
                relevance_score += min(0.15, econ_count * 0.03)
        
        # High impact news gets a boost
        if self.is_high_impact:
            relevance_score += 0.1
        
        # Keyword matching
        symbol_keywords = self._get_symbol_keywords(symbol_upper)
        if self.keywords:
            keywords_lower = [kw.lower() for kw in self.keywords]
            keyword_matches = sum(1 for kw in symbol_keywords if kw in keywords_lower)
            if keyword_matches > 0:
                relevance_score += min(0.2, keyword_matches * 0.05)
        
        return min(1.0, relevance_score)
    
    def _get_symbol_keywords(self, symbol: str) -> List[str]:
        """Get relevant keywords for a symbol"""
        keywords = []
        
        if symbol == "EURUSD":
            keywords = ["euro", "eur", "european", "ecb", "eurozone", "dollar", "usd", "fed"]
        elif symbol == "GBPUSD":
            keywords = ["pound", "gbp", "sterling", "boe", "uk", "british", "dollar", "usd"]
        elif symbol == "USDJPY":
            keywords = ["dollar", "usd", "yen", "jpy", "japan", "boj", "fed"]
        elif symbol == "AUDUSD":
            keywords = ["aussie", "aud", "australian", "rba", "dollar", "usd"]
        elif symbol == "USDCAD":
            keywords = ["dollar", "usd", "canadian", "cad", "loonie", "boc"]
        elif symbol == "XAUUSD":
            keywords = ["gold", "xau", "precious", "metal", "commodity", "dollar"]
        
        return [kw.lower() for kw in keywords]


class MarketAuxResponse(BaseModel):
    """API response wrapper"""
    meta: Dict[str, Any] = Field(default_factory=dict)
    data: List[MarketAuxArticle] = Field(default_factory=list)
    error: Optional[str] = None
    
    @property
    def found(self) -> int:
        """Number of articles found"""
        return self.meta.get('found', len(self.data))
    
    @property
    def page(self) -> int:
        """Current page number"""
        return self.meta.get('page', 1)
    
    @property
    def limit(self) -> int:
        """Results per page"""
        return self.meta.get('limit', len(self.data))
    
    @property
    def has_more(self) -> bool:
        """Whether more pages are available"""
        total = self.meta.get('found', 0)
        returned = self.page * self.limit
        return returned < total


class ApiUsageStats(BaseModel):
    """API usage tracking"""
    date: datetime = Field(default_factory=datetime.now)
    requests_today: int = Field(default=0, ge=0)
    requests_this_hour: int = Field(default=0, ge=0)
    last_request_time: Optional[datetime] = None
    daily_limit: int = Field(default=100)
    hourly_limit: int = Field(default=10)
    
    @property
    def remaining_today(self) -> int:
        """Remaining requests for today"""
        return max(0, self.daily_limit - self.requests_today)
    
    @property
    def remaining_this_hour(self) -> int:
        """Remaining requests for this hour"""
        return max(0, self.hourly_limit - self.requests_this_hour)
    
    @property
    def can_make_request(self) -> bool:
        """Check if we can make another request"""
        return self.remaining_today > 0 and self.remaining_this_hour > 0
    
    def update_for_request(self):
        """Update stats after making a request"""
        now = datetime.now()
        
        # Reset daily counter if new day
        if self.date.date() != now.date():
            self.date = now
            self.requests_today = 0
            self.requests_this_hour = 0
        
        # Reset hourly counter if new hour
        if self.last_request_time and self.last_request_time.hour != now.hour:
            self.requests_this_hour = 0
        
        self.requests_today += 1
        self.requests_this_hour += 1
        self.last_request_time = now