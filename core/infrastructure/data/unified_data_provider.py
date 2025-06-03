# core/infrastructure/data/unified_data_provider.py
"""
Unified data provider that handles both real-time and historical data requests.
Single source of truth for all market data operations.
"""

import asyncio
import logging
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
from functools import lru_cache

from core.domain.models import MarketData, Candle
from core.domain.enums import TimeFrame
from core.domain.exceptions import DataError, InsufficientDataError
from core.infrastructure.mt5.client import MT5Client

logger = logging.getLogger(__name__)


@dataclass
class DataRequest:
    """Standardized data request format"""
    symbol: str
    timeframe: TimeFrame
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    num_bars: Optional[int] = None
    
    def __post_init__(self):
        # Ensure timezone awareness
        if self.start_date and self.start_date.tzinfo is None:
            self.start_date = self.start_date.replace(tzinfo=timezone.utc)
        if self.end_date and self.end_date.tzinfo is None:
            self.end_date = self.end_date.replace(tzinfo=timezone.utc)


class TimeframeCalculator:
    """Handles all timeframe and bar calculations"""
    
    TIMEFRAME_MINUTES = {
        TimeFrame.M1: 1,
        TimeFrame.M5: 5,
        TimeFrame.M15: 15,
        TimeFrame.M30: 30,
        TimeFrame.H1: 60,
        TimeFrame.H4: 240,
        TimeFrame.D1: 1440,
        TimeFrame.W1: 10080,
        TimeFrame.MN1: 43200
    }
    
    @classmethod
    def calculate_bars_needed(
        cls,
        start_date: datetime,
        end_date: datetime,
        timeframe: TimeFrame,
        buffer_bars: int = 200
    ) -> int:
        """Calculate exact number of bars needed for date range"""
        duration = end_date - start_date
        minutes = duration.total_seconds() / 60
        timeframe_minutes = cls.TIMEFRAME_MINUTES[timeframe]
        
        # Account for market closure (weekends)
        if timeframe in [TimeFrame.M1, TimeFrame.M5, TimeFrame.M15, TimeFrame.M30, TimeFrame.H1, TimeFrame.H4]:
            # Forex markets closed ~48 hours per week
            weeks = duration.days / 7
            market_closed_hours = weeks * 48
            minutes -= market_closed_hours * 60
        
        bars_needed = int(minutes / timeframe_minutes) + buffer_bars
        return max(bars_needed, buffer_bars)
    
    @classmethod
    def get_date_from_bars(
        cls,
        reference_date: datetime,
        num_bars: int,
        timeframe: TimeFrame,
        direction: str = 'backward'
    ) -> datetime:
        """Calculate date from number of bars"""
        timeframe_minutes = cls.TIMEFRAME_MINUTES[timeframe]
        total_minutes = num_bars * timeframe_minutes
        
        if direction == 'backward':
            return reference_date - timedelta(minutes=total_minutes)
        else:
            return reference_date + timedelta(minutes=total_minutes)


class DataCache:
    """Thread-safe data cache with TTL"""
    
    def __init__(self, ttl_seconds: int = 300):
        self._cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}
        self.ttl_seconds = ttl_seconds
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get cached data if not expired"""
        if key in self._cache:
            data, timestamp = self._cache[key]
            if (datetime.now() - timestamp).total_seconds() < self.ttl_seconds:
                logger.debug(f"Cache hit for {key}")
                return data.copy()
            else:
                del self._cache[key]
                logger.debug(f"Cache expired for {key}")
        return None
    
    def set(self, key: str, data: pd.DataFrame):
        """Cache data with current timestamp"""
        self._cache[key] = (data.copy(), datetime.now())
        logger.debug(f"Cached data for {key}")
    
    def clear(self):
        """Clear all cached data"""
        self._cache.clear()


class UnifiedDataProvider:
    """
    Unified data provider for all market data needs.
    Handles both real-time and historical data requests with intelligent caching.
    """
    
    def __init__(self, mt5_client: MT5Client, cache_ttl: int = 300):
        self.mt5_client = mt5_client
        self.calculator = TimeframeCalculator()
        self.cache = DataCache(cache_ttl)
        self._available_data_cache: Dict[str, Tuple[datetime, datetime]] = {}
    
    async def get_data(self, request: DataRequest) -> MarketData:
        """
        Main entry point for all data requests.
        Automatically determines the best approach based on request parameters.
        """
        # Validate request
        self._validate_request(request)
        
        # Determine request type and fetch accordingly
        if request.start_date and request.end_date:
            # Historical request with specific date range
            df = await self._get_historical_data(
                request.symbol,
                request.start_date,
                request.end_date,
                request.timeframe
            )
        elif request.num_bars:
            # Recent bars request
            df = await self._get_recent_bars(
                request.symbol,
                request.timeframe,
                request.num_bars
            )
        else:
            raise ValueError("Request must specify either date range or number of bars")
        
        # Convert to MarketData
        return self._dataframe_to_market_data(df, request.symbol, request.timeframe)
    
    async def _get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: TimeFrame
    ) -> pd.DataFrame:
        """Get historical data for specific date range"""
        
        # Create cache key
        cache_key = f"{symbol}_{timeframe}_{start_date.isoformat()}_{end_date.isoformat()}"
        
        # Check cache first
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Calculate bars needed
        bars_needed = self.calculator.calculate_bars_needed(
            start_date, end_date, timeframe
        )
        
        logger.info(
            f"Fetching {bars_needed} bars for {symbol} {timeframe.value} "
            f"from {start_date.date()} to {end_date.date()}"
        )
        
        # Fetch data from MT5
        df = await self._fetch_from_mt5(symbol, timeframe, bars_needed)
        
        if df.empty:
            raise InsufficientDataError(f"No data available for {symbol}")
        
        # Check data availability
        actual_start = df.index.min()
        actual_end = df.index.max()
        
        logger.info(
            f"Received {len(df)} bars from {actual_start} to {actual_end}"
        )
        
        # Validate we have the requested range
        if actual_start > start_date:
            logger.warning(
                f"Requested start {start_date} not available. "
                f"Actual start: {actual_start}"
            )
            # Optionally raise an error or adjust dates
            if (actual_start - start_date).days > 7:  # More than a week difference
                raise InsufficientDataError(
                    f"Requested start date {start_date} not available. "
                    f"Earliest available: {actual_start}"
                )
        
        # Filter to requested range
        df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]
        
        if df_filtered.empty:
            raise InsufficientDataError(
                f"No data in requested range {start_date} to {end_date}"
            )
        
        # Cache the result
        self.cache.set(cache_key, df_filtered)
        
        # Update available data cache
        self._available_data_cache[symbol] = (actual_start, actual_end)
        
        return df_filtered
    
    async def _get_recent_bars(
        self,
        symbol: str,
        timeframe: TimeFrame,
        num_bars: int
    ) -> pd.DataFrame:
        """Get recent bars (for real-time trading)"""
        
        # For recent bars, always fetch fresh data
        logger.debug(f"Fetching {num_bars} recent bars for {symbol} {timeframe.value}")
        
        df = await self._fetch_from_mt5(symbol, timeframe, num_bars)
        
        if df.empty:
            raise InsufficientDataError(f"No data available for {symbol}")
        
        return df.tail(num_bars)
    
    async def _fetch_from_mt5(
        self,
        symbol: str,
        timeframe: TimeFrame,
        num_bars: int
    ) -> pd.DataFrame:
        """Fetch data from MT5 and convert to DataFrame"""
        
        # Import here to avoid circular import
        from core.domain.enums import TIMEFRAME_TO_MT5
        
        mt5_timeframe = TIMEFRAME_TO_MT5.get(timeframe)
        if not mt5_timeframe:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        # Fetch extra bars for indicator calculation (at least 250 for EMA200)
        # We need more bars than requested to calculate indicators properly
        bars_to_fetch = max(num_bars + 200, 250)
        
        # Fetch from MT5 with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                rates = self.mt5_client.copy_rates(
                    symbol=symbol,
                    timeframe=mt5_timeframe,
                    start_pos=0,
                    count=bars_to_fetch
                )
                
                if rates:
                    df = self._rates_to_dataframe(rates)
                    df = self._calculate_indicators(df)
                    # Return only the requested number of bars after indicator calculation
                    return df.tail(num_bars)
                
            except Exception as e:
                logger.error(f"MT5 fetch attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)  # Brief pause before retry
                else:
                    raise DataError(f"Failed to fetch data after {max_retries} attempts")
        
        return pd.DataFrame()
    
    def _rates_to_dataframe(self, rates: List[Dict]) -> pd.DataFrame:
        """Convert MT5 rates to pandas DataFrame"""
        df = pd.DataFrame(rates)
        
        # Convert time to datetime index
        df['timestamp'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('timestamp', inplace=True)
        
        # Rename columns for consistency
        column_mapping = {
            'tick_volume': 'volume',
            'real_volume': 'real_volume'
        }
        df.rename(columns=column_mapping, inplace=True)
        
        # Ensure proper data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'spread']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        return df
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # Keep existing indicator calculation logic
        # but ensure it handles edge cases properly
        
        if len(df) < 200:
            logger.warning(f"Only {len(df)} bars available for indicator calculation (200+ recommended for EMA200)")
        
        # Use try-except for each indicator to handle failures gracefully
        try:
            df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
        except Exception as e:
            logger.error(f"EMA calculation failed: {e}")
        
        try:
            # RSI calculation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi14'] = 100 - (100 / (1 + rs))
        except Exception as e:
            logger.error(f"RSI calculation failed: {e}")
        
        try:
            # ATR calculation
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr14'] = true_range.rolling(window=14).mean()
        except Exception as e:
            logger.error(f"ATR calculation failed: {e}")
        
        return df
    
    def _dataframe_to_market_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: TimeFrame
    ) -> MarketData:
        """Convert DataFrame to MarketData object"""
        candles = []
        
        for idx, row in df.iterrows():
            candle = Candle(
                timestamp=idx,
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=int(row.get('volume', 0)),
                spread=float(row.get('spread', 0)) if pd.notna(row.get('spread')) else None,
                ema50=float(row['ema50']) if pd.notna(row.get('ema50')) else None,
                ema200=float(row['ema200']) if pd.notna(row.get('ema200')) else None,
                rsi14=float(row['rsi14']) if pd.notna(row.get('rsi14')) else None,
                atr14=float(row['atr14']) if pd.notna(row.get('atr14')) else None
            )
            candles.append(candle)
        
        return MarketData(
            symbol=symbol,
            timeframe=timeframe.value,
            candles=candles,
            timestamp=datetime.now(timezone.utc)
        )
    
    def _validate_request(self, request: DataRequest):
        """Validate data request parameters"""
        if not request.symbol:
            raise ValueError("Symbol is required")
        
        if not request.timeframe:
            raise ValueError("Timeframe is required")
        
        if request.start_date and request.end_date:
            if request.start_date >= request.end_date:
                raise ValueError("Start date must be before end date")
            
            # Check if date range is reasonable
            max_days = 365 * 5  # 5 years max
            if (request.end_date - request.start_date).days > max_days:
                raise ValueError(f"Date range exceeds maximum of {max_days} days")
        
        if request.num_bars and request.num_bars <= 0:
            raise ValueError("Number of bars must be positive")
    
    async def get_available_date_range(self, symbol: str) -> Tuple[datetime, datetime]:
        """Get available date range for symbol (cached)"""
        if symbol in self._available_data_cache:
            return self._available_data_cache[symbol]
        
        # Fetch minimal data to determine range
        try:
            # Get oldest data
            df_old = await self._fetch_from_mt5(symbol, TimeFrame.D1, 10000)
            
            if df_old.empty:
                raise DataError(f"No data available for {symbol}")
            
            oldest = df_old.index.min()
            newest = df_old.index.max()
            
            self._available_data_cache[symbol] = (oldest, newest)
            return oldest, newest
            
        except Exception as e:
            logger.error(f"Failed to determine date range for {symbol}: {e}")
            raise
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        self._available_data_cache.clear()
        self.get_available_date_range.cache_clear()