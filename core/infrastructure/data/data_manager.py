# core/infrastructure/data/data_manager.py
"""
Unified data management system for live trading and backtesting.
Handles data sourcing, caching, and fallback strategies.
"""

import asyncio
from abc import ABC, abstractmethod
from asyncio.log import logger
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
from core.infrastructure.data.unified_data_provider import DataRequest, UnifiedDataProvider
import h5py

from core.domain.models import MarketData
from core.domain.enums import TimeFrame
from core.domain.exceptions import DataError


class DataSource(ABC):
    """Abstract base for data sources"""
    
    @abstractmethod
    async def get_data(
        self, 
        symbol: str, 
        timeframe: TimeFrame,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Get data from this source"""
        pass
    
    @abstractmethod
    async def get_available_range(self, symbol: str) -> Tuple[datetime, datetime]:
        """Get available date range for symbol"""
        pass
    
    @abstractmethod
    def get_priority(self) -> int:
        """Priority for this source (higher = preferred)"""
        pass


class MT5DataSource(DataSource):
    """MT5 live/demo data source"""
    
    def __init__(self, mt5_client):
        self.mt5_client = mt5_client
        self.unified_provider = UnifiedDataProvider(mt5_client)
    
    async def get_data(
        self, 
        symbol: str, 
        timeframe: TimeFrame,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        request = DataRequest(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        market_data = await self.unified_provider.get_data(request)
        return self._market_data_to_dataframe(market_data)
    
    async def get_available_range(self, symbol: str) -> Tuple[datetime, datetime]:
        return self.unified_provider.get_available_date_range(symbol)
    
    def get_priority(self) -> int:
        return 100  # Highest priority for live data


class HistoricalDataSource(DataSource):
    """Historical data from files (HDF5/Parquet)"""
    
    def __init__(self, data_dir: str = "data/historical"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._metadata = self._load_metadata()
    
    async def get_data(
        self, 
        symbol: str, 
        timeframe: TimeFrame,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        file_path = self.data_dir / f"{symbol}_{timeframe.value}.h5"
        
        if not file_path.exists():
            raise DataError(f"No historical data for {symbol} {timeframe}")
        
        # Read from HDF5
        with h5py.File(file_path, 'r') as f:
            df = pd.read_hdf(file_path, 'data')
        
        # Filter date range
        return df[(df.index >= start_date) & (df.index <= end_date)]
    
    async def get_available_range(self, symbol: str) -> Tuple[datetime, datetime]:
        if symbol in self._metadata:
            return self._metadata[symbol]
        raise DataError(f"No metadata for {symbol}")
    
    def get_priority(self) -> int:
        return 50  # Lower priority than live data


class DataManager:
    """
    Central data management system that intelligently routes requests
    to appropriate sources and handles caching/fallbacks.
    """
    
    def __init__(self):
        self.sources: List[DataSource] = []
        self._cache = {}
        self._availability_cache = {}
    
    def register_source(self, source: DataSource):
        """Register a data source"""
        self.sources.append(source)
        self.sources.sort(key=lambda s: s.get_priority(), reverse=True)
    
    async def get_data(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start_date: datetime,
        end_date: datetime,
        allow_partial: bool = True
    ) -> pd.DataFrame:
        """
        Get data from the best available source.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start_date: Start of period
            end_date: End of period
            allow_partial: Whether to accept partial data
            
        Returns:
            DataFrame with OHLCV data and indicators
        """
        # Try each source in priority order
        last_error = None
        best_data = None
        best_coverage = 0
        
        for source in self.sources:
            try:
                # Check if source has data for this symbol
                available_start, available_end = await source.get_available_range(symbol)
                
                # Calculate coverage
                request_duration = (end_date - start_date).total_seconds()
                actual_start = max(start_date, available_start)
                actual_end = min(end_date, available_end)
                
                if actual_start >= actual_end:
                    continue  # No overlap
                
                actual_duration = (actual_end - actual_start).total_seconds()
                coverage = actual_duration / request_duration
                
                # If we have full coverage, use this source
                if coverage >= 0.99:  # Allow 1% tolerance
                    return await source.get_data(symbol, timeframe, start_date, end_date)
                
                # Otherwise, keep track of best partial coverage
                if allow_partial and coverage > best_coverage:
                    data = await source.get_data(symbol, timeframe, actual_start, actual_end)
                    best_data = data
                    best_coverage = coverage
                    
            except Exception as e:
                last_error = e
                continue
        
        # Return best partial data if available
        if best_data is not None and allow_partial:
            if best_coverage < 0.8:  # Warn if coverage is low
                logger.warning(
                    f"Only {best_coverage:.1%} coverage for {symbol} "
                    f"from {start_date} to {end_date}"
                )
            return best_data
        
        # No suitable data found
        if last_error:
            raise DataError(f"No data available: {last_error}")
        else:
            raise DataError(f"No data sources available for {symbol}")
    
    async def get_unified_availability(self) -> Dict[str, Tuple[datetime, datetime]]:
        """Get combined availability across all sources"""
        availability = {}
        
        for source in self.sources:
            try:
                # This would need to be implemented per source
                source_availability = await source.get_all_symbols_availability()
                
                for symbol, (start, end) in source_availability.items():
                    if symbol in availability:
                        # Extend range if this source has more data
                        current_start, current_end = availability[symbol]
                        availability[symbol] = (
                            min(start, current_start),
                            max(end, current_end)
                        )
                    else:
                        availability[symbol] = (start, end)
                        
            except Exception as e:
                logger.error(f"Failed to get availability from {source.__class__.__name__}: {e}")
                
        return availability