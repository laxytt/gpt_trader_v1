"""
MT5 data provider for market data retrieval and technical analysis.
Handles candle data fetching, indicator calculations, and chart generation.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import pandas as pd

from core.infrastructure.data.unified_data_provider import DataRequest, UnifiedDataProvider
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("TA-Lib not available. Using pandas for basic indicators.")
import numpy as np

from core.infrastructure.mt5.client import MT5Client
from core.domain.models import MarketData, Candle
from core.domain.exceptions import (
    MT5DataError, InsufficientDataError, ErrorContext
)
from core.domain.enums import MT5TimeFrame, TIMEFRAME_TO_MT5, TimeFrame
from core.utils.chart_utils import ChartGenerator


logger = logging.getLogger(__name__)


class MT5DataProvider:
    """
    MT5 data provider that delegates to UnifiedDataProvider.
    Maintains backward compatibility while using unified logic.    """
    
    def __init__(self, mt5_client: MT5Client, chart_generator: Optional[ChartGenerator] = None):
        self.mt5_client = mt5_client
        self.chart_generator = chart_generator
        self.unified_provider = UnifiedDataProvider(mt5_client)
        
        # Technical analysis parameters
        self.indicator_warmup_bars = 250  # Bars needed for indicator stability
        self.min_required_bars = 20      # Minimum bars for analysis
    
    async def get_market_data(
        self, 
        symbol: str, 
        timeframe: TimeFrame, 
        bars: int = 100,
        include_indicators: bool = True
    ) -> MarketData:
        """
        Get market data with optional technical indicators.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for data
            bars: Number of bars to retrieve
            include_indicators: Whether to calculate technical indicators
            
        Returns:
            MarketData object with candles and indicators
            
        Raises:
            MT5DataError: If data retrieval fails
            InsufficientDataError: If not enough data available
        """

        # Create request for recent bars
        request = DataRequest(
            symbol=symbol,
            timeframe=timeframe,
            num_bars=bars
        )
        
         # Use unified provider
        market_data = await self.unified_provider.get_data(request)
        
        # Indicators are already included from unified provider
        return market_data
    
    def _rates_to_dataframe(self, rates_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert MT5 rates data to pandas DataFrame"""
        df = pd.DataFrame(rates_data)
        
        # Convert time to datetime
        df['timestamp'] = pd.to_datetime(df['time'], unit='s', utc=True)
        
        # Rename tick_volume to volume for consistency
        if 'tick_volume' in df.columns:
            df = df.rename(columns={'tick_volume': 'volume'})
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise MT5DataError(f"Missing required column: {col}")
        
        # Add spread calculation if not present
        if 'spread' not in df.columns:
            df['spread'] = np.nan  # Will be filled by real spread data if available
        
        return df
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the DataFrame"""
        try:
            if TALIB_AVAILABLE:
                # Use TA-Lib for better performance and accuracy
                df['ema50'] = talib.EMA(df['close'].values, timeperiod=50)
                df['ema200'] = talib.EMA(df['close'].values, timeperiod=200)
                df['rsi14'] = talib.RSI(df['close'].values, timeperiod=14)
                df['atr14'] = talib.ATR(
                    df['high'].values, 
                    df['low'].values, 
                    df['close'].values, 
                    timeperiod=14
                )
                df['true_range'] = talib.TRANGE(
                    df['high'].values,
                    df['low'].values, 
                    df['close'].values
                )
            else:
                # Use pandas for basic indicators
                df['ema50'] = df['close'].ewm(span=50).mean()
                df['ema200'] = df['close'].ewm(span=200).mean()
                
                # Simple RSI calculation
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi14'] = 100 - (100 / (1 + rs))
                
                # Simple ATR calculation
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                df['atr14'] = true_range.rolling(window=14).mean()
                df['true_range'] = true_range
            
            # Common calculations
            df['rsi_slope'] = df['rsi14'].diff()
            
            # Additional indicators for VSA analysis
            df['sma_volume'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['sma_volume']
            
            # Price range analysis
            df['body_size'] = abs(df['close'] - df['open'])
            df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
            df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
            
            return df
            
        except Exception as e:
            logger.error(f"Indicator calculation failed: {e}")
            raise MT5DataError(f"Technical indicator calculation failed: {str(e)}")
    
    def _dataframe_to_candles(self, df: pd.DataFrame) -> List[Candle]:
        """Convert DataFrame to list of Candle objects"""
        candles = []
        
        for _, row in df.iterrows():
            candle = Candle(
                timestamp=row['timestamp'],
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=int(row['volume']),
                spread=float(row.get('spread', 0)) if pd.notna(row.get('spread')) else None,
                
                # Technical indicators (handle NaN values)
                ema50=float(row['ema50']) if pd.notna(row.get('ema50')) else None,
                ema200=float(row['ema200']) if pd.notna(row.get('ema200')) else None,
                rsi14=float(row['rsi14']) if pd.notna(row.get('rsi14')) else None,
                atr14=float(row['atr14']) if pd.notna(row.get('atr14')) else None,
                rsi_slope=float(row['rsi_slope']) if pd.notna(row.get('rsi_slope')) else None
            )
            candles.append(candle)
        
        return candles
    
    async def get_multi_timeframe_data(
        self, 
        symbol: str, 
        timeframes: List[TimeFrame],
        bars: int = 100
    ) -> Dict[str, MarketData]:
        """
        Get market data for multiple timeframes.
        
        Args:
            symbol: Trading symbol
            timeframes: List of timeframes
            bars: Number of bars per timeframe
            
        Returns:
            Dict mapping timeframe to MarketData
        """
        result = {}
        
        for tf in timeframes:
            try:
                market_data = await self.get_market_data(symbol, tf, bars)
                result[tf.value] = market_data
            except Exception as e:
                logger.error(f"Failed to get data for {symbol} {tf}: {e}")
                # Continue with other timeframes
        
        return result 
    
    def get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get current bid/ask prices for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with bid/ask prices or None if failed
        """
        tick = self.mt5_client.get_symbol_tick(symbol)
        if not tick:
            return None
        
        return {
            'bid': tick['bid'],
            'ask': tick['ask'],
            'last': tick.get('last', tick['bid']),
            'time': datetime.fromtimestamp(tick['time'], tz=timezone.utc)
        }
    
    def calculate_volatility_metrics(self, market_data: MarketData) -> Dict[str, float]:
        """
        Calculate volatility metrics from market data.
        
        Args:
            market_data: Market data to analyze
            
        Returns:
            Dict with volatility metrics
        """
        if not market_data.candles:
            return {}
        
        # Get ATR values
        atr_values = [c.atr14 for c in market_data.candles if c.atr14 is not None]
        
        if not atr_values:
            return {}
        
        current_atr = atr_values[-1] if atr_values else 0
        avg_atr = sum(atr_values) / len(atr_values)
        
        # Calculate volume metrics
        volumes = [c.volume for c in market_data.candles]
        avg_volume = sum(volumes) / len(volumes) if volumes else 0
        current_volume = volumes[-1] if volumes else 0
        
        return {
            'current_atr': current_atr,
            'average_atr': avg_atr,
            'atr_ratio': current_atr / avg_atr if avg_atr > 0 else 1,
            'current_volume': current_volume,
            'average_volume': avg_volume,
            'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1,
            'volatility_level': self._classify_volatility(current_atr, avg_atr)
        }
    
    def _classify_volatility(self, current_atr: float, avg_atr: float) -> str:
        """Classify volatility level based on ATR"""
        if avg_atr == 0:
            return "unknown"
        
        ratio = current_atr / avg_atr
        
        if ratio > 1.3:
            return "high"
        elif ratio < 0.7:
            return "low"
        else:
            return "medium"
    
    async def validate_symbol_data(self, symbol: str, timeframe: TimeFrame) -> bool:
        """
        Validate that symbol has sufficient data for analysis.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe to check
            
        Returns:
            bool: True if symbol has sufficient data
        """
        try:
            market_data = await self.get_market_data(symbol, timeframe, bars=self.min_required_bars)
            return len(market_data.candles) >= self.min_required_bars
        except Exception:
            return False


# Export main class
__all__ = ['MT5DataProvider']