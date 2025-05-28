# core/services/chart_service.py
"""Unified chart generation service"""

import logging
from typing import Optional, Dict, List, Any
from pathlib import Path
import pandas as pd

from core.domain.models import MarketData
from core.domain.exceptions import ChartGenerationError, ErrorContext
from core.utils.chart_utils import ChartGenerator, PLOTTING_AVAILABLE

logger = logging.getLogger(__name__)


class ChartService:
    """
    Unified service for all chart generation needs.
    Consolidates chart generation logic from multiple places.
    """
    
    def __init__(self, chart_generator: Optional[ChartGenerator] = None):
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting libraries not available. Chart generation disabled.")
            self.enabled = False
            self.generator = None
        else:
            self.enabled = True
            self.generator = chart_generator or ChartGenerator()
    
    def generate_market_chart(
        self,
        market_data: MarketData,
        output_path: str,
        title: Optional[str] = None,
        include_volume: bool = True,
        include_rsi: bool = True
    ) -> Optional[str]:
        """
        Generate chart from market data.
        
        Args:
            market_data: Market data to chart
            output_path: Path to save chart
            title: Optional chart title
            include_volume: Include volume panel
            include_rsi: Include RSI panel
            
        Returns:
            Path to generated chart or None if failed
        """
        if not self.enabled or not market_data.candles:
            return None
        
        with ErrorContext("Chart generation", symbol=market_data.symbol):
            try:
                # Convert to DataFrame
                df = self._market_data_to_dataframe(market_data)
                
                # Generate title
                if not title:
                    title = f"{market_data.symbol} {market_data.timeframe}"
                
                # Generate chart
                return self.generator.generate_chart_with_indicators(
                    df=df,
                    output_path=output_path,
                    title=title,
                    include_volume=include_volume,
                    include_rsi=include_rsi
                )
                
            except Exception as e:
                logger.error(f"Chart generation failed: {e}")
                return None
    
    def generate_multi_timeframe_chart(
        self,
        market_data_dict: Dict[str, MarketData],
        output_path: str,
        title: str = "Multi-Timeframe Analysis"
    ) -> Optional[str]:
        """
        Generate comparison chart for multiple timeframes.
        
        Args:
            market_data_dict: Dict of timeframe -> MarketData
            output_path: Path to save chart
            title: Chart title
            
        Returns:
            Path to generated chart or None if failed
        """
        if not self.enabled:
            return None
        
        try:
            # Convert each MarketData to DataFrame
            df_dict = {}
            for timeframe, data in market_data_dict.items():
                df_dict[f"{data.symbol} {timeframe}"] = self._market_data_to_dataframe(data)
            
            # Generate comparison chart
            return self.generator.create_comparison_chart(
                data_dict=df_dict,
                output_path=output_path,
                title=title
            )
            
        except Exception as e:
            logger.error(f"Multi-timeframe chart generation failed: {e}")
            return None
    
    def generate_vsa_analysis_chart(
        self,
        market_data: MarketData,
        output_path: str,
        vsa_patterns: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[str]:
        """
        Generate chart with VSA analysis annotations.
        
        Args:
            market_data: Market data to analyze
            output_path: Path to save chart
            vsa_patterns: Optional VSA patterns to highlight
            
        Returns:
            Path to generated chart or None if failed
        """
        if not self.enabled:
            return None
        
        try:
            df = self._market_data_to_dataframe(market_data)
            
            return self.generator.generate_vsa_chart(
                df=df,
                output_path=output_path,
                title=f"{market_data.symbol} VSA Analysis",
                highlight_patterns=vsa_patterns
            )
            
        except Exception as e:
            logger.error(f"VSA chart generation failed: {e}")
            return None
    
    def _market_data_to_dataframe(self, market_data: MarketData) -> pd.DataFrame:
        """Convert MarketData to DataFrame for charting"""
        data = []
        for candle in market_data.candles:
            data.append({
                'timestamp': candle.timestamp,
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume,
                'ema50': candle.ema50,
                'ema200': candle.ema200,
                'rsi14': candle.rsi14,
                'atr14': candle.atr14
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df