"""
Chart generation utilities for market data visualization.
Handles candlestick charts with technical indicators and VSA analysis.
"""

import logging
import base64
from typing import Optional, Dict, List, Any
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import mplfinance as mpf
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Plotting libraries not available. Chart generation disabled.")

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

from core.domain.exceptions import ChartGenerationError, ErrorContext
from core.domain.models import MarketData, Candle


logger = logging.getLogger(__name__)


class ChartGenerator:
    """
    Generates financial charts with technical indicators and VSA analysis.
    """
    
    def __init__(self):
        if not PLOTTING_AVAILABLE:
            raise ChartGenerationError("Required plotting libraries not installed")
        
        # Chart styling configuration
        self.default_style = {
            'type': 'candle',
            'style': 'yahoo',
            'volume': True,
            'figscale': 1.3,
            'figratio': (12, 8),
            'panel_ratios': (6, 2, 2)  # Main, Volume, RSI
        }
        
        # Color scheme
        self.colors = {
            'ema50': '#1f77b4',      # Blue
            'ema200': '#ff7f0e',     # Orange  
            'rsi': '#9467bd',        # Purple
            'volume_up': '#2ca02c',   # Green
            'volume_down': '#d62728', # Red
            'background': '#f8f9fa',  # Light gray
            'grid': '#e0e0e0'        # Gray
        }
    
    def generate_chart_with_indicators(
        self,
        df: pd.DataFrame,
        output_path: str,
        title: str = "Market Analysis",
        include_volume: bool = True,
        include_rsi: bool = True,
        width: int = 1200,
        height: int = 800
    ) -> str:
        """
        Generate candlestick chart with technical indicators.
        
        Args:
            df: DataFrame with OHLCV data and indicators
            output_path: Path to save the chart
            title: Chart title
            include_volume: Whether to include volume panel
            include_rsi: Whether to include RSI panel
            width: Chart width in pixels
            height: Chart height in pixels
            
        Returns:
            Path to generated chart file
            
        Raises:
            ChartGenerationError: If chart generation fails
        """
        with ErrorContext("Chart generation") as ctx:
            ctx.add_detail("output_path", output_path)
            ctx.add_detail("data_points", len(df))
            
            # Validate data
            self._validate_dataframe(df)
            
            # Prepare data for mplfinance
            chart_data = self._prepare_chart_data(df)
            
            # Create additional plots
            additional_plots = self._create_additional_plots(chart_data, include_rsi)
            
            # Configure chart style
            style_config = self._get_style_config(include_volume, include_rsi)
            
            # Generate chart
            try:
                # Set figure size
                fig_size = (width/100, height/100)  # Convert pixels to inches (approx)
                
                mpf.plot(
                    chart_data,
                    addplot=additional_plots,
                    title=title,
                    savefig=dict(
                        fname=output_path,
                        dpi=100,
                        bbox_inches='tight',
                        facecolor=self.colors['background']
                    ),
                    figsize=fig_size,
                    **style_config
                )
                
                logger.info(f"Chart generated successfully: {output_path}")
                return output_path
                
            except Exception as e:
                logger.error(f"Chart generation failed: {e}")
                raise ChartGenerationError(f"Failed to generate chart: {str(e)}")
    
    def generate_vsa_chart(
        self,
        df: pd.DataFrame,
        output_path: str,
        title: str = "VSA Analysis",
        highlight_patterns: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Generate chart with VSA (Volume Spread Analysis) annotations.
        
        Args:
            df: DataFrame with OHLCV data
            output_path: Path to save the chart
            title: Chart title
            highlight_patterns: List of VSA patterns to highlight
            
        Returns:
            Path to generated chart file
        """
        with ErrorContext("VSA chart generation"):
            # Add VSA-specific indicators
            vsa_df = self._add_vsa_indicators(df.copy())
            
            # Create base chart
            chart_path = self.generate_chart_with_indicators(
                vsa_df, output_path, title
            )
            
            # Add VSA annotations if patterns provided
            if highlight_patterns:
                self._add_vsa_annotations(chart_path, highlight_patterns)
            
            return chart_path
    
    def _validate_dataframe(self, df: pd.DataFrame):
        """Validate DataFrame has required columns and proper index"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ChartGenerationError(f"Missing required columns: {missing_columns}")
        
        if df.empty:
            raise ChartGenerationError("DataFrame is empty")
        
        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            else:
                raise ChartGenerationError("DataFrame must have DatetimeIndex or 'timestamp' column")
    
    def _prepare_chart_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for mplfinance"""
        # Create a copy with only OHLCV data for mplfinance
        chart_data = df[['open', 'high', 'low', 'close', 'volume']].copy()
        
        # Ensure proper data types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            chart_data[col] = pd.to_numeric(chart_data[col], errors='coerce')
        
        # Remove any NaN rows
        chart_data = chart_data.dropna()
        
        return chart_data
    
    def _create_additional_plots(
        self, 
        df: pd.DataFrame, 
        include_rsi: bool = True
    ) -> List:
        """Create additional plots for indicators"""
        additional_plots = []
        
        # Add EMAs to main panel
        if 'ema50' in df.columns:
            ema50_data = df['ema50'].dropna()
            if not ema50_data.empty:
                additional_plots.append(
                    mpf.make_addplot(
                        ema50_data,
                        color=self.colors['ema50'],
                        width=1,
                        panel=0
                    )
                )
        
        if 'ema200' in df.columns:
            ema200_data = df['ema200'].dropna()
            if not ema200_data.empty:
                additional_plots.append(
                    mpf.make_addplot(
                        ema200_data,
                        color=self.colors['ema200'],
                        width=1,
                        panel=0
                    )
                )
        
        # Add RSI to separate panel
        if include_rsi and 'rsi14' in df.columns:
            rsi_data = df['rsi14'].dropna()
            if not rsi_data.empty:
                additional_plots.append(
                    mpf.make_addplot(
                        rsi_data,
                        color=self.colors['rsi'],
                        panel=2,
                        ylabel='RSI',
                        ylim=(0, 100),
                        secondary_y=False
                    )
                )
                
                # Add RSI reference lines
                rsi_70 = pd.Series([70] * len(rsi_data), index=rsi_data.index)
                rsi_30 = pd.Series([30] * len(rsi_data), index=rsi_data.index)
                
                additional_plots.extend([
                    mpf.make_addplot(rsi_70, color='red', linestyle='--', width=0.5, panel=2),
                    mpf.make_addplot(rsi_30, color='green', linestyle='--', width=0.5, panel=2)
                ])
        
        return additional_plots
    
    def _get_style_config(self, include_volume: bool, include_rsi: bool) -> Dict[str, Any]:
        """Get style configuration for the chart"""
        config = self.default_style.copy()
        
        # Adjust panel ratios based on what's included
        if include_volume and include_rsi:
            config['panel_ratios'] = (6, 2, 2)  # Main, Volume, RSI
        elif include_volume:
            config['panel_ratios'] = (4, 1)     # Main, Volume
        elif include_rsi:
            config['panel_ratios'] = (4, 1)     # Main, RSI
        else:
            config['panel_ratios'] = (1,)       # Main only
        
        config['volume'] = include_volume
        
        return config
    
    def _add_vsa_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add VSA-specific indicators to DataFrame"""
        if not TA_AVAILABLE:
            logger.warning("TA library not available, skipping VSA indicators")
            return df
        
        try:
            # Volume moving average for comparison
            if 'sma_volume' not in df.columns:
                df['sma_volume'] = ta.trend.SMAIndicator(df['volume'], window=20).sma_indicator()
            
            # Volume ratio (current vs average)
            df['volume_ratio'] = df['volume'] / df['sma_volume']
            
            # Price spread (high - low)
            df['spread'] = df['high'] - df['low']
            df['spread_sma'] = ta.trend.SMAIndicator(df['spread'], window=10).sma_indicator()
            
            # Body size and position
            df['body_size'] = abs(df['close'] - df['open'])
            df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
            df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
            
            # Close position in range (0 = at low, 1 = at high)
            df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding VSA indicators: {e}")
            return df
    
    def _add_vsa_annotations(self, chart_path: str, patterns: List[Dict[str, Any]]):
        """Add VSA pattern annotations to existing chart"""
        # This would require reopening and modifying the saved chart
        # For now, we'll log the patterns that would be annotated
        logger.info(f"VSA patterns to annotate: {len(patterns)}")
        for pattern in patterns:
            logger.debug(f"Pattern: {pattern}")
    
    def create_comparison_chart(
        self,
        data_dict: Dict[str, pd.DataFrame],
        output_path: str,
        title: str = "Symbol Comparison"
    ) -> str:
        """
        Create comparison chart for multiple symbols.
        
        Args:
            data_dict: Dictionary mapping symbol names to DataFrames
            output_path: Path to save the chart
            title: Chart title
            
        Returns:
            Path to generated chart file
        """
        if not PLOTTING_AVAILABLE:
            raise ChartGenerationError("Plotting libraries not available")
        
        try:
            fig, axes = plt.subplots(len(data_dict), 1, figsize=(12, 4 * len(data_dict)))
            if len(data_dict) == 1:
                axes = [axes]
            
            for idx, (symbol, df) in enumerate(data_dict.items()):
                ax = axes[idx]
                
                # Plot closing prices
                ax.plot(df.index, df['close'], label=f'{symbol} Close', linewidth=1.5)
                
                # Add EMAs if available
                if 'ema50' in df.columns:
                    ax.plot(df.index, df['ema50'], label='EMA50', alpha=0.7)
                if 'ema200' in df.columns:
                    ax.plot(df.index, df['ema200'], label='EMA200', alpha=0.7)
                
                ax.set_title(f'{symbol} Price Chart')
                ax.set_ylabel('Price')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Comparison chart generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Comparison chart generation failed: {e}")
            raise ChartGenerationError(f"Failed to generate comparison chart: {str(e)}")


def encode_image_as_b64(image_path: str) -> Optional[Dict[str, Any]]:
    """
    Encode image as base64 for GPT vision analysis.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with base64 encoded image or None if failed
    """
    try:
        path = Path(image_path)
        if not path.exists():
            logger.warning(f"Image file not found: {image_path}")
            return None
        
        if path.stat().st_size == 0:
            logger.warning(f"Image file is empty: {image_path}")
            return None
        
        with open(path, "rb") as image_file:
            image_data = image_file.read()
            b64_string = base64.b64encode(image_data).decode('utf-8')
            
            # Determine image type from extension
            mime_type = "image/png"
            if path.suffix.lower() in ['.jpg', '.jpeg']:
                mime_type = "image/jpeg"
            elif path.suffix.lower() == '.gif':
                mime_type = "image/gif"
            
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{b64_string}"
                }
            }
            
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return None


def create_market_data_chart(
    market_data: MarketData,
    output_path: str,
    title: Optional[str] = None
) -> Optional[str]:
    """
    Create chart from MarketData object.
    
    Args:
        market_data: MarketData object to chart
        output_path: Path to save chart
        title: Optional chart title
        
    Returns:
        Path to generated chart or None if failed
    """
    if not market_data.candles:
        logger.warning("No candles in market data")
        return None
    
    try:
        # Convert candles to DataFrame
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
        
        # Generate chart
        chart_title = title or f"{market_data.symbol} {market_data.timeframe}"
        generator = ChartGenerator()
        
        return generator.generate_chart_with_indicators(
            df=df,
            output_path=output_path,
            title=chart_title
        )
        
    except Exception as e:
        logger.error(f"Failed to create chart from MarketData: {e}")
        return None


# Export main classes and functions
__all__ = [
    'ChartGenerator',
    'encode_image_as_b64', 
    'create_market_data_chart',
    'PLOTTING_AVAILABLE'
]