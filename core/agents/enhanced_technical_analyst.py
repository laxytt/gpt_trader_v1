"""
Enhanced Technical Analyst with VSA Integration
Combines traditional TA with Volume Spread Analysis insights
"""

import logging
from typing import List, Dict, Any, Optional

from core.agents.technical_analyst import TechnicalAnalyst
from core.domain.models import MarketData

logger = logging.getLogger(__name__)


class EnhancedTechnicalAnalyst(TechnicalAnalyst):
    """
    Technical Analyst enhanced with VSA principles
    Adds volume analysis to traditional technical analysis
    """
    
    def _create_analysis_prompt(
        self,
        h1_data: MarketData,
        h4_data: MarketData,
        news_context: List[str]
    ) -> str:
        """Create enhanced prompt with VSA elements"""
        
        h1_latest = h1_data.latest_candle
        h4_latest = h4_data.latest_candle
        
        # Get volume context
        volume_context = self._analyze_volume_patterns(h1_data)
        
        # Base technical analysis prompt
        base_prompt = f"""Analyze {h1_data.symbol} using technical AND volume spread analysis.

Current Price: {h1_latest.close:.5f}

Technical Indicators (H1):
- EMA 50: {h1_latest.ema50:.5f}
- EMA 200: {h1_latest.ema200:.5f}
- RSI: {h1_latest.rsi14:.2f}
- ATR: {h1_latest.atr14:.5f}

H4 Background:
- Trend: {"Bullish" if h4_latest.close > h4_latest.ema50 > h4_latest.ema200 else "Bearish" if h4_latest.close < h4_latest.ema50 < h4_latest.ema200 else "Neutral"}

Volume Analysis:
{volume_context}

Apply BOTH traditional TA and VSA principles:
1. Identify chart patterns and support/resistance
2. Analyze volume in relation to price movement
3. Look for VSA signals (stopping volume, no supply/demand, climax)
4. Consider smart money accumulation/distribution

News Context: {'; '.join(news_context[:3]) if news_context else 'No significant news'}

Format your response as:
PATTERN: [chart pattern + any VSA pattern]
KEY_LEVELS: [support: X.XXXX, resistance: X.XXXX]
VOLUME_SIGNAL: [VSA signal if any, e.g., "stopping volume", "no demand", "climax"]
INDICATORS: [technical indicators summary]
TREND: [overall trend assessment combining TA + volume]
RECOMMENDATION: [BUY/SELL/WAIT]
CONFIDENCE: [0-100]
ENTRY: [price]
STOP_LOSS: [price]
TAKE_PROFIT: [price]
CONCERN: [main concern from TA or VSA perspective]
"""
        
        return base_prompt
    
    def _analyze_volume_patterns(self, market_data: MarketData) -> str:
        """Analyze volume patterns for VSA insights"""
        
        if len(market_data.candles) < 5:
            return "Insufficient data for volume analysis"
        
        recent_candles = market_data.candles[-5:]
        volume_insights = []
        
        # Calculate average volume
        volumes = [c.tick_volume for c in recent_candles if hasattr(c, 'tick_volume')]
        if not volumes:
            return "Volume data not available"
        
        avg_volume = sum(volumes) / len(volumes)
        latest = recent_candles[-1]
        prev = recent_candles[-2]
        
        # Analyze latest candle
        spread = latest.high - latest.low
        close_position = (latest.close - latest.low) / spread if spread > 0 else 0.5
        
        # VSA Pattern Detection
        if hasattr(latest, 'tick_volume'):
            volume_ratio = latest.tick_volume / avg_volume if avg_volume > 0 else 1
            
            # High volume analysis
            if volume_ratio > 1.5:
                if close_position < 0.3 and latest.close < latest.open:
                    volume_insights.append("Possible SELLING CLIMAX (high volume, close near low)")
                elif close_position > 0.7 and latest.close > latest.open:
                    volume_insights.append("Strong BUYING (high volume, close near high)")
                elif 0.3 <= close_position <= 0.7:
                    volume_insights.append("SUPPLY entering (high volume, close mid-range)")
            
            # Low volume analysis
            elif volume_ratio < 0.5:
                if latest.close < latest.open:
                    volume_insights.append("NO SUPPLY test (low volume down bar)")
                else:
                    volume_insights.append("NO DEMAND (low volume up bar)")
            
            # Stopping volume
            if (latest.close < latest.open and hasattr(prev, 'tick_volume') and 
                latest.tick_volume > prev.tick_volume * 1.5 and close_position > 0.5):
                volume_insights.append("STOPPING VOLUME detected")
        
        # Volume trend
        if len(volumes) >= 3:
            if volumes[-1] > volumes[-2] > volumes[-3]:
                volume_insights.append("Volume INCREASING (trend likely to continue)")
            elif volumes[-1] < volumes[-2] < volumes[-3]:
                volume_insights.append("Volume DECREASING (potential exhaustion)")
        
        return " | ".join(volume_insights) if volume_insights else "Normal volume activity"