"""
Enhanced Base Agent with Professional Trading Metrics
Provides comprehensive market data analysis for algorithmic trading
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from core.domain.models import MarketData, Candle

@dataclass
class EnhancedMarketMetrics:
    """Professional-grade market metrics for algorithmic trading"""
    
    # Price Action Metrics
    vwap: float
    vwap_deviation: float
    average_true_range: float
    true_range_percentile: float
    
    # Volume Profile
    volume_profile: Dict[float, float]  # price -> volume
    point_of_control: float  # Price with highest volume
    value_area_high: float
    value_area_low: float
    volume_weighted_spread: float
    
    # Market Microstructure
    bid_ask_imbalance: float
    order_flow_imbalance: float
    absorption_ratio: float  # Large orders being absorbed
    
    # Volatility Analysis
    realized_volatility: float
    garch_forecast: float
    volatility_regime: str  # "low", "normal", "high", "extreme"
    volatility_percentile: float
    
    # Advanced Momentum
    rate_of_change: Dict[int, float]  # period -> ROC
    force_index: float
    accumulation_distribution: float
    money_flow_index: float
    
    # Market Breadth
    advance_decline_ratio: float
    mcclellan_oscillator: float
    new_highs_lows: Tuple[int, int]
    
    # Sentiment Indicators
    put_call_ratio: float
    vix_correlation: float
    term_structure: float
    
    # Statistical Measures
    z_score: float
    percentile_rank: float
    half_life: float  # Mean reversion half-life
    hurst_exponent: float  # Trending vs mean-reverting
    
    # Risk Metrics
    value_at_risk: float
    expected_shortfall: float
    max_drawdown: float
    calmar_ratio: float
    information_ratio: float


class ProfessionalMarketAnalyzer:
    """Professional market analysis tools for trading agents"""
    
    @staticmethod
    def calculate_vwap(candles: List[Candle]) -> Tuple[float, float]:
        """Calculate VWAP and current price deviation"""
        if not candles:
            return 0, 0
            
        total_volume = sum(c.volume for c in candles)
        if total_volume == 0:
            return candles[-1].close, 0
            
        vwap = sum((c.high + c.low + c.close) / 3 * c.volume for c in candles) / total_volume
        deviation = (candles[-1].close - vwap) / vwap * 100
        
        return vwap, deviation
    
    @staticmethod
    def calculate_volume_profile(candles: List[Candle], bins: int = 20) -> Dict[str, any]:
        """Calculate volume profile with POC and value area"""
        if not candles:
            return {}
            
        prices = []
        volumes = []
        
        for candle in candles:
            # Use typical price
            typical = (candle.high + candle.low + candle.close) / 3
            prices.append(typical)
            volumes.append(candle.volume)
        
        # Create price bins
        price_min = min(c.low for c in candles)
        price_max = max(c.high for c in candles)
        bin_size = (price_max - price_min) / bins
        
        # Build volume profile
        profile = {}
        for i in range(bins):
            level = price_min + (i + 0.5) * bin_size
            profile[level] = 0
        
        # Assign volumes to bins
        for price, volume in zip(prices, volumes):
            bin_idx = min(int((price - price_min) / bin_size), bins - 1)
            level = price_min + (bin_idx + 0.5) * bin_size
            profile[level] += volume
        
        # Find POC (Point of Control)
        poc = max(profile.keys(), key=lambda x: profile[x])
        
        # Calculate value area (70% of volume)
        sorted_levels = sorted(profile.items(), key=lambda x: x[1], reverse=True)
        total_volume = sum(profile.values())
        target_volume = total_volume * 0.7
        
        accumulated = 0
        value_levels = []
        for level, vol in sorted_levels:
            accumulated += vol
            value_levels.append(level)
            if accumulated >= target_volume:
                break
        
        return {
            'profile': profile,
            'poc': poc,
            'value_area_high': max(value_levels),
            'value_area_low': min(value_levels)
        }
    
    @staticmethod
    def calculate_order_flow_imbalance(candles: List[Candle]) -> float:
        """Calculate order flow imbalance from candle data"""
        if len(candles) < 2:
            return 0
            
        buying_pressure = 0
        selling_pressure = 0
        
        for i in range(1, len(candles)):
            # Estimate buying/selling from candle structure
            candle = candles[i]
            range_size = candle.high - candle.low
            
            if range_size > 0:
                # Buying pressure: close near high
                buying_ratio = (candle.close - candle.low) / range_size
                # Selling pressure: close near low
                selling_ratio = (candle.high - candle.close) / range_size
                
                buying_pressure += buying_ratio * candle.volume
                selling_pressure += selling_ratio * candle.volume
        
        total_pressure = buying_pressure + selling_pressure
        if total_pressure == 0:
            return 0
            
        return (buying_pressure - selling_pressure) / total_pressure
    
    @staticmethod
    def calculate_realized_volatility(candles: List[Candle], period: int = 20) -> float:
        """Calculate realized volatility (annualized)"""
        if len(candles) < period + 1:
            return 0
            
        returns = []
        for i in range(len(candles) - period, len(candles)):
            if i > 0:
                ret = np.log(candles[i].close / candles[i-1].close)
                returns.append(ret)
        
        if not returns:
            return 0
            
        # Annualize (assuming 252 trading days)
        return np.std(returns) * np.sqrt(252 * 24)  # 24 hours for forex
    
    @staticmethod
    def calculate_hurst_exponent(candles: List[Candle]) -> float:
        """Calculate Hurst exponent to determine trending vs mean-reverting"""
        if len(candles) < 100:
            return 0.5  # Random walk
            
        prices = [c.close for c in candles[-100:]]
        
        # R/S analysis
        lags = range(2, 20)
        tau = []
        
        for lag in lags:
            series = np.array(prices[:lag])
            mean = series.mean()
            
            # Calculate cumulative deviations
            deviations = series - mean
            Z = np.cumsum(deviations)
            
            # Range
            R = Z.max() - Z.min()
            
            # Standard deviation
            S = series.std()
            
            if S != 0:
                tau.append(R / S)
        
        # Fit log(R/S) = log(c) + H*log(lag)
        if tau:
            log_lags = np.log(list(lags))
            log_rs = np.log(tau)
            
            # Linear regression
            A = np.vstack([log_lags, np.ones(len(log_lags))]).T
            hurst, c = np.linalg.lstsq(A, log_rs, rcond=None)[0]
            
            return max(0, min(1, hurst))
        
        return 0.5
    
    @staticmethod
    def calculate_market_regime(candles: List[Candle]) -> Dict[str, any]:
        """Identify current market regime"""
        if len(candles) < 50:
            return {'regime': 'unknown', 'confidence': 0}
            
        # Calculate various metrics
        hurst = ProfessionalMarketAnalyzer.calculate_hurst_exponent(candles)
        volatility = ProfessionalMarketAnalyzer.calculate_realized_volatility(candles)
        
        # Recent price action
        recent_trend = (candles[-1].close - candles[-20].close) / candles[-20].close
        
        # Determine regime
        if hurst > 0.6 and abs(recent_trend) > 0.02:
            regime = 'trending'
        elif hurst < 0.4:
            regime = 'mean_reverting'
        elif volatility > 0.3:  # 30% annualized
            regime = 'volatile'
        else:
            regime = 'ranging'
            
        return {
            'regime': regime,
            'hurst': hurst,
            'volatility': volatility,
            'trend_strength': abs(recent_trend),
            'confidence': abs(hurst - 0.5) * 2  # 0 to 1
        }


def format_enhanced_prompt(base_prompt: str, market_data: Dict[str, MarketData], 
                          include_microstructure: bool = True) -> str:
    """Format prompt with comprehensive market data"""
    
    h1_data = market_data.get('h1')
    h4_data = market_data.get('h4')
    
    if not h1_data:
        return base_prompt
    
    # Calculate all metrics
    analyzer = ProfessionalMarketAnalyzer()
    
    # VWAP
    vwap, vwap_dev = analyzer.calculate_vwap(h1_data.candles[-50:])
    
    # Volume Profile
    vol_profile = analyzer.calculate_volume_profile(h1_data.candles[-100:])
    
    # Order Flow
    order_flow = analyzer.calculate_order_flow_imbalance(h1_data.candles[-20:])
    
    # Volatility
    volatility = analyzer.calculate_realized_volatility(h1_data.candles)
    
    # Market Regime
    regime = analyzer.calculate_market_regime(h1_data.candles)
    
    # Add to prompt
    enhanced_prompt = base_prompt + f"""

COMPREHENSIVE MARKET DATA:

Price Structure (Last 100 bars):
- VWAP: {vwap:.5f} (Price deviation: {vwap_dev:.2f}%)
- Point of Control: {vol_profile.get('poc', 0):.5f}
- Value Area: {vol_profile.get('value_area_low', 0):.5f} - {vol_profile.get('value_area_high', 0):.5f}

Market Microstructure:
- Order Flow Imbalance: {order_flow:.2%}
- Realized Volatility: {volatility:.1%} annualized
- Market Regime: {regime['regime']} (confidence: {regime['confidence']:.1%})
- Hurst Exponent: {regime['hurst']:.3f}

Recent Candles (Last 10):
"""
    
    # Add recent candles with volume
    for i, candle in enumerate(h1_data.candles[-10:]):
        direction = "🟢" if candle.close > candle.open else "🔴"
        body = abs(candle.close - candle.open)
        upper_wick = candle.high - max(candle.open, candle.close)
        lower_wick = min(candle.open, candle.close) - candle.low
        
        enhanced_prompt += f"\n{direction} O:{candle.open:.5f} H:{candle.high:.5f} L:{candle.low:.5f} C:{candle.close:.5f} V:{candle.volume:.0f} Body:{body:.5f} Upper:{upper_wick:.5f} Lower:{lower_wick:.5f}"
    
    return enhanced_prompt