"""
Professional Trading Configuration
Defines optimal data requirements for each trading agent
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class AgentDataRequirements:
    """Data requirements for each agent type"""
    
    # Technical Analyst needs extensive history for pattern recognition
    TECHNICAL_ANALYST = {
        'h1_candles': 200,  # For proper pattern recognition
        'h4_candles': 100,  # For higher timeframe context
        'd1_candles': 50,   # For major levels
        'indicators': [
            'EMA(20, 50, 200)',
            'RSI(14)',
            'MACD(12, 26, 9)',
            'ATR(14)',
            'Bollinger Bands(20, 2)',
            'Stochastic(14, 3, 3)',
            'ADX(14)',
            'CCI(20)',
            'Williams %R(14)',
            'Parabolic SAR',
            'Ichimoku Cloud'
        ],
        'patterns': [
            'Head and Shoulders',
            'Double Top/Bottom',
            'Triangles',
            'Flags/Pennants',
            'Wedges',
            'Channels',
            'Harmonic Patterns'
        ]
    }
    
    # Momentum Trader needs sufficient data for momentum calculation
    MOMENTUM_TRADER = {
        'h1_candles': 100,  # For momentum calculations
        'h4_candles': 50,   # For trend confirmation
        'm15_candles': 100, # For entry timing
        'indicators': [
            'ROC(10, 20, 50)',
            'Momentum Oscillator',
            'ADX(14)',
            'Force Index',
            'TSI(25, 13)',
            'Know Sure Thing',
            'Coppock Curve'
        ],
        'volume_analysis': True,
        'momentum_periods': [5, 10, 20, 50, 100]
    }
    
    # Sentiment Reader needs market breadth data
    SENTIMENT_READER = {
        'h1_candles': 100,
        'h4_candles': 50,
        'd1_candles': 20,
        'sentiment_data': [
            'Put/Call Ratio',
            'VIX Level',
            'Commitment of Traders',
            'Retail Sentiment',
            'News Sentiment Score',
            'Social Media Sentiment'
        ],
        'volume_profile': True,
        'market_breadth': True
    }
    
    # Risk Manager needs comprehensive risk metrics
    RISK_MANAGER = {
        'h1_candles': 250,  # For proper volatility calculation
        'h4_candles': 100,
        'd1_candles': 60,
        'risk_metrics': [
            'ATR(14, 20)',
            'Historical Volatility',
            'Implied Volatility',
            'Value at Risk (95%, 99%)',
            'Expected Shortfall',
            'Maximum Drawdown',
            'Correlation Matrix'
        ],
        'position_data': True,
        'account_metrics': True
    }
    
    # Contrarian Trader needs extremes and divergences
    CONTRARIAN_TRADER = {
        'h1_candles': 150,  # For finding extremes
        'h4_candles': 75,
        'd1_candles': 30,
        'divergence_periods': [10, 20, 50],
        'extremes': [
            'Bollinger Band Width',
            'RSI Extremes',
            'Stochastic Extremes',
            'Volume Extremes',
            'Volatility Extremes'
        ],
        'mean_reversion_stats': True
    }
    
    # Fundamental Analyst needs economic data
    FUNDAMENTAL_ANALYST = {
        'h1_candles': 50,   # Less focus on technicals
        'h4_candles': 25,
        'd1_candles': 20,
        'w1_candles': 10,
        'economic_data': [
            'Interest Rate Differentials',
            'GDP Growth Rates',
            'Inflation Data',
            'Employment Data',
            'Trade Balance',
            'Central Bank Policies',
            'Political Events'
        ],
        'news_history_days': 7
    }


@dataclass 
class ProfessionalTradingMetrics:
    """Professional metrics that all agents should consider"""
    
    MARKET_MICROSTRUCTURE = [
        'Bid-Ask Spread',
        'Market Depth',
        'Order Book Imbalance',
        'Trade Size Distribution',
        'Quote Frequency',
        'Effective Spread'
    ]
    
    VOLATILITY_METRICS = [
        'Realized Volatility',
        'GARCH Forecast',
        'Volatility Smile',
        'Term Structure',
        'Jump Detection',
        'Volatility Regime'
    ]
    
    EXECUTION_METRICS = [
        'Expected Slippage',
        'Market Impact',
        'Optimal Trade Size',
        'VWAP Deviation',
        'Implementation Shortfall'
    ]
    
    STATISTICAL_MEASURES = [
        'Autocorrelation',
        'Hurst Exponent',
        'Lyapunov Exponent',
        'Entropy',
        'Fractal Dimension',
        'Detrended Fluctuation Analysis'
    ]


class DataQualityValidator:
    """Ensures data quality for professional trading"""
    
    @staticmethod
    def validate_candle_data(candles: List, required_count: int) -> Tuple[bool, str]:
        """Validate candle data quality"""
        if not candles:
            return False, "No candle data provided"
            
        if len(candles) < required_count:
            return False, f"Insufficient candles: {len(candles)} < {required_count}"
            
        # Check for data gaps
        gaps = []
        for i in range(1, len(candles)):
            time_diff = (candles[i].timestamp - candles[i-1].timestamp).total_seconds()
            expected_diff = 3600  # 1 hour for H1
            
            if time_diff > expected_diff * 1.5:
                gaps.append(i)
        
        if gaps:
            return False, f"Data gaps detected at indices: {gaps}"
            
        # Check for anomalies
        for i, candle in enumerate(candles):
            if candle.high < candle.low:
                return False, f"Invalid candle at index {i}: high < low"
                
            if candle.close <= 0 or candle.open <= 0:
                return False, f"Invalid price at index {i}"
                
            if candle.volume < 0:
                return False, f"Negative volume at index {i}"
        
        return True, "Data quality OK"
    
    @staticmethod
    def calculate_data_quality_score(candles: List) -> float:
        """Calculate overall data quality score (0-100)"""
        if not candles:
            return 0
            
        score = 100
        
        # Penalize for missing data
        expected_candles = 100
        if len(candles) < expected_candles:
            score -= (expected_candles - len(candles)) * 0.5
            
        # Check for zero volume candles
        zero_volume = sum(1 for c in candles if c.volume == 0)
        score -= zero_volume * 0.2
        
        # Check for price spikes
        for i in range(1, len(candles)):
            price_change = abs((candles[i].close - candles[i-1].close) / candles[i-1].close)
            if price_change > 0.1:  # 10% spike
                score -= 5
                
        return max(0, score)


# Response format standardization
STANDARDIZED_RESPONSE_FORMAT = """
SIGNAL: [BUY/SELL/WAIT]
CONFIDENCE: [0-100]
ENTRY: [price or N/A]
STOP_LOSS: [price or N/A]
TAKE_PROFIT: [price or N/A]
RISK_REWARD: [ratio or N/A]
POSITION_SIZE: [lots or N/A]
ANALYSIS: [key points in bullet format]
RISKS: [main risks in bullet format]
"""