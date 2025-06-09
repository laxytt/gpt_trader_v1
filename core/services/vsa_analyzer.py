"""
Volume Spread Analysis (VSA) analyzer for improved signal generation.
Based on professional VSA trading principles.
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from core.domain.models import Candle, MarketData, TradingSignal, SignalType, RiskClass


logger = logging.getLogger(__name__)


class VSASignalType(Enum):
    """VSA signal types based on video tutorials"""
    # Strength signals
    STOPPING_VOLUME = "stopping_volume"
    SHAKEOUT = "shakeout"
    NO_SUPPLY = "no_supply"
    TWO_BAR_REVERSAL_UP = "two_bar_reversal_up"
    SELLING_CLIMAX = "selling_climax"
    
    # Weakness signals
    UPTHRUST = "upthrust"
    NO_DEMAND = "no_demand"
    TWO_BAR_REVERSAL_DOWN = "two_bar_reversal_down"
    BUYING_CLIMAX = "buying_climax"
    SUPPLY_COMING_IN = "supply_coming_in"
    TRAP_UP_MOVE = "trap_up_move"
    
    # Neutral
    TEST = "test"
    NONE = "none"


@dataclass
class VSASignal:
    """VSA signal with context"""
    signal_type: VSASignalType
    strength: float  # 0-1
    description: str
    volume_context: str
    entry_suggestion: Optional[str] = None


class VSAAnalyzer:
    """Analyzes market data using Volume Spread Analysis principles"""
    
    def __init__(self, lookback_periods: int = 20):
        self.lookback_periods = lookback_periods
    
    def analyze(self, market_data: MarketData) -> VSASignal:
        """Perform comprehensive VSA analysis"""
        candles = market_data.candles
        
        if len(candles) < self.lookback_periods:
            return VSASignal(
                signal_type=VSASignalType.NONE,
                strength=0.0,
                description="Insufficient data for VSA analysis",
                volume_context="Unknown"
            )
        
        # Get recent candles for analysis
        recent_candles = candles[-self.lookback_periods:]
        current = candles[-1]
        prev = candles[-2]
        prev2 = candles[-3] if len(candles) > 2 else None
        
        # Calculate volume metrics
        avg_volume = self._calculate_average_volume(recent_candles[:-1])
        volume_ratio = current.volume / avg_volume if avg_volume > 0 else 1
        
        # Check for strength signals first (higher priority)
        strength_signal = self._check_strength_signals(current, prev, prev2, recent_candles, avg_volume)
        if strength_signal.signal_type != VSASignalType.NONE:
            return strength_signal
        
        # Check for weakness signals
        weakness_signal = self._check_weakness_signals(current, prev, prev2, recent_candles, avg_volume)
        if weakness_signal.signal_type != VSASignalType.NONE:
            return weakness_signal
        
        # Check for test signals
        test_signal = self._check_test_signals(current, prev, recent_candles, avg_volume)
        if test_signal.signal_type != VSASignalType.NONE:
            return test_signal
        
        return VSASignal(
            signal_type=VSASignalType.NONE,
            strength=0.0,
            description="No clear VSA signal",
            volume_context=self._get_volume_context(volume_ratio)
        )
    
    def _check_strength_signals(
        self, 
        current: Candle, 
        prev: Candle,
        prev2: Optional[Candle],
        candles: List[Candle],
        avg_volume: float
    ) -> VSASignal:
        """Check for bullish VSA signals"""
        
        # 1. Stopping Volume (Video 9)
        if (self._is_down_bar(prev) and 
            prev.volume > avg_volume * 1.5 and
            self._is_up_bar(current) and
            current.close > prev.close and
            current.close > (current.high + current.low) / 2):
            
            return VSASignal(
                signal_type=VSASignalType.STOPPING_VOLUME,
                strength=0.85,
                description="Stopping Volume - Strong buying after decline",
                volume_context="High volume on down bar followed by reversal",
                entry_suggestion="Consider long entry on confirmation"
            )
        
        # 2. Shakeout (Video 10)
        if (current.low < min(c.low for c in candles[-5:-1]) and  # New low
            current.close > (current.high + current.low) * 0.66 and  # Close in upper third
            current.volume > avg_volume * 1.3 and
            self._calculate_spread(current) > self._calculate_average_spread(candles) * 1.5):
            
            return VSASignal(
                signal_type=VSASignalType.SHAKEOUT,
                strength=0.80,
                description="Shakeout - False breakdown with strong recovery",
                volume_context="High volume on wide spread recovery",
                entry_suggestion="Long entry after shakeout confirmation"
            )
        
        # 3. No Supply (Video 12)
        if (self._is_down_bar(current) and
            current.volume < avg_volume * 0.5 and
            current.close > current.low + (current.high - current.low) * 0.5 and
            prev2 and self._was_in_downtrend(candles[-10:-1])):
            
            return VSASignal(
                signal_type=VSASignalType.NO_SUPPLY,
                strength=0.70,
                description="No Supply - Lack of selling pressure",
                volume_context="Very low volume on down move",
                entry_suggestion="Wait for upbar confirmation"
            )
        
        # 4. Two Bar Reversal Up (Video 11)
        if (self._is_down_bar(prev) and
            self._is_up_bar(current) and
            current.volume >= prev.volume and
            current.close > prev.open and
            current.close > (current.high + current.low) * 0.6):
            
            return VSASignal(
                signal_type=VSASignalType.TWO_BAR_REVERSAL_UP,
                strength=0.75,
                description="Two Bar Reversal Up - Bullish reversal pattern",
                volume_context="Equal or higher volume on reversal",
                entry_suggestion="Long entry on pattern completion"
            )
        
        # 5. Selling Climax (Video 8)
        if (self._is_down_bar(current) and
            self._calculate_spread(current) > self._calculate_average_spread(candles) * 2 and
            current.volume > avg_volume * 2 and
            current.close > current.low + (current.high - current.low) * 0.4):
            
            return VSASignal(
                signal_type=VSASignalType.SELLING_CLIMAX,
                strength=0.85,
                description="Selling Climax - Potential end of downtrend",
                volume_context="Extreme volume on wide spread",
                entry_suggestion="Wait for test of lows"
            )
        
        return VSASignal(signal_type=VSASignalType.NONE, strength=0.0, description="", volume_context="")
    
    def _check_weakness_signals(
        self,
        current: Candle,
        prev: Candle,
        prev2: Optional[Candle],
        candles: List[Candle],
        avg_volume: float
    ) -> VSASignal:
        """Check for bearish VSA signals"""
        
        # 1. Upthrust (Video 20)
        if (current.high > max(c.high for c in candles[-10:-1]) and
            current.close < (current.high + current.low) * 0.33 and
            self._has_upper_shadow(current) and
            current.volume > avg_volume):
            
            return VSASignal(
                signal_type=VSASignalType.UPTHRUST,
                strength=0.85,
                description="Upthrust - Failed breakout with selling pressure",
                volume_context="Volume on failed breakout",
                entry_suggestion="Consider short on confirmation"
            )
        
        # 2. No Demand (Video 19)
        if (self._is_up_bar(current) and
            current.volume < avg_volume * 0.5 and
            current.close < current.high - (current.high - current.low) * 0.3 and
            self._was_in_uptrend(candles[-10:-1])):
            
            return VSASignal(
                signal_type=VSASignalType.NO_DEMAND,
                strength=0.70,
                description="No Demand - Lack of buying interest",
                volume_context="Low volume on up move",
                entry_suggestion="Wait for downbar confirmation"
            )
        
        # 3. Supply Coming In (Video 15)
        if (current.volume > avg_volume * 1.2 and
            current.volume < max(c.volume for c in candles[-10:-1]) and
            self._has_long_upper_shadow(current) and
            current.close < (current.high + current.low) * 0.5):
            
            return VSASignal(
                signal_type=VSASignalType.SUPPLY_COMING_IN,
                strength=0.65,
                description="Supply Coming In - Selling entering market",
                volume_context="Increased volume with upper shadow",
                entry_suggestion="Watch for further weakness"
            )
        
        # 4. Two Bar Reversal Down (Video 18)
        if (self._is_up_bar(prev) and
            self._is_down_bar(current) and
            current.volume >= prev.volume and
            current.close < prev.open):
            
            return VSASignal(
                signal_type=VSASignalType.TWO_BAR_REVERSAL_DOWN,
                strength=0.75,
                description="Two Bar Reversal Down - Bearish reversal",
                volume_context="Equal or higher volume on reversal",
                entry_suggestion="Short entry on pattern completion"
            )
        
        # 5. Trap Up Move (Video 16)
        if (prev and self._is_up_bar(prev) and
            self._has_long_upper_shadow(current) and
            current.close < current.open and
            current.close < (current.high + current.low) * 0.33 and
            current.volume > avg_volume * 1.3):
            
            return VSASignal(
                signal_type=VSASignalType.TRAP_UP_MOVE,
                strength=0.80,
                description="Trap Up Move - Bull trap formation",
                volume_context="High volume on rejection",
                entry_suggestion="Short after trap confirmation"
            )
        
        return VSASignal(signal_type=VSASignalType.NONE, strength=0.0, description="", volume_context="")
    
    def _check_test_signals(
        self,
        current: Candle,
        prev: Candle,
        candles: List[Candle],
        avg_volume: float
    ) -> VSASignal:
        """Check for test signals (Video 13)"""
        
        # Test requires previous accumulation/distribution
        if not self._has_previous_significant_action(candles[:-5]):
            return VSASignal(signal_type=VSASignalType.NONE, strength=0.0, description="", volume_context="")
        
        # Test characteristics
        if (self._calculate_spread(current) < self._calculate_average_spread(candles) and
            current.volume < avg_volume * 0.6 and
            self._has_lower_shadow(current)):
            
            return VSASignal(
                signal_type=VSASignalType.TEST,
                strength=0.60,
                description="Test - Testing previous levels",
                volume_context="Low volume on test",
                entry_suggestion="Wait for successful test confirmation"
            )
        
        return VSASignal(signal_type=VSASignalType.NONE, strength=0.0, description="", volume_context="")
    
    # Helper methods
    def _is_up_bar(self, candle: Candle) -> bool:
        """Check if candle is an up bar"""
        return candle.close > candle.open
    
    def _is_down_bar(self, candle: Candle) -> bool:
        """Check if candle is a down bar"""
        return candle.close < candle.open
    
    def _calculate_spread(self, candle: Candle) -> float:
        """Calculate candle spread"""
        return candle.high - candle.low
    
    def _calculate_average_spread(self, candles: List[Candle]) -> float:
        """Calculate average spread"""
        if not candles:
            return 0.0
        return sum(self._calculate_spread(c) for c in candles) / len(candles)
    
    def _calculate_average_volume(self, candles: List[Candle]) -> float:
        """Calculate average volume"""
        if not candles:
            return 0.0
        return sum(c.volume for c in candles) / len(candles)
    
    def _has_upper_shadow(self, candle: Candle) -> bool:
        """Check if candle has significant upper shadow"""
        body_size = abs(candle.close - candle.open)
        upper_shadow = candle.high - max(candle.close, candle.open)
        return upper_shadow > body_size * 0.5
    
    def _has_long_upper_shadow(self, candle: Candle) -> bool:
        """Check if candle has long upper shadow"""
        body_size = abs(candle.close - candle.open)
        upper_shadow = candle.high - max(candle.close, candle.open)
        return upper_shadow > body_size
    
    def _has_lower_shadow(self, candle: Candle) -> bool:
        """Check if candle has significant lower shadow"""
        body_size = abs(candle.close - candle.open)
        lower_shadow = min(candle.close, candle.open) - candle.low
        return lower_shadow > body_size * 0.5
    
    def _was_in_uptrend(self, candles: List[Candle]) -> bool:
        """Check if market was in uptrend"""
        if len(candles) < 3:
            return False
        closes = [c.close for c in candles]
        return closes[-1] > closes[0] and sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1]) > len(closes) * 0.6
    
    def _was_in_downtrend(self, candles: List[Candle]) -> bool:
        """Check if market was in downtrend"""
        if len(candles) < 3:
            return False
        closes = [c.close for c in candles]
        return closes[-1] < closes[0] and sum(1 for i in range(1, len(closes)) if closes[i] < closes[i-1]) > len(closes) * 0.6
    
    def _has_previous_significant_action(self, candles: List[Candle]) -> bool:
        """Check if there was significant volume action previously"""
        if len(candles) < 5:
            return False
        avg_volume = self._calculate_average_volume(candles)
        return any(c.volume > avg_volume * 1.5 for c in candles[-5:])
    
    def _get_volume_context(self, volume_ratio: float) -> str:
        """Get volume context description"""
        if volume_ratio > 2:
            return "Very high volume"
        elif volume_ratio > 1.5:
            return "High volume"
        elif volume_ratio > 1.0:
            return "Above average volume"
        elif volume_ratio > 0.7:
            return "Average volume"
        elif volume_ratio > 0.5:
            return "Below average volume"
        else:
            return "Very low volume"


def create_signal_from_vsa(
    market_data: MarketData,
    vsa_signal: VSASignal,
    base_score: float = 0.5
) -> TradingSignal:
    """Create trading signal from VSA analysis"""
    
    # Combine VSA strength with base score
    combined_score = (vsa_signal.strength + base_score) / 2
    
    # Map VSA signals to trading signals
    if vsa_signal.signal_type in [
        VSASignalType.STOPPING_VOLUME,
        VSASignalType.SHAKEOUT,
        VSASignalType.NO_SUPPLY,
        VSASignalType.TWO_BAR_REVERSAL_UP,
        VSASignalType.SELLING_CLIMAX
    ]:
        if combined_score > 0.7:
            return _create_buy_signal(market_data, vsa_signal, combined_score)
    
    elif vsa_signal.signal_type in [
        VSASignalType.UPTHRUST,
        VSASignalType.NO_DEMAND,
        VSASignalType.TWO_BAR_REVERSAL_DOWN,
        VSASignalType.BUYING_CLIMAX,
        VSASignalType.SUPPLY_COMING_IN,
        VSASignalType.TRAP_UP_MOVE
    ]:
        if combined_score > 0.7:
            return _create_sell_signal(market_data, vsa_signal, combined_score)
    
    # Default to wait
    return TradingSignal(
        symbol=market_data.symbol,
        signal=SignalType.WAIT,
        reason=vsa_signal.description or "No clear VSA signal",
        risk_class=RiskClass.C
    )


def _create_buy_signal(
    market_data: MarketData,
    vsa_signal: VSASignal,
    score: float
) -> TradingSignal:
    """Create buy signal with proper risk management"""
    latest = market_data.latest_candle
    candles = market_data.candles[-20:]
    
    # Calculate ATR for dynamic stops
    atr = latest.atr14 or _calculate_simple_atr(candles)
    
    # Limit ATR to reasonable values (max 50 pips)
    atr = min(atr, 0.0050)
    
    # Find recent swing low for stop loss
    recent_low = min(c.low for c in candles[-10:])
    
    # Entry at current price
    entry = latest.close
    
    # Stop loss placement based on VSA signal type
    if vsa_signal.signal_type == VSASignalType.SHAKEOUT:
        stop_loss = latest.low - (0.5 * atr)  # Tight stop after shakeout
    elif vsa_signal.signal_type == VSASignalType.STOPPING_VOLUME:
        stop_loss = min(recent_low, latest.low) - atr
    else:
        stop_loss = recent_low - atr
    
    # Ensure minimum and maximum stop distance
    stop_distance = entry - stop_loss
    min_stop = max(atr * 0.5, 0.0010)  # At least 10 pips
    max_stop = 0.0050  # Max 50 pips
    
    if stop_distance < min_stop:
        stop_loss = entry - min_stop
    elif stop_distance > max_stop:
        stop_loss = entry - max_stop  # Cap stop distance
    
    # Take profit with good risk/reward
    risk = entry - stop_loss
    take_profit = entry + (risk * 3)  # 3:1 RR
    
    # Adjust risk class based on signal strength
    if score > 0.85:
        risk_class = RiskClass.A
    elif score > 0.75:
        risk_class = RiskClass.B
    else:
        risk_class = RiskClass.C
    
    return TradingSignal(
        symbol=market_data.symbol,
        signal=SignalType.BUY,
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_reward=3.0,
        risk_class=risk_class,
        reason=f"{vsa_signal.description} | {vsa_signal.volume_context}"
    )


def _create_sell_signal(
    market_data: MarketData,
    vsa_signal: VSASignal,
    score: float
) -> TradingSignal:
    """Create sell signal with proper risk management"""
    latest = market_data.latest_candle
    candles = market_data.candles[-20:]
    
    # Calculate ATR for dynamic stops
    atr = latest.atr14 or _calculate_simple_atr(candles)
    
    # Limit ATR to reasonable values (max 50 pips)
    atr = min(atr, 0.0050)
    
    # Find recent swing high for stop loss
    recent_high = max(c.high for c in candles[-10:])
    
    # Entry at current price
    entry = latest.close
    
    # Stop loss placement based on VSA signal type
    if vsa_signal.signal_type == VSASignalType.UPTHRUST:
        stop_loss = latest.high + (0.5 * atr)  # Tight stop after upthrust
    elif vsa_signal.signal_type == VSASignalType.TRAP_UP_MOVE:
        stop_loss = max(recent_high, latest.high) + atr
    else:
        stop_loss = recent_high + atr
    
    # Ensure minimum and maximum stop distance
    stop_distance = stop_loss - entry
    min_stop = max(atr * 0.5, 0.0010)  # At least 10 pips
    max_stop = 0.0050  # Max 50 pips
    
    if stop_distance < min_stop:
        stop_loss = entry + min_stop
    elif stop_distance > max_stop:
        stop_loss = entry + max_stop  # Cap stop distance
    
    # Take profit with good risk/reward
    risk = stop_loss - entry
    take_profit = entry - (risk * 3)  # 3:1 RR
    
    # Adjust risk class based on signal strength
    if score > 0.85:
        risk_class = RiskClass.A
    elif score > 0.75:
        risk_class = RiskClass.B
    else:
        risk_class = RiskClass.C
    
    return TradingSignal(
        symbol=market_data.symbol,
        signal=SignalType.SELL,
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_reward=3.0,
        risk_class=risk_class,
        reason=f"{vsa_signal.description} | {vsa_signal.volume_context}"
    )


def _calculate_simple_atr(candles: List[Candle], period: int = 14) -> float:
    """Calculate simple ATR when not available"""
    if len(candles) < period:
        # Fallback to simple range average
        ranges = [c.high - c.low for c in candles]
        return sum(ranges) / len(ranges) if ranges else 0.0001
    
    # Calculate true ranges
    true_ranges = []
    for i in range(1, len(candles)):
        high_low = candles[i].high - candles[i].low
        high_close = abs(candles[i].high - candles[i-1].close)
        low_close = abs(candles[i].low - candles[i-1].close)
        true_ranges.append(max(high_low, high_close, low_close))
    
    # Return average of last 'period' true ranges
    if len(true_ranges) >= period:
        return sum(true_ranges[-period:]) / period
    else:
        return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0001