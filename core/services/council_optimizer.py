"""
Council optimization service for dynamic debate depth and early stopping.
Reduces API costs by adapting council complexity to market clarity.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np

from core.domain.models import SignalType, MarketData
from core.agents.council import CouncilDecision, AgentAnalysis
from config.settings import TradingSettings

logger = logging.getLogger(__name__)


class CouncilOptimizer:
    """
    Optimizes council decision process based on market conditions and agent agreement.
    """
    
    def __init__(self, trading_config: TradingSettings):
        self.trading_config = trading_config
        
        # Optimization thresholds
        self.high_agreement_threshold = 0.85  # 6/7 agents strongly agree
        self.obvious_trade_confidence = 90.0  # Very high confidence
        self.low_confidence_threshold = 40.0  # Too uncertain
        self.risk_veto_immediate = True      # Risk manager can veto immediately
        
        # Obvious trade patterns
        self.obvious_patterns = {
            'strong_breakout': {
                'trend_alignment': True,
                'volume_surge': 2.0,  # 2x average
                'momentum_strong': True,
                'no_news_conflict': True
            },
            'clear_reversal': {
                'extreme_rsi': [20, 80],  # Oversold/overbought
                'divergence': True,
                'support_resistance': True,
                'volume_confirmation': True
            }
        }
        
        # Statistics
        self.stats = {
            'full_debates': 0,
            'early_stops': 0,
            'immediate_decisions': 0,
            'average_rounds': 1.0
        }
    
    def needs_full_debate(
        self, 
        initial_signals: Dict[str, Dict[str, Any]],
        market_data: MarketData
    ) -> Tuple[bool, str]:
        """
        Determine if full 3-round debate is needed based on initial agent responses.
        
        Args:
            initial_signals: Initial recommendations from each agent
            market_data: Current market data
            
        Returns:
            Tuple of (needs_full_debate, reason)
        """
        # Check for risk manager veto
        if self.risk_veto_immediate:
            risk_signal = initial_signals.get('risk_manager', {})
            if risk_signal.get('signal') == SignalType.WAIT and risk_signal.get('confidence', 0) > 80:
                logger.info("Risk Manager immediate veto - skipping debate")
                return False, "risk_veto"
        
        # Calculate agreement score
        agreement_score, signal_distribution = self._calculate_agreement(initial_signals)
        
        # High agreement - skip additional rounds
        if agreement_score >= self.high_agreement_threshold:
            dominant_signal = max(signal_distribution.items(), key=lambda x: x[1])[0]
            logger.info(f"High agreement ({agreement_score:.1%}) on {dominant_signal} - skip debate")
            return False, "high_agreement"
        
        # Check for obvious trade patterns
        if self._is_obvious_trade(market_data, initial_signals):
            logger.info("Obvious trade pattern detected - minimal debate needed")
            return False, "obvious_pattern"
        
        # Low confidence across board
        avg_confidence = self._average_confidence(initial_signals)
        if avg_confidence < self.low_confidence_threshold:
            logger.info(f"Low confidence across agents ({avg_confidence:.1f}%) - skip debate")
            return False, "low_confidence"
        
        # Mixed signals or moderate confidence - need full debate
        return True, "mixed_signals"
    
    def _calculate_agreement(self, signals: Dict[str, Dict[str, Any]]) -> Tuple[float, Dict[str, int]]:
        """Calculate agreement score and signal distribution"""
        signal_counts = {}
        total_weighted = 0
        total_weight = 0
        
        for agent, data in signals.items():
            signal = data.get('signal', SignalType.WAIT)
            confidence = data.get('confidence', 50) / 100.0
            
            # Count signals
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
            
            # Weighted agreement
            if signal != SignalType.WAIT:
                total_weighted += confidence
                total_weight += 1
        
        # Calculate agreement as percentage of agents with same signal
        if signal_counts:
            max_count = max(signal_counts.values())
            agreement = max_count / len(signals)
        else:
            agreement = 0
        
        return agreement, signal_counts
    
    def _average_confidence(self, signals: Dict[str, Dict[str, Any]]) -> float:
        """Calculate average confidence across all agents"""
        confidences = [
            data.get('confidence', 50) 
            for data in signals.values()
        ]
        return np.mean(confidences) if confidences else 50.0
    
    def _is_obvious_trade(
        self, 
        market_data: MarketData,
        signals: Dict[str, Dict[str, Any]]
    ) -> bool:
        """Check if market shows obvious trade pattern"""
        if not market_data.h1_candles or len(market_data.h1_candles) < 20:
            return False
        
        recent_candles = market_data.h1_candles[-20:]
        last_candle = recent_candles[-1]
        
        # Strong breakout pattern
        if self._check_strong_breakout(recent_candles):
            # Verify agents agree
            tech_signal = signals.get('technical_analyst', {}).get('signal')
            momentum_signal = signals.get('momentum_trader', {}).get('signal')
            
            if tech_signal == momentum_signal and tech_signal != SignalType.WAIT:
                return True
        
        # Clear reversal pattern
        if self._check_clear_reversal(recent_candles):
            # Verify contrarian agrees
            contrarian_signal = signals.get('contrarian_trader', {}).get('signal')
            if contrarian_signal != SignalType.WAIT:
                return True
        
        return False
    
    def _check_strong_breakout(self, candles: List[Any]) -> bool:
        """Check for strong breakout pattern"""
        if len(candles) < 10:
            return False
        
        last = candles[-1]
        
        # Price breakout
        recent_high = max(c.high for c in candles[-10:-1])
        recent_low = min(c.low for c in candles[-10:-1])
        
        is_breakout_up = last.close > recent_high
        is_breakout_down = last.close < recent_low
        
        if not (is_breakout_up or is_breakout_down):
            return False
        
        # Volume confirmation
        avg_volume = np.mean([c.volume for c in candles[-10:-1]])
        volume_surge = last.volume > (avg_volume * 1.5)
        
        # Momentum confirmation
        if is_breakout_up:
            momentum_confirmed = last.rsi > 60 and last.close > last.ema_50
        else:
            momentum_confirmed = last.rsi < 40 and last.close < last.ema_50
        
        return volume_surge and momentum_confirmed
    
    def _check_clear_reversal(self, candles: List[Any]) -> bool:
        """Check for clear reversal pattern"""
        if len(candles) < 5:
            return False
        
        last = candles[-1]
        
        # Extreme RSI
        rsi_extreme = last.rsi < 25 or last.rsi > 75
        
        if not rsi_extreme:
            return False
        
        # Check for divergence
        price_trend = candles[-1].close - candles[-5].close
        rsi_trend = candles[-1].rsi - candles[-5].rsi
        
        # Bullish divergence: price down, RSI up
        bullish_div = price_trend < 0 and rsi_trend > 0 and last.rsi < 35
        
        # Bearish divergence: price up, RSI down
        bearish_div = price_trend > 0 and rsi_trend < 0 and last.rsi > 65
        
        return bullish_div or bearish_div
    
    def optimize_debate_rounds(
        self,
        round_results: List[Dict[str, Any]],
        target_confidence: float = 75.0
    ) -> Tuple[bool, str]:
        """
        Determine if more debate rounds are needed based on results so far.
        
        Args:
            round_results: Results from completed rounds
            target_confidence: Minimum confidence threshold
            
        Returns:
            Tuple of (continue_debate, reason)
        """
        if not round_results:
            return True, "no_rounds_yet"
        
        latest_round = round_results[-1]
        confidence = latest_round.get('confidence', 0)
        consensus = latest_round.get('consensus_level', 0)
        
        # High confidence reached - stop
        if confidence >= target_confidence and consensus >= 0.7:
            return False, "target_reached"
        
        # No improvement between rounds - stop
        if len(round_results) >= 2:
            prev_confidence = round_results[-2].get('confidence', 0)
            if abs(confidence - prev_confidence) < 5:  # Less than 5% change
                return False, "no_improvement"
        
        # Maximum rounds reached
        if len(round_results) >= self.trading_config.council_debate_rounds:
            return False, "max_rounds"
        
        # Continue debate
        return True, "improving"
    
    def create_batched_analysis_prompt(
        self,
        symbols: List[str],
        market_data_dict: Dict[str, MarketData]
    ) -> str:
        """
        Create a single prompt for analyzing multiple symbols.
        Reduces API calls by batching similar market conditions.
        """
        prompt_parts = ["Analyze the following symbols for trading opportunities:\n"]
        
        for symbol in symbols:
            market_data = market_data_dict.get(symbol)
            if not market_data:
                continue
            
            # Add symbol section
            prompt_parts.append(f"\n## {symbol}")
            
            # Add key metrics
            if market_data.h1_candles:
                last = market_data.h1_candles[-1]
                prompt_parts.append(
                    f"- Price: {last.close}, RSI: {last.rsi:.1f}, "
                    f"ATR: {last.atr:.5f}, Volume: {last.volume}"
                )
                
                # Trend
                if last.close > last.ema_50 > last.ema_200:
                    prompt_parts.append("- Trend: Bullish (price > EMA50 > EMA200)")
                elif last.close < last.ema_50 < last.ema_200:
                    prompt_parts.append("- Trend: Bearish (price < EMA50 < EMA200)")
                else:
                    prompt_parts.append("- Trend: Mixed")
        
        prompt_parts.append("\nProvide trading signals for each symbol.")
        
        return "\n".join(prompt_parts)
    
    def should_batch_symbols(self, symbols: List[str], market_conditions: Dict[str, Any]) -> List[List[str]]:
        """
        Group symbols that can be analyzed together based on similar conditions.
        
        Returns:
            List of symbol groups for batched analysis
        """
        # Group by market characteristics
        groups = {
            'trending_up': [],
            'trending_down': [],
            'ranging': [],
            'low_volatility': []
        }
        
        for symbol in symbols:
            condition = market_conditions.get(symbol, {})
            
            if condition.get('trend') == 'up':
                groups['trending_up'].append(symbol)
            elif condition.get('trend') == 'down':
                groups['trending_down'].append(symbol)
            elif condition.get('volatility', 1) < 0.5:
                groups['low_volatility'].append(symbol)
            else:
                groups['ranging'].append(symbol)
        
        # Create batches (max 3 symbols per batch for quality)
        batches = []
        for group_symbols in groups.values():
            if group_symbols:
                for i in range(0, len(group_symbols), 3):
                    batch = group_symbols[i:i+3]
                    if batch:
                        batches.append(batch)
        
        return batches
    
    def update_stats(self, decision_type: str, rounds_used: int = 1):
        """Update optimization statistics"""
        if decision_type == 'full_debate':
            self.stats['full_debates'] += 1
        elif decision_type == 'early_stop':
            self.stats['early_stops'] += 1
        elif decision_type == 'immediate':
            self.stats['immediate_decisions'] += 1
        
        # Update average rounds
        total_decisions = sum([
            self.stats['full_debates'],
            self.stats['early_stops'],
            self.stats['immediate_decisions']
        ])
        
        if total_decisions > 0:
            total_rounds = (
                self.stats['full_debates'] * 3 +
                self.stats['early_stops'] * rounds_used +
                self.stats['immediate_decisions'] * 1
            )
            self.stats['average_rounds'] = total_rounds / total_decisions
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        total = sum([
            self.stats['full_debates'],
            self.stats['early_stops'],
            self.stats['immediate_decisions']
        ])
        
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'optimization_rate': f"{(self.stats['early_stops'] + self.stats['immediate_decisions']) / total:.1%}",
            'rounds_saved': f"{(3.0 - self.stats['average_rounds']) / 3.0:.1%}"
        }