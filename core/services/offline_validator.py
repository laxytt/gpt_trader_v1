"""
Offline signal validation service for pre-filtering and enriching trading signals.
Provides deterministic validation before expensive GPT API calls.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from enum import Enum

import numpy as np
import pandas as pd

from core.domain.models import MarketData, Candle
from core.domain.exceptions import ValidationError
from config.symbols import get_symbol_spec


logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a single validation issue"""
    severity: ValidationSeverity
    category: str
    message: str
    impact_score: float  # 0-1, how much this affects trading
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Result of validation check"""
    passed: bool
    score: float  # 0-1, overall quality score
    issues: List[ValidationIssue]
    metadata: Dict[str, Any]
    recommendation: str


class BaseValidator(ABC):
    """Base class for all validators"""
    
    @abstractmethod
    async def validate(self, market_data: MarketData) -> ValidationResult:
        """Perform validation on market data"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Validator name for logging"""
        pass


class SpreadValidator(BaseValidator):
    """Validates spread conditions for tradability"""
    
    @property
    def name(self) -> str:
        return "SpreadValidator"
    
    async def validate(self, market_data: MarketData) -> ValidationResult:
        """Check if spread is acceptable for trading"""
        issues = []
        score = 1.0
        metadata = {}
        
        if not market_data.candles:
            return ValidationResult(
                passed=False,
                score=0.0,
                issues=[ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="data",
                    message="No market data available",
                    impact_score=1.0
                )],
                metadata={},
                recommendation="Skip - No data"
            )
        
        # Get symbol specifications
        symbol_spec = get_symbol_spec(market_data.symbol)
        typical_spread = symbol_spec.typical_spread
        
        # Calculate current spread
        latest_candle = market_data.latest_candle
        if latest_candle and latest_candle.spread is not None:
            current_spread = latest_candle.spread
            spread_ratio = current_spread / typical_spread if typical_spread > 0 else 999
            
            metadata['current_spread'] = current_spread
            metadata['typical_spread'] = typical_spread
            metadata['spread_ratio'] = spread_ratio
            
            # Evaluate spread
            if spread_ratio > 3.0:
                score -= 0.8
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="spread",
                    message=f"Spread is {spread_ratio:.1f}x typical ({current_spread:.1f} pips)",
                    impact_score=0.8,
                    details={'spread_ratio': spread_ratio}
                ))
            elif spread_ratio > 2.0:
                score -= 0.4
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="spread",
                    message=f"Spread is elevated ({spread_ratio:.1f}x typical)",
                    impact_score=0.4
                ))
            elif spread_ratio > 1.5:
                score -= 0.2
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="spread",
                    message=f"Spread slightly elevated ({spread_ratio:.1f}x typical)",
                    impact_score=0.2
                ))
        
        # Check spread volatility
        spread_volatility = self._calculate_spread_volatility(market_data)
        metadata['spread_volatility'] = spread_volatility
        
        if spread_volatility > 0.5:
            score -= 0.3
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="spread",
                message="High spread volatility detected",
                impact_score=0.3,
                details={'volatility': spread_volatility}
            ))
        
        recommendation = self._generate_recommendation(score, issues)
        
        return ValidationResult(
            passed=score >= 0.5,
            score=max(0, score),
            issues=issues,
            metadata=metadata,
            recommendation=recommendation
        )
    
    def _calculate_spread_volatility(self, market_data: MarketData) -> float:
        """Calculate spread volatility over recent candles"""
        spreads = [c.spread for c in market_data.candles[-10:] if c.spread is not None]
        
        if len(spreads) < 3:
            return 0.0
        
        spread_series = pd.Series(spreads)
        return spread_series.std() / spread_series.mean() if spread_series.mean() > 0 else 0.0
    
    def _generate_recommendation(self, score: float, issues: List[ValidationIssue]) -> str:
        """Generate recommendation based on validation results"""
        if score >= 0.8:
            return "Excellent spread conditions for trading"
        elif score >= 0.5:
            return "Acceptable spread conditions, proceed with caution"
        else:
            critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
            if critical_issues:
                return f"Poor spread conditions: {critical_issues[0].message}"
            return "Suboptimal spread conditions for trading"


class VolumeValidator(BaseValidator):
    """Validates volume patterns and liquidity"""
    
    @property
    def name(self) -> str:
        return "VolumeValidator"
    
    async def validate(self, market_data: MarketData) -> ValidationResult:
        """Check volume patterns for tradability"""
        issues = []
        score = 1.0
        metadata = {}
        
        if len(market_data.candles) < 20:
            return ValidationResult(
                passed=False,
                score=0.0,
                issues=[ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="data",
                    message="Insufficient data for volume analysis",
                    impact_score=1.0
                )],
                metadata={},
                recommendation="Skip - Insufficient data"
            )
        
        # Calculate volume metrics
        volumes = [c.volume for c in market_data.candles[-20:]]
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[:-1])
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        
        metadata.update({
            'current_volume': current_volume,
            'average_volume': avg_volume,
            'volume_ratio': volume_ratio,
            'volume_trend': self._calculate_volume_trend(volumes)
        })
        
        # Get minimum volume threshold
        symbol_spec = get_symbol_spec(market_data.symbol)
        min_volume = symbol_spec.min_volume
        
        # Validate volume levels
        if current_volume < min_volume:
            score -= 0.7
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="volume",
                message=f"Volume below minimum threshold ({current_volume} < {min_volume})",
                impact_score=0.7
            ))
        
        if volume_ratio < 0.3:
            score -= 0.5
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="volume",
                message="Extremely low volume compared to average",
                impact_score=0.5
            ))
        elif volume_ratio < 0.5:
            score -= 0.3
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="volume",
                message="Low volume period",
                impact_score=0.3
            ))
        
        # Check for volume spikes (potential news/events)
        if volume_ratio > 3.0:
            score -= 0.2
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="volume",
                message="High volume spike detected - possible news event",
                impact_score=0.2,
                details={'spike_ratio': volume_ratio}
            ))
        
        # Analyze volume pattern quality
        volume_quality = self._analyze_volume_pattern(market_data)
        metadata['volume_quality'] = volume_quality
        
        if volume_quality < 0.4:
            score -= 0.3
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="volume",
                message="Poor volume pattern quality",
                impact_score=0.3
            ))
        
        recommendation = self._generate_recommendation(score, metadata)
        
        return ValidationResult(
            passed=score >= 0.4,
            score=max(0, score),
            issues=issues,
            metadata=metadata,
            recommendation=recommendation
        )
    
    def _calculate_volume_trend(self, volumes: List[int]) -> str:
        """Determine volume trend direction"""
        if len(volumes) < 5:
            return "unknown"
        
        recent = np.mean(volumes[-5:])
        older = np.mean(volumes[-10:-5])
        
        if recent > older * 1.2:
            return "increasing"
        elif recent < older * 0.8:
            return "decreasing"
        return "stable"
    
    def _analyze_volume_pattern(self, market_data: MarketData) -> float:
        """Analyze volume pattern quality (0-1)"""
        if len(market_data.candles) < 10:
            return 0.5
        
        # Check volume-price correlation
        prices = [c.close for c in market_data.candles[-10:]]
        volumes = [c.volume for c in market_data.candles[-10:]]
        
        # Positive volume on up moves, lower volume on pullbacks is good
        price_changes = np.diff(prices)
        volume_changes = np.diff(volumes)
        
        correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
        
        # Good pattern: positive correlation (volume confirms price)
        quality_score = (correlation + 1) / 2  # Normalize to 0-1
        
        return quality_score
    
    def _generate_recommendation(self, score: float, metadata: Dict) -> str:
        """Generate volume-based recommendation"""
        if score >= 0.8:
            return "Strong volume profile supports trading"
        elif score >= 0.5:
            if metadata.get('volume_ratio', 0) > 2:
                return "High volume detected - check for news events"
            return "Adequate volume for trading"
        else:
            return "Poor volume conditions - consider waiting for better liquidity"


class TechnicalSetupValidator(BaseValidator):
    """Validates technical analysis setup quality"""
    
    @property
    def name(self) -> str:
        return "TechnicalSetupValidator"
    
    async def validate(self, market_data: MarketData) -> ValidationResult:
        """Validate technical setup strength"""
        issues = []
        score = 1.0
        metadata = {}
        
        if len(market_data.candles) < 50:
            return ValidationResult(
                passed=False,
                score=0.0,
                issues=[ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="data",
                    message="Insufficient data for technical analysis",
                    impact_score=1.0
                )],
                metadata={},
                recommendation="Skip - Insufficient data"
            )
        
        latest = market_data.latest_candle
        
        # Check EMA alignment
        ema_analysis = self._analyze_ema_setup(market_data)
        metadata['ema_setup'] = ema_analysis
        
        if ema_analysis['quality'] == 'poor':
            score -= 0.4
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="technical",
                message=ema_analysis['issue'],
                impact_score=0.4
            ))
        elif ema_analysis['quality'] == 'mixed':
            score -= 0.2
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="technical",
                message=ema_analysis['issue'],
                impact_score=0.2
            ))
        
        # Check RSI conditions
        rsi_analysis = self._analyze_rsi(latest)
        metadata['rsi_analysis'] = rsi_analysis
        
        if rsi_analysis['overbought']:
            score -= 0.3
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="technical",
                message="RSI indicates overbought conditions",
                impact_score=0.3,
                details={'rsi': rsi_analysis['value']}
            ))
        elif rsi_analysis['oversold']:
            score -= 0.3
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="technical",
                message="RSI indicates oversold conditions",
                impact_score=0.3,
                details={'rsi': rsi_analysis['value']}
            ))
        
        # Check price action quality
        pa_quality = self._analyze_price_action(market_data)
        metadata['price_action_quality'] = pa_quality
        
        if pa_quality['choppy']:
            score -= 0.4
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="technical",
                message="Choppy price action detected",
                impact_score=0.4,
                details=pa_quality
            ))
        
        # Check for overextension
        extension = self._check_overextension(market_data)
        metadata['extension_analysis'] = extension
        
        if extension['overextended']:
            score -= 0.3
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="technical",
                message=f"Price overextended from {extension['from']}",
                impact_score=0.3,
                details=extension
            ))
        
        recommendation = self._generate_recommendation(score, metadata)
        
        return ValidationResult(
            passed=score >= 0.4,
            score=max(0, score),
            issues=issues,
            metadata=metadata,
            recommendation=recommendation
        )
    
    def _analyze_ema_setup(self, market_data: MarketData) -> Dict[str, Any]:
        """Analyze EMA configuration quality"""
        latest = market_data.latest_candle
        
        if not latest.ema50 or not latest.ema200:
            return {'quality': 'unknown', 'issue': 'EMAs not calculated'}
        
        price = latest.close
        ema50 = latest.ema50
        ema200 = latest.ema200
        
        # Calculate distances
        price_to_ema50 = abs(price - ema50) / price
        ema50_to_ema200 = abs(ema50 - ema200) / ema50
        
        # Check for tangled EMAs
        if ema50_to_ema200 < 0.001:  # Less than 0.1% apart
            return {
                'quality': 'poor',
                'issue': 'EMAs are tangled - no clear trend',
                'trend': 'neutral'
            }
        
        # Determine trend
        if ema50 > ema200:
            trend = 'bullish'
            if price < ema50:
                return {
                    'quality': 'mixed',
                    'issue': 'Price below EMA50 in uptrend',
                    'trend': trend
                }
        else:
            trend = 'bearish'
            if price > ema50:
                return {
                    'quality': 'mixed',
                    'issue': 'Price above EMA50 in downtrend',
                    'trend': trend
                }
        
        return {
            'quality': 'good',
            'trend': trend,
            'price_to_ema50_distance': price_to_ema50,
            'ema_separation': ema50_to_ema200
        }
    
    def _analyze_rsi(self, candle: Candle) -> Dict[str, Any]:
        """Analyze RSI conditions"""
        if not candle.rsi14:
            return {'value': None, 'overbought': False, 'oversold': False}
        
        rsi = candle.rsi14
        
        return {
            'value': rsi,
            'overbought': rsi > 70,
            'oversold': rsi < 30,
            'neutral_zone': 40 <= rsi <= 60
        }
    
    def _analyze_price_action(self, market_data: MarketData) -> Dict[str, Any]:
        """Analyze price action quality"""
        candles = market_data.candles[-20:]
        
        # Calculate swing highs and lows
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        
        # Count direction changes
        direction_changes = 0
        for i in range(2, len(candles)):
            if (candles[i].close > candles[i-1].close and 
                candles[i-1].close < candles[i-2].close):
                direction_changes += 1
            elif (candles[i].close < candles[i-1].close and 
                  candles[i-1].close > candles[i-2].close):
                direction_changes += 1
        
        # Choppy if too many direction changes
        choppy = direction_changes > len(candles) * 0.6
        
        # Calculate average range
        avg_range = np.mean([c.high - c.low for c in candles])
        current_range = candles[-1].high - candles[-1].low
        
        return {
            'choppy': choppy,
            'direction_changes': direction_changes,
            'range_expansion': current_range > avg_range * 1.5,
            'range_contraction': current_range < avg_range * 0.5
        }
    
    def _check_overextension(self, market_data: MarketData) -> Dict[str, Any]:
        """Check if price is overextended from key levels"""
        latest = market_data.latest_candle
        price = latest.close
        
        result = {'overextended': False, 'from': None, 'distance': 0}
        
        # Check distance from EMAs
        if latest.ema50:
            distance_from_ema50 = abs(price - latest.ema50) / latest.ema50
            if distance_from_ema50 > 0.02:  # More than 2%
                result = {
                    'overextended': True,
                    'from': 'EMA50',
                    'distance': distance_from_ema50,
                    'direction': 'above' if price > latest.ema50 else 'below'
                }
        
        # Check ATR-based extension
        if latest.atr14:
            atr_multiples = abs(price - latest.ema50) / latest.atr14 if latest.ema50 else 0
            if atr_multiples > 2.5:
                result['atr_extension'] = atr_multiples
        
        return result
    
    def _generate_recommendation(self, score: float, metadata: Dict) -> str:
        """Generate technical setup recommendation"""
        if score >= 0.8:
            return "Strong technical setup for entry"
        elif score >= 0.6:
            return "Decent technical setup, monitor closely"
        elif score >= 0.4:
            return "Weak technical setup, consider waiting"
        else:
            return "Poor technical conditions for trading"


class MarketStructureValidator(BaseValidator):
    """Validates market structure and support/resistance levels"""
    
    @property
    def name(self) -> str:
        return "MarketStructureValidator"
    
    async def validate(self, market_data: MarketData) -> ValidationResult:
        """Validate market structure quality"""
        issues = []
        score = 1.0
        metadata = {}
        
        if len(market_data.candles) < 100:
            return ValidationResult(
                passed=True,  # Don't block, just note limited data
                score=0.7,
                issues=[ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="data",
                    message="Limited data for structure analysis",
                    impact_score=0.3
                )],
                metadata={},
                recommendation="Proceed with limited structure data"
            )
        
        # Identify key levels
        levels = self._identify_key_levels(market_data)
        metadata['key_levels'] = levels
        
        # Check proximity to levels
        latest_price = market_data.latest_candle.close
        nearest_resistance = self._find_nearest_level(latest_price, levels['resistance'])
        nearest_support = self._find_nearest_level(latest_price, levels['support'])
        
        metadata['nearest_resistance'] = nearest_resistance
        metadata['nearest_support'] = nearest_support
        
        # Validate based on proximity
        if nearest_resistance and nearest_resistance['distance_percent'] < 0.3:
            score -= 0.4
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="structure",
                message=f"Very close to resistance at {nearest_resistance['level']:.5f}",
                impact_score=0.4,
                details=nearest_resistance
            ))
        
        if nearest_support and nearest_support['distance_percent'] < 0.2:
            score -= 0.2
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="structure",
                message=f"Close to support at {nearest_support['level']:.5f}",
                impact_score=0.2,
                details=nearest_support
            ))
        
        # Check trend structure quality
        trend_quality = self._analyze_trend_structure(market_data)
        metadata['trend_structure'] = trend_quality
        
        if trend_quality['quality'] == 'poor':
            score -= 0.3
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="structure",
                message="Poor trend structure",
                impact_score=0.3,
                details=trend_quality
            ))
        
        recommendation = self._generate_recommendation(score, metadata)
        
        return ValidationResult(
            passed=score >= 0.5,
            score=max(0, score),
            issues=issues,
            metadata=metadata,
            recommendation=recommendation
        )
    
    def _identify_key_levels(self, market_data: MarketData) -> Dict[str, List[float]]:
        """Identify key support and resistance levels"""
        candles = market_data.candles
        
        # Simple implementation - find local highs/lows
        resistance_levels = []
        support_levels = []
        
        for i in range(10, len(candles) - 10):
            # Check if local high
            if all(candles[i].high >= candles[j].high for j in range(i-5, i+6) if j != i):
                resistance_levels.append(candles[i].high)
            
            # Check if local low
            if all(candles[i].low <= candles[j].low for j in range(i-5, i+6) if j != i):
                support_levels.append(candles[i].low)
        
        # Cluster nearby levels
        resistance_levels = self._cluster_levels(resistance_levels)
        support_levels = self._cluster_levels(support_levels)
        
        return {
            'resistance': sorted(resistance_levels, reverse=True)[:5],
            'support': sorted(support_levels)[:5]
        }
    
    def _cluster_levels(self, levels: List[float], threshold: float = 0.001) -> List[float]:
        """Cluster nearby levels together"""
        if not levels:
            return []
        
        sorted_levels = sorted(levels)
        clusters = []
        current_cluster = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            if level - current_cluster[-1] < threshold:
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        
        if current_cluster:
            clusters.append(np.mean(current_cluster))
        
        return clusters
    
    def _find_nearest_level(self, price: float, levels: List[float]) -> Optional[Dict[str, Any]]:
        """Find nearest level to current price"""
        if not levels:
            return None
        
        distances = [(abs(level - price), level) for level in levels]
        distance, nearest_level = min(distances)
        
        return {
            'level': nearest_level,
            'distance': distance,
            'distance_percent': (distance / price) * 100,
            'type': 'resistance' if nearest_level > price else 'support'
        }
    
    def _analyze_trend_structure(self, market_data: MarketData) -> Dict[str, Any]:
        """Analyze trend structure quality"""
        candles = market_data.candles[-50:]
        
        # Find swing highs and lows
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(candles) - 2):
            # Swing high
            if (candles[i].high > candles[i-1].high and 
                candles[i].high > candles[i-2].high and
                candles[i].high > candles[i+1].high and 
                candles[i].high > candles[i+2].high):
                swing_highs.append((i, candles[i].high))
            
            # Swing low
            if (candles[i].low < candles[i-1].low and 
                candles[i].low < candles[i-2].low and
                candles[i].low < candles[i+1].low and 
                candles[i].low < candles[i+2].low):
                swing_lows.append((i, candles[i].low))
        
        # Analyze structure
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {'quality': 'poor', 'reason': 'Insufficient swing points'}
        
        # Check for higher highs/higher lows (uptrend) or lower highs/lower lows (downtrend)
        hh_hl = self._check_trend_pattern(swing_highs, swing_lows, 'up')
        ll_lh = self._check_trend_pattern(swing_highs, swing_lows, 'down')
        
        if hh_hl:
            return {'quality': 'good', 'type': 'uptrend', 'strength': 'strong'}
        elif ll_lh:
            return {'quality': 'good', 'type': 'downtrend', 'strength': 'strong'}
        else:
            return {'quality': 'mixed', 'type': 'ranging', 'strength': 'weak'}
    
    def _check_trend_pattern(self, highs: List[Tuple], lows: List[Tuple], direction: str) -> bool:
        """Check if swings follow trend pattern"""
        if direction == 'up':
            # Check for higher highs
            for i in range(1, len(highs)):
                if highs[i][1] <= highs[i-1][1]:
                    return False
            # Check for higher lows
            for i in range(1, len(lows)):
                if lows[i][1] <= lows[i-1][1]:
                    return False
            return True
        else:
            # Check for lower highs
            for i in range(1, len(highs)):
                if highs[i][1] >= highs[i-1][1]:
                    return False
            # Check for lower lows
            for i in range(1, len(lows)):
                if lows[i][1] >= lows[i-1][1]:
                    return False
            return True
    
    def _generate_recommendation(self, score: float, metadata: Dict) -> str:
        """Generate structure-based recommendation"""
        if score >= 0.8:
            return "Clean market structure for trading"
        elif score >= 0.6:
            return "Acceptable structure with minor concerns"
        elif score >= 0.5:
            if 'nearest_resistance' in metadata and metadata['nearest_resistance']['distance_percent'] < 0.5:
                return "Caution: Near resistance level"
            return "Mixed structure - trade carefully"
        else:
            return "Poor market structure - avoid trading"


class OfflineSignalValidator:
    """
    Main offline signal validation service.
    Coordinates multiple validators and provides comprehensive pre-trade analysis.
    """
    
    def __init__(self):
        self.validators = [
            SpreadValidator(),
            VolumeValidator(),
            TechnicalSetupValidator(),
            MarketStructureValidator()
        ]
        
        # Weights for each validator (can be adjusted based on strategy)
        self.validator_weights = {
            'SpreadValidator': 0.20,
            'VolumeValidator': 0.25,
            'TechnicalSetupValidator': 0.35,
            'MarketStructureValidator': 0.20
        }
    
    async def validate_market_data(
        self, 
        market_data: MarketData,
        min_score_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        Run all validators and compile results.
        
        Args:
            market_data: Market data to validate
            min_score_threshold: Minimum score to proceed with signal generation
            
        Returns:
            Comprehensive validation result with GPT context
        """
        logger.info(f"Starting offline validation for {market_data.symbol}")
        
        # Run all validators in parallel
        validation_tasks = [
            validator.validate(market_data) 
            for validator in self.validators
        ]
        
        results = await asyncio.gather(*validation_tasks)
        
        # Compile results
        validation_summary = self._compile_results(results)
        
        # Determine if we should proceed
        should_proceed = validation_summary['weighted_score'] >= min_score_threshold
        
        # Generate context for GPT
        gpt_context = self._generate_gpt_context(validation_summary, should_proceed)
        
        # Log summary
        logger.info(
            f"Validation complete for {market_data.symbol}: "
            f"Score={validation_summary['weighted_score']:.2f}, "
            f"Proceed={should_proceed}"
        )
        
        return {
            'should_proceed': should_proceed,
            'validation_summary': validation_summary,
            'gpt_context': gpt_context,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _compile_results(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Compile individual validation results into summary"""
        summary = {
            'individual_scores': {},
            'all_issues': [],
            'critical_issues': [],
            'weighted_score': 0.0,
            'validators_passed': 0,
            'total_validators': len(results),
            'metadata': {}
        }
        
        # Process each validator result
        for i, result in enumerate(results):
            validator_name = self.validators[i].name
            weight = self.validator_weights.get(validator_name, 0.25)
            
            # Store individual scores
            summary['individual_scores'][validator_name] = {
                'score': result.score,
                'passed': result.passed,
                'weight': weight
            }
            
            # Accumulate weighted score
            summary['weighted_score'] += result.score * weight
            
            # Count passed validators
            if result.passed:
                summary['validators_passed'] += 1
            
            # Collect issues
            summary['all_issues'].extend([
                {'validator': validator_name, **issue.__dict__}
                for issue in result.issues
            ])
            
            # Identify critical issues
            critical = [
                issue for issue in result.issues 
                if issue.severity == ValidationSeverity.CRITICAL
            ]
            if critical:
                summary['critical_issues'].extend([
                    {'validator': validator_name, **issue.__dict__}
                    for issue in critical
                ])
            
            # Merge metadata
            summary['metadata'][validator_name] = result.metadata
        
        # Sort issues by impact score
        summary['all_issues'].sort(key=lambda x: x.get('impact_score', 0), reverse=True)
        
        return summary
    
    def _generate_gpt_context(
        self, 
        validation_summary: Dict[str, Any], 
        should_proceed: bool
    ) -> Dict[str, Any]:
        """Generate enriched context for GPT signal generation"""
        
        # Build quality indicators
        quality_indicators = {
            'overall_quality_score': validation_summary['weighted_score'],
            'validators_passed_ratio': (
                validation_summary['validators_passed'] / 
                validation_summary['total_validators']
            ),
            'has_critical_issues': len(validation_summary['critical_issues']) > 0,
            'issue_count': len(validation_summary['all_issues'])
        }
        
        # Extract key insights from metadata
        insights = self._extract_key_insights(validation_summary['metadata'])
        
        # Build issue summary
        issue_summary = self._summarize_issues(validation_summary['all_issues'])
        
        # Create structured context for GPT
        gpt_context = {
            'pre_validation_performed': True,
            'validation_recommendation': 'PROCEED' if should_proceed else 'CAUTION',
            'quality_indicators': quality_indicators,
            'key_insights': insights,
            'issue_summary': issue_summary,
            'detailed_scores': validation_summary['individual_scores']
        }
        
        # Add specific warnings if needed
        if validation_summary['critical_issues']:
            gpt_context['critical_warnings'] = [
                issue['message'] for issue in validation_summary['critical_issues'][:3]
            ]
        
        return gpt_context
    
    def _extract_key_insights(self, metadata: Dict[str, Dict]) -> Dict[str, Any]:
        """Extract key insights from validator metadata"""
        insights = {}
        
        # Spread insights
        if 'SpreadValidator' in metadata:
            spread_data = metadata['SpreadValidator']
            if 'spread_ratio' in spread_data:
                insights['spread_condition'] = (
                    'tight' if spread_data['spread_ratio'] < 1.2 
                    else 'wide' if spread_data['spread_ratio'] > 2 
                    else 'normal'
                )
        
        # Volume insights
        if 'VolumeValidator' in metadata:
            volume_data = metadata['VolumeValidator']
            insights['volume_profile'] = {
                'current_vs_average': volume_data.get('volume_ratio', 1.0),
                'trend': volume_data.get('volume_trend', 'unknown')
            }
        
        # Technical insights
        if 'TechnicalSetupValidator' in metadata:
            tech_data = metadata['TechnicalSetupValidator']
            if 'ema_setup' in tech_data:
                insights['trend_alignment'] = tech_data['ema_setup'].get('trend', 'unknown')
            if 'rsi_analysis' in tech_data:
                insights['momentum_condition'] = (
                    'overbought' if tech_data['rsi_analysis'].get('overbought')
                    else 'oversold' if tech_data['rsi_analysis'].get('oversold')
                    else 'neutral'
                )
        
        # Structure insights
        if 'MarketStructureValidator' in metadata:
            structure_data = metadata['MarketStructureValidator']
            if 'nearest_resistance' in structure_data and structure_data['nearest_resistance']:
                insights['nearest_resistance_distance'] = structure_data['nearest_resistance']['distance_percent']
            if 'trend_structure' in structure_data:
                insights['market_structure_type'] = structure_data['trend_structure'].get('type', 'unknown')
        
        return insights
    
    def _summarize_issues(self, issues: List[Dict]) -> Dict[str, Any]:
        """Summarize issues for quick reference"""
        if not issues:
            return {'summary': 'No issues detected', 'top_concerns': []}
        
        # Count by severity
        severity_counts = {
            'critical': sum(1 for i in issues if i.get('severity') == ValidationSeverity.CRITICAL),
            'warning': sum(1 for i in issues if i.get('severity') == ValidationSeverity.WARNING),
            'info': sum(1 for i in issues if i.get('severity') == ValidationSeverity.INFO)
        }
        
        # Get top 3 issues by impact
        top_issues = sorted(issues, key=lambda x: x.get('impact_score', 0), reverse=True)[:3]
        
        return {
            'summary': f"{len(issues)} issues found ({severity_counts['critical']} critical)",
            'severity_breakdown': severity_counts,
            'top_concerns': [
                {
                    'message': issue['message'],
                    'impact': issue.get('impact_score', 0),
                    'category': issue.get('category', 'general')
                }
                for issue in top_issues
            ]
        }
    
    def get_validator_weights(self) -> Dict[str, float]:
        """Get current validator weights"""
        return self.validator_weights.copy()
    
    def update_validator_weights(self, new_weights: Dict[str, float]):
        """Update validator weights (must sum to 1.0)"""
        if abs(sum(new_weights.values()) - 1.0) > 0.001:
            raise ValueError("Validator weights must sum to 1.0")
        
        self.validator_weights.update(new_weights)
        logger.info(f"Updated validator weights: {self.validator_weights}")


# Export main classes
__all__ = [
    'OfflineSignalValidator',
    'ValidationResult',
    'ValidationIssue',
    'ValidationSeverity',
    'SpreadValidator',
    'VolumeValidator',
    'TechnicalSetupValidator',
    'MarketStructureValidator'
]