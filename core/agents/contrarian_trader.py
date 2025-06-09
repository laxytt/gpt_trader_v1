"""
Contrarian Trader Agent
Specializes in fading extremes and mean reversion
"""

import logging
from typing import Dict, List, Optional, Any

from core.agents.base_agent import TradingAgent, AgentType, AgentAnalysis, DebateResponse
from core.domain.models import MarketData, SignalType
from core.infrastructure.gpt.client import GPTClient

logger = logging.getLogger(__name__)


class ContrarianTrader(TradingAgent):
    """The Skeptic - fades extremes and crowds"""
    
    def __init__(self, gpt_client: GPTClient):
        super().__init__(
            agent_type=AgentType.CONTRARIAN_TRADER,
            personality="skeptical and contrarian fade trader",
            specialty="reversals, extremes, mean reversion, failed moves, and divergences"
        )
        self.gpt_client = gpt_client
    
    def analyze(
        self,
        market_data: Dict[str, MarketData],
        news_context: Optional[List[str]] = None,
        ml_context: Optional[Dict[str, Any]] = None
    ) -> AgentAnalysis:
        """Look for contrarian opportunities with professional mean reversion analysis"""
        
        h1_data = market_data.get('h1')
        h4_data = market_data.get('h4')
        d1_data = market_data.get('d1')
        
        if not h1_data:
            raise ValueError("H1 data required for contrarian analysis")
        
        # Import enhanced analyzer
        try:
            from core.agents.enhanced_base_agent import ProfessionalMarketAnalyzer
            import numpy as np
            analyzer = ProfessionalMarketAnalyzer()
            
            # Need 150+ candles for proper mean reversion analysis
            h1_candles = h1_data.candles[-150:] if len(h1_data.candles) >= 150 else h1_data.candles
            
            # Market regime for contrarian context
            regime = analyzer.calculate_market_regime(h1_candles)
            hurst = regime.get('hurst', 0.5)
            
            # Volume profile for liquidity zones
            vol_profile = analyzer.calculate_volume_profile(h1_candles[-100:])
            
            use_enhanced = True
        except:
            use_enhanced = False
            hurst = 0.5
            vol_profile = {}
        
        candles = h1_data.candles
        current_price = h1_data.latest_candle.close
        
        # Professional mean reversion metrics
        zscore_20 = self._calculate_zscore(candles, 20)
        zscore_50 = self._calculate_zscore(candles, 50)
        
        # Bollinger Band analysis
        bb_position, bb_width = self._calculate_bollinger_position(candles)
        bb_squeeze = self._detect_bollinger_squeeze(candles)
        
        # RSI divergence analysis
        rsi_divergence = self._calculate_rsi_divergence(candles)
        
        # Failed breakout detection
        failed_breakouts = self._detect_failed_breakouts(candles)
        
        # Volume analysis for exhaustion
        volume_climax = self._detect_volume_climax(candles)
        volume_divergence = self._detect_volume_divergence(candles)
        
        # Statistical mean reversion
        mean_reversion_prob = self._calculate_mean_reversion_probability(candles, hurst)
        half_life = self._calculate_half_life(candles)
        
        # Extreme detection
        percentile_rank = self._calculate_percentile_rank(candles)
        consecutive_moves = self._count_consecutive_moves(candles)
        
        # Support/Resistance for reversal zones
        reversal_zones = self._identify_reversal_zones(candles, vol_profile)
        
        analysis_prompt = self._build_prompt("""
Analyze {symbol} for professional contrarian/mean reversion opportunities.

STATISTICAL EXTREMES (150 bars):
- Z-Score (20/50): {zscore20:.2f} / {zscore50:.2f}
- Percentile Rank: {percentile:.0f}th
- Consecutive Moves: {consecutive} bars in same direction
- Mean Reversion Probability: {mr_prob:.1%}
- Half-Life: {half_life:.1f} bars

BOLLINGER BAND ANALYSIS:
- Position: {bb_position:.1%} (0=lower, 100=upper)
- Band Width: {bb_width:.3f} (squeeze: {bb_squeeze})
- Current Price: {price:.5f}

DIVERGENCE ANALYSIS:
- RSI Divergence: {rsi_div}
- Volume Divergence: {vol_div}
- Failed Breakouts (20 bars): {failed_breaks}

EXHAUSTION SIGNALS:
- Volume Climax: {vol_climax}
- Hurst Exponent: {hurst:.3f} (mean reverting if <0.5)
- Market Regime: {regime}

REVERSAL ZONES:
- Key Levels: {reversal_zones}
- Volume POC: {poc:.5f}
- Value Area: {val_low:.5f} - {val_high:.5f}

SENTIMENT FACTORS:
- RSI(14): {rsi:.1f}
- Price vs 20MA: {vs_ma20:.2f}%
- Price vs 50MA: {vs_ma50:.2f}%

Professional Contrarian Assessment:
1. Statistical edge for mean reversion
2. Exhaustion/climax indicators
3. Failed moves and traps
4. Divergence quality
5. Risk/reward at extremes
6. Timing the reversal

Provide:
OPPORTUNITY: [Strong/Moderate/Weak/None]
SETUP_TYPE: [Statistical extreme/Failed breakout/Divergence/Exhaustion/Squeeze play]
CROWD_ERROR: [specific psychological error]
RECOMMENDATION: [BUY/SELL/WAIT]
CONFIDENCE: [0-100]
ENTRY_ZONE: [specific price zone]
REVERSAL_TRIGGERS: [specific catalysts]
INVALIDATION: [specific level/condition]
POSITION_SIZE: [0.1-0.5 for contrarian trades]

Format your response with exact labels above.
""",
            symbol=h1_data.symbol,
            price=current_price,
            zscore20=zscore_20,
            zscore50=zscore_50,
            percentile=percentile_rank,
            consecutive=consecutive_moves,
            mr_prob=mean_reversion_prob,
            half_life=half_life,
            bb_position=bb_position,
            bb_width=bb_width,
            bb_squeeze="Active" if bb_squeeze else "None",
            rsi_div=rsi_divergence,
            vol_div=volume_divergence,
            failed_breaks=failed_breakouts,
            vol_climax=volume_climax,
            hurst=hurst,
            regime=regime.get('regime', 'unknown') if use_enhanced else 'unknown',
            reversal_zones=", ".join([f"{z:.5f}" for z in reversal_zones[:3]]),
            poc=vol_profile.get('poc', current_price),
            val_low=vol_profile.get('value_area_low', current_price * 0.995),
            val_high=vol_profile.get('value_area_high', current_price * 1.005),
            rsi=h1_data.latest_candle.rsi14,
            vs_ma20=self._price_vs_ma(candles, 20),
            vs_ma50=self._price_vs_ma(candles, 50)
        )
        
        response = self.gpt_client.analyze_with_response(
            analysis_prompt,
            temperature=0.6,  # Higher for creative contrarian thinking
            agent_type=self.agent_type.value,
            symbol=h1_data.symbol
        )
        
        return self._parse_analysis(response, current_price)
    
    def debate(
        self,
        other_analyses: List[AgentAnalysis],
        round_number: int,
        previous_responses: Optional[List[DebateResponse]] = None
    ) -> DebateResponse:
        """Participate in debate with contrarian perspective"""
        
        my_position = self.analysis_history[-1]
        
        # Count how many agree - contrarian loves to fade consensus
        consensus_count = sum(1 for a in other_analyses if a.recommendation == my_position.recommendation)
        
        if round_number == 1:
            statement = self._generate_opening_statement(my_position, len(other_analyses) - consensus_count)
        elif round_number == 2:
            # Challenge the momentum trader specifically
            momentum = next((a for a in other_analyses if a.agent_type == AgentType.MOMENTUM_TRADER), None)
            if momentum and momentum.recommendation != SignalType.WAIT and momentum.recommendation != my_position.recommendation:
                statement = "The momentum trader sees strength, but that's exactly when reversals happen. The rubber band is stretched too far."
            else:
                statement = self._challenge_consensus(other_analyses)
        else:
            statement = self._generate_closing_statement(other_analyses)
        
        # Contrarian gets more confident when everyone disagrees
        updated_confidence = my_position.confidence
        
        if consensus_count <= 1:  # Only contrarian sees this
            updated_confidence = min(100, my_position.confidence + 10)
        elif consensus_count >= 5:  # Everyone agrees (suspicious to contrarian)
            updated_confidence = max(30, my_position.confidence - 20)
        
        return DebateResponse(
            agent_type=self.agent_type,
            round=round_number,
            statement=statement,
            maintains_position=True,  # Contrarians don't follow the crowd
            updated_confidence=updated_confidence
        )
    
    def _parse_analysis(self, response: str, current_price: float) -> AgentAnalysis:
        """Parse GPT response into AgentAnalysis using safe parsing"""
        try:
            # Use the safe parsing method from base class
            parsed_data = self._safe_parse_response(response, current_price)
            
            # Extract contrarian-specific fields
            opportunity = parsed_data.get('metadata', {}).get('OPPORTUNITY', 'None')
            setup_type = parsed_data.get('metadata', {}).get('SETUP_TYPE', '')
            position_size = parsed_data.get('metadata', {}).get('POSITION_SIZE', 0.2)
            entry_zone = parsed_data.get('metadata', {}).get('ENTRY_ZONE', '')
            
            # Add position size to metadata
            parsed_data['metadata']['position_size'] = float(position_size) if isinstance(position_size, str) else position_size
            
            # Build comprehensive reasoning
            reasoning = []
            if setup_type:
                reasoning.append(f"Setup: {setup_type}")
            if parsed_data.get('metadata', {}).get('CROWD_ERROR'):
                reasoning.append(f"Crowd error: {parsed_data['metadata']['CROWD_ERROR']}")
            if entry_zone:
                reasoning.append(f"Entry zone: {entry_zone}")
            
            analysis = AgentAnalysis(
                agent_type=self.agent_type,
                recommendation=parsed_data['recommendation'],
                confidence=parsed_data['confidence'],
                reasoning=reasoning[:3] or ["Contrarian analysis completed"],
                concerns=parsed_data.get('concerns', []),
                entry_price=parsed_data.get('entry', current_price),
                stop_loss=parsed_data.get('stop_loss', 0),
                take_profit=parsed_data.get('take_profit', 0),
                risk_reward_ratio=self._calculate_risk_reward(
                    parsed_data.get('entry', current_price),
                    parsed_data.get('stop_loss', 0),
                    parsed_data.get('take_profit', 0)
                ) if parsed_data.get('stop_loss') and parsed_data.get('take_profit') else None,
                metadata=parsed_data['metadata']
            )
            
            self.analysis_history.append(analysis)
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to parse contrarian analysis: {e}")
            default_analysis = AgentAnalysis(
                agent_type=self.agent_type,
                recommendation=SignalType.WAIT,
                confidence=50.0,
                reasoning=["Error parsing contrarian analysis"],
                concerns=["Analysis parsing failed"],
                entry_price=current_price,
                metadata={'error': str(e)}
            )
            self.analysis_history.append(default_analysis)
            return default_analysis
    
    def _generate_opening_statement(self, analysis: AgentAnalysis, disagreement_count: int) -> str:
        """Generate opening debate statement"""
        setup = analysis.metadata.get('setup_type', 'contrarian setup')
        
        if analysis.recommendation != SignalType.WAIT:
            intro = f"While others see strength, I see exhaustion. {setup} suggests {analysis.recommendation.value}."
            if disagreement_count >= 4:
                intro += " The fact that most disagree only strengthens my conviction."
            return intro
        else:
            return "No contrarian opportunity here. Sometimes the trend is genuine, not ready to fade yet."
    
    def _challenge_consensus(self, analyses: List[AgentAnalysis]) -> str:
        """Challenge the consensus view"""
        buy_count = sum(1 for a in analyses if a.recommendation == SignalType.BUY)
        sell_count = sum(1 for a in analyses if a.recommendation == SignalType.SELL)
        
        if buy_count >= 4:
            return "Everyone's bullish? That's precisely when tops form. The last buyer is in."
        elif sell_count >= 4:
            return "Universal bearishness is a contrarian's dream. Bottoms are made in despair."
        else:
            return "The market lacks conviction here. Perhaps that's the real signal - confusion before a move."
    
    def _generate_closing_statement(self, all_analyses: List[AgentAnalysis]) -> str:
        """Generate closing statement"""
        my_position = self.analysis_history[-1]
        crowd_error = my_position.metadata.get('crowd_error', 'following the herd')
        
        if my_position.recommendation != SignalType.WAIT:
            return f"The crowd is {crowd_error}. Contrarian trades are uncomfortable by design - that's why they work."
        else:
            return "No clear extreme to fade. A good contrarian knows when NOT to be contrarian."
    
    def _calculate_zscore(self, candles: List, period: int) -> float:
        """Calculate Z-score for mean reversion"""
        if len(candles) < period:
            return 0
            
        prices = [c.close for c in candles[-period:]]
        mean = sum(prices) / len(prices)
        
        # Calculate standard deviation
        variance = sum((p - mean) ** 2 for p in prices) / len(prices)
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return 0
            
        current = candles[-1].close
        return (current - mean) / std_dev
    
    def _calculate_bollinger_position(self, candles: List) -> tuple:
        """Calculate position within Bollinger Bands"""
        if len(candles) < 20:
            return 0.5, 1.0
            
        # 20-period SMA and standard deviation
        prices = [c.close for c in candles[-20:]]
        mean = sum(prices) / len(prices)
        variance = sum((p - mean) ** 2 for p in prices) / len(prices)
        std_dev = variance ** 0.5
        
        upper = mean + (2 * std_dev)
        lower = mean - (2 * std_dev)
        current = candles[-1].close
        
        # Position (0 = at lower band, 1 = at upper band)
        position = (current - lower) / (upper - lower) if upper != lower else 0.5
        
        # Band width
        width = (upper - lower) / mean if mean > 0 else 0
        
        return position, width
    
    def _detect_bollinger_squeeze(self, candles: List) -> bool:
        """Detect Bollinger Band squeeze"""
        if len(candles) < 100:
            return False
            
        # Calculate band widths over time
        widths = []
        for i in range(20, 100):
            prices = [c.close for c in candles[-i-20:-i]]
            mean = sum(prices) / len(prices)
            variance = sum((p - mean) ** 2 for p in prices) / len(prices)
            std_dev = variance ** 0.5
            width = (4 * std_dev) / mean if mean > 0 else 0
            widths.append(width)
        
        # Current width
        current_prices = [c.close for c in candles[-20:]]
        current_mean = sum(current_prices) / len(current_prices)
        current_var = sum((p - current_mean) ** 2 for p in current_prices) / len(current_prices)
        current_std = current_var ** 0.5
        current_width = (4 * current_std) / current_mean if current_mean > 0 else 0
        
        # Squeeze if current width is in bottom 20th percentile
        widths.sort()
        percentile_20 = widths[int(len(widths) * 0.2)]
        
        return current_width < percentile_20
    
    def _calculate_rsi_divergence(self, candles: List) -> str:
        """Detect RSI divergence"""
        if len(candles) < 20:
            return "None"
            
        # Find recent price peaks/troughs
        price_highs = []
        price_lows = []
        rsi_highs = []
        rsi_lows = []
        
        for i in range(5, 20):
            idx = len(candles) - i
            
            # Price peak
            if candles[idx].high > candles[idx-1].high and candles[idx].high > candles[idx+1].high:
                price_highs.append((idx, candles[idx].high, candles[idx].rsi14))
            
            # Price trough
            if candles[idx].low < candles[idx-1].low and candles[idx].low < candles[idx+1].low:
                price_lows.append((idx, candles[idx].low, candles[idx].rsi14))
        
        # Check for divergences
        if len(price_highs) >= 2:
            # Bearish divergence: higher price high, lower RSI high
            if price_highs[-1][1] > price_highs[-2][1] and price_highs[-1][2] < price_highs[-2][2]:
                return "Bearish divergence"
        
        if len(price_lows) >= 2:
            # Bullish divergence: lower price low, higher RSI low
            if price_lows[-1][1] < price_lows[-2][1] and price_lows[-1][2] > price_lows[-2][2]:
                return "Bullish divergence"
        
        return "None"
    
    def _detect_failed_breakouts(self, candles: List) -> int:
        """Count failed breakout attempts"""
        if len(candles) < 30:
            return 0
            
        failed = 0
        
        # 20-bar high/low
        for i in range(10, 25):
            idx = len(candles) - i
            high_20 = max(c.high for c in candles[idx-20:idx])
            low_20 = min(c.low for c in candles[idx-20:idx])
            
            # Failed upside breakout
            if candles[idx].close > high_20:
                # Check if it failed within next 3 bars
                if any(c.close < high_20 * 0.998 for c in candles[idx+1:idx+4]):
                    failed += 1
            
            # Failed downside breakout
            if candles[idx].close < low_20:
                # Check if it failed within next 3 bars
                if any(c.close > low_20 * 1.002 for c in candles[idx+1:idx+4]):
                    failed += 1
        
        return failed
    
    def _detect_volume_climax(self, candles: List) -> str:
        """Detect volume climax"""
        if len(candles) < 50:
            return "None"
            
        # Volume statistics
        volumes = [c.volume for c in candles[-50:]]
        avg_vol = sum(volumes) / len(volumes)
        
        # Recent volume
        recent_vol = candles[-1].volume
        
        if recent_vol > avg_vol * 3:
            # Check if it's buying or selling climax
            if candles[-1].close > candles[-1].open:
                # But price isn't making new highs
                if candles[-1].high < max(c.high for c in candles[-10:]):
                    return "Buying exhaustion"
                else:
                    return "Buying climax"
            else:
                # But price isn't making new lows
                if candles[-1].low > min(c.low for c in candles[-10:]):
                    return "Selling exhaustion"
                else:
                    return "Selling climax"
        
        return "None"
    
    def _detect_volume_divergence(self, candles: List) -> str:
        """Detect price/volume divergence"""
        if len(candles) < 20:
            return "None"
            
        # Recent trend
        price_change = (candles[-1].close - candles[-10].close) / candles[-10].close
        
        # Volume trend
        recent_avg_vol = sum(c.volume for c in candles[-5:]) / 5
        older_avg_vol = sum(c.volume for c in candles[-20:-10]) / 10
        
        # Divergence detection
        if price_change > 0.01:  # Price rising
            if recent_avg_vol < older_avg_vol * 0.8:  # Volume falling
                return "Bearish (rising price, falling volume)"
        elif price_change < -0.01:  # Price falling
            if recent_avg_vol < older_avg_vol * 0.8:  # Volume falling
                return "Bullish (falling price, drying volume)"
        
        return "None"
    
    def _calculate_mean_reversion_probability(self, candles: List, hurst: float) -> float:
        """Calculate probability of mean reversion"""
        if len(candles) < 50:
            return 0.5
            
        # Factors that increase mean reversion probability
        prob = 0.5
        
        # Hurst exponent (lower = more mean reverting)
        if hurst < 0.4:
            prob += 0.2
        elif hurst < 0.5:
            prob += 0.1
        elif hurst > 0.6:
            prob -= 0.2
        
        # Z-score extremes
        zscore = self._calculate_zscore(candles, 20)
        if abs(zscore) > 2:
            prob += 0.15
        elif abs(zscore) > 1.5:
            prob += 0.1
        
        # Failed breakouts
        failed = self._detect_failed_breakouts(candles)
        if failed >= 2:
            prob += 0.1
        
        # Cap probability
        return max(0, min(1, prob))
    
    def _calculate_half_life(self, candles: List) -> float:
        """Calculate mean reversion half-life"""
        if len(candles) < 50:
            return 10.0
            
        # Simplified half-life calculation using autocorrelation
        prices = [c.close for c in candles[-50:]]
        
        # Calculate lag-1 autocorrelation
        mean = sum(prices) / len(prices)
        
        numerator = 0
        denominator = 0
        for i in range(1, len(prices)):
            numerator += (prices[i] - mean) * (prices[i-1] - mean)
            denominator += (prices[i-1] - mean) ** 2
        
        if denominator == 0:
            return 10.0
            
        autocorr = numerator / denominator
        
        # Half-life = -log(2) / log(autocorrelation)
        if autocorr > 0 and autocorr < 1:
            import math
            half_life = -math.log(2) / math.log(autocorr)
            return min(50, max(1, half_life))  # Cap between 1 and 50
        
        return 10.0
    
    def _calculate_percentile_rank(self, candles: List) -> float:
        """Calculate current price percentile rank"""
        if len(candles) < 100:
            return 50.0
            
        prices = [c.close for c in candles[-100:]]
        current = candles[-1].close
        
        # Count how many prices are below current
        below = sum(1 for p in prices if p < current)
        
        return (below / len(prices)) * 100
    
    def _count_consecutive_moves(self, candles: List) -> int:
        """Count consecutive bars in same direction"""
        if len(candles) < 2:
            return 0
            
        count = 1
        direction = 1 if candles[-1].close > candles[-1].open else -1
        
        for i in range(len(candles) - 2, max(0, len(candles) - 20), -1):
            bar_dir = 1 if candles[i].close > candles[i].open else -1
            if bar_dir == direction:
                count += 1
            else:
                break
        
        return count
    
    def _identify_reversal_zones(self, candles: List, vol_profile: Dict) -> List[float]:
        """Identify key reversal zones"""
        zones = []
        
        if len(candles) < 50:
            return zones
            
        # Previous swing highs/lows
        for i in range(10, 40):
            idx = len(candles) - i
            
            # Swing high
            if candles[idx].high > max(c.high for c in candles[idx-5:idx]) and \
               candles[idx].high > max(c.high for c in candles[idx+1:idx+6]):
                zones.append(candles[idx].high)
            
            # Swing low
            if candles[idx].low < min(c.low for c in candles[idx-5:idx]) and \
               candles[idx].low < min(c.low for c in candles[idx+1:idx+6]):
                zones.append(candles[idx].low)
        
        # Add volume-based levels
        if vol_profile and 'poc' in vol_profile:
            zones.append(vol_profile['poc'])
        
        # Sort by distance from current price
        current = candles[-1].close
        zones = list(set(zones))  # Remove duplicates
        zones.sort(key=lambda x: abs(x - current))
        
        return zones[:5]
    
    def _price_vs_ma(self, candles: List, period: int) -> float:
        """Calculate price vs moving average percentage"""
        if len(candles) < period:
            return 0
            
        ma = sum(c.close for c in candles[-period:]) / period
        current = candles[-1].close
        
        return ((current - ma) / ma) * 100 if ma > 0 else 0