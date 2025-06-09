"""
Momentum Trader Agent
Specializes in trend following and momentum strategies
"""

import logging
from typing import Dict, List, Optional, Any

from core.agents.base_agent import TradingAgent, AgentType, AgentAnalysis, DebateResponse
from core.domain.models import MarketData, SignalType
from core.infrastructure.gpt.client import GPTClient

logger = logging.getLogger(__name__)


class MomentumTrader(TradingAgent):
    """The Trend Rider - follows strong momentum and trends"""
    
    def __init__(self, gpt_client: GPTClient):
        super().__init__(
            agent_type=AgentType.MOMENTUM_TRADER,
            personality="aggressive and trend-following momentum trader",
            specialty="breakouts, strong trends, continuation patterns, and momentum acceleration"
        )
        self.gpt_client = gpt_client
    
    def analyze(
        self,
        market_data: Dict[str, MarketData],
        news_context: Optional[List[str]] = None,
        ml_context: Optional[Dict[str, Any]] = None
    ) -> AgentAnalysis:
        """Analyze momentum and trend strength with professional metrics"""
        
        # Check if we're in position trading mode
        is_position_mode = self._is_position_trading_mode(market_data)
        
        if is_position_mode:
            return self._analyze_position_momentum(market_data, news_context, ml_context)
        
        # Standard momentum analysis for scalping/day trading
        h1_data = market_data.get('h1')
        h4_data = market_data.get('h4')
        m15_data = market_data.get('m15')  # For fine-tuned entry
        
        if not h1_data or not h4_data:
            raise ValueError("Both H1 and H4 data required for momentum analysis")
        
        # Import enhanced analyzer
        try:
            from core.agents.enhanced_base_agent import ProfessionalMarketAnalyzer
            analyzer = ProfessionalMarketAnalyzer()
            
            # Calculate professional momentum metrics (need 100+ candles)
            h1_candles = h1_data.candles[-100:] if len(h1_data.candles) >= 100 else h1_data.candles
            h4_candles = h4_data.candles[-50:] if len(h4_data.candles) >= 50 else h4_data.candles
            
            # Order flow for momentum confirmation
            order_flow = analyzer.calculate_order_flow_imbalance(h1_candles[-20:])
            
            # Market regime (trending vs ranging)
            regime = analyzer.calculate_market_regime(h1_candles)
            
            # Volume analysis
            vol_profile = analyzer.calculate_volume_profile(h1_candles[-50:])
            
            use_enhanced = True
        except:
            use_enhanced = False
            order_flow = 0
            regime = {'regime': 'unknown', 'hurst': 0.5}
            vol_profile = {}
        
        # Multi-period momentum calculations
        latest = h1_data.latest_candle
        candles = h1_data.candles
        
        # Calculate ROC for multiple periods
        momentum_5 = self._calculate_roc(candles, 5)
        momentum_10 = self._calculate_roc(candles, 10)
        momentum_20 = self._calculate_roc(candles, 20)
        momentum_50 = self._calculate_roc(candles, 50)
        
        # Momentum acceleration (2nd derivative)
        momentum_accel = momentum_10 - self._calculate_roc(candles[:-10], 10) if len(candles) > 20 else 0
        
        # Force Index (price change * volume)
        force_index = self._calculate_force_index(candles, 13)
        
        # ADX for trend strength
        adx_value = self._estimate_adx(candles, 14)
        
        # Volume-weighted momentum
        vw_momentum = self._calculate_volume_weighted_momentum(candles, 20)
        
        # Trend quality metrics
        h4_trend_bars = self._count_trend_bars(h4_candles)
        trend_consistency = self._calculate_trend_consistency(h1_candles)
        
        analysis_prompt = self._build_prompt("""
Analyze {symbol} for professional momentum trading opportunities.

MOMENTUM METRICS (100 bars):
- ROC(5/10/20/50): {mom5:.2f}% / {mom10:.2f}% / {mom20:.2f}% / {mom50:.2f}%
- Momentum Acceleration: {mom_accel:.3f}%/bar
- Force Index(13): {force_index:.2f}
- Volume-Weighted Momentum: {vw_momentum:.2f}%
- ADX(14): {adx:.1f} (Trend Strength)

MARKET STRUCTURE:
- Current Price: {price:.5f}
- Market Regime: {regime} (Hurst: {hurst:.3f})
- Order Flow Imbalance: {order_flow:.2%}
- Value Area: {val_low:.5f} - {val_high:.5f}
- Point of Control: {poc:.5f}

TREND ANALYSIS:
- H1 Trend Consistency: {trend_consistency:.1%}
- H4 Consecutive Bars: {h4_trend_bars}
- Price vs EMA(20/50/200): {ema20_pos} / {ema50_pos} / {ema200_pos}
- RSI(14): H1={h1_rsi:.1f}, H4={h4_rsi:.1f}

VOLUME ANALYSIS:
- Current vs 20-bar Avg: {vol_ratio:.1%}
- Volume Trend: {vol_trend}
- High Volume Nodes: {high_vol_nodes}

Professional Assessment Required:
1. Momentum quality (smooth acceleration vs choppy)
2. Trend maturity (early/mid/late stage)
3. Volume confirmation of momentum
4. Optimal entry for trend continuation
5. Risk of momentum exhaustion
6. Position sizing based on momentum strength

Provide:
MOMENTUM_STATE: [Strong Buy/Strong Sell/Neutral/Weakening]
TREND_QUALITY: [Excellent/Good/Poor]
ENTRY_STRATEGY: [specific entry method]
RECOMMENDATION: [BUY/SELL/WAIT]
CONFIDENCE: [0-100]
TARGETS: [3 momentum-based targets]
INVALIDATION: [specific level/condition]
POSITION_SIZE: [0.1-1.0 based on momentum quality]

Format your response with exact labels above.
""",
            symbol=h1_data.symbol,
            price=latest.close,
            mom5=momentum_5,
            mom10=momentum_10,
            mom20=momentum_20,
            mom50=momentum_50,
            mom_accel=momentum_accel,
            force_index=force_index,
            vw_momentum=vw_momentum,
            adx=adx_value,
            regime=regime.get('regime', 'unknown'),
            hurst=regime.get('hurst', 0.5),
            order_flow=order_flow,
            val_low=vol_profile.get('value_area_low', latest.close * 0.995),
            val_high=vol_profile.get('value_area_high', latest.close * 1.005),
            poc=vol_profile.get('poc', latest.close),
            trend_consistency=trend_consistency,
            h4_trend_bars=h4_trend_bars,
            ema20_pos="Above" if latest.close > latest.ema50 * 0.996 else "Below",
            ema50_pos="Above" if latest.close > latest.ema50 else "Below",
            ema200_pos="Above" if latest.close > latest.ema200 else "Below",
            h1_rsi=latest.rsi14,
            h4_rsi=h4_data.latest_candle.rsi14,
            vol_ratio=(candles[-1].volume / (sum(c.volume for c in candles[-20:]) / 20) - 1) if len(candles) >= 20 else 0,
            vol_trend=self._analyze_volume_trend(candles),
            high_vol_nodes=self._find_high_volume_nodes(vol_profile)
        )
        
        response = self.gpt_client.analyze_with_response(
            analysis_prompt,
            temperature=0.5,
            agent_type=self.agent_type.value,
            symbol=h1_data.symbol
        )
        
        return self._parse_analysis(response, h1_data.latest_candle.close)
    
    def debate(
        self,
        other_analyses: List[AgentAnalysis],
        round_number: int,
        previous_responses: Optional[List[DebateResponse]] = None
    ) -> DebateResponse:
        """Participate in debate with momentum arguments"""
        
        my_position = self.analysis_history[-1]
        
        if round_number == 1:
            statement = self._generate_opening_statement(my_position)
        elif round_number == 2:
            # Address contrarian if they oppose
            contrarian = next((a for a in other_analyses if a.agent_type == AgentType.CONTRARIAN_TRADER), None)
            risk_manager = next((a for a in other_analyses if a.agent_type == AgentType.RISK_MANAGER), None)
            
            if contrarian and contrarian.recommendation != my_position.recommendation:
                statement = "The contrarian may fade this move, but the trend is your friend until it ends. Momentum is clearly present."
            elif risk_manager and risk_manager.recommendation == SignalType.WAIT:
                statement = "Risk concerns are valid, but strong trends offer the best risk/reward when properly managed. We can use tight stops."
            else:
                statement = self._reinforce_momentum_case(other_analyses)
        else:
            statement = self._generate_closing_statement(other_analyses)
        
        # Momentum trader may reduce confidence if technical structure is poor
        updated_confidence = my_position.confidence
        tech_analysis = next((a for a in other_analyses if a.agent_type == AgentType.TECHNICAL_ANALYST), None)
        
        if tech_analysis and tech_analysis.recommendation != my_position.recommendation:
            updated_confidence = max(40, my_position.confidence - 15)
        
        return DebateResponse(
            agent_type=self.agent_type,
            round=round_number,
            statement=statement,
            maintains_position=True,  # Momentum traders stick to trends
            updated_confidence=updated_confidence
        )
    
    def _count_trend_bars(self, candles: List) -> int:
        """Count consecutive bars in same direction"""
        if len(candles) < 2:
            return 0
        
        count = 1
        direction = 1 if candles[-1].close > candles[-1].open else -1
        
        for i in range(len(candles) - 2, -1, -1):
            candle_dir = 1 if candles[i].close > candles[i].open else -1
            if candle_dir == direction:
                count += 1
            else:
                break
        
        return count * direction  # Positive for bullish, negative for bearish
    
    def _parse_analysis(self, response: str, current_price: float) -> AgentAnalysis:
        """Parse GPT response into AgentAnalysis using safe parsing"""
        try:
            # Use the safe parsing method from base class
            parsed_data = self._safe_parse_response(response, current_price)
            
            # Extract momentum-specific fields
            momentum_state = parsed_data.get('metadata', {}).get('MOMENTUM_STATE', 'Neutral')
            trend_quality = parsed_data.get('metadata', {}).get('TREND_QUALITY', 'Poor')
            position_size = parsed_data.get('metadata', {}).get('POSITION_SIZE', 0.1)
            
            # Add position size to metadata
            parsed_data['metadata']['position_size'] = float(position_size) if isinstance(position_size, str) else position_size
            
            # Build comprehensive reasoning
            reasoning = [f"Momentum: {momentum_state}", f"Trend quality: {trend_quality}"]
            if parsed_data.get('metadata', {}).get('ENTRY_STRATEGY'):
                reasoning.append(f"Entry: {parsed_data['metadata']['ENTRY_STRATEGY']}")
            
            analysis = AgentAnalysis(
                agent_type=self.agent_type,
                recommendation=parsed_data['recommendation'],
                confidence=parsed_data['confidence'],
                reasoning=reasoning,
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
            logger.error(f"Failed to parse momentum analysis: {e}")
            default_analysis = AgentAnalysis(
                agent_type=self.agent_type,
                recommendation=SignalType.WAIT,
                confidence=50.0,
                reasoning=["Error parsing momentum analysis"],
                concerns=["Analysis parsing failed"],
                entry_price=current_price,
                metadata={'error': str(e)}
            )
            self.analysis_history.append(default_analysis)
            return default_analysis
    
    def _calculate_roc(self, candles: List, period: int) -> float:
        """Calculate Rate of Change"""
        if len(candles) < period + 1:
            return 0
        current = candles[-1].close
        past = candles[-period-1].close
        return ((current - past) / past) * 100 if past > 0 else 0
    
    def _calculate_force_index(self, candles: List, period: int) -> float:
        """Calculate Force Index (price change * volume)"""
        if len(candles) < period + 1:
            return 0
        
        force_values = []
        for i in range(len(candles) - period, len(candles)):
            if i > 0:
                price_change = candles[i].close - candles[i-1].close
                force = price_change * candles[i].volume
                force_values.append(force)
        
        return sum(force_values) / len(force_values) if force_values else 0
    
    def _estimate_adx(self, candles: List, period: int = 14) -> float:
        """Estimate ADX (simplified)"""
        if len(candles) < period * 2:
            return 0
        
        # Simplified ADX calculation
        plus_dm = []
        minus_dm = []
        tr_values = []
        
        for i in range(1, len(candles)):
            high_diff = candles[i].high - candles[i-1].high
            low_diff = candles[i-1].low - candles[i].low
            
            plus_dm.append(high_diff if high_diff > 0 and high_diff > low_diff else 0)
            minus_dm.append(low_diff if low_diff > 0 and low_diff > high_diff else 0)
            
            tr = max(
                candles[i].high - candles[i].low,
                abs(candles[i].high - candles[i-1].close),
                abs(candles[i].low - candles[i-1].close)
            )
            tr_values.append(tr)
        
        if len(tr_values) < period:
            return 0
            
        # Smooth the values
        avg_plus = sum(plus_dm[-period:]) / period
        avg_minus = sum(minus_dm[-period:]) / period
        avg_tr = sum(tr_values[-period:]) / period
        
        if avg_tr == 0:
            return 0
            
        plus_di = (avg_plus / avg_tr) * 100
        minus_di = (avg_minus / avg_tr) * 100
        
        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 0
            
        dx = abs(plus_di - minus_di) / di_sum * 100
        return min(100, dx)  # Cap at 100
    
    def _calculate_volume_weighted_momentum(self, candles: List, period: int) -> float:
        """Calculate volume-weighted momentum"""
        if len(candles) < period + 1:
            return 0
            
        total_volume = sum(c.volume for c in candles[-period:])
        if total_volume == 0:
            return 0
            
        weighted_sum = 0
        for i in range(len(candles) - period, len(candles)):
            price_change = (candles[i].close - candles[i-period].close) / candles[i-period].close
            weight = candles[i].volume / total_volume
            weighted_sum += price_change * weight
            
        return weighted_sum * 100
    
    def _calculate_trend_consistency(self, candles: List) -> float:
        """Calculate how consistent the trend is (0-1)"""
        if len(candles) < 20:
            return 0
            
        # Count bars moving in trend direction
        trend_bars = 0
        total_bars = min(50, len(candles) - 1)
        
        # Determine overall trend
        trend_direction = 1 if candles[-1].close > candles[-total_bars-1].close else -1
        
        for i in range(len(candles) - total_bars, len(candles)):
            bar_direction = 1 if candles[i].close > candles[i-1].close else -1
            if bar_direction == trend_direction:
                trend_bars += 1
                
        return trend_bars / total_bars
    
    def _analyze_volume_trend(self, candles: List) -> str:
        """Analyze volume trend"""
        if len(candles) < 20:
            return "Unknown"
            
        recent_vol = sum(c.volume for c in candles[-5:]) / 5
        older_vol = sum(c.volume for c in candles[-20:-10]) / 10
        
        if recent_vol > older_vol * 1.2:
            return "Increasing"
        elif recent_vol < older_vol * 0.8:
            return "Decreasing"
        else:
            return "Stable"
    
    def _find_high_volume_nodes(self, vol_profile: Dict) -> str:
        """Find high volume price levels"""
        if not vol_profile or 'profile' not in vol_profile:
            return "None identified"
            
        profile = vol_profile['profile']
        if not profile:
            return "None identified"
            
        # Sort by volume
        sorted_levels = sorted(profile.items(), key=lambda x: x[1], reverse=True)
        
        # Get top 3 levels
        top_levels = sorted_levels[:3]
        if not top_levels:
            return "None identified"
            
        return ", ".join([f"{level:.5f}" for level, _ in top_levels])
    
    def _generate_opening_statement(self, analysis: AgentAnalysis) -> str:
        """Generate opening debate statement"""
        momentum = analysis.metadata.get('momentum_state', 'Neutral')
        quality = analysis.metadata.get('trend_quality', 'Poor')
        
        if analysis.recommendation != SignalType.WAIT:
            return f"{momentum} momentum detected with {quality} trend quality. This is a clear opportunity to ride the trend. {analysis.reasoning[2]}"
        else:
            return "No clear momentum present. As a momentum trader, I prefer to wait for strong directional moves."
    
    def _reinforce_momentum_case(self, analyses: List[AgentAnalysis]) -> str:
        """Reinforce the momentum argument"""
        my_rec = self.analysis_history[-1].recommendation
        if my_rec != SignalType.WAIT:
            return "The momentum is undeniable here. Missing this move would be leaving money on the table. We can manage risk with trailing stops."
        else:
            return "Without clear momentum, there's no edge. Patience until a strong trend emerges."
    
    def _generate_closing_statement(self, all_analyses: List[AgentAnalysis]) -> str:
        """Generate closing statement"""
        my_position = self.analysis_history[-1]
        
        if my_position.recommendation != SignalType.WAIT:
            targets = my_position.metadata.get('targets', 'momentum targets')
            return f"Strong trends don't come often. When momentum aligns, we must act. Target: {targets}."
        else:
            return "No momentum, no trade. Waiting for the next strong directional move is the disciplined approach."
    
    def _is_position_trading_mode(self, market_data: Dict[str, MarketData]) -> bool:
        """Detect if using daily/weekly timeframes for position trading"""
        return 'd1' in market_data or 'w1' in market_data
    
    def _analyze_position_momentum(self, market_data: Dict[str, MarketData],
                                  news_context: Optional[List[str]],
                                  ml_context: Optional[Dict[str, Any]]) -> AgentAnalysis:
        """Analyze momentum for position trading on daily/weekly timeframes"""
        
        d1_data = market_data.get('d1')
        w1_data = market_data.get('w1')
        h4_data = market_data.get('h4')  # For refined entry timing
        
        if not d1_data:
            raise ValueError("Daily data required for position momentum analysis")
        
        # Import enhanced analyzer
        try:
            from core.agents.enhanced_base_agent import ProfessionalMarketAnalyzer
            analyzer = ProfessionalMarketAnalyzer()
            
            # Use longer history for position trading
            d1_candles = d1_data.candles[-250:] if len(d1_data.candles) >= 250 else d1_data.candles
            w1_candles = w1_data.candles[-52:] if w1_data and len(w1_data.candles) >= 52 else []
            
            # Calculate long-term momentum metrics
            order_flow = analyzer.calculate_order_flow_imbalance(d1_candles[-20:])
            regime = analyzer.calculate_market_regime(d1_candles)
            vol_profile = analyzer.calculate_volume_profile(d1_candles[-100:])
            
            use_enhanced = True
        except:
            use_enhanced = False
            order_flow = 0
            regime = {'regime': 'unknown', 'hurst': 0.5}
            vol_profile = {}
        
        latest = d1_data.latest_candle
        candles = d1_data.candles
        
        # Multi-period momentum for position trading
        momentum_5d = self._calculate_roc(candles, 5)
        momentum_20d = self._calculate_roc(candles, 20)
        momentum_50d = self._calculate_roc(candles, 50)
        momentum_100d = self._calculate_roc(candles, 100)
        
        # Weekly momentum if available
        weekly_momentum = self._calculate_weekly_momentum(w1_data) if w1_data else {'4w': 0, '12w': 0, '26w': 0}
        
        # Long-term trend strength
        daily_adx = self._estimate_adx(candles, 14)
        trend_duration = self._calculate_trend_duration(candles)
        trend_quality = self._assess_position_trend_quality(candles)
        
        # Momentum divergence analysis
        divergence = self._check_momentum_divergence(candles)
        
        # Volume confirmation for position trades
        volume_trend = self._analyze_position_volume_trend(candles)
        
        analysis_prompt = self._build_prompt("""
Analyze {symbol} for POSITION TRADING momentum opportunities (multi-day/week holding).

DAILY MOMENTUM METRICS (250 bars):
- ROC(5d/20d/50d/100d): {mom5d:.2f}% / {mom20d:.2f}% / {mom50d:.2f}% / {mom100d:.2f}%
- ADX(14): {adx:.1f} (Trend Strength)
- Trend Duration: {trend_duration} days
- Trend Quality: {trend_quality}

WEEKLY MOMENTUM:
- 4-week: {w4_mom:.2f}%
- 12-week: {w12_mom:.2f}%
- 26-week: {w26_mom:.2f}%

MARKET STRUCTURE:
- Current Price: {price:.5f}
- Market Regime: {regime} (Hurst: {hurst:.3f})
- Order Flow (20d): {order_flow:.2%}
- Volume Trend: {volume_trend}

POSITION INDICATORS:
- Price vs EMA(20/50/200): {ema20_pos} / {ema50_pos} / {ema200_pos}
- Daily RSI(14): {daily_rsi:.1f}
- Weekly RSI(14): {weekly_rsi:.1f}
- Momentum Divergence: {divergence}

TREND CONTINUATION ANALYSIS:
- 50-day High: {high_50d:.5f} ({high_50d_dist:.2%} away)
- 50-day Low: {low_50d:.5f} ({low_50d_dist:.2%} away)
- Breakout Status: {breakout_status}

Position Trading Assessment Required:
1. Multi-timeframe momentum alignment
2. Trend maturity (early/middle/late stage)
3. Probability of multi-week continuation
4. Optimal position entry strategy
5. Risk of major trend reversal
6. Position sizing for extended holding

Provide:
MOMENTUM_STATE: [Accelerating/Strong/Weakening/Reversing]
TREND_STAGE: [Early/Middle/Late/Exhaustion]
CONTINUATION_PROBABILITY: [High/Medium/Low]
ENTRY_STRATEGY: [Breakout/Pullback/Accumulation]
RECOMMENDATION: [BUY/SELL/WAIT]
CONFIDENCE: [0-100]
POSITION_TARGETS: [3 targets with timeframes]
TRAILING_STOP: [Method and initial level]
TIME_STOP: [Maximum holding period]

Format your response with exact labels above.
""",
            symbol=d1_data.symbol,
            price=latest.close,
            mom5d=momentum_5d,
            mom20d=momentum_20d,
            mom50d=momentum_50d,
            mom100d=momentum_100d,
            w4_mom=weekly_momentum['4w'],
            w12_mom=weekly_momentum['12w'],
            w26_mom=weekly_momentum['26w'],
            adx=daily_adx,
            trend_duration=trend_duration,
            trend_quality=trend_quality,
            regime=regime.get('regime', 'unknown'),
            hurst=regime.get('hurst', 0.5),
            order_flow=order_flow,
            volume_trend=volume_trend,
            ema20_pos="Above" if latest.close > self._calculate_ema(candles, 20) else "Below",
            ema50_pos="Above" if latest.close > latest.ema50 else "Below",
            ema200_pos="Above" if latest.close > latest.ema200 else "Below",
            daily_rsi=latest.rsi14,
            weekly_rsi=w1_data.latest_candle.rsi14 if w1_data else 50,
            divergence=divergence,
            high_50d=max(c.high for c in candles[-50:]) if len(candles) >= 50 else latest.high,
            high_50d_dist=((max(c.high for c in candles[-50:]) - latest.close) / latest.close * 100) if len(candles) >= 50 else 0,
            low_50d=min(c.low for c in candles[-50:]) if len(candles) >= 50 else latest.low,
            low_50d_dist=((latest.close - min(c.low for c in candles[-50:])) / latest.close * 100) if len(candles) >= 50 else 0,
            breakout_status=self._check_breakout_status(candles)
        )
        
        response = self.gpt_client.analyze_with_response(
            analysis_prompt,
            temperature=0.4,  # Slightly lower for position trading
            agent_type=self.agent_type.value,
            symbol=d1_data.symbol
        )
        
        return self._parse_position_momentum(response, latest.close)
    
    def _calculate_weekly_momentum(self, weekly_data: Optional[MarketData]) -> Dict[str, float]:
        """Calculate weekly timeframe momentum"""
        if not weekly_data or len(weekly_data.candles) < 26:
            return {'4w': 0, '12w': 0, '26w': 0}
        
        candles = weekly_data.candles
        current = candles[-1].close
        
        mom_4w = ((current - candles[-5].close) / candles[-5].close * 100) if len(candles) >= 5 else 0
        mom_12w = ((current - candles[-13].close) / candles[-13].close * 100) if len(candles) >= 13 else 0
        mom_26w = ((current - candles[-27].close) / candles[-27].close * 100) if len(candles) >= 27 else 0
        
        return {'4w': mom_4w, '12w': mom_12w, '26w': mom_26w}
    
    def _calculate_trend_duration(self, candles: List) -> int:
        """Calculate how many days the current trend has lasted"""
        if len(candles) < 20:
            return 0
        
        # Find when trend started (last major reversal)
        trend_days = 0
        current_trend = 1 if candles[-1].close > candles[-20].close else -1
        
        for i in range(len(candles) - 2, max(0, len(candles) - 100), -1):
            # Check if trend continues
            if current_trend > 0:
                if candles[i].close > candles[max(0, i-20)].close:
                    trend_days += 1
                else:
                    break
            else:
                if candles[i].close < candles[max(0, i-20)].close:
                    trend_days += 1
                else:
                    break
        
        return trend_days
    
    def _assess_position_trend_quality(self, candles: List) -> str:
        """Assess trend quality for position trading"""
        if len(candles) < 50:
            return "Unknown"
        
        # Calculate trend consistency over different periods
        consistency_20d = self._calculate_trend_consistency(candles[-20:])
        consistency_50d = self._calculate_trend_consistency(candles[-50:])
        
        # Check for smooth vs choppy trend
        atr_changes = []
        for i in range(len(candles) - 20, len(candles)):
            if i > 0 and candles[i].atr14 and candles[i-1].atr14:
                atr_changes.append(abs(candles[i].atr14 - candles[i-1].atr14) / candles[i-1].atr14)
        
        volatility_stability = sum(atr_changes) / len(atr_changes) if atr_changes else 0
        
        # Assess quality
        if consistency_50d > 0.7 and volatility_stability < 0.1:
            return "Excellent - Smooth and consistent"
        elif consistency_50d > 0.6:
            return "Good - Mostly consistent"
        elif consistency_50d > 0.5:
            return "Fair - Some choppiness"
        else:
            return "Poor - Choppy/ranging"
    
    def _check_momentum_divergence(self, candles: List) -> str:
        """Check for momentum divergence on daily timeframe"""
        if len(candles) < 50:
            return "No divergence"
        
        # Find recent swing highs/lows
        price_highs = []
        price_lows = []
        rsi_at_highs = []
        rsi_at_lows = []
        
        for i in range(10, len(candles) - 10):
            # Swing high
            if candles[i].high == max(c.high for c in candles[i-10:i+11]):
                price_highs.append((i, candles[i].high))
                rsi_at_highs.append(candles[i].rsi14)
            
            # Swing low
            if candles[i].low == min(c.low for c in candles[i-10:i+11]):
                price_lows.append((i, candles[i].low))
                rsi_at_lows.append(candles[i].rsi14)
        
        # Check for divergence
        if len(price_highs) >= 2 and len(rsi_at_highs) >= 2:
            if price_highs[-1][1] > price_highs[-2][1] and rsi_at_highs[-1] < rsi_at_highs[-2]:
                return "Bearish divergence detected"
        
        if len(price_lows) >= 2 and len(rsi_at_lows) >= 2:
            if price_lows[-1][1] < price_lows[-2][1] and rsi_at_lows[-1] > rsi_at_lows[-2]:
                return "Bullish divergence detected"
        
        return "No divergence"
    
    def _analyze_position_volume_trend(self, candles: List) -> str:
        """Analyze volume trend for position trading"""
        if len(candles) < 50:
            return "Unknown"
        
        # Compare recent vs historical volume
        recent_vol = sum(c.volume for c in candles[-10:]) / 10
        historical_vol = sum(c.volume for c in candles[-50:-10]) / 40
        
        # Check if volume confirms price trend
        price_trend = 1 if candles[-1].close > candles[-20].close else -1
        
        # Volume should increase in trend direction
        bullish_volume = sum(c.volume for c in candles[-20:] if c.close > c.open)
        bearish_volume = sum(c.volume for c in candles[-20:] if c.close < c.open)
        
        volume_ratio = recent_vol / historical_vol if historical_vol > 0 else 1
        
        if volume_ratio > 1.5 and price_trend > 0 and bullish_volume > bearish_volume:
            return "Strong bullish volume"
        elif volume_ratio > 1.5 and price_trend < 0 and bearish_volume > bullish_volume:
            return "Strong bearish volume"
        elif volume_ratio > 1.2:
            return "Increasing volume"
        elif volume_ratio < 0.8:
            return "Declining volume (caution)"
        else:
            return "Normal volume"
    
    def _check_breakout_status(self, candles: List) -> str:
        """Check breakout status for position trading"""
        if len(candles) < 50:
            return "Unknown"
        
        current = candles[-1].close
        high_20d = max(c.high for c in candles[-20:])
        low_20d = min(c.low for c in candles[-20:])
        high_50d = max(c.high for c in candles[-50:])
        low_50d = min(c.low for c in candles[-50:])
        
        if current > high_50d:
            return "Breaking 50-day highs"
        elif current > high_20d and current > high_50d * 0.98:
            return "Approaching 50-day highs"
        elif current < low_50d:
            return "Breaking 50-day lows"
        elif current < low_20d and current < low_50d * 1.02:
            return "Approaching 50-day lows"
        else:
            return "Within range"
    
    def _calculate_ema(self, candles: List, period: int) -> float:
        """Calculate EMA manually"""
        if len(candles) < period:
            return candles[-1].close
        
        # Simple approximation using SMA
        return sum(c.close for c in candles[-period:]) / period
    
    def _parse_position_momentum(self, response: str, current_price: float) -> AgentAnalysis:
        """Parse position momentum trading response"""
        try:
            # Use the safe parsing method from base class
            parsed_data = self._safe_parse_response(response, current_price)
            raw_parsed = parsed_data['metadata'].get('raw_parsed', {})
            
            # Extract position-specific fields
            momentum_state = raw_parsed.get('MOMENTUM_STATE', 'Unknown')
            trend_stage = raw_parsed.get('TREND_STAGE', 'Unknown')
            continuation_prob = raw_parsed.get('CONTINUATION_PROBABILITY', 'Low')
            entry_strategy = raw_parsed.get('ENTRY_STRATEGY', 'Market')
            
            # Parse position targets
            targets_str = raw_parsed.get('POSITION_TARGETS', '')
            targets = []
            if targets_str:
                import re
                # Match patterns like "1.2345 (10 days)"
                matches = re.findall(r'([\d.]+)\s*\(([^)]+)\)', targets_str)
                for price, timeframe in matches:
                    targets.append({'price': float(price), 'timeframe': timeframe})
            
            # Update metadata
            parsed_data['metadata'].update({
                'momentum_state': momentum_state,
                'trend_stage': trend_stage,
                'continuation_probability': continuation_prob,
                'entry_strategy': entry_strategy,
                'trailing_stop': raw_parsed.get('TRAILING_STOP', 'ATR-based'),
                'time_stop': raw_parsed.get('TIME_STOP', '4 weeks'),
                'position_targets': targets,
                'position_mode': True
            })
            
            # Build reasoning
            reasoning = [
                f"Position momentum: {momentum_state}",
                f"Trend stage: {trend_stage}",
                f"Continuation probability: {continuation_prob}"
            ]
            
            # Adjust confidence based on trend stage
            if trend_stage == 'Late' or trend_stage == 'Exhaustion':
                parsed_data['confidence'] = max(40, parsed_data['confidence'] - 20)
            
            analysis = AgentAnalysis(
                agent_type=self.agent_type,
                recommendation=parsed_data['recommendation'],
                confidence=parsed_data['confidence'],
                reasoning=reasoning,
                concerns=parsed_data.get('concerns', []),
                entry_price=parsed_data['entry'],
                stop_loss=parsed_data['stop_loss'],
                take_profit=targets[0]['price'] if targets else parsed_data['take_profit'],
                risk_reward_ratio=self._calculate_risk_reward(
                    parsed_data['entry'],
                    parsed_data['stop_loss'],
                    targets[0]['price'] if targets else parsed_data['take_profit']
                ) if parsed_data['stop_loss'] and (targets or parsed_data['take_profit']) else None,
                metadata=parsed_data['metadata']
            )
            
            self.analysis_history.append(analysis)
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to parse position momentum analysis: {e}")
            default_analysis = AgentAnalysis(
                agent_type=self.agent_type,
                recommendation=SignalType.WAIT,
                confidence=50.0,
                reasoning=["Error parsing position momentum analysis"],
                concerns=["Analysis parsing failed"],
                entry_price=current_price,
                metadata={'error': str(e), 'position_mode': True}
            )
            self.analysis_history.append(default_analysis)
            return default_analysis