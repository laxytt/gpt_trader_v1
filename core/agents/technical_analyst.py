"""
Technical Analyst Agent
Specializes in chart patterns, indicators, and price action
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from core.agents.base_agent import TradingAgent, AgentType, AgentAnalysis, DebateResponse
from core.domain.models import MarketData, SignalType
from core.infrastructure.gpt.client import GPTClient

logger = logging.getLogger(__name__)


class TechnicalAnalyst(TradingAgent):
    """The Chartist - focuses on technical analysis"""
    
    def __init__(self, gpt_client: GPTClient):
        super().__init__(
            agent_type=AgentType.TECHNICAL_ANALYST,
            personality="methodical and pattern-focused chartist",
            specialty="price action, support/resistance, technical indicators, and chart patterns"
        )
        self.gpt_client = gpt_client
    
    def analyze(
        self,
        market_data: Dict[str, MarketData],
        news_context: Optional[List[str]] = None,
        ml_context: Optional[Dict[str, Any]] = None
    ) -> AgentAnalysis:
        """Perform technical analysis"""
        
        # Check if we're in position trading mode
        is_position_mode = self._is_position_trading_mode(market_data)
        
        if is_position_mode:
            return self._analyze_position_trading(market_data, news_context, ml_context)
        
        # Standard scalping/day trading analysis
        h1_data = market_data.get('h1')
        h4_data = market_data.get('h4')
        
        if not h1_data or not h4_data:
            raise ValueError("Both H1 and H4 data required for technical analysis")
        
        # Extract key technical data
        h1_latest = h1_data.latest_candle
        h4_latest = h4_data.latest_candle
        symbol = h1_data.symbol
        
        # Import enhanced analyzer
        try:
            from core.agents.enhanced_base_agent import ProfessionalMarketAnalyzer, format_enhanced_prompt
            
            # Calculate professional metrics
            analyzer = ProfessionalMarketAnalyzer()
            vwap, vwap_dev = analyzer.calculate_vwap(h1_data.candles[-50:])
            vol_profile = analyzer.calculate_volume_profile(h1_data.candles[-100:])
            order_flow = analyzer.calculate_order_flow_imbalance(h1_data.candles[-20:])
            regime = analyzer.calculate_market_regime(h1_data.candles)
            
            use_enhanced = True
        except:
            use_enhanced = False
            vwap = vwap_dev = 0
            vol_profile = {'poc': 0, 'value_area_high': 0, 'value_area_low': 0}
            order_flow = 0
            regime = {'regime': 'unknown', 'hurst': 0.5}
        
        analysis_prompt = self._build_prompt("""
Analyze the {symbol} chart from a pure technical perspective.

MARKET STRUCTURE (100 bars):
- Current Price: {h1_price}
- VWAP: {vwap:.5f} (Deviation: {vwap_dev:.2f}%)
- Point of Control: {poc:.5f}
- Value Area: {val_low:.5f} - {val_high:.5f}
- Market Regime: {regime} (Hurst: {hurst:.3f})
- Order Flow Imbalance: {order_flow:.2%}

H1 TECHNICAL INDICATORS:
- EMA20: {h1_ema20}, EMA50: {h1_ema50}, EMA200: {h1_ema200}
- RSI(14): {h1_rsi} 
- ATR(14): {h1_atr} ({atr_pips:.1f} pips)
- 20-bar High: {h1_high}, Low: {h1_low}
- Momentum (10/20): {mom10:.2f}% / {mom20:.2f}%

H4 TREND CONTEXT:
- Price: {h4_price}
- EMA50: {h4_ema50}, EMA200: {h4_ema200}
- RSI: {h4_rsi}
- Trend: {h4_trend}

RECENT PRICE ACTION (Last 10 bars):
{h1_recent_candles}

Pattern Analysis: {h1_price_action}

Provide:
1. Chart pattern identification (if any)
2. Key support and resistance levels
3. Indicator confluence analysis
4. Trend structure assessment
5. Your trading recommendation (BUY/SELL/WAIT)
6. Confidence level (0-100)
7. Specific entry, stop loss, and take profit levels
8. Main technical concern

Format your response as:
PATTERN: [identified pattern or "None clear"]
KEY_LEVELS: [support: X.XXXX, resistance: X.XXXX]
INDICATORS: [summary of indicator signals]
TREND: [trend assessment]
RECOMMENDATION: [BUY/SELL/WAIT]
CONFIDENCE: [0-100]
ENTRY: [price]
STOP_LOSS: [price]
TAKE_PROFIT: [price]
CONCERN: [main technical concern]
""",
            symbol=h1_data.symbol,
            h1_price=h1_latest.close,
            h1_ema50=h1_latest.ema50,
            h1_ema200=h1_latest.ema200,
            h1_rsi=h1_latest.rsi14,
            h1_atr=h1_latest.atr14,
            h1_high=max(c.high for c in h1_data.candles[-20:]),
            h1_low=min(c.low for c in h1_data.candles[-20:]),
            h4_price=h4_latest.close,
            h4_ema50=h4_latest.ema50,
            h4_ema200=h4_latest.ema200,
            h4_rsi=h4_latest.rsi14,
            h4_trend="Bullish" if h4_latest.close > h4_latest.ema50 > h4_latest.ema200 else "Bearish" if h4_latest.close < h4_latest.ema50 < h4_latest.ema200 else "Neutral",
            h1_recent_candles=self._format_recent_candles(h1_data.candles[-10:]),  # Show 10 candles
            h1_price_action=self._analyze_price_action(h1_data.candles[-20:]),
            h4_recent_candles=self._format_recent_candles(h4_data.candles[-3:]),
            # New professional metrics
            vwap=vwap,
            vwap_dev=vwap_dev,
            poc=vol_profile.get('poc', h1_latest.close),
            val_low=vol_profile.get('value_area_low', h1_latest.close * 0.995),
            val_high=vol_profile.get('value_area_high', h1_latest.close * 1.005),
            regime=regime['regime'],
            hurst=regime.get('hurst', 0.5),
            order_flow=order_flow,
            h1_ema20=h1_latest.ema50 * 0.996,  # Approximate for now
            atr_pips=h1_latest.atr14 * 10000,
            mom10=((h1_latest.close - h1_data.candles[-10].close) / h1_data.candles[-10].close * 100) if len(h1_data.candles) >= 10 else 0,
            mom20=((h1_latest.close - h1_data.candles[-20].close) / h1_data.candles[-20].close * 100) if len(h1_data.candles) >= 20 else 0
        )
        
        # Get GPT analysis
        response = self.gpt_client.analyze_with_response(
            analysis_prompt,
            temperature=0.3,  # Lower temperature for technical analysis
            agent_type=self.agent_type.value,
            symbol=symbol
        )
        
        # Parse response
        return self._parse_analysis(response, h1_latest.close)
    
    def debate(
        self,
        other_analyses: List[AgentAnalysis],
        round_number: int,
        previous_responses: Optional[List[DebateResponse]] = None
    ) -> DebateResponse:
        """Participate in debate with technical arguments"""
        
        # Find opposing views
        opposing_views = [
            a for a in other_analyses 
            if a.recommendation != self.analysis_history[-1].recommendation
            and a.agent_type != self.agent_type
        ]
        
        if round_number == 1:
            # Initial position
            statement = self._generate_opening_statement(self.analysis_history[-1])
        elif round_number == 2 and opposing_views:
            # Counter-argument
            statement = self._generate_counter_argument(opposing_views[0])
        else:
            # Final position
            statement = self._generate_closing_statement(other_analyses)
        
        maintains_position = True  # Technical analyst rarely changes based on non-technical factors
        
        return DebateResponse(
            agent_type=self.agent_type,
            round=round_number,
            statement=statement,
            maintains_position=maintains_position,
            updated_confidence=self.analysis_history[-1].confidence
        )
    
    def _parse_analysis(self, response: str, current_price: float) -> AgentAnalysis:
        """Parse GPT response into AgentAnalysis using safe parsing"""
        try:
            # Log the raw response for debugging
            logger.debug(f"Technical Analyst raw response: {response[:200]}...")
            
            # Use the safe parsing method from base class
            parsed_data = self._safe_parse_response(response, current_price)
            
            # Extract concerns
            concerns = []
            if parsed_data.get('concern'):
                concerns.append(parsed_data['concern'])
            
            # Create analysis object
            analysis = AgentAnalysis(
                agent_type=self.agent_type,
                recommendation=parsed_data['recommendation'],
                confidence=parsed_data['confidence'],
                reasoning=parsed_data['reasoning'] or ["Technical analysis completed"],
                concerns=concerns,
                entry_price=parsed_data['entry'],
                stop_loss=parsed_data['stop_loss'],
                take_profit=parsed_data['take_profit'],
                risk_reward_ratio=self._calculate_risk_reward(
                    parsed_data['entry'], 
                    parsed_data['stop_loss'], 
                    parsed_data['take_profit']
                ) if parsed_data['stop_loss'] and parsed_data['take_profit'] else None,
                metadata=parsed_data['metadata']
            )
            
            self.analysis_history.append(analysis)
            return analysis
            
        except Exception as e:
            # If all else fails, return a safe default
            default_analysis = AgentAnalysis(
                agent_type=self.agent_type,
                recommendation=SignalType.WAIT,
                confidence=50.0,
                reasoning=["Error parsing analysis - defaulting to WAIT"],
                concerns=["Technical analysis parsing failed"],
                entry_price=current_price,
                stop_loss=0,
                take_profit=0,
                risk_reward_ratio=None,
                metadata={'error': str(e)}
            )
            self.analysis_history.append(default_analysis)
            return default_analysis
    
    def _generate_opening_statement(self, analysis: AgentAnalysis) -> str:
        """Generate opening debate statement"""
        key_point = analysis.reasoning[0] if analysis.reasoning else "Technical setup present"
        return f"The charts clearly show a {analysis.recommendation.value} opportunity. {key_point}. My confidence is {analysis.confidence}% based on pure technical factors."
    
    def _generate_counter_argument(self, opposing: AgentAnalysis) -> str:
        """Generate counter-argument"""
        if opposing.agent_type == AgentType.FUNDAMENTAL_ANALYST:
            return "While fundamentals matter long-term, the technical setup is clear right now. Price action doesn't lie."
        elif opposing.agent_type == AgentType.RISK_MANAGER:
            return f"The risk is well-defined with stops at {self.analysis_history[-1].stop_loss}. Technical levels provide clear invalidation."
        else:
            return "The technical picture overrides other concerns in this timeframe. Charts reflect all known information."
    
    def _generate_closing_statement(self, all_analyses: List[AgentAnalysis]) -> str:
        """Generate closing statement"""
        agreement_count = sum(1 for a in all_analyses if a.recommendation == self.analysis_history[-1].recommendation)
        if agreement_count >= 4:
            return "The technical analysis aligns with the majority view. This strengthens the setup."
        else:
            return "Despite differing opinions, the technical setup remains valid. I maintain my position."
    
    def _format_recent_candles(self, candles) -> str:
        """Format recent candles for analysis"""
        if not candles:
            return "No candle data"
        
        formatted = []
        for candle in candles:
            direction = "🟢" if candle.close > candle.open else "🔴"
            formatted.append(f"{direction} O:{candle.open:.5f} H:{candle.high:.5f} L:{candle.low:.5f} C:{candle.close:.5f}")
        
        return " | ".join(formatted)
    
    def _analyze_price_action(self, candles) -> str:
        """Analyze recent price action"""
        if len(candles) < 2:
            return "Insufficient data"
        
        # Count bullish vs bearish candles
        bullish = sum(1 for c in candles if c.close > c.open)
        bearish = len(candles) - bullish
        
        # Check for patterns
        last_3 = candles[-3:]
        if all(c.close > c.open for c in last_3):
            pattern = "Strong bullish momentum (3 green candles)"
        elif all(c.close < c.open for c in last_3):
            pattern = "Strong bearish momentum (3 red candles)"
        else:
            pattern = f"Mixed action ({bullish} bullish/{bearish} bearish)"
        
        return pattern
    
    def _is_position_trading_mode(self, market_data: Dict[str, MarketData]) -> bool:
        """Detect if using daily/weekly timeframes for position trading"""
        return 'd1' in market_data or 'w1' in market_data
    
    def _analyze_position_trading(self, market_data: Dict[str, MarketData], 
                                  news_context: Optional[List[str]], 
                                  ml_context: Optional[Dict[str, Any]]) -> AgentAnalysis:
        """Analyze for position trading on daily/weekly timeframes"""
        
        d1_data = market_data.get('d1')
        w1_data = market_data.get('w1')
        h4_data = market_data.get('h4')  # For refined entry
        
        if not d1_data:
            raise ValueError("Daily data required for position trading analysis")
        
        # Import enhanced analyzer
        try:
            from core.agents.enhanced_base_agent import ProfessionalMarketAnalyzer, format_enhanced_prompt
            
            # Calculate professional metrics on daily timeframe
            analyzer = ProfessionalMarketAnalyzer()
            d1_candles = d1_data.candles[-200:] if len(d1_data.candles) >= 200 else d1_data.candles
            
            vwap, vwap_dev = analyzer.calculate_vwap(d1_candles[-50:])
            vol_profile = analyzer.calculate_volume_profile(d1_candles[-100:])
            order_flow = analyzer.calculate_order_flow_imbalance(d1_candles[-20:])
            regime = analyzer.calculate_market_regime(d1_candles)
            
            use_enhanced = True
        except:
            use_enhanced = False
            vwap = vwap_dev = 0
            vol_profile = {'poc': 0, 'value_area_high': 0, 'value_area_low': 0}
            order_flow = 0
            regime = {'regime': 'unknown', 'hurst': 0.5}
        
        daily_latest = d1_data.latest_candle
        weekly_bias = self._get_weekly_bias(w1_data) if w1_data else "Neutral"
        daily_structure = self._analyze_daily_structure(d1_data)
        
        # Major support/resistance levels
        major_levels = self._find_major_levels(d1_data.candles[-100:])
        
        analysis_prompt = self._build_prompt("""
Analyze {symbol} for POSITION TRADING (holding days to weeks).

DAILY TIMEFRAME ANALYSIS (200 bars):
- Current Price: {daily_price}
- EMA20: {daily_ema20}, EMA50: {daily_ema50}, EMA200: {daily_ema200}
- RSI(14): {daily_rsi}
- ATR(14): {daily_atr} ({daily_atr_pct:.2%} of price)
- VWAP: {vwap:.5f} (Deviation: {vwap_dev:.2f}%)
- Market Regime: {regime} (Hurst: {hurst:.3f})

WEEKLY BIAS:
- Trend: {weekly_bias}
- Structure: {weekly_structure}

DAILY STRUCTURE:
{daily_structure}

MAJOR LEVELS:
- Key Support: {major_support}
- Key Resistance: {major_resistance}
- Point of Control: {poc:.5f}
- Value Area: {val_low:.5f} - {val_high:.5f}

POSITION TRADING PATTERNS:
- Daily Chart Pattern: {daily_pattern}
- Breakout Potential: {breakout_analysis}
- Moving Average Configuration: {ma_config}

Provide position trading analysis:
1. Major trend assessment (primary trend)
2. Key breakout/breakdown levels
3. Position trade setup quality
4. Entry strategy for multi-day hold
5. Wide stop placement (3-5x daily ATR)
6. Multiple profit targets for scaling out
7. Time-based exit considerations

Format your response as:
TREND: [Primary daily/weekly trend]
PATTERN: [Major pattern if any]
KEY_LEVELS: [Critical support: X.XXXX, resistance: X.XXXX]
SETUP_QUALITY: [Excellent/Good/Poor]
RECOMMENDATION: [BUY/SELL/WAIT]
CONFIDENCE: [0-100]
ENTRY_STRATEGY: [Specific entry method]
POSITION_STOP: [Wide stop level]
TARGETS: [T1: X.XXXX (X days), T2: X.XXXX (X days), T3: X.XXXX (X days)]
HOLD_DURATION: [Expected days/weeks]
EXIT_TRIGGER: [Time or price condition]
""",
            symbol=d1_data.symbol,
            daily_price=daily_latest.close,
            daily_ema20=self._calculate_ema(d1_data.candles, 20),
            daily_ema50=daily_latest.ema50,
            daily_ema200=daily_latest.ema200,
            daily_rsi=daily_latest.rsi14,
            daily_atr=daily_latest.atr14,
            daily_atr_pct=daily_latest.atr14 / daily_latest.close,
            vwap=vwap,
            vwap_dev=vwap_dev,
            regime=regime['regime'],
            hurst=regime.get('hurst', 0.5),
            weekly_bias=weekly_bias,
            weekly_structure=self._get_weekly_structure(w1_data) if w1_data else "Unknown",
            daily_structure=daily_structure,
            major_support=", ".join([f"{s:.5f}" for s in major_levels['support'][:2]]),
            major_resistance=", ".join([f"{r:.5f}" for r in major_levels['resistance'][:2]]),
            poc=vol_profile.get('poc', daily_latest.close),
            val_low=vol_profile.get('value_area_low', daily_latest.close * 0.98),
            val_high=vol_profile.get('value_area_high', daily_latest.close * 1.02),
            daily_pattern=self._identify_daily_pattern(d1_data.candles[-50:]),
            breakout_analysis=self._analyze_breakout_potential(d1_data),
            ma_config=self._analyze_ma_configuration(daily_latest)
        )
        
        # Use lower temperature for position trading (more conservative)
        response = self.gpt_client.analyze_with_response(
            analysis_prompt,
            temperature=0.2,
            agent_type=self.agent_type.value,
            symbol=d1_data.symbol
        )
        
        return self._parse_position_analysis(response, daily_latest.close)
    
    def _analyze_daily_structure(self, daily_data: MarketData) -> str:
        """Analyze daily timeframe structure"""
        if len(daily_data.candles) < 50:
            return "Insufficient daily data"
        
        candles = daily_data.candles
        latest = daily_data.latest_candle
        
        # Higher highs/lower lows analysis
        swing_highs = []
        swing_lows = []
        
        for i in range(5, len(candles) - 5):
            # Swing high
            if candles[i].high > max(c.high for c in candles[i-5:i]) and \
               candles[i].high > max(c.high for c in candles[i+1:i+6]):
                swing_highs.append((i, candles[i].high))
            
            # Swing low
            if candles[i].low < min(c.low for c in candles[i-5:i]) and \
               candles[i].low < min(c.low for c in candles[i+1:i+6]):
                swing_lows.append((i, candles[i].low))
        
        # Determine structure
        structure_parts = []
        
        if len(swing_highs) >= 2:
            if swing_highs[-1][1] > swing_highs[-2][1]:
                structure_parts.append("Higher highs")
            else:
                structure_parts.append("Lower highs")
        
        if len(swing_lows) >= 2:
            if swing_lows[-1][1] > swing_lows[-2][1]:
                structure_parts.append("higher lows")
            else:
                structure_parts.append("lower lows")
        
        # Trend strength
        above_200ema = latest.close > latest.ema200
        trend_strength = "Strong uptrend" if above_200ema and "Higher highs" in " ".join(structure_parts) else \
                        "Strong downtrend" if not above_200ema and "Lower highs" in " ".join(structure_parts) else \
                        "Consolidating"
        
        return f"{trend_strength}: {', '.join(structure_parts) if structure_parts else 'No clear structure'}"
    
    def _get_weekly_bias(self, weekly_data: Optional[MarketData]) -> str:
        """Get directional bias from weekly timeframe"""
        if not weekly_data or len(weekly_data.candles) < 10:
            return "Neutral"
        
        latest = weekly_data.latest_candle
        
        # Simple bias based on EMAs and close
        if latest.close > latest.ema50 > latest.ema200:
            return "Strongly Bullish"
        elif latest.close > latest.ema50:
            return "Bullish"
        elif latest.close < latest.ema50 < latest.ema200:
            return "Strongly Bearish"
        elif latest.close < latest.ema50:
            return "Bearish"
        else:
            return "Neutral"
    
    def _get_weekly_structure(self, weekly_data: Optional[MarketData]) -> str:
        """Analyze weekly structure"""
        if not weekly_data or len(weekly_data.candles) < 20:
            return "Insufficient data"
        
        candles = weekly_data.candles[-20:]
        
        # Count trend weeks
        bullish_weeks = sum(1 for c in candles if c.close > c.open)
        
        # Check for momentum
        recent_move = (candles[-1].close - candles[-4].close) / candles[-4].close * 100
        
        if bullish_weeks > 15:
            structure = "Strong bullish momentum"
        elif bullish_weeks > 12:
            structure = "Bullish momentum"
        elif bullish_weeks < 5:
            structure = "Strong bearish momentum"
        elif bullish_weeks < 8:
            structure = "Bearish momentum"
        else:
            structure = "Balanced/ranging"
        
        return f"{structure} ({bullish_weeks}/20 bullish weeks, {recent_move:.1f}% 4-week move)"
    
    def _find_major_levels(self, candles: List) -> Dict[str, List[float]]:
        """Find major support/resistance levels for position trading"""
        levels = {'support': [], 'resistance': []}
        
        if len(candles) < 20:
            return levels
        
        current_price = candles[-1].close
        
        # Find significant swing points
        for i in range(10, len(candles) - 10):
            # Major swing high (10 bars each side)
            if candles[i].high == max(c.high for c in candles[i-10:i+11]):
                if candles[i].high > current_price:
                    levels['resistance'].append(candles[i].high)
                else:
                    levels['support'].append(candles[i].high)
            
            # Major swing low
            if candles[i].low == min(c.low for c in candles[i-10:i+11]):
                if candles[i].low < current_price:
                    levels['support'].append(candles[i].low)
                else:
                    levels['resistance'].append(candles[i].low)
        
        # Sort by distance from current price
        levels['support'].sort(key=lambda x: current_price - x)
        levels['resistance'].sort(key=lambda x: x - current_price)
        
        # Keep only the most relevant levels
        levels['support'] = levels['support'][:5]
        levels['resistance'] = levels['resistance'][:5]
        
        return levels
    
    def _identify_daily_pattern(self, candles: List) -> str:
        """Identify major patterns on daily timeframe"""
        if len(candles) < 20:
            return "No clear pattern"
        
        # Simple pattern recognition
        recent_candles = candles[-20:]
        highs = [c.high for c in recent_candles]
        lows = [c.low for c in recent_candles]
        
        # Ascending triangle
        if len(set(highs[-5:])) <= 2 and lows[-1] > lows[-10]:
            return "Ascending Triangle"
        
        # Descending triangle
        if len(set(lows[-5:])) <= 2 and highs[-1] < highs[-10]:
            return "Descending Triangle"
        
        # Flag/pennant
        if max(highs[:10]) - min(lows[:10]) > 2 * (max(highs[-5:]) - min(lows[-5:])):
            return "Flag/Pennant consolidation"
        
        # Range
        if max(highs) - min(lows) < candles[-1].atr14 * 5:
            return "Tight range"
        
        return "No clear pattern"
    
    def _analyze_breakout_potential(self, daily_data: MarketData) -> str:
        """Analyze breakout potential for position trades"""
        if len(daily_data.candles) < 50:
            return "Unknown"
        
        candles = daily_data.candles
        latest = daily_data.latest_candle
        
        # Check distance to recent highs/lows
        recent_high = max(c.high for c in candles[-20:])
        recent_low = min(c.low for c in candles[-20:])
        fifty_day_high = max(c.high for c in candles[-50:])
        fifty_day_low = min(c.low for c in candles[-50:])
        
        # Calculate position in range
        range_position = (latest.close - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
        
        # Volume analysis
        recent_volume = sum(c.volume for c in candles[-5:]) / 5
        avg_volume = sum(c.volume for c in candles[-20:]) / 20
        volume_surge = recent_volume / avg_volume if avg_volume > 0 else 1
        
        if range_position > 0.8 and volume_surge > 1.2:
            return f"High breakout potential (at {range_position:.0%} of range, {volume_surge:.1f}x volume)"
        elif range_position < 0.2 and volume_surge > 1.2:
            return f"High breakdown potential (at {range_position:.0%} of range, {volume_surge:.1f}x volume)"
        elif latest.close > fifty_day_high * 0.98:
            return "Testing 50-day highs"
        elif latest.close < fifty_day_low * 1.02:
            return "Testing 50-day lows"
        else:
            return f"Mid-range (at {range_position:.0%} of 20-day range)"
    
    def _analyze_ma_configuration(self, candle) -> str:
        """Analyze moving average configuration"""
        # Get approximate EMA20 (using EMA50 as proxy)
        ema20_approx = candle.ema50 * 0.996
        
        if candle.close > ema20_approx > candle.ema50 > candle.ema200:
            return "Perfect bullish alignment"
        elif candle.close < ema20_approx < candle.ema50 < candle.ema200:
            return "Perfect bearish alignment"
        elif candle.close > candle.ema50 > candle.ema200:
            return "Bullish alignment"
        elif candle.close < candle.ema50 < candle.ema200:
            return "Bearish alignment"
        else:
            return "Mixed/Neutral"
    
    def _calculate_ema(self, candles: List, period: int) -> float:
        """Calculate EMA manually"""
        if len(candles) < period:
            return candles[-1].close
        
        # Simple approximation using SMA for initial value
        sma = sum(c.close for c in candles[-period:]) / period
        return sma
    
    def _parse_position_analysis(self, response: str, current_price: float) -> AgentAnalysis:
        """Parse position trading analysis response"""
        try:
            # Use the safe parsing method from base class
            parsed_data = self._safe_parse_response(response, current_price)
            
            # Extract position-specific fields
            raw_parsed = parsed_data['metadata'].get('raw_parsed', {})
            
            # Parse targets with timeframes
            targets_str = raw_parsed.get('TARGETS', '')
            targets = []
            if targets_str:
                # Parse targets like "T1: 1.2345 (5 days)"
                import re
                target_matches = re.findall(r'T\d+:\s*([\d.]+)\s*\(([^)]+)\)', targets_str)
                for price, timeframe in target_matches:
                    targets.append({'price': float(price), 'timeframe': timeframe})
            
            # Extract position-specific metadata
            parsed_data['metadata'].update({
                'trend': raw_parsed.get('TREND', 'Unknown'),
                'setup_quality': raw_parsed.get('SETUP_QUALITY', 'Poor'),
                'entry_strategy': raw_parsed.get('ENTRY_STRATEGY', 'Market order'),
                'hold_duration': raw_parsed.get('HOLD_DURATION', 'Unknown'),
                'exit_trigger': raw_parsed.get('EXIT_TRIGGER', 'Target or stop'),
                'targets': targets,
                'position_mode': True
            })
            
            # Calculate position stop (wider for position trading)
            stop_distance = abs(parsed_data['stop_loss'] - current_price) if parsed_data['stop_loss'] else current_price * 0.03
            
            # Update reasoning for position trading
            reasoning = []
            reasoning.append(f"Position trade setup: {parsed_data['metadata']['setup_quality']}")
            reasoning.append(f"Primary trend: {parsed_data['metadata']['trend']}")
            if parsed_data['metadata'].get('pattern'):
                reasoning.append(f"Pattern: {parsed_data['metadata']['pattern']}")
            
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
            logger.error(f"Failed to parse position trading analysis: {e}")
            default_analysis = AgentAnalysis(
                agent_type=self.agent_type,
                recommendation=SignalType.WAIT,
                confidence=50.0,
                reasoning=["Error parsing position analysis - defaulting to WAIT"],
                concerns=["Position analysis parsing failed"],
                entry_price=current_price,
                stop_loss=0,
                take_profit=0,
                risk_reward_ratio=None,
                metadata={'error': str(e), 'position_mode': True}
            )
            self.analysis_history.append(default_analysis)
            return default_analysis