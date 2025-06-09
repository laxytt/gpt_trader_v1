"""
Sentiment Reader Agent
Specializes in market psychology and sentiment analysis
"""

import logging
from typing import Dict, List, Optional, Any

from core.agents.base_agent import TradingAgent, AgentType, AgentAnalysis, DebateResponse
from core.domain.models import MarketData, SignalType
from core.infrastructure.gpt.client import GPTClient

logger = logging.getLogger(__name__)


class SentimentReader(TradingAgent):
    """The Psychologist - reads market sentiment and crowd behavior"""
    
    def __init__(self, gpt_client: GPTClient):
        super().__init__(
            agent_type=AgentType.SENTIMENT_READER,
            personality="intuitive and mood-sensitive market psychologist",
            specialty="market psychology, fear/greed dynamics, positioning extremes, and crowd behavior"
        )
        self.gpt_client = gpt_client
    
    def analyze(
        self,
        market_data: Dict[str, MarketData],
        news_context: Optional[List[str]] = None,
        ml_context: Optional[Dict[str, Any]] = None
    ) -> AgentAnalysis:
        """Analyze market sentiment and psychology with professional metrics"""
        
        h1_data = market_data.get('h1')
        h4_data = market_data.get('h4')
        d1_data = market_data.get('d1')
        
        if not h1_data:
            raise ValueError("H1 data required for sentiment analysis")
        
        # Import enhanced analyzer
        try:
            from core.agents.enhanced_base_agent import ProfessionalMarketAnalyzer
            analyzer = ProfessionalMarketAnalyzer()
            
            # Need 100+ candles for proper sentiment analysis
            h1_candles = h1_data.candles[-100:] if len(h1_data.candles) >= 100 else h1_data.candles
            
            # Calculate market microstructure sentiment
            order_flow = analyzer.calculate_order_flow_imbalance(h1_candles[-20:])
            vol_profile = analyzer.calculate_volume_profile(h1_candles[-50:])
            regime = analyzer.calculate_market_regime(h1_candles)
            
            use_enhanced = True
        except:
            use_enhanced = False
            order_flow = 0
            vol_profile = {}
            regime = {'regime': 'unknown'}
        
        # Professional sentiment metrics
        candles = h1_data.candles
        latest = h1_data.latest_candle
        
        # Fear & Greed indicators
        rsi_extreme = self._calculate_rsi_extreme(latest.rsi14)
        bollinger_squeeze = self._calculate_bollinger_squeeze(candles)
        volume_climax = self._detect_volume_climax(candles)
        
        # Market breadth and participation
        breadth_ratio = self._calculate_market_breadth(candles)
        participation = self._calculate_participation_rate(candles)
        
        # Trap detection
        bull_trap, bear_trap = self._detect_traps(candles)
        failed_breakouts = self._count_failed_breakouts(candles)
        
        # Exhaustion signals
        exhaustion_gap = self._detect_exhaustion_gap(candles)
        divergence = self._calculate_price_momentum_divergence(candles)
        
        # Psychological levels
        round_numbers = self._find_round_numbers(latest.close)
        swing_levels = self._find_swing_levels(candles)
        
        # Capitulation/Euphoria detection
        volume_spike_percentile = self._calculate_volume_percentile(candles)
        range_expansion = self._calculate_range_expansion(candles)
        
        analysis_prompt = self._build_prompt("""
Analyze {symbol} market psychology like a professional sentiment trader.

SENTIMENT INDICATORS (100 bars):
- RSI Extreme: {rsi_extreme} (14: {rsi:.1f})
- Order Flow Imbalance: {order_flow:.2%}
- Volume Climax: {volume_climax}
- Bollinger Band Squeeze: {bb_squeeze:.2f}σ

MARKET BREADTH & PARTICIPATION:
- Breadth Ratio: {breadth:.1%} (advancing vs declining bars)
- Participation Rate: {participation:.1%}
- Volume Percentile: {vol_percentile:.0f}th percentile
- Range Expansion: {range_exp:.1f}x normal

TRAP DETECTION:
- Bull Traps: {bull_traps}
- Bear Traps: {bear_traps}  
- Failed Breakouts (20 bars): {failed_breakouts}

EXHAUSTION SIGNALS:
- Exhaustion Gap: {exhaustion_gap}
- Price/Momentum Divergence: {divergence}
- Market Regime: {regime}

PSYCHOLOGICAL LEVELS:
- Current Price: {price:.5f}
- Nearest Round: {round_number:.5f} ({round_dist:.1f} pips)
- Key Swing Levels: {swing_levels}
- Value Area: {val_low:.5f} - {val_high:.5f}

VOLUME PROFILE:
- Point of Control: {poc:.5f}
- Current vs POC: {poc_dist:.1f} pips

NEWS SENTIMENT:
{news_summary}

Professional Psychology Assessment:
1. Fear/Greed reading (contrarian opportunities)
2. Crowd positioning (overleveraged longs/shorts)
3. Smart money vs retail divergence
4. Capitulation or euphoria signs
5. Key levels where stops are clustered
6. Sentiment shift catalysts

Provide:
MOOD: [extreme fear/fear/neutral/greed/extreme greed]
CROWD: [heavily long/long/balanced/short/heavily short]
PSYCH_LEVELS: [list key stop-loss clusters]
SENTIMENT_SIGNS: [professional interpretation]
RECOMMENDATION: [BUY/SELL/WAIT]
CONFIDENCE: [0-100]
TRIGGERS: [specific levels/events that shift sentiment]
CONCERN: [main psychological risk]

Format your response with exact labels above.
""",
            symbol=h1_data.symbol,
            price=latest.close,
            rsi=latest.rsi14,
            rsi_extreme=rsi_extreme,
            order_flow=order_flow,
            volume_climax=volume_climax,
            bb_squeeze=bollinger_squeeze,
            breadth=breadth_ratio,
            participation=participation,
            vol_percentile=volume_spike_percentile,
            range_exp=range_expansion,
            bull_traps=bull_trap,
            bear_traps=bear_trap,
            failed_breakouts=failed_breakouts,
            exhaustion_gap=exhaustion_gap,
            divergence=divergence,
            regime=regime.get('regime', 'unknown'),
            round_number=round_numbers[0] if round_numbers else latest.close,
            round_dist=abs(latest.close - round_numbers[0]) * 10000 if round_numbers else 0,
            swing_levels=", ".join([f"{level:.5f}" for level in swing_levels[:3]]),
            poc=vol_profile.get('poc', latest.close),
            poc_dist=(latest.close - vol_profile.get('poc', latest.close)) * 10000,
            val_low=vol_profile.get('value_area_low', latest.close * 0.995),
            val_high=vol_profile.get('value_area_high', latest.close * 1.005),
            news_summary=self._summarize_news_sentiment(news_context) if news_context else "No recent news"
        )
        
        response = self.gpt_client.analyze_with_response(
            analysis_prompt,
            temperature=0.6,  # Higher temperature for intuitive reading
            agent_type=self.agent_type.value,
            symbol=h1_data.symbol
        )
        
        return self._parse_analysis(response, h1_data.latest_candle.close if h1_data else 0)
    
    def debate(
        self,
        other_analyses: List[AgentAnalysis],
        round_number: int,
        previous_responses: Optional[List[DebateResponse]] = None
    ) -> DebateResponse:
        """Participate in debate with sentiment insights"""
        
        my_position = self.analysis_history[-1]
        
        if round_number == 1:
            statement = self._generate_opening_statement(my_position)
        elif round_number == 2:
            # Find who to address
            momentum_trader = next((a for a in other_analyses if a.agent_type == AgentType.MOMENTUM_TRADER), None)
            if momentum_trader and momentum_trader.recommendation != my_position.recommendation:
                statement = f"The {momentum_trader.agent_type.value} may see momentum, but I sense exhaustion. The crowd is already all-in."
            else:
                statement = self._provide_psychological_insight(other_analyses)
        else:
            statement = self._generate_closing_statement(other_analyses)
        
        # Sentiment reader may change view if overwhelming technical evidence
        maintains_position = True
        updated_confidence = my_position.confidence
        
        tech_analysis = next((a for a in other_analyses if a.agent_type == AgentType.TECHNICAL_ANALYST), None)
        if tech_analysis and tech_analysis.confidence > 80:
            if tech_analysis.recommendation != my_position.recommendation:
                updated_confidence = max(30, my_position.confidence - 20)
                if updated_confidence < 50:
                    maintains_position = False
        
        return DebateResponse(
            agent_type=self.agent_type,
            round=round_number,
            statement=statement,
            maintains_position=maintains_position,
            updated_confidence=updated_confidence
        )
    
    def _parse_analysis(self, response: str, current_price: float) -> AgentAnalysis:
        """Parse GPT response into AgentAnalysis using safe parsing"""
        try:
            # Use the safe parsing method from base class
            parsed_data = self._safe_parse_response(response, current_price)
            raw_parsed = parsed_data['metadata']['raw_parsed']
            
            # Determine recommendation based on mood and crowd
            mood = raw_parsed.get('MOOD', 'neutral').lower()
            crowd = raw_parsed.get('CROWD', '').lower()
            
            # Contrarian logic
            if 'fear' in mood and 'oversold' in crowd:
                recommendation = SignalType.BUY
            elif 'greed' in mood and 'overbought' in crowd:
                recommendation = SignalType.SELL
            else:
                recommendation = parsed_data['recommendation']
            
            # Build custom reasoning for sentiment
            reasoning = []
            if raw_parsed.get('MOOD'):
                reasoning.append(f"Market mood: {raw_parsed['MOOD']}")
            if raw_parsed.get('CROWD'):
                reasoning.append(f"Crowd positioning: {raw_parsed['CROWD']}")
            if raw_parsed.get('SENTIMENT_SIGNS'):
                reasoning.append(f"Sentiment signs: {raw_parsed['SENTIMENT_SIGNS']}")
            
            concerns = []
            if raw_parsed.get('CONCERN'):
                concerns.append(raw_parsed['CONCERN'])
            
            analysis = AgentAnalysis(
                agent_type=self.agent_type,
                recommendation=recommendation,
                confidence=parsed_data['confidence'],
                reasoning=reasoning[:3] or ["Sentiment analysis completed"],
                concerns=concerns,
                entry_price=current_price,
                metadata={
                    'mood': raw_parsed.get('MOOD'),
                    'crowd_positioning': raw_parsed.get('CROWD'),
                    'psychological_levels': raw_parsed.get('PSYCH_LEVELS'),
                    'triggers': raw_parsed.get('TRIGGERS')
                }
            )
            
            self.analysis_history.append(analysis)
            return analysis
            
        except Exception as e:
            # If all else fails, return a safe default
            default_analysis = AgentAnalysis(
                agent_type=self.agent_type,
                recommendation=SignalType.WAIT,
                confidence=50.0,
                reasoning=["Error parsing sentiment analysis - defaulting to WAIT"],
                concerns=["Sentiment analysis parsing failed"],
                entry_price=current_price,
                metadata={'error': str(e)}
            )
            self.analysis_history.append(default_analysis)
            return default_analysis
    
    def _generate_opening_statement(self, analysis: AgentAnalysis) -> str:
        """Generate opening debate statement"""
        mood = analysis.metadata.get('mood', 'uncertain')
        return f"I sense {mood} in the market. {analysis.reasoning[0]}. The psychological setup suggests {analysis.recommendation.value}."
    
    def _provide_psychological_insight(self, analyses: List[AgentAnalysis]) -> str:
        """Provide psychological insight to the debate"""
        agreements = sum(1 for a in analyses if a.recommendation == self.analysis_history[-1].recommendation)
        
        if agreements >= 4:
            return "The council's agreement itself is telling - when everyone sees the same thing, the market often does the opposite."
        else:
            return "This disagreement reflects market uncertainty. Perhaps that's the real signal - stay cautious."
    
    def _generate_closing_statement(self, all_analyses: List[AgentAnalysis]) -> str:
        """Generate closing statement"""
        my_mood = self.analysis_history[-1].metadata.get('mood', 'uncertain')
        
        if my_mood in ['fear', 'greed']:
            return f"Extreme {my_mood} creates opportunities. The crowd's emotion is our edge."
        else:
            return "The market psychology isn't clear. Without a strong sentiment read, perhaps we should be patient."
    
    def _calculate_rsi_extreme(self, rsi: float) -> str:
        """Classify RSI extreme"""
        if rsi >= 80:
            return "Extreme overbought"
        elif rsi >= 70:
            return "Overbought"
        elif rsi <= 20:
            return "Extreme oversold"
        elif rsi <= 30:
            return "Oversold"
        else:
            return "Neutral"
    
    def _calculate_bollinger_squeeze(self, candles: List) -> float:
        """Calculate Bollinger Band squeeze (volatility contraction)"""
        if len(candles) < 20:
            return 1.0
            
        # Calculate 20-period standard deviation
        closes = [c.close for c in candles[-20:]]
        mean = sum(closes) / len(closes)
        variance = sum((x - mean) ** 2 for x in closes) / len(closes)
        std_dev = variance ** 0.5
        
        # Compare to historical volatility
        if len(candles) >= 100:
            historical_stds = []
            for i in range(20, 100):
                hist_closes = [c.close for c in candles[-i-20:-i]]
                hist_mean = sum(hist_closes) / len(hist_closes)
                hist_var = sum((x - hist_mean) ** 2 for x in hist_closes) / len(hist_closes)
                historical_stds.append(hist_var ** 0.5)
            
            avg_historical_std = sum(historical_stds) / len(historical_stds)
            return std_dev / avg_historical_std if avg_historical_std > 0 else 1.0
        
        return 1.0
    
    def _detect_volume_climax(self, candles: List) -> str:
        """Detect volume climax conditions"""
        if len(candles) < 20:
            return "No climax"
            
        recent_vol = candles[-1].volume
        avg_vol = sum(c.volume for c in candles[-20:]) / 20
        
        if recent_vol > avg_vol * 3:
            if candles[-1].close > candles[-1].open:
                return "Buying climax"
            else:
                return "Selling climax"
        elif recent_vol > avg_vol * 2:
            return "High volume"
        else:
            return "Normal"
    
    def _calculate_market_breadth(self, candles: List) -> float:
        """Calculate market breadth (advancing vs declining bars)"""
        if len(candles) < 20:
            return 0.5
            
        advancing = sum(1 for c in candles[-20:] if c.close > c.open)
        return advancing / 20
    
    def _calculate_participation_rate(self, candles: List) -> float:
        """Calculate participation rate (volume consistency)"""
        if len(candles) < 20:
            return 0.5
            
        volumes = [c.volume for c in candles[-20:]]
        avg_vol = sum(volumes) / len(volumes)
        
        # Count bars with above-average volume
        high_vol_bars = sum(1 for v in volumes if v > avg_vol * 0.8)
        return high_vol_bars / len(volumes)
    
    def _detect_traps(self, candles: List) -> tuple:
        """Detect bull and bear traps"""
        if len(candles) < 10:
            return 0, 0
            
        bull_traps = 0
        bear_traps = 0
        
        for i in range(len(candles) - 10, len(candles) - 2):
            # Bull trap: breakout above previous high that fails
            if candles[i].high > max(c.high for c in candles[i-5:i]):
                if candles[i+1].close < candles[i].open:
                    bull_traps += 1
            
            # Bear trap: breakdown below previous low that fails
            if candles[i].low < min(c.low for c in candles[i-5:i]):
                if candles[i+1].close > candles[i].open:
                    bear_traps += 1
        
        return bull_traps, bear_traps
    
    def _count_failed_breakouts(self, candles: List) -> int:
        """Count failed breakout attempts"""
        if len(candles) < 20:
            return 0
            
        failed = 0
        high_20 = max(c.high for c in candles[-20:])
        low_20 = min(c.low for c in candles[-20:])
        
        for i in range(len(candles) - 20, len(candles) - 2):
            # Failed upside breakout
            if candles[i].close > high_20 * 0.999:
                if candles[i+1].close < high_20:
                    failed += 1
            
            # Failed downside breakout
            if candles[i].close < low_20 * 1.001:
                if candles[i+1].close > low_20:
                    failed += 1
        
        return failed
    
    def _detect_exhaustion_gap(self, candles: List) -> str:
        """Detect exhaustion gaps"""
        if len(candles) < 5:
            return "None"
            
        # Check last candle for gap
        gap_up = candles[-1].low > candles[-2].high
        gap_down = candles[-1].high < candles[-2].low
        
        if gap_up and candles[-1].volume > sum(c.volume for c in candles[-5:-1]) / 4 * 1.5:
            return "Exhaustion gap up"
        elif gap_down and candles[-1].volume > sum(c.volume for c in candles[-5:-1]) / 4 * 1.5:
            return "Exhaustion gap down"
        else:
            return "None"
    
    def _calculate_price_momentum_divergence(self, candles: List) -> str:
        """Detect price/momentum divergence"""
        if len(candles) < 20:
            return "No divergence"
            
        # Simple momentum using price change
        price_highs = []
        momentum_values = []
        
        for i in range(10, 20):
            price_highs.append(candles[-i].high)
            momentum = (candles[-i].close - candles[-i-10].close) / candles[-i-10].close
            momentum_values.append(momentum)
        
        # Check for divergence
        if price_highs[-1] > price_highs[0] and momentum_values[-1] < momentum_values[0]:
            return "Bearish divergence"
        elif price_highs[-1] < price_highs[0] and momentum_values[-1] > momentum_values[0]:
            return "Bullish divergence"
        else:
            return "No divergence"
    
    def _find_round_numbers(self, price: float) -> List[float]:
        """Find nearby psychological round numbers"""
        # Different scales for different price ranges
        if price < 1:
            increment = 0.001  # 10 pips for pairs like EURUSD
        elif price < 10:
            increment = 0.01   # 100 pips
        elif price < 100:
            increment = 0.1    # 1000 pips
        else:
            increment = 1.0
        
        rounds = []
        base = int(price / increment) * increment
        
        for i in range(-2, 3):
            level = base + (i * increment)
            if level > 0:
                rounds.append(level)
        
        return sorted(rounds, key=lambda x: abs(x - price))
    
    def _find_swing_levels(self, candles: List) -> List[float]:
        """Find key swing high/low levels"""
        if len(candles) < 20:
            return []
            
        swings = []
        
        # Find swing highs
        for i in range(5, len(candles) - 5):
            if candles[i].high == max(c.high for c in candles[i-5:i+5]):
                swings.append(candles[i].high)
        
        # Find swing lows
        for i in range(5, len(candles) - 5):
            if candles[i].low == min(c.low for c in candles[i-5:i+5]):
                swings.append(candles[i].low)
        
        # Remove duplicates and sort by distance from current price
        unique_swings = list(set(swings))
        current_price = candles[-1].close
        return sorted(unique_swings, key=lambda x: abs(x - current_price))[:5]
    
    def _calculate_volume_percentile(self, candles: List) -> float:
        """Calculate current volume percentile"""
        if len(candles) < 100:
            return 50
            
        volumes = sorted([c.volume for c in candles[-100:]])
        current_vol = candles[-1].volume
        
        # Find percentile
        position = 0
        for v in volumes:
            if v < current_vol:
                position += 1
        
        return (position / len(volumes)) * 100
    
    def _calculate_range_expansion(self, candles: List) -> float:
        """Calculate range expansion relative to average"""
        if len(candles) < 20:
            return 1.0
            
        current_range = candles[-1].high - candles[-1].low
        avg_range = sum(c.high - c.low for c in candles[-20:]) / 20
        
        return current_range / avg_range if avg_range > 0 else 1.0
    
    def _summarize_news_sentiment(self, news_context: List[str]) -> str:
        """Summarize news sentiment"""
        if not news_context:
            return "No recent news"
            
        # Simple keyword-based sentiment
        positive_words = ['bullish', 'positive', 'growth', 'rise', 'gain', 'improve']
        negative_words = ['bearish', 'negative', 'decline', 'fall', 'loss', 'concern']
        
        positive_count = 0
        negative_count = 0
        
        for news in news_context[:5]:  # Check first 5 news items
            news_lower = news.lower()
            positive_count += sum(1 for word in positive_words if word in news_lower)
            negative_count += sum(1 for word in negative_words if word in news_lower)
        
        if positive_count > negative_count * 1.5:
            return "Positive news sentiment"
        elif negative_count > positive_count * 1.5:
            return "Negative news sentiment"
        else:
            return "Mixed news sentiment"