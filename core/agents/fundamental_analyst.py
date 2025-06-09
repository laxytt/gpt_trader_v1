"""
Fundamental Analyst Agent
Specializes in economic data, news, and macro trends
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from core.agents.base_agent import TradingAgent, AgentType, AgentAnalysis, DebateResponse
from core.domain.models import MarketData, SignalType
from core.infrastructure.gpt.client import GPTClient

logger = logging.getLogger(__name__)


class FundamentalAnalyst(TradingAgent):
    """The Economist - focuses on fundamental analysis"""
    
    def __init__(self, gpt_client: GPTClient):
        super().__init__(
            agent_type=AgentType.FUNDAMENTAL_ANALYST,
            personality="news-driven and macro-focused economist",
            specialty="economic data, central bank policy, news impact, and cross-market correlations"
        )
        self.gpt_client = gpt_client
    
    def analyze(
        self,
        market_data: Dict[str, MarketData],
        news_context: Optional[List[str]] = None,
        ml_context: Optional[Dict[str, Any]] = None
    ) -> AgentAnalysis:
        """Perform comprehensive fundamental analysis"""
        
        h1_data = market_data.get('h1')
        h4_data = market_data.get('h4')
        d1_data = market_data.get('d1')
        w1_data = market_data.get('w1')  # Weekly for long-term trends
        
        if not h1_data:
            raise ValueError("H1 data required for fundamental analysis")
            
        symbol = h1_data.symbol
        
        # Extract currency pair components
        base_currency = symbol[:3] if len(symbol) >= 6 else "XXX"
        quote_currency = symbol[3:6] if len(symbol) >= 6 else "XXX"
        
        # Need fewer candles for fundamental analysis
        h1_candles = h1_data.candles[-50:] if len(h1_data.candles) >= 50 else h1_data.candles
        h4_candles = h4_data.candles[-25:] if h4_data and len(h4_data.candles) >= 25 else []
        d1_candles = d1_data.candles[-20:] if d1_data and len(d1_data.candles) >= 20 else []
        
        # Calculate price momentum for different timeframes
        momentum_1w = self._calculate_momentum(h1_candles, 168) if len(h1_candles) >= 168 else self._calculate_momentum(d1_candles, 7)
        momentum_1m = self._calculate_momentum(d1_candles, 20) if len(d1_candles) >= 20 else 0
        
        # Interest rate differentials (simplified)
        rate_diff = self._estimate_rate_differential(base_currency, quote_currency)
        
        # Risk sentiment analysis
        risk_sentiment = self._analyze_risk_sentiment(symbol, h1_candles)
        
        # Correlation analysis
        correlations = self._analyze_correlations(symbol)
        
        # Format news with advanced parsing
        news_analysis = self._analyze_news_impact(news_context)
        
        # Economic calendar impact
        calendar_impact = self._assess_economic_calendar(base_currency, quote_currency)
        
        analysis_prompt = self._build_prompt("""
Analyze {symbol} from a professional fundamental perspective.

CURRENCY ANALYSIS:
- Base Currency: {base} 
- Quote Currency: {quote}
- Current Price: {current_price:.5f}

MOMENTUM & TRENDS:
- 1-Week Change: {mom_1w:.2f}%
- 1-Month Change: {mom_1m:.2f}%
- Price vs 20DMA: {vs_20dma:.2f}%
- Price vs 50DMA: {vs_50dma:.2f}%

FUNDAMENTAL FACTORS:
- Interest Rate Differential: {rate_diff}
- Risk Sentiment: {risk_sentiment}
- Key Correlations: {correlations}

NEWS ANALYSIS:
{news_analysis}

ECONOMIC CALENDAR:
{calendar_impact}

MARKET SENTIMENT:
- Overall Sentiment: {sentiment_score}
- News Volume: {news_volume}
- Sentiment Trend: {sentiment_trend}

TECHNICAL CONTEXT:
- Daily Range %: {daily_range:.2f}%
- Weekly Range %: {weekly_range:.2f}%
- Volatility Regime: {vol_regime}

Professional Fundamental Assessment:
1. Central bank policy divergence
2. Economic growth differentials
3. Inflation expectations
4. Capital flows and positioning
5. Geopolitical risk factors
6. Cross-asset correlations
7. Market sentiment alignment

Provide:
BASE_BIAS: [Strong Bullish/Bullish/Neutral/Bearish/Strong Bearish] - [specific reason]
QUOTE_BIAS: [Strong Bullish/Bullish/Neutral/Bearish/Strong Bearish] - [specific reason]
MACRO_DRIVERS: [top 3 drivers in order of importance]
NEWS_IMPACT: [quantify impact: High/Medium/Low positive/negative]
CORRELATIONS: [key correlation trades]
RECOMMENDATION: [BUY/SELL/WAIT]
CONFIDENCE: [0-100]
TIME_HORIZON: [intraday/days/weeks]
RISK: [specific fundamental risk]
POSITION_SIZE: [0.1-0.5 based on conviction]

Format your response with exact labels above.
""",
            symbol=symbol,
            base=base_currency,
            quote=quote_currency,
            current_price=h1_data.latest_candle.close,
            mom_1w=momentum_1w,
            mom_1m=momentum_1m,
            vs_20dma=self._price_vs_ma(h1_candles, 20) if len(h1_candles) >= 20 else 0,
            vs_50dma=self._price_vs_ma(h1_candles, 50) if len(h1_candles) >= 50 else 0,
            rate_diff=rate_diff,
            risk_sentiment=risk_sentiment,
            correlations=correlations,
            news_analysis=news_analysis['summary'],
            calendar_impact=calendar_impact,
            sentiment_score=news_analysis['sentiment_score'],
            news_volume=news_analysis['volume'],
            sentiment_trend=news_analysis['trend'],
            daily_range=self._calculate_range_pct(h1_candles[-24:]) if len(h1_candles) >= 24 else 0,
            weekly_range=self._calculate_range_pct(h1_candles[-168:]) if len(h1_candles) >= 168 else self._calculate_range_pct(d1_candles[-7:]) if len(d1_candles) >= 7 else 0,
            vol_regime=self._classify_volatility(h1_data.latest_candle.atr14)
        )
        
        # Get GPT analysis
        response = self.gpt_client.analyze_with_response(
            analysis_prompt,
            temperature=0.4,
            agent_type=self.agent_type.value,
            symbol=symbol
        )
        
        return self._parse_analysis(response, h1_data.latest_candle.close if h1_data else 0)
    
    def debate(
        self,
        other_analyses: List[AgentAnalysis],
        round_number: int,
        previous_responses: Optional[List[DebateResponse]] = None
    ) -> DebateResponse:
        """Participate in debate with fundamental arguments"""
        
        my_position = self.analysis_history[-1]
        
        if round_number == 1:
            statement = self._generate_opening_statement(my_position)
        elif round_number == 2:
            # Address technical analyst if they disagree
            tech_analysis = next((a for a in other_analyses if a.agent_type == AgentType.TECHNICAL_ANALYST), None)
            if tech_analysis and tech_analysis.recommendation != my_position.recommendation:
                statement = f"The technical picture may suggest {tech_analysis.recommendation.value}, but fundamentals drive the bigger moves. {my_position.reasoning[0]}"
            else:
                statement = self._support_allied_view(other_analyses)
        else:
            statement = self._generate_closing_statement(other_analyses)
        
        # Fundamental analyst may adjust confidence based on technical alignment
        updated_confidence = my_position.confidence
        tech_analysis = next((a for a in other_analyses if a.agent_type == AgentType.TECHNICAL_ANALYST), None)
        if tech_analysis:
            if tech_analysis.recommendation == my_position.recommendation:
                updated_confidence = min(100, my_position.confidence + 10)
            else:
                updated_confidence = max(0, my_position.confidence - 5)
        
        return DebateResponse(
            agent_type=self.agent_type,
            round=round_number,
            statement=statement,
            maintains_position=True,
            updated_confidence=updated_confidence
        )
    
    def _parse_analysis(self, response: str, current_price: float) -> AgentAnalysis:
        """Parse GPT response into AgentAnalysis using safe parsing"""
        try:
            # Use the safe parsing method from base class
            parsed_data = self._safe_parse_response(response, current_price)
            raw_parsed = parsed_data['metadata']['raw_parsed']
            
            # Determine recommendation based on biases
            base_bias = raw_parsed.get('BASE_BIAS', '')
            quote_bias = raw_parsed.get('QUOTE_BIAS', '')
            
            if 'Bullish' in base_bias and 'Bearish' in quote_bias:
                recommendation = SignalType.BUY
            elif 'Bearish' in base_bias and 'Bullish' in quote_bias:
                recommendation = SignalType.SELL
            else:
                recommendation = parsed_data['recommendation']
            
            # Extract confidence with sentiment adjustment
            base_confidence = parsed_data['confidence']
            sentiment_impact = raw_parsed.get('SENTIMENT_IMPACT', '')
            
            if sentiment_impact:
                if 'strong' in sentiment_impact.lower() and 'support' in sentiment_impact.lower():
                    confidence = min(100, base_confidence + 10)
                elif 'conflict' in sentiment_impact.lower() or 'against' in sentiment_impact.lower():
                    confidence = max(0, base_confidence - 10)
                else:
                    confidence = base_confidence
            else:
                confidence = base_confidence
            
            # Build custom reasoning for fundamental analysis
            reasoning = []
            if raw_parsed.get('BASE_BIAS'):
                reasoning.append(f"Base currency: {raw_parsed['BASE_BIAS']}")
            if raw_parsed.get('MACRO_DRIVERS'):
                reasoning.append(f"Macro drivers: {raw_parsed['MACRO_DRIVERS']}")
            if raw_parsed.get('NEWS_IMPACT'):
                reasoning.append(f"News impact: {raw_parsed['NEWS_IMPACT']}")
            if raw_parsed.get('SENTIMENT_IMPACT'):
                reasoning.append(f"Sentiment: {raw_parsed['SENTIMENT_IMPACT']}")
            
            # Extract concerns
            concerns = []
            if raw_parsed.get('RISK'):
                concerns.append(raw_parsed['RISK'])
            
            analysis = AgentAnalysis(
                agent_type=self.agent_type,
                recommendation=recommendation,
                confidence=confidence,
                reasoning=reasoning[:3] or ["Fundamental analysis completed"],
                concerns=concerns,
                entry_price=current_price,  # Fundamental doesn't specify exact entry
                metadata={
                    'base_bias': raw_parsed.get('BASE_BIAS'),
                    'quote_bias': raw_parsed.get('QUOTE_BIAS'),
                    'time_horizon': raw_parsed.get('TIME_HORIZON', 'medium-term'),
                    'macro_drivers': raw_parsed.get('MACRO_DRIVERS'),
                    'sentiment_impact': raw_parsed.get('SENTIMENT_IMPACT', '')
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
                reasoning=["Error parsing fundamental analysis - defaulting to WAIT"],
                concerns=["Fundamental analysis parsing failed"],
                entry_price=current_price,
                metadata={'error': str(e)}
            )
            self.analysis_history.append(default_analysis)
            return default_analysis
    
    def _generate_opening_statement(self, analysis: AgentAnalysis) -> str:
        """Generate opening debate statement"""
        horizon = analysis.metadata.get('time_horizon', 'medium-term')
        return f"From a fundamental perspective, {analysis.recommendation.value} is the clear choice. {analysis.reasoning[0]}. This view is based on {horizon} fundamentals."
    
    def _support_allied_view(self, analyses: List[AgentAnalysis]) -> str:
        """Support agents who agree"""
        my_rec = self.analysis_history[-1].recommendation
        allies = [a for a in analyses if a.recommendation == my_rec and a.agent_type != self.agent_type]
        
        if allies:
            ally = allies[0]
            return f"I agree with the {ally.agent_type.value}. The fundamentals support this view, adding conviction to the setup."
        else:
            return "The fundamental picture is clear despite mixed technical signals. Major trends are driven by fundamentals."
    
    def _generate_closing_statement(self, all_analyses: List[AgentAnalysis]) -> str:
        """Generate closing statement"""
        my_analysis = self.analysis_history[-1]
        time_horizon = my_analysis.metadata.get('time_horizon', 'medium-term')
        
        if time_horizon in ['hours', 'short-term']:
            return "While fundamentals favor this direction, be aware this is a shorter-term fundamental view. Manage risk accordingly."
        else:
            return f"The fundamental thesis is strong for the {time_horizon}. This provides a solid foundation for the trade."
    
    def _calculate_momentum(self, candles: List, periods: int) -> float:
        """Calculate price momentum over periods"""
        if len(candles) < periods + 1:
            return 0
            
        if periods > len(candles):
            # Use daily candles approximation
            periods = min(periods, len(candles) - 1)
            
        current = candles[-1].close
        past = candles[-periods-1].close if periods < len(candles) else candles[0].close
        
        return ((current - past) / past * 100) if past > 0 else 0
    
    def _estimate_rate_differential(self, base: str, quote: str) -> str:
        """Estimate interest rate differential (simplified)"""
        # Simplified rate estimates (would use real data in production)
        rates = {
            'USD': 5.50,  # Fed Funds
            'EUR': 4.50,  # ECB
            'GBP': 5.25,  # BoE
            'JPY': -0.10, # BoJ
            'CHF': 1.75,  # SNB
            'CAD': 5.00,  # BoC
            'AUD': 4.35,  # RBA
            'NZD': 5.50   # RBNZ
        }
        
        base_rate = rates.get(base, 0)
        quote_rate = rates.get(quote, 0)
        diff = base_rate - quote_rate
        
        if abs(diff) < 0.5:
            return f"{diff:+.2f}% (Neutral)"
        elif diff > 2:
            return f"{diff:+.2f}% (Strong {base} positive)"
        elif diff > 0:
            return f"{diff:+.2f}% ({base} positive)"
        elif diff < -2:
            return f"{diff:+.2f}% (Strong {quote} positive)"
        else:
            return f"{diff:+.2f}% ({quote} positive)"
    
    def _analyze_risk_sentiment(self, symbol: str, candles: List) -> str:
        """Analyze risk sentiment based on symbol characteristics"""
        # Safe havens
        safe_havens = ['JPY', 'CHF', 'USD']
        risk_currencies = ['AUD', 'NZD', 'CAD', 'GBP']
        
        # Check recent volatility
        if len(candles) >= 20:
            recent_range = max(c.high for c in candles[-20:]) - min(c.low for c in candles[-20:])
            avg_range = sum(c.high - c.low for c in candles[-20:]) / 20
            
            if recent_range > avg_range * 2:
                vol_state = "High volatility"
            else:
                vol_state = "Normal volatility"
        else:
            vol_state = "Unknown"
        
        # Determine risk sentiment
        if any(safe in symbol for safe in safe_havens):
            if 'JPY' in symbol and symbol.startswith(('USD', 'EUR', 'GBP')):
                return f"Risk-off benefits JPY ({vol_state})"
            elif 'CHF' in symbol:
                return f"Safe haven flow potential ({vol_state})"
            else:
                return f"Mixed risk sentiment ({vol_state})"
        elif any(risk in symbol for risk in risk_currencies):
            return f"Risk-on benefits {symbol[:3]} ({vol_state})"
        else:
            return f"Neutral risk sentiment ({vol_state})"
    
    def _analyze_correlations(self, symbol: str) -> str:
        """Analyze key correlations for the currency pair"""
        correlations = []
        
        # USD correlations
        if 'USD' in symbol:
            correlations.append("DXY (inverse if USD quote)")
            correlations.append("US 10Y yields (positive)")
        
        # Commodity correlations
        if 'AUD' in symbol:
            correlations.append("Iron ore & copper (positive)")
        if 'CAD' in symbol:
            correlations.append("Oil prices (positive)")
        if 'NZD' in symbol:
            correlations.append("Dairy prices (positive)")
        
        # Risk correlations
        if 'JPY' in symbol:
            correlations.append("S&P 500 (inverse)")
            correlations.append("VIX (positive for JPY)")
        
        # EUR correlations
        if 'EUR' in symbol:
            correlations.append("German 10Y yields (positive)")
            correlations.append("EUR stocks (mixed)")
        
        return ", ".join(correlations[:3]) if correlations else "No major correlations"
    
    def _analyze_news_impact(self, news_context: Optional[List[str]]) -> Dict[str, Any]:
        """Analyze news sentiment and impact"""
        if not news_context:
            return {
                'summary': 'No recent news',
                'sentiment_score': 'Neutral',
                'volume': 'Low',
                'trend': 'Stable'
            }
        
        # Extract sentiment if provided
        sentiment_line = None
        news_items = []
        
        for line in news_context:
            if "Market Sentiment:" in line:
                sentiment_line = line
            else:
                news_items.append(line)
        
        # Analyze news content
        positive_keywords = ['growth', 'rise', 'gain', 'improve', 'strong', 'beat', 'rally', 'surge']
        negative_keywords = ['decline', 'fall', 'weak', 'concern', 'miss', 'cut', 'crisis', 'recession']
        
        positive_count = 0
        negative_count = 0
        
        for news in news_items[:10]:  # Analyze up to 10 news items
            news_lower = news.lower()
            positive_count += sum(1 for word in positive_keywords if word in news_lower)
            negative_count += sum(1 for word in negative_keywords if word in news_lower)
        
        # Determine sentiment
        if positive_count > negative_count * 1.5:
            sentiment = "Positive"
        elif negative_count > positive_count * 1.5:
            sentiment = "Negative"
        else:
            sentiment = "Mixed"
        
        # News volume
        volume = "High" if len(news_items) > 5 else "Medium" if len(news_items) > 2 else "Low"
        
        # Trend (would need historical comparison in production)
        trend = "Intensifying" if volume == "High" else "Stable"
        
        # Format summary
        summary = f"Recent news ({len(news_items)} items):\n"
        summary += "\n".join(f"- {news[:100]}..." for news in news_items[:5])
        
        if sentiment_line:
            summary = f"{sentiment_line}\n\n{summary}"
        
        return {
            'summary': summary,
            'sentiment_score': sentiment,
            'volume': volume,
            'trend': trend
        }
    
    def _assess_economic_calendar(self, base: str, quote: str) -> str:
        """Assess upcoming economic events impact"""
        # Simplified calendar assessment
        import datetime
        weekday = datetime.datetime.now().weekday()
        
        # Major events by day (simplified)
        events = []
        
        if weekday == 0:  # Monday
            events.append("Week opening positioning")
        elif weekday == 2:  # Wednesday
            if base == 'USD' or quote == 'USD':
                events.append("FOMC minutes risk")
        elif weekday == 3:  # Thursday
            if base == 'EUR' or quote == 'EUR':
                events.append("ECB meeting risk")
            if base == 'GBP' or quote == 'GBP':
                events.append("BoE decision risk")
        elif weekday == 4:  # Friday
            if base == 'USD' or quote == 'USD':
                events.append("NFP/US data risk")
            events.append("Weekend position squaring")
        
        if not events:
            return "No major events today"
        
        return f"Key risks: {', '.join(events)}"
    
    def _price_vs_ma(self, candles: List, period: int) -> float:
        """Calculate price vs moving average"""
        if len(candles) < period:
            return 0
            
        ma = sum(c.close for c in candles[-period:]) / period
        current = candles[-1].close
        
        return ((current - ma) / ma * 100) if ma > 0 else 0
    
    def _calculate_range_pct(self, candles: List) -> float:
        """Calculate price range as percentage"""
        if not candles:
            return 0
            
        high = max(c.high for c in candles)
        low = min(c.low for c in candles)
        mid = (high + low) / 2
        
        return ((high - low) / mid * 100) if mid > 0 else 0
    
    def _classify_volatility(self, atr: float) -> str:
        """Classify volatility regime based on ATR"""
        # Convert ATR to pips (approximate)
        atr_pips = atr * 10000
        
        if atr_pips < 5:
            return "Ultra Low"
        elif atr_pips < 10:
            return "Low"
        elif atr_pips < 20:
            return "Normal"
        elif atr_pips < 30:
            return "High"
        else:
            return "Extreme"