"""
Index Specialist Agent
Specializes in trading stock indices (US30, US100, US500, DAX, etc.)
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, time

from core.agents.base_agent import BaseAgent, AgentType, AgentAnalysis
from core.domain.models import SignalType, RiskClass
from core.domain.exceptions import AgentError

logger = logging.getLogger(__name__)


class IndexSpecialist(BaseAgent):
    """
    Specialist agent for index trading with deep understanding of:
    - Market opening/closing dynamics
    - Sector rotation
    - Risk-on/risk-off sentiment
    - Options expiry impacts
    - Correlation with economic data
    """
    
    def __init__(self, gpt_client):
        super().__init__(
            agent_type=AgentType.TECHNICAL_ANALYST,  # Reuse type for now
            gpt_client=gpt_client
        )
        self.agent_name = "Index Specialist"
        
    def get_analysis_prompt(self, market_data: Dict, news_context: Optional[List[str]] = None) -> str:
        """Generate index-specific analysis prompt"""
        
        symbol = market_data.get('symbol', 'UNKNOWN')
        h1_data = market_data.get('h1', {})
        h4_data = market_data.get('h4', {})
        
        # Determine index region
        index_region = self._get_index_region(symbol)
        market_session = self._get_current_session(symbol)
        
        prompt = f"""You are an Index Trading Specialist analyzing {symbol}.

INDEX TYPE: {index_region} Index
CURRENT SESSION: {market_session}

MARKET DATA:
- Current Level: {h1_data.get('close', 'N/A')}
- H1 ATR: {h1_data.get('atr', 'N/A')}
- H1 Volume: {h1_data.get('volume', 'N/A')}
- H4 Trend: {self._determine_trend(h4_data)}

SPECIALIZED INDEX ANALYSIS:

1. SESSION ANALYSIS:
{self._get_session_analysis(symbol, market_session)}

2. MARKET SENTIMENT:
{self._analyze_risk_sentiment(symbol, market_data)}

3. SECTOR ROTATION:
{self._get_sector_analysis(symbol, index_region)}

4. KEY LEVELS & GAPS:
{self._analyze_gaps_and_levels(symbol, market_data)}

5. CORRELATION ANALYSIS:
{self._get_correlation_context(symbol, index_region)}

6. ECONOMIC CALENDAR:
{self._get_economic_impact(symbol, news_context)}

TRADING RECOMMENDATION:
Based on your specialized index analysis, provide:
- Signal: BUY/SELL/WAIT
- Confidence: 0-100%
- Key reasons (max 3 bullet points)
- Primary risk factor

Remember:
- Indices gap at open frequently
- Strong intraday momentum is common
- Options expiry causes volatility
- Economic data releases drive movement
- Respect session-specific behavior
"""

        return prompt
    
    def _get_index_region(self, symbol: str) -> str:
        """Categorize index by region"""
        symbol_upper = symbol.upper()
        
        if any(x in symbol_upper for x in ['US30', 'US100', 'US500', 'DOW', 'NASDAQ', 'SPX']):
            return "US"
        elif any(x in symbol_upper for x in ['GER', 'DAX', 'UK100', 'FTSE', 'FR40', 'CAC', 'EU50', 'STOXX']):
            return "European"
        elif any(x in symbol_upper for x in ['JP225', 'NIKKEI', 'HK50', 'HANG', 'CN50', 'CHINA']):
            return "Asian"
        else:
            return "Global"
    
    def _get_current_session(self, symbol: str) -> str:
        """Determine current trading session"""
        now = datetime.utcnow()
        hour = now.hour
        
        # Define session times in UTC
        sessions = {
            'Asian': (23, 8),      # 23:00 - 08:00 UTC
            'European': (7, 16),   # 07:00 - 16:00 UTC
            'US': (13, 21),        # 13:00 - 21:00 UTC
        }
        
        region = self._get_index_region(symbol)
        
        # Check primary session for the index
        if region == "US":
            if 13 <= hour <= 21:
                return "US Session (Active)"
            elif 7 <= hour < 13:
                return "European Session (Pre-market)"
            else:
                return "Asian Session (Overnight)"
        elif region == "European":
            if 7 <= hour <= 16:
                return "European Session (Active)"
            elif 13 <= hour <= 21:
                return "US Session (Correlation)"
            else:
                return "Asian Session (Quiet)"
        elif region == "Asian":
            if 23 <= hour or hour < 8:
                return "Asian Session (Active)"
            else:
                return "European/US Session (Follow-through)"
        
        return "Global Trading"
    
    def _get_session_analysis(self, symbol: str, session: str) -> str:
        """Get session-specific trading dynamics"""
        
        session_dynamics = {
            "US Session (Active)": """
- Market open (9:30 EST): High volatility, gaps common
- First hour: Establish daily range
- Lunch hour (12:00-13:00 EST): Lower volume
- Last hour: Institutional positioning
- Watch for 3:30 PM reversals
""",
            "European Session (Active)": """
- Opening hour: Gap fills from overnight
- Mid-session: Steady trends develop
- US pre-market impact starts 13:00 UTC
- ECB announcements at 12:45 UTC
- Close positioning before US open
""",
            "Asian Session (Active)": """
- Tokyo open: Initial volatility
- Chinese markets influence (CSI open)
- Typically range-bound movement
- Watch for overnight gaps in US/EU indices
- Lower overall volume
""",
            "Pre-market": """
- Thin liquidity, wider spreads
- Futures driven movement
- Gap risk building
- Early economic data impact
- Institutional positioning
""",
            "Overnight": """
- Very low liquidity
- Wider spreads
- News driven spikes
- Technical levels less reliable
- Higher slippage risk
"""
        }
        
        for key, value in session_dynamics.items():
            if key in session:
                return value
        
        return "Standard session - normal liquidity"
    
    def _analyze_risk_sentiment(self, symbol: str, market_data: Dict) -> str:
        """Analyze risk-on/risk-off sentiment"""
        
        # Simple sentiment indicators
        vix_level = market_data.get('vix', 20)  # Default VIX
        
        if vix_level < 15:
            sentiment = "STRONG RISK-ON: Low volatility, bullish indices"
        elif vix_level < 20:
            sentiment = "MILD RISK-ON: Normal market conditions"
        elif vix_level < 25:
            sentiment = "NEUTRAL: Mixed sentiment"
        elif vix_level < 30:
            sentiment = "RISK-OFF: Elevated fear, bearish bias"
        else:
            sentiment = "EXTREME RISK-OFF: High volatility, defensive positioning"
        
        # Add index-specific sentiment
        if "US100" in symbol.upper() or "NASDAQ" in symbol.upper():
            sentiment += "\n- Tech-heavy index, sensitive to growth concerns"
        elif "US30" in symbol.upper() or "DOW" in symbol.upper():
            sentiment += "\n- Value-oriented index, defensive in nature"
        
        return sentiment
    
    def _get_sector_analysis(self, symbol: str, region: str) -> str:
        """Analyze sector rotation impacts"""
        
        sector_impacts = {
            "US": {
                "US100": """
- Technology sector dominates (40%+)
- FAANG stocks drive movement
- Growth vs Value rotation key
- Interest rate sensitive
""",
                "US30": """
- Industrials and financials heavy
- Less tech exposure
- Dividend stocks influence
- Economic cycle sensitive
""",
                "US500": """
- Broad market representation
- All sectors included
- Best overall market gauge
- Follows sector rotation trends
"""
            },
            "European": {
                "GER40": """
- Manufacturing/export focused
- Auto sector important
- China trade sensitive
- EUR/USD impacts earnings
""",
                "UK100": """
- Commodity/mining heavy
- Banking sector large weight
- GBP movements impact
- Brexit news sensitive
"""
            }
        }
        
        # Get specific sector analysis
        regional_sectors = sector_impacts.get(region, {})
        
        for key, analysis in regional_sectors.items():
            if key in symbol.upper():
                return analysis
        
        return "Diversified index - monitor broad market trends"
    
    def _analyze_gaps_and_levels(self, symbol: str, market_data: Dict) -> str:
        """Analyze gaps and key levels"""
        
        h1_data = market_data.get('h1', {})
        previous_close = h1_data.get('previous_close', 0)
        current_open = h1_data.get('open', 0)
        
        if previous_close and current_open:
            gap_size = ((current_open - previous_close) / previous_close) * 100
            
            if abs(gap_size) > 0.5:
                gap_type = "UP" if gap_size > 0 else "DOWN"
                return f"""
GAP DETECTED: {gap_type} gap of {abs(gap_size):.2f}%
- Gap fill target: {previous_close}
- Typical fill probability: 70% within 2 days
- Trade with gap direction initially
- Watch for exhaustion at gap fill
"""
            else:
                return "No significant gap - normal open"
        
        return """
Key levels to monitor:
- Yesterday's high/low
- Weekly pivot points
- Round numbers (psychological)
- Previous day's close
"""
    
    def _get_correlation_context(self, symbol: str, region: str) -> str:
        """Get correlation context for indices"""
        
        correlations = {
            "US": """
- Positive: Other US indices, risk assets
- Negative: VIX, bonds (during risk-off)
- Watch: Dollar strength impact
- Sector rotation between indices
""",
            "European": """
- Positive: US futures overnight
- Impact: EUR/USD on export stocks
- Watch: US market open at 15:30 CET
- ECB policy impacts all EU indices
""",
            "Asian": """
- Follow: US close sentiment
- Lead: European open direction
- China policy impacts regional indices
- Yen strength affects Nikkei
"""
        }
        
        return correlations.get(region, "Monitor global risk sentiment")
    
    def _get_economic_impact(self, symbol: str, news_context: List[str]) -> str:
        """Extract economic data impact"""
        
        if not news_context:
            return "No major economic data pending"
        
        # Key economic indicators for indices
        key_indicators = [
            "GDP", "Employment", "CPI", "PPI", "Retail Sales",
            "Manufacturing PMI", "Services PMI", "Consumer Confidence",
            "Fed", "ECB", "BOJ", "Interest Rate"
        ]
        
        relevant_news = []
        for news_item in news_context:
            if any(indicator in news_item for indicator in key_indicators):
                relevant_news.append(news_item)
        
        if relevant_news:
            return "HIGH IMPACT DATA:\n" + "\n".join(relevant_news[:3])
        else:
            return "No high-impact economic data in next 4 hours"
    
    def analyze(
        self,
        market_data: Dict[str, Any],
        news_context: Optional[List[str]] = None,
        ml_context: Optional[Dict[str, Any]] = None
    ) -> AgentAnalysis:
        """Perform index specialist analysis"""
        
        try:
            # Get the analysis prompt
            prompt = self.get_analysis_prompt(market_data, news_context)
            
            # Get GPT analysis
            response = self._get_gpt_response(prompt)
            
            # Parse the response
            signal, confidence, reasoning = self._parse_index_response(response)
            
            # Adjust confidence based on index-specific factors
            adjusted_confidence = self._adjust_index_confidence(
                confidence,
                market_data.get('symbol', ''),
                market_data
            )
            
            # Create analysis
            analysis = AgentAnalysis(
                agent_type=self.agent_type,
                signal=signal,
                confidence=adjusted_confidence,
                reasoning=reasoning,
                risk_assessment=self._assess_index_risk(market_data),
                key_levels=self._extract_key_levels(market_data),
                metadata={
                    'index_region': self._get_index_region(market_data.get('symbol', '')),
                    'session': self._get_current_session(market_data.get('symbol', '')),
                    'gap_analysis': self._check_gap(market_data)
                }
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Index specialist analysis failed: {e}")
            raise AgentError(f"Index analysis failed: {str(e)}")
    
    def _parse_index_response(self, response: str) -> tuple:
        """Parse index specialist response"""
        lines = response.strip().split('\n')
        
        signal = SignalType.WAIT
        confidence = 50.0
        reasoning = []
        
        for line in lines:
            line = line.strip()
            
            if 'Signal:' in line or 'SIGNAL:' in line:
                if 'BUY' in line.upper():
                    signal = SignalType.BUY
                elif 'SELL' in line.upper():
                    signal = SignalType.SELL
                    
            elif 'Confidence:' in line or 'CONFIDENCE:' in line:
                try:
                    conf_str = line.split(':')[1].strip().replace('%', '')
                    confidence = float(conf_str)
                except:
                    pass
                    
            elif line.startswith('-') or line.startswith('•'):
                reasoning.append(line.strip('- •'))
        
        return signal, confidence, reasoning[:3]
    
    def _adjust_index_confidence(self, base_confidence: float, symbol: str, market_data: Dict) -> float:
        """Adjust confidence based on index-specific factors"""
        
        adjusted = base_confidence
        
        # Check session quality
        session = self._get_current_session(symbol)
        if "Overnight" in session or "Quiet" in session:
            adjusted *= 0.8  # Reduce confidence in low liquidity
        elif "Active" in session:
            adjusted *= 1.05  # Slight boost during main session
        
        # Check if near market open/close (volatile)
        if self._near_session_boundary(symbol):
            adjusted *= 0.85  # Reduce confidence near open/close
        
        # Gap trades typically have higher probability
        if self._check_gap(market_data) != "No gap":
            adjusted *= 1.1  # Boost for gap scenarios
        
        # Options expiry dates (monthly)
        if self._is_options_expiry():
            adjusted *= 0.9  # Reduce on expiry days
        
        return min(100, max(0, adjusted))
    
    def _near_session_boundary(self, symbol: str) -> bool:
        """Check if near market open or close"""
        now = datetime.utcnow()
        region = self._get_index_region(symbol)
        
        # Define open/close times in UTC
        boundaries = {
            "US": [(13, 30), (20, 45)],      # 9:30 EST, 3:45 EST
            "European": [(7, 0), (15, 30)],   # Market open, close
            "Asian": [(23, 0), (7, 0)]        # Tokyo open, close
        }
        
        region_boundaries = boundaries.get(region, [])
        current_time = now.hour * 60 + now.minute
        
        for hour, minute in region_boundaries:
            boundary_time = hour * 60 + minute
            if abs(current_time - boundary_time) <= 30:  # Within 30 minutes
                return True
        
        return False
    
    def _check_gap(self, market_data: Dict) -> str:
        """Check for gap at open"""
        h1_data = market_data.get('h1', {})
        
        previous_close = h1_data.get('previous_close', 0)
        current_open = h1_data.get('open', 0)
        
        if previous_close and current_open:
            gap_percent = ((current_open - previous_close) / previous_close) * 100
            
            if gap_percent > 0.3:
                return f"Gap up {gap_percent:.2f}%"
            elif gap_percent < -0.3:
                return f"Gap down {abs(gap_percent):.2f}%"
        
        return "No gap"
    
    def _is_options_expiry(self) -> bool:
        """Check if today is options expiry (3rd Friday)"""
        today = datetime.now()
        
        # Find third Friday
        first_day = today.replace(day=1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(weeks=2)
        
        return today.date() == third_friday.date()
    
    def _assess_index_risk(self, market_data: Dict) -> str:
        """Assess index-specific risks"""
        
        symbol = market_data.get('symbol', '').upper()
        session = self._get_current_session(symbol)
        
        risk_factors = []
        
        # Session risk
        if "Overnight" in session:
            risk_factors.append("Low liquidity - wider spreads")
        
        # Volatility risk
        if "US100" in symbol or "NASDAQ" in symbol:
            risk_factors.append("High volatility - tech sensitive")
        
        # Gap risk
        gap = self._check_gap(market_data)
        if gap != "No gap":
            risk_factors.append(f"Gap risk - {gap}")
        
        # Economic data risk
        if self._has_pending_data(market_data):
            risk_factors.append("Economic data pending")
        
        if risk_factors:
            return "RISKS: " + ", ".join(risk_factors)
        else:
            return "MODERATE RISK - Standard index volatility"
    
    def _has_pending_data(self, market_data: Dict) -> bool:
        """Check for pending economic data"""
        # Simplified check - would connect to economic calendar
        now = datetime.utcnow()
        
        # Major data usually at specific times
        data_times = [
            (12, 30),  # 8:30 EST
            (14, 0),   # 10:00 EST  
            (18, 0),   # 14:00 EST (Fed)
        ]
        
        current_minutes = now.hour * 60 + now.minute
        
        for hour, minute in data_times:
            data_minutes = hour * 60 + minute
            if 0 <= data_minutes - current_minutes <= 30:
                return True
        
        return False