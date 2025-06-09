"""
Commodity Specialist Agent
Specializes in trading energy, metals, and agricultural commodities
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from core.agents.base_agent import BaseAgent, AgentType, AgentAnalysis
from core.domain.models import SignalType, RiskClass
from core.domain.exceptions import AgentError

logger = logging.getLogger(__name__)


class CommoditySpecialist(BaseAgent):
    """
    Specialist agent for commodity trading with deep understanding of:
    - Supply/demand dynamics
    - Seasonal patterns
    - Inventory reports
    - Geopolitical impacts
    - Dollar correlation
    """
    
    def __init__(self, gpt_client):
        super().__init__(
            agent_type=AgentType.TECHNICAL_ANALYST,  # Reuse type for now
            gpt_client=gpt_client
        )
        self.agent_name = "Commodity Specialist"
        
    def get_analysis_prompt(self, market_data: Dict, news_context: Optional[List[str]] = None) -> str:
        """Generate commodity-specific analysis prompt"""
        
        symbol = market_data.get('symbol', 'UNKNOWN')
        h1_data = market_data.get('h1', {})
        h4_data = market_data.get('h4', {})
        
        # Determine commodity type
        commodity_type = self._get_commodity_type(symbol)
        
        prompt = f"""You are a Commodity Trading Specialist analyzing {symbol}.

COMMODITY TYPE: {commodity_type}

MARKET DATA:
- Current Price: {h1_data.get('close', 'N/A')}
- H1 ATR: {h1_data.get('atr', 'N/A')}
- H1 Volume: {h1_data.get('volume', 'N/A')}
- H4 Trend: {self._determine_trend(h4_data)}

SPECIALIZED ANALYSIS REQUIRED:

1. SUPPLY/DEMAND ANALYSIS:
{self._get_supply_demand_factors(symbol, commodity_type)}

2. SEASONAL PATTERNS:
{self._get_seasonal_analysis(symbol, commodity_type)}

3. DOLLAR CORRELATION:
- Analyze inverse correlation with USD strength
- Current USD trend impact on commodity

4. INVENTORY/FUNDAMENTAL DATA:
{self._get_inventory_context(symbol, commodity_type, news_context)}

5. TECHNICAL PATTERNS:
- Commodities often trend stronger than forex
- Look for breakout/momentum patterns
- Volume analysis is crucial

6. RISK FACTORS:
{self._get_commodity_risks(symbol, commodity_type)}

TRADING RECOMMENDATION:
Based on your specialized commodity analysis, provide:
- Signal: BUY/SELL/WAIT
- Confidence: 0-100%
- Key reasons (max 3 bullet points)
- Primary risk factor

Remember:
- Commodities are more volatile than forex
- Position sizes should be reduced
- Wider stops are necessary
- News events cause dramatic spikes
"""

        return prompt
    
    def _get_commodity_type(self, symbol: str) -> str:
        """Categorize commodity type"""
        symbol_upper = symbol.upper()
        
        if symbol_upper in ['XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD']:
            return "Precious Metal"
        elif symbol_upper in ['WTIUSD', 'UKOUSD', 'NATGAS']:
            return "Energy"
        elif symbol_upper in ['CORN', 'WHEAT', 'SOYBEAN', 'COFFEE', 'COCOA']:
            return "Agricultural"
        else:
            return "Unknown Commodity"
    
    def _get_supply_demand_factors(self, symbol: str, commodity_type: str) -> str:
        """Get relevant supply/demand factors"""
        
        factors = {
            "Energy": """
- OPEC production decisions
- US shale production levels
- Global demand growth (China, India)
- Strategic reserve releases
- Refinery capacity/maintenance
""",
            "Precious Metal": """
- Central bank buying/selling
- Jewelry demand (seasonal)
- Industrial demand
- Mining supply constraints
- ETF inflows/outflows
""",
            "Agricultural": """
- Weather conditions in key regions
- Planting/harvest progress
- Export demand changes
- Government subsidy changes
- Storage/inventory levels
"""
        }
        
        return factors.get(commodity_type, "Standard supply/demand analysis")
    
    def _get_seasonal_analysis(self, symbol: str, commodity_type: str) -> str:
        """Get seasonal patterns for commodity"""
        
        month = datetime.now().month
        
        seasonal_patterns = {
            "NATGAS": {
                "bullish_months": [10, 11, 12, 1, 2],  # Winter demand
                "bearish_months": [5, 6, 7, 8],        # Summer low demand
                "pattern": "Strong winter demand, weak summer"
            },
            "WTIUSD": {
                "bullish_months": [5, 6, 7, 8],        # Driving season
                "bearish_months": [2, 3, 4],           # Refinery maintenance
                "pattern": "Summer driving season strength"
            },
            "CORN": {
                "bullish_months": [5, 6, 7],           # Planting uncertainty
                "bearish_months": [10, 11, 12],        # Harvest pressure
                "pattern": "Weather premium in growing season"
            },
            "XAUUSD": {
                "bullish_months": [8, 9, 10, 11],      # Indian wedding season
                "bearish_months": [3, 4, 5],           # Typically weak
                "pattern": "Q4 jewelry demand, summer doldrums"
            }
        }
        
        pattern_info = seasonal_patterns.get(symbol.upper(), {})
        
        if pattern_info:
            is_bullish_season = month in pattern_info.get("bullish_months", [])
            is_bearish_season = month in pattern_info.get("bearish_months", [])
            
            if is_bullish_season:
                return f"BULLISH SEASONAL PERIOD - {pattern_info['pattern']}"
            elif is_bearish_season:
                return f"BEARISH SEASONAL PERIOD - {pattern_info['pattern']}"
            else:
                return f"NEUTRAL SEASONAL PERIOD - {pattern_info['pattern']}"
        
        return "No strong seasonal pattern currently"
    
    def _get_inventory_context(self, symbol: str, commodity_type: str, news_context: List[str]) -> str:
        """Extract inventory/fundamental context from news"""
        
        if not news_context:
            return "No recent inventory data available"
        
        # Keywords to look for in news
        inventory_keywords = {
            "Energy": ["EIA", "inventory", "stockpile", "storage", "API", "crude stocks"],
            "Precious Metal": ["ETF holdings", "COMEX", "vault", "delivery"],
            "Agricultural": ["USDA", "crop report", "harvest", "planting", "stocks"]
        }
        
        relevant_news = []
        keywords = inventory_keywords.get(commodity_type, [])
        
        for news_item in news_context:
            if any(keyword.lower() in news_item.lower() for keyword in keywords):
                relevant_news.append(news_item)
        
        if relevant_news:
            return "\n".join(relevant_news[:3])  # Top 3 relevant items
        else:
            return "No specific inventory news, trade with caution"
    
    def _get_commodity_risks(self, symbol: str, commodity_type: str) -> str:
        """Get commodity-specific risk factors"""
        
        risks = {
            "NATGAS": """
- EXTREME VOLATILITY: Can move 5-10% in minutes
- Weather forecast changes
- Storage report surprises
- Pipeline disruptions
""",
            "WTIUSD": """
- OPEC surprise announcements
- Geopolitical events (Middle East)
- Dollar strength changes
- Economic data (demand proxy)
""",
            "XAUUSD": """
- FOMC decisions (inverse correlation)
- Real yield changes
- Risk-on/off sentiment shifts
- Central bank buying/selling
""",
            "Agricultural": """
- Weather disasters
- Trade policy changes
- Currency fluctuations (exporters)
- Crop report surprises
"""
        }
        
        specific_risk = risks.get(symbol.upper(), "")
        general_risk = risks.get(commodity_type, "Standard commodity volatility risk")
        
        return specific_risk or general_risk
    
    def analyze(
        self,
        market_data: Dict[str, Any],
        news_context: Optional[List[str]] = None,
        ml_context: Optional[Dict[str, Any]] = None
    ) -> AgentAnalysis:
        """Perform commodity specialist analysis"""
        
        try:
            # Get the analysis prompt
            prompt = self.get_analysis_prompt(market_data, news_context)
            
            # Get GPT analysis
            response = self._get_gpt_response(prompt)
            
            # Parse the response
            signal, confidence, reasoning = self._parse_commodity_response(response)
            
            # Adjust confidence based on commodity-specific factors
            adjusted_confidence = self._adjust_commodity_confidence(
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
                risk_assessment=self._assess_commodity_risk(market_data),
                key_levels=self._extract_key_levels(market_data),
                metadata={
                    'commodity_type': self._get_commodity_type(market_data.get('symbol', '')),
                    'original_confidence': confidence,
                    'seasonal_bias': self._get_seasonal_bias(market_data.get('symbol', ''))
                }
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Commodity specialist analysis failed: {e}")
            raise AgentError(f"Commodity analysis failed: {str(e)}")
    
    def _parse_commodity_response(self, response: str) -> tuple:
        """Parse commodity specialist response"""
        # Similar to base parsing but with commodity-specific handling
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
        
        return signal, confidence, reasoning[:3]  # Max 3 reasons
    
    def _adjust_commodity_confidence(self, base_confidence: float, symbol: str, market_data: Dict) -> float:
        """Adjust confidence based on commodity-specific factors"""
        
        adjusted = base_confidence
        
        # High volatility commodities need higher confidence threshold
        high_vol_commodities = ['NATGAS', 'XPDUSD', 'COCOA']
        if symbol.upper() in high_vol_commodities:
            # Reduce confidence by 10% for high volatility
            adjusted *= 0.9
        
        # Check if we're near inventory report time (dangerous)
        if self._near_inventory_report(symbol):
            adjusted *= 0.8  # 20% confidence reduction
        
        # Boost confidence if strong seasonal pattern
        seasonal_bias = self._get_seasonal_bias(symbol)
        if seasonal_bias != 'neutral':
            adjusted *= 1.1  # 10% boost for seasonal alignment
        
        return min(100, max(0, adjusted))
    
    def _near_inventory_report(self, symbol: str) -> bool:
        """Check if near major inventory report"""
        
        now = datetime.now()
        day_of_week = now.weekday()
        hour = now.hour
        
        # EIA Crude report: Wednesday 10:30 EST
        if symbol.upper() in ['WTIUSD', 'UKOUSD'] and day_of_week == 2:
            if 14 <= hour <= 16:  # Around 10:30 EST in UTC
                return True
        
        # Natural Gas storage: Thursday 10:30 EST  
        if symbol.upper() == 'NATGAS' and day_of_week == 3:
            if 14 <= hour <= 16:
                return True
                
        return False
    
    def _get_seasonal_bias(self, symbol: str) -> str:
        """Get current seasonal bias"""
        month = datetime.now().month
        
        seasonal_biases = {
            'NATGAS': {
                'bullish': [10, 11, 12, 1, 2],
                'bearish': [5, 6, 7, 8]
            },
            'WTIUSD': {
                'bullish': [5, 6, 7, 8],
                'bearish': [2, 3, 4]
            },
            'XAUUSD': {
                'bullish': [8, 9, 10, 11],
                'bearish': [3, 4, 5]
            }
        }
        
        bias_config = seasonal_biases.get(symbol.upper(), {})
        
        if month in bias_config.get('bullish', []):
            return 'bullish'
        elif month in bias_config.get('bearish', []):
            return 'bearish'
        else:
            return 'neutral'
    
    def _assess_commodity_risk(self, market_data: Dict) -> str:
        """Assess commodity-specific risks"""
        
        symbol = market_data.get('symbol', '').upper()
        
        risk_assessments = {
            'NATGAS': "EXTREME RISK - Use 0.3x position size",
            'WTIUSD': "HIGH RISK - Use 0.5x position size", 
            'XAUUSD': "MODERATE RISK - Use 0.7x position size",
            'CORN': "MODERATE RISK - Weather dependent"
        }
        
        return risk_assessments.get(symbol, "ELEVATED RISK - Reduce position size")