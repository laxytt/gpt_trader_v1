"""
Market Type Detector
Determines the type of market (forex, commodity, index) and selects appropriate specialists
"""

import logging
from typing import Dict, List, Optional, Tuple
from config.ftmo_symbols import (
    FTMO_COMMODITIES, FTMO_INDICES, FTMO_EXOTIC_FOREX,
    get_ftmo_symbol_spec
)

logger = logging.getLogger(__name__)


class MarketTypeDetector:
    """
    Detects market type and provides specialized configuration
    """
    
    def __init__(self):
        # Build lookup dictionaries
        self.commodities = set(FTMO_COMMODITIES.keys())
        self.indices = set(FTMO_INDICES.keys())
        self.exotic_forex = set(FTMO_EXOTIC_FOREX.keys())
        
        # Major forex pairs
        self.major_forex = {
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF',
            'AUDUSD', 'USDCAD', 'NZDUSD'
        }
        
        # Cross pairs
        self.cross_pairs = {
            'EURJPY', 'GBPJPY', 'EURGBP', 'EURAUD',
            'EURCAD', 'GBPAUD', 'GBPCAD', 'AUDNZD'
        }
    
    def detect_market_type(self, symbol: str) -> Tuple[str, Dict[str, any]]:
        """
        Detect market type and return configuration
        
        Returns:
            Tuple of (market_type, config_dict)
        """
        symbol_upper = symbol.upper()
        
        # Check each market type
        if symbol_upper in self.commodities:
            spec = get_ftmo_symbol_spec(symbol_upper)
            return 'commodity', {
                'type': 'commodity',
                'subtype': self._get_commodity_subtype(symbol_upper),
                'volatility': spec.volatility_profile if spec else 'high',
                'risk_factor': spec.risk_factor if spec else 1.5,
                'use_specialists': ['commodity_specialist', 'vsa_trader'],
                'position_size_multiplier': self._get_position_multiplier(symbol_upper),
                'special_rules': self._get_commodity_rules(symbol_upper)
            }
            
        elif symbol_upper in self.indices:
            spec = get_ftmo_symbol_spec(symbol_upper)
            return 'index', {
                'type': 'index',
                'region': self._get_index_region(symbol_upper),
                'volatility': spec.volatility_profile if spec else 'medium',
                'risk_factor': spec.risk_factor if spec else 1.2,
                'use_specialists': ['index_specialist', 'momentum_trader'],
                'position_size_multiplier': 0.6,
                'special_rules': self._get_index_rules(symbol_upper)
            }
            
        elif symbol_upper in self.major_forex:
            return 'forex_major', {
                'type': 'forex',
                'subtype': 'major',
                'volatility': 'low',
                'risk_factor': 1.0,
                'use_specialists': ['technical_analyst', 'momentum_trader'],
                'position_size_multiplier': 1.0,
                'special_rules': self._get_forex_rules(symbol_upper)
            }
            
        elif symbol_upper in self.cross_pairs:
            return 'forex_cross', {
                'type': 'forex',
                'subtype': 'cross',
                'volatility': 'medium',
                'risk_factor': 1.2,
                'use_specialists': ['technical_analyst', 'contrarian_trader'],
                'position_size_multiplier': 0.8,
                'special_rules': self._get_forex_rules(symbol_upper)
            }
            
        elif symbol_upper in self.exotic_forex:
            spec = get_ftmo_symbol_spec(symbol_upper)
            return 'forex_exotic', {
                'type': 'forex',
                'subtype': 'exotic',
                'volatility': spec.volatility_profile if spec else 'high',
                'risk_factor': spec.risk_factor if spec else 1.7,
                'use_specialists': ['risk_manager', 'contrarian_trader'],
                'position_size_multiplier': 0.4,
                'special_rules': self._get_exotic_rules(symbol_upper)
            }
            
        else:
            # Default to forex major treatment
            logger.warning(f"Unknown symbol type: {symbol}, treating as forex")
            return 'unknown', {
                'type': 'forex',
                'subtype': 'unknown',
                'volatility': 'medium',
                'risk_factor': 1.3,
                'use_specialists': ['technical_analyst', 'risk_manager'],
                'position_size_multiplier': 0.7,
                'special_rules': {}
            }
    
    def _get_commodity_subtype(self, symbol: str) -> str:
        """Get commodity subtype"""
        if symbol in ['XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD']:
            return 'precious_metal'
        elif symbol in ['WTIUSD', 'UKOUSD', 'NATGAS']:
            return 'energy'
        elif symbol in ['CORN', 'WHEAT', 'SOYBEAN', 'COFFEE', 'COCOA']:
            return 'agricultural'
        return 'other'
    
    def _get_index_region(self, symbol: str) -> str:
        """Get index region"""
        if any(x in symbol for x in ['US30', 'US100', 'US500']):
            return 'US'
        elif any(x in symbol for x in ['GER', 'UK100', 'FR40', 'EU50']):
            return 'Europe'
        elif any(x in symbol for x in ['JP225', 'HK50', 'CN50']):
            return 'Asia'
        return 'Global'
    
    def _get_position_multiplier(self, symbol: str) -> float:
        """Get position size multiplier based on symbol risk"""
        
        # Extreme volatility - very small positions
        if symbol in ['NATGAS', 'USDTRY']:
            return 0.3
            
        # High volatility
        elif symbol in ['WTIUSD', 'UKOUSD', 'XPDUSD', 'USDMXN', 'USDZAR']:
            return 0.5
            
        # Moderate volatility
        elif symbol in ['XAUUSD', 'US30.cash', 'GER40.cash']:
            return 0.7
            
        # Normal volatility
        else:
            return 1.0
    
    def _get_commodity_rules(self, symbol: str) -> Dict:
        """Get commodity-specific trading rules"""
        
        rules = {
            'min_atr_multiplier': 1.5,  # Wider stops needed
            'news_blackout_minutes': 30,  # Inventory reports
            'avoid_times': [],
            'preferred_sessions': ['London', 'New York'],
            'special_events': []
        }
        
        # Energy specific
        if symbol in ['WTIUSD', 'UKOUSD']:
            rules['special_events'] = ['EIA Wednesday 10:30 EST']
            rules['avoid_times'] = [(14, 30, 15, 30)]  # UTC
            
        elif symbol == 'NATGAS':
            rules['special_events'] = ['Storage Report Thursday 10:30 EST']
            rules['min_atr_multiplier'] = 2.5  # Extra wide stops
            
        # Precious metals
        elif symbol in ['XAUUSD', 'XAGUSD']:
            rules['correlation_watch'] = 'USD strength (inverse)'
            rules['special_events'] = ['FOMC meetings', 'NFP']
            
        return rules
    
    def _get_index_rules(self, symbol: str) -> Dict:
        """Get index-specific trading rules"""
        
        rules = {
            'gap_trading': True,
            'avoid_times': [],
            'key_times': [],
            'correlation_indices': [],
            'options_expiry_impact': True
        }
        
        # US indices
        if any(x in symbol for x in ['US30', 'US100', 'US500']):
            rules['key_times'] = [
                (13, 30),  # Market open 9:30 EST
                (20, 0),   # Last hour 3:00 EST
            ]
            rules['avoid_times'] = [(13, 30, 14, 30)]  # First hour volatility
            rules['correlation_indices'] = ['VIX (inverse)']
            
        # European indices
        elif any(x in symbol for x in ['GER', 'UK100', 'FR40']):
            rules['key_times'] = [(8, 0), (15, 30)]  # Open and US impact
            rules['correlation_indices'] = ['US futures']
            
        return rules
    
    def _get_forex_rules(self, symbol: str) -> Dict:
        """Get forex-specific trading rules"""
        
        return {
            'spread_threshold': 2.0,  # Max spread in pips
            'news_pairs': self._get_news_currencies(symbol),
            'best_sessions': self._get_forex_sessions(symbol),
            'correlation_pairs': self._get_correlated_pairs(symbol)
        }
    
    def _get_exotic_rules(self, symbol: str) -> Dict:
        """Get exotic forex rules"""
        
        rules = {
            'wider_spreads': True,
            'reduced_liquidity': True,
            'political_risk': True,
            'max_position_pct': 30,  # Max 30% of normal position
        }
        
        if symbol == 'USDTRY':
            rules['extreme_volatility'] = True
            rules['central_bank_risk'] = 'High'
            
        elif symbol in ['USDMXN', 'USDZAR']:
            rules['commodity_correlation'] = True
            
        return rules
    
    def _get_news_currencies(self, symbol: str) -> List[str]:
        """Get currencies affected by news"""
        # Extract base and quote currencies
        if len(symbol) >= 6:
            return [symbol[:3], symbol[3:6]]
        return []
    
    def _get_forex_sessions(self, symbol: str) -> List[str]:
        """Get optimal trading sessions for forex pair"""
        
        sessions_map = {
            'EUR': ['London', 'New York'],
            'GBP': ['London', 'New York'],
            'USD': ['London', 'New York'],
            'JPY': ['Tokyo', 'London'],
            'AUD': ['Sydney', 'Tokyo'],
            'NZD': ['Sydney', 'Tokyo'],
            'CAD': ['New York'],
            'CHF': ['London']
        }
        
        sessions = set()
        currencies = self._get_news_currencies(symbol)
        
        for currency in currencies:
            if currency in sessions_map:
                sessions.update(sessions_map[currency])
        
        return list(sessions)
    
    def _get_correlated_pairs(self, symbol: str) -> List[str]:
        """Get correlated currency pairs"""
        
        correlations = {
            'EURUSD': ['GBPUSD', 'USDCHF (inverse)'],
            'GBPUSD': ['EURUSD', 'GBPJPY'],
            'USDJPY': ['US indices', 'Risk sentiment'],
            'AUDUSD': ['NZDUSD', 'Commodities'],
            'USDCAD': ['Oil (inverse)', 'AUDUSD (inverse)'],
            'XAUUSD': ['EURUSD', 'Silver', 'USD (inverse)']
        }
        
        return correlations.get(symbol, [])
    
    def get_risk_parameters(self, symbol: str) -> Dict[str, float]:
        """
        Get risk parameters for a symbol
        
        Returns dict with:
        - position_size_pct: Percentage of normal position size
        - stop_loss_atr: ATR multiplier for stop loss
        - min_rr_ratio: Minimum risk/reward ratio
        - max_spread_atr: Maximum spread as ATR percentage
        """
        
        market_type, config = self.detect_market_type(symbol)
        
        # Base parameters
        params = {
            'position_size_pct': config['position_size_multiplier'] * 100,
            'stop_loss_atr': 1.5,
            'min_rr_ratio': 2.0,
            'max_spread_atr': 0.3
        }
        
        # Adjust based on market type
        if market_type == 'commodity':
            params['stop_loss_atr'] = 2.0
            params['min_rr_ratio'] = 2.5
            
            if config['subtype'] == 'energy':
                params['stop_loss_atr'] = 2.5
                
        elif market_type == 'index':
            params['stop_loss_atr'] = 1.8
            params['max_spread_atr'] = 0.2
            
        elif market_type == 'forex_exotic':
            params['stop_loss_atr'] = 3.0
            params['min_rr_ratio'] = 3.0
            params['max_spread_atr'] = 0.5
        
        return params
    
    def should_use_specialist(self, symbol: str, agent_type: str) -> bool:
        """Check if a specialist agent should be used for this symbol"""
        
        market_type, config = self.detect_market_type(symbol)
        return agent_type in config.get('use_specialists', [])


# Singleton instance
_detector_instance = None

def get_market_detector() -> MarketTypeDetector:
    """Get singleton instance of market detector"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = MarketTypeDetector()
    return _detector_instance