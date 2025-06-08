"""
Professional Response Parser for Trading Agents
Ensures consistent and reliable parsing of GPT responses
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from core.domain.models import SignalType

logger = logging.getLogger(__name__)


class ProfessionalResponseParser:
    """Robust parser for agent responses with validation"""
    
    # Expected fields for each agent type
    AGENT_RESPONSE_FIELDS = {
        'technical_analyst': {
            'required': ['PATTERN', 'KEY_LEVELS', 'INDICATORS', 'TREND', 'RECOMMENDATION', 'CONFIDENCE'],
            'optional': ['ENTRY', 'STOP_LOSS', 'TAKE_PROFIT', 'CONCERN'],
            'types': {
                'CONFIDENCE': float,
                'ENTRY': float,
                'STOP_LOSS': float,
                'TAKE_PROFIT': float
            }
        },
        'momentum_trader': {
            'required': ['MOMENTUM_STATE', 'TREND_QUALITY', 'ENTRY_STRATEGY', 'RECOMMENDATION', 'CONFIDENCE'],
            'optional': ['TARGETS', 'INVALIDATION'],
            'types': {
                'CONFIDENCE': float
            }
        },
        'sentiment_reader': {
            'required': ['MOOD', 'CROWD', 'PSYCH_LEVELS', 'SENTIMENT_SIGNS', 'RECOMMENDATION', 'CONFIDENCE'],
            'optional': ['TRIGGERS', 'CONCERN'],
            'types': {
                'CONFIDENCE': float
            }
        },
        'risk_manager': {
            'required': ['RISK_LEVEL', 'POSITION_SIZE', 'STOP_DISTANCE', 'RISK_REWARD', 'RECOMMENDATION', 'CONFIDENCE'],
            'optional': ['MAX_LOSS', 'KEY_RISKS'],
            'types': {
                'POSITION_SIZE': float,
                'STOP_DISTANCE': float,
                'RISK_REWARD': float,
                'CONFIDENCE': float,
                'MAX_LOSS': float
            }
        },
        'contrarian_trader': {
            'required': ['OPPORTUNITY', 'SETUP_TYPE', 'CROWD_ERROR', 'RECOMMENDATION', 'CONFIDENCE'],
            'optional': ['REVERSAL_TRIGGERS', 'INVALIDATION'],
            'types': {
                'CONFIDENCE': float
            }
        },
        'fundamental_analyst': {
            'required': ['BASE_BIAS', 'QUOTE_BIAS', 'MACRO_DRIVERS', 'RECOMMENDATION', 'CONFIDENCE'],
            'optional': ['NEWS_IMPACT', 'KEY_EVENTS', 'CORRELATIONS'],
            'types': {
                'CONFIDENCE': float
            }
        }
    }
    
    @staticmethod
    def parse_response(response: str, agent_type: str, current_price: float) -> Dict[str, Any]:
        """
        Parse agent response with validation and error handling
        
        Args:
            response: Raw GPT response
            agent_type: Type of agent (for validation)
            current_price: Current market price (for defaults)
            
        Returns:
            Parsed and validated response dictionary
        """
        logger.debug(f"Parsing {agent_type} response: {response[:200]}...")
        
        # Get expected fields
        expected = ProfessionalResponseParser.AGENT_RESPONSE_FIELDS.get(
            agent_type, 
            {'required': ['RECOMMENDATION', 'CONFIDENCE'], 'optional': [], 'types': {}}
        )
        
        # Parse response
        parsed = ProfessionalResponseParser._extract_fields(response)
        
        # Validate required fields
        missing = [field for field in expected['required'] if field not in parsed]
        if missing:
            logger.warning(f"{agent_type} response missing required fields: {missing}")
            # Return safe defaults
            return ProfessionalResponseParser._get_safe_defaults(agent_type, current_price)
        
        # Type conversion and validation
        for field, expected_type in expected.get('types', {}).items():
            if field in parsed:
                try:
                    if expected_type == float:
                        # Handle special cases
                        if parsed[field].upper() in ['N/A', 'NONE', '-']:
                            parsed[field] = None
                        else:
                            parsed[field] = float(parsed[field].replace('%', '').replace(',', ''))
                except (ValueError, AttributeError):
                    logger.warning(f"Failed to convert {field} to {expected_type}: {parsed[field]}")
                    parsed[field] = None
        
        # Validate recommendation
        recommendation = parsed.get('RECOMMENDATION', 'WAIT').upper()
        if recommendation not in ['BUY', 'SELL', 'WAIT']:
            logger.warning(f"Invalid recommendation: {recommendation}")
            parsed['RECOMMENDATION'] = 'WAIT'
        
        # Validate confidence
        confidence = parsed.get('CONFIDENCE', 50)
        if isinstance(confidence, (int, float)):
            parsed['CONFIDENCE'] = max(0, min(100, confidence))
        else:
            parsed['CONFIDENCE'] = 50
        
        # Add metadata
        parsed['_agent_type'] = agent_type
        parsed['_parse_success'] = True
        parsed['_missing_fields'] = [f for f in expected['optional'] if f not in parsed]
        
        return parsed
    
    @staticmethod
    def _extract_fields(response: str) -> Dict[str, Any]:
        """Extract key-value pairs from response"""
        parsed = {}
        
        # Try structured format first (FIELD: value)
        lines = response.split('\n')
        for line in lines:
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip().upper()
                    value = parts[1].strip()
                    
                    # Clean up common formatting issues
                    value = value.strip('[]').strip('"').strip("'")
                    
                    # Handle multi-line values
                    if key and value:
                        parsed[key] = value
        
        # Fallback patterns for common fields
        if 'RECOMMENDATION' not in parsed:
            # Look for BUY/SELL/WAIT in response
            if re.search(r'\b(recommend|signal).*?\bBUY\b', response, re.IGNORECASE):
                parsed['RECOMMENDATION'] = 'BUY'
            elif re.search(r'\b(recommend|signal).*?\bSELL\b', response, re.IGNORECASE):
                parsed['RECOMMENDATION'] = 'SELL'
            else:
                parsed['RECOMMENDATION'] = 'WAIT'
        
        if 'CONFIDENCE' not in parsed:
            # Look for confidence patterns
            conf_match = re.search(r'confidence[:\s]+(\d+(?:\.\d+)?)', response, re.IGNORECASE)
            if conf_match:
                parsed['CONFIDENCE'] = conf_match.group(1)
        
        return parsed
    
    @staticmethod
    def _get_safe_defaults(agent_type: str, current_price: float) -> Dict[str, Any]:
        """Get safe default values for failed parsing"""
        defaults = {
            'RECOMMENDATION': 'WAIT',
            'CONFIDENCE': 50,
            'ENTRY': current_price,
            'STOP_LOSS': 0,
            'TAKE_PROFIT': 0,
            '_agent_type': agent_type,
            '_parse_success': False,
            '_error': 'Failed to parse response'
        }
        
        # Agent-specific defaults
        if agent_type == 'risk_manager':
            defaults.update({
                'RISK_LEVEL': 'High',
                'POSITION_SIZE': 0.1,
                'STOP_DISTANCE': 50,
                'RISK_REWARD': 1.0
            })
        elif agent_type == 'technical_analyst':
            defaults.update({
                'PATTERN': 'None clear',
                'KEY_LEVELS': f'support: {current_price * 0.99:.5f}, resistance: {current_price * 1.01:.5f}',
                'INDICATORS': 'Mixed signals',
                'TREND': 'Unclear'
            })
        elif agent_type == 'momentum_trader':
            defaults.update({
                'MOMENTUM_STATE': 'Neutral',
                'TREND_QUALITY': 'Poor',
                'ENTRY_STRATEGY': 'Wait for clarity'
            })
        
        return defaults
    
    @staticmethod
    def validate_trade_parameters(parsed: Dict[str, Any], current_price: float) -> Tuple[bool, List[str]]:
        """
        Validate trade parameters for execution
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if parsed.get('RECOMMENDATION') in ['BUY', 'SELL']:
            # Check entry price
            entry = parsed.get('ENTRY')
            if not entry or entry <= 0:
                issues.append("Invalid or missing entry price")
            elif abs(entry - current_price) / current_price > 0.05:  # 5% away
                issues.append(f"Entry price too far from current: {entry} vs {current_price}")
            
            # Check stop loss
            stop = parsed.get('STOP_LOSS')
            if not stop or stop <= 0:
                issues.append("Invalid or missing stop loss")
            elif parsed['RECOMMENDATION'] == 'BUY' and stop >= entry:
                issues.append("Buy stop loss must be below entry")
            elif parsed['RECOMMENDATION'] == 'SELL' and stop <= entry:
                issues.append("Sell stop loss must be above entry")
            
            # Check take profit
            tp = parsed.get('TAKE_PROFIT')
            if tp and tp > 0:
                if parsed['RECOMMENDATION'] == 'BUY' and tp <= entry:
                    issues.append("Buy take profit must be above entry")
                elif parsed['RECOMMENDATION'] == 'SELL' and tp >= entry:
                    issues.append("Sell take profit must be below entry")
            
            # Check risk/reward
            if entry and stop and tp:
                risk = abs(entry - stop)
                reward = abs(tp - entry)
                if risk > 0:
                    rr = reward / risk
                    if rr < 0.5:
                        issues.append(f"Poor risk/reward ratio: {rr:.2f}")
        
        return len(issues) == 0, issues


class ResponseValidator:
    """Validates consistency across agent responses"""
    
    @staticmethod
    def check_response_consistency(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check for logical consistency across agent responses"""
        issues = []
        
        # Extract recommendations
        recommendations = {}
        for resp in responses:
            agent = resp.get('_agent_type', 'unknown')
            rec = resp.get('RECOMMENDATION', 'WAIT')
            conf = resp.get('CONFIDENCE', 50)
            recommendations[agent] = (rec, conf)
        
        # Check for extreme disagreements
        buy_agents = [a for a, (r, c) in recommendations.items() if r == 'BUY']
        sell_agents = [a for a, (r, c) in recommendations.items() if r == 'SELL']
        
        if buy_agents and sell_agents:
            issues.append(f"Strong disagreement: {buy_agents} say BUY, {sell_agents} say SELL")
        
        # Check risk manager veto
        risk_rec = recommendations.get('risk_manager', ('WAIT', 50))
        if risk_rec[0] == 'WAIT' and risk_rec[1] > 80:
            other_trades = [a for a, (r, c) in recommendations.items() 
                          if a != 'risk_manager' and r != 'WAIT']
            if other_trades:
                issues.append(f"Risk manager strongly opposes trading but {other_trades} want to trade")
        
        # Check confidence consistency
        confidences = [c for r, c in recommendations.values()]
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            high_conf = [a for a, (r, c) in recommendations.items() if c > 85]
            low_conf = [a for a, (r, c) in recommendations.items() if c < 30]
            
            if high_conf and low_conf:
                issues.append(f"Confidence mismatch: {high_conf} very confident, {low_conf} very uncertain")
        
        return {
            'is_consistent': len(issues) == 0,
            'issues': issues,
            'recommendations': recommendations,
            'average_confidence': avg_conf if confidences else 0
        }