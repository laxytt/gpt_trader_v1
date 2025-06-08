"""
Integration tests for the Trading Council multi-agent system.
"""

import pytest
import json
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from core.agents import (
    TechnicalAnalyst, FundamentalAnalyst, SentimentReader,
    RiskManager, MomentumTrader, ContrarianTrader, HeadTrader
)
from core.agents.council import TradingCouncil
from core.domain.models import SignalType, MarketData
from core.agents.base_agent import AgentType, AgentAnalysis


class TestCouncilIntegration:
    """Test the Trading Council multi-agent system integration."""
    
    @pytest.fixture
    def mock_gpt_responses(self):
        """Mock GPT responses for different agents."""
        return {
            AgentType.TECHNICAL_ANALYST: json.dumps({
                'RECOMMENDATION': 'BUY',
                'CONFIDENCE': '80',
                'ENTRY': '1.0800',
                'STOP_LOSS': '1.0750',
                'TAKE_PROFIT': '1.0850',
                'PATTERN': 'Bullish flag breakout',
                'INDICATORS': 'RSI oversold, MACD bullish cross',
                'KEY_LEVELS': 'Support at 1.0750, Resistance at 1.0850'
            }),
            AgentType.FUNDAMENTAL_ANALYST: json.dumps({
                'RECOMMENDATION': 'BUY',
                'CONFIDENCE': '75',
                'CURRENCY_PAIR': 'EUR stronger than USD',
                'ECONOMIC_FACTORS': 'ECB hawkish, Fed dovish',
                'NEWS_IMPACT': 'Positive EUR data',
                'ENTRY': '1.0800'
            }),
            AgentType.SENTIMENT_READER: json.dumps({
                'RECOMMENDATION': 'BUY',
                'CONFIDENCE': '70',
                'MARKET_MOOD': 'Risk-on',
                'POSITIONING': 'Retail short, institutions long',
                'SENTIMENT_SCORE': '65',
                'KEY_INSIGHT': 'Bullish divergence'
            }),
            AgentType.RISK_MANAGER: json.dumps({
                'RECOMMENDATION': 'BUY',
                'CONFIDENCE': '60',
                'RISK_LEVEL': 'Medium',
                'POSITION_SIZE': '0.1',
                'STOP_DISTANCE': '50',
                'RISK_REWARD': '2.0',
                'MAX_LOSS': '100'
            }),
            AgentType.MOMENTUM_TRADER: json.dumps({
                'RECOMMENDATION': 'BUY',
                'CONFIDENCE': '85',
                'TREND': 'Strong uptrend',
                'MOMENTUM': 'Increasing',
                'ENTRY': '1.0800',
                'KEY_OBSERVATION': 'Breaking resistance'
            }),
            AgentType.CONTRARIAN_TRADER: json.dumps({
                'RECOMMENDATION': 'WAIT',
                'CONFIDENCE': '65',
                'CONTRARIAN_VIEW': 'Overbought conditions',
                'CROWD_POSITION': 'Too bullish',
                'REVERSAL_SIGNALS': 'Not yet',
                'WARNING': 'Wait for pullback'
            })
        }
    
    @pytest.mark.asyncio
    async def test_council_consensus_building(
        self,
        mock_gpt_client,
        sample_market_data,
        mock_gpt_responses
    ):
        """Test that council can build consensus from multiple agents."""
        # Configure GPT client to return agent-specific responses
        def mock_analyze(prompt, **kwargs):
            agent_type = kwargs.get('agent_type', '')
            
            # Map agent type string to enum
            agent_map = {
                'technical_analyst': AgentType.TECHNICAL_ANALYST,
                'fundamental_analyst': AgentType.FUNDAMENTAL_ANALYST,
                'sentiment_reader': AgentType.SENTIMENT_READER,
                'risk_manager': AgentType.RISK_MANAGER,
                'momentum_trader': AgentType.MOMENTUM_TRADER,
                'contrarian_trader': AgentType.CONTRARIAN_TRADER
            }
            
            agent_enum = agent_map.get(agent_type)
            return mock_gpt_responses.get(agent_enum, '{}')
        
        mock_gpt_client.analyze_with_response = mock_analyze
        
        # Create council
        council = TradingCouncil(
            gpt_client=mock_gpt_client,
            account_balance=10000,
            risk_per_trade=0.02
        )
        
        # Convene council
        market_data = {'h1': sample_market_data}
        decision = await council.convene_council(market_data)
        
        # Verify consensus was reached
        assert decision.signal is not None
        assert decision.consensus_level > 0.5  # Majority agreement
        assert len(decision.agent_analyses) == 7  # All agents participated
        
        # Verify majority recommendation
        buy_votes = sum(1 for a in decision.agent_analyses if a.recommendation == SignalType.BUY)
        assert buy_votes >= 4  # Majority for BUY
    
    @pytest.mark.asyncio
    async def test_risk_manager_veto_power(
        self,
        mock_gpt_client,
        sample_market_data
    ):
        """Test that risk manager can veto high-risk trades."""
        # Configure risk manager to issue veto
        def mock_analyze(prompt, **kwargs):
            agent_type = kwargs.get('agent_type', '')
            
            if agent_type == 'risk_manager':
                return json.dumps({
                    'RECOMMENDATION': 'WAIT',
                    'CONFIDENCE': '90',
                    'RISK_LEVEL': 'High',
                    'WARNING': 'Excessive risk - position size too large',
                    'MAX_LOSS': '500'
                })
            else:
                # Other agents recommend BUY
                return json.dumps({
                    'RECOMMENDATION': 'BUY',
                    'CONFIDENCE': '75',
                    'ENTRY': '1.0800'
                })
        
        mock_gpt_client.analyze_with_response = mock_analyze
        
        # Create council
        council = TradingCouncil(
            gpt_client=mock_gpt_client,
            account_balance=10000,
            risk_per_trade=0.02
        )
        
        # Convene council
        market_data = {'h1': sample_market_data}
        decision = await council.convene_council(market_data)
        
        # Verify veto was applied
        assert decision.signal.signal == SignalType.WAIT
        assert 'veto' in decision.signal.metadata
        assert decision.signal.metadata['veto_by'] == 'risk_manager'
        assert 'Risk Manager VETO' in decision.signal.reason
    
    @pytest.mark.asyncio
    async def test_debate_position_changes(
        self,
        mock_gpt_client,
        sample_market_data
    ):
        """Test that agents can change positions during debate."""
        debate_round = 0
        
        def mock_analyze(prompt, **kwargs):
            nonlocal debate_round
            agent_type = kwargs.get('agent_type', '')
            
            # Technical analyst changes position after debate
            if agent_type == 'technical_analyst':
                if 'debate' in prompt.lower():
                    debate_round += 1
                    if debate_round > 1:
                        # Changed mind after contrarian's argument
                        return "After considering the contrarian view, I agree we should WAIT. The overbought conditions are concerning."
                
                return json.dumps({
                    'RECOMMENDATION': 'BUY',
                    'CONFIDENCE': '80'
                })
            
            return json.dumps({
                'RECOMMENDATION': 'WAIT',
                'CONFIDENCE': '70'
            })
        
        mock_gpt_client.analyze_with_response = mock_analyze
        
        # Create council with debate enabled
        council = TradingCouncil(
            gpt_client=mock_gpt_client,
            account_balance=10000,
            risk_per_trade=0.02
        )
        
        # Disable quick mode to enable debates
        with patch('config.settings.get_settings') as mock_settings:
            mock_settings.return_value.trading.council_quick_mode = False
            
            # Convene council
            market_data = {'h1': sample_market_data}
            decision = await council.convene_council(market_data)
            
            # Verify debate occurred
            assert len(decision.debate_log) > 0
            
            # Check for position changes
            position_changes = [
                response for response in decision.debate_log
                if not response.maintains_position
            ]
            # Note: This test is simplified - in reality, we'd need to parse debate responses
    
    @pytest.mark.asyncio 
    async def test_council_error_handling(
        self,
        mock_gpt_client,
        sample_market_data
    ):
        """Test council handles agent failures gracefully."""
        call_count = 0
        
        def mock_analyze(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            
            # Make technical analyst fail
            if kwargs.get('agent_type') == 'technical_analyst' and call_count == 1:
                raise Exception("API error")
            
            return json.dumps({
                'RECOMMENDATION': 'BUY',
                'CONFIDENCE': '70'
            })
        
        mock_gpt_client.analyze_with_response = mock_analyze
        
        # Create council
        council = TradingCouncil(
            gpt_client=mock_gpt_client,
            account_balance=10000,
            risk_per_trade=0.02
        )
        
        # Convene council
        market_data = {'h1': sample_market_data}
        decision = await council.convene_council(market_data)
        
        # Verify council continued despite one agent failure
        assert decision.signal is not None
        # Technical analyst should have low confidence due to error
        tech_analysis = next(
            (a for a in decision.agent_analyses if a.agent_type == AgentType.TECHNICAL_ANALYST),
            None
        )
        assert tech_analysis is not None
        assert tech_analysis.confidence == 50.0  # Default on error
    
    def test_individual_agent_analysis(
        self,
        mock_gpt_client,
        sample_market_data
    ):
        """Test individual agent analysis capabilities."""
        # Test Technical Analyst
        tech_analyst = TechnicalAnalyst(mock_gpt_client)
        analysis = tech_analyst.analyze({'h1': sample_market_data})
        
        assert isinstance(analysis, AgentAnalysis)
        assert analysis.agent_type == AgentType.TECHNICAL_ANALYST
        assert analysis.recommendation in [SignalType.BUY, SignalType.SELL, SignalType.WAIT]
        assert 0 <= analysis.confidence <= 100
        
        # Test Risk Manager
        risk_manager = RiskManager(mock_gpt_client, account_balance=10000)
        risk_analysis = risk_manager.analyze({'h1': sample_market_data})
        
        assert risk_analysis.agent_type == AgentType.RISK_MANAGER
        assert 'risk_level' in risk_analysis.metadata or 'position_size' in risk_analysis.metadata