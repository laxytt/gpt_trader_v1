"""
Integration tests for the complete trading flow.
Tests the interaction between all major components.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch

from core.services.trading_orchestrator import TradingOrchestrator, TradingState
from core.services.council_signal_service import CouncilSignalService
from core.services.trade_service import TradeService
from core.services.market_service import MarketService
from core.domain.models import SignalType, TradeStatus
from core.infrastructure.mt5.order_manager import MT5OrderManager


class TestTradingFlow:
    """Test the complete trading flow from signal generation to trade execution."""
    
    @pytest.mark.asyncio
    async def test_successful_trade_flow(
        self,
        test_settings,
        mock_mt5_client,
        mock_gpt_client,
        sample_market_data,
        mock_news_service,
        mock_memory_service,
        trade_repository,
        signal_repository
    ):
        """Test successful trade execution flow."""
        # Setup order manager mock
        order_manager = Mock(spec=MT5OrderManager)
        order_manager.execute_signal.return_value = Mock(
            id="trade-123",
            signal_id="signal-123",
            symbol="EURUSD",
            side="BUY",
            entry_price=1.0800,
            stop_loss=1.0750,
            take_profit=1.0850,
            position_size=0.1,
            ticket=12345,
            status=TradeStatus.OPEN,
            open_time=datetime.now(timezone.utc)
        )
        
        # Create services
        with patch('core.services.council_signal_service.TradingCouncil') as mock_council:
            # Mock council decision
            mock_council_instance = mock_council.return_value
            mock_council_instance.convene_council = AsyncMock(return_value=Mock(
                signal=Mock(
                    symbol="EURUSD",
                    signal=SignalType.BUY,
                    entry_price=1.0800,
                    stop_loss=1.0750,
                    take_profit=1.0850,
                    confidence=75.0,
                    is_actionable=True
                )
            ))
            
            # Create signal service
            signal_service = CouncilSignalService(
                data_provider=Mock(get_market_data=AsyncMock(return_value={
                    'h1': sample_market_data,
                    'h4': sample_market_data
                })),
                gpt_client=mock_gpt_client,
                news_service=mock_news_service,
                memory_service=mock_memory_service,
                signal_repository=signal_repository,
                trading_config=test_settings.trading,
                chart_generator=Mock(),
                screenshots_dir=str(test_settings.paths.screenshots_dir)
            )
            
            # Create trade service
            trade_service = TradeService(
                order_manager=order_manager,
                trade_repository=trade_repository,
                news_service=mock_news_service,
                memory_service=mock_memory_service,
                gpt_client=mock_gpt_client,
                trading_config=test_settings.trading,
                portfolio_risk_manager=Mock(can_open_position=Mock(return_value=True))
            )
            
            # Create market service
            market_service = MarketService(
                data_provider=Mock(),
                trading_config=test_settings.trading
            )
            
            # Create orchestrator
            orchestrator = TradingOrchestrator(
                signal_service=signal_service,
                trade_service=trade_service,
                news_service=mock_news_service,
                memory_service=mock_memory_service,
                market_service=market_service,
                mt5_client=mock_mt5_client,
                trading_config=test_settings.trading,
                settings=test_settings
            )
            
            # Mock trading hours
            with patch.object(orchestrator.scheduler, 'is_trading_hours', return_value=True):
                # Execute one trading cycle
                orchestrator.state = TradingState.RUNNING
                await orchestrator._execute_trading_cycle()
            
            # Verify signal was generated
            signals = signal_repository.get_recent_signals(hours=1)
            assert len(signals) > 0
            signal = signals[0]
            assert signal.symbol == "EURUSD"
            assert signal.signal == SignalType.BUY
            
            # Verify trade was executed
            trades = trade_repository.get_open_trades()
            assert len(trades) == 1
            trade = trades[0]
            assert trade.symbol == "EURUSD"
            assert trade.side == "BUY"
            assert trade.entry_price == 1.0800
    
    @pytest.mark.asyncio
    async def test_risk_manager_veto(
        self,
        test_settings,
        mock_mt5_client,
        mock_gpt_client,
        sample_market_data,
        mock_news_service,
        mock_memory_service,
        signal_repository
    ):
        """Test that risk manager can veto trades."""
        with patch('core.services.council_signal_service.TradingCouncil') as mock_council:
            # Mock council decision with risk manager veto
            mock_council_instance = mock_council.return_value
            mock_council_instance.convene_council = AsyncMock(return_value=Mock(
                signal=Mock(
                    symbol="EURUSD",
                    signal=SignalType.WAIT,
                    confidence=90.0,
                    reason="Risk Manager VETO: High risk detected",
                    is_actionable=False,
                    metadata={'veto': True, 'veto_by': 'risk_manager'}
                )
            ))
            
            # Create signal service
            signal_service = CouncilSignalService(
                data_provider=Mock(get_market_data=AsyncMock(return_value={
                    'h1': sample_market_data,
                    'h4': sample_market_data
                })),
                gpt_client=mock_gpt_client,
                news_service=mock_news_service,
                memory_service=mock_memory_service,
                signal_repository=signal_repository,
                trading_config=test_settings.trading,
                chart_generator=Mock(),
                screenshots_dir=str(test_settings.paths.screenshots_dir)
            )
            
            # Generate signal
            signal = await signal_service.generate_signal("EURUSD")
            
            # Verify veto
            assert signal.signal == SignalType.WAIT
            assert not signal.is_actionable
            assert 'veto' in signal.metadata
            assert signal.metadata['veto_by'] == 'risk_manager'
    
    @pytest.mark.asyncio
    async def test_error_recovery(
        self,
        test_settings,
        mock_mt5_client,
        mock_gpt_client,
        sample_market_data,
        mock_news_service,
        mock_memory_service,
        trade_repository,
        signal_repository
    ):
        """Test system recovery from errors."""
        # Create orchestrator with mocked dependencies
        orchestrator = TradingOrchestrator(
            signal_service=Mock(generate_signal=AsyncMock(side_effect=Exception("Test error"))),
            trade_service=Mock(),
            news_service=mock_news_service,
            memory_service=mock_memory_service,
            market_service=Mock(
                get_market_overview=AsyncMock(return_value={
                    'session': {'current_session': 'London'},
                    'summary': {'excellent_conditions': 1, 'good_conditions': 1, 'average_score': 75}
                }),
                get_market_intelligence=AsyncMock(return_value={}),
                filter_tradeable_symbols=AsyncMock(return_value=["EURUSD"])
            ),
            mt5_client=mock_mt5_client,
            trading_config=test_settings.trading,
            settings=test_settings
        )
        
        # Execute cycle with error
        orchestrator.state = TradingState.RUNNING
        
        # Should handle error without crashing
        with patch.object(orchestrator.scheduler, 'is_trading_hours', return_value=True):
            await orchestrator._execute_trading_cycle()
        
        # Verify error count increased
        assert orchestrator.error_count == 1
        assert orchestrator.state == TradingState.RUNNING  # Should still be running
    
    @pytest.mark.asyncio
    async def test_concurrent_symbol_processing(
        self,
        test_settings,
        mock_mt5_client,
        mock_gpt_client,
        sample_market_data,
        mock_news_service,
        mock_memory_service,
        signal_repository
    ):
        """Test processing multiple symbols concurrently."""
        test_settings.trading.symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        
        signal_results = []
        
        async def mock_generate_signal(symbol):
            await asyncio.sleep(0.1)  # Simulate processing time
            signal_results.append(symbol)
            return Mock(
                symbol=symbol,
                signal=SignalType.WAIT,
                is_actionable=False
            )
        
        # Create orchestrator
        orchestrator = TradingOrchestrator(
            signal_service=Mock(generate_signal=mock_generate_signal),
            trade_service=Mock(),
            news_service=mock_news_service,
            memory_service=mock_memory_service,
            market_service=Mock(
                get_market_overview=AsyncMock(return_value={
                    'session': {'current_session': 'London'},
                    'summary': {'excellent_conditions': 3, 'good_conditions': 0, 'average_score': 85}
                }),
                get_market_intelligence=AsyncMock(return_value={
                    symbol: Mock(score=85, condition=Mock(value='excellent'), data_quality='good')
                    for symbol in test_settings.trading.symbols
                })
            ),
            mt5_client=mock_mt5_client,
            trading_config=test_settings.trading,
            settings=test_settings
        )
        
        # Process symbols
        with patch.object(orchestrator.scheduler, 'is_trading_hours', return_value=True):
            await orchestrator._execute_trading_cycle()
        
        # Verify all symbols were processed
        assert len(signal_results) == 3
        assert set(signal_results) == {"EURUSD", "GBPUSD", "USDJPY"}