"""
Async trade service for non-blocking trade operations
"""

import logging
from typing import Optional, List
from datetime import datetime, timezone

from core.domain.models import Trade, TradingSignal, TradeStatus
from core.domain.exceptions import TradeExecutionError
from core.infrastructure.database.async_repositories import AsyncTradeRepository, AsyncSignalRepository
from core.infrastructure.mt5.order_manager import MT5OrderManager
from core.services.news_service import NewsService
from core.services.memory_service import MemoryService
from core.infrastructure.gpt.client import GPTClient
from config.settings import TradingSettings

logger = logging.getLogger(__name__)


class AsyncTradeService:
    """Async version of trade service for high-performance operations"""
    
    def __init__(
        self,
        order_manager: MT5OrderManager,
        trade_repository: AsyncTradeRepository,
        signal_repository: AsyncSignalRepository,
        news_service: NewsService,
        memory_service: MemoryService,
        gpt_client: GPTClient,
        trading_config: TradingSettings,
        portfolio_risk_manager: Optional[Any] = None
    ):
        self.order_manager = order_manager
        self.trade_repo = trade_repository
        self.signal_repo = signal_repository
        self.news_service = news_service
        self.memory_service = memory_service
        self.gpt_client = gpt_client
        self.trading_config = trading_config
        self.portfolio_risk_manager = portfolio_risk_manager
    
    async def execute_signal(self, signal: TradingSignal) -> Optional[Trade]:
        """Execute a trading signal asynchronously"""
        if not signal.is_actionable:
            logger.warning(f"Cannot execute non-actionable signal for {signal.symbol}")
            return None
        
        try:
            # Check portfolio risk
            if self.portfolio_risk_manager:
                risk_check = await self._check_portfolio_risk(signal)
                if not risk_check['allowed']:
                    logger.warning(f"Trade blocked by portfolio risk: {risk_check['reason']}")
                    return None
            
            # Execute order (this is still sync as MT5 is sync)
            trade = self.order_manager.execute_signal(signal)
            
            if trade:
                # Save trade asynchronously
                await self.trade_repo.save_trade(trade)
                
                # Save signal and mark as executed
                signal_id = await self.signal_repo.save_signal(signal)
                await self.signal_repo.mark_signal_executed(signal_id, trade.id)
                
                # Update memory asynchronously
                await self._update_memory_async(trade)
                
                logger.info(f"Trade executed and saved: {trade.id}")
                return trade
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            raise TradeExecutionError(f"Failed to execute signal: {str(e)}")
        
        return None
    
    async def get_open_trades(self) -> List[Trade]:
        """Get all open trades asynchronously"""
        return await self.trade_repo.get_open_trades()
    
    async def get_trade_by_symbol(self, symbol: str) -> Optional[Trade]:
        """Get open trade for a symbol asynchronously"""
        return await self.trade_repo.get_trade_by_symbol(symbol)
    
    async def get_recent_trades(self, limit: int = 10) -> List[Trade]:
        """Get recent trades asynchronously"""
        return await self.trade_repo.get_recent_trades(limit)
    
    async def manage_trade(self, trade: Trade) -> bool:
        """Manage an existing trade asynchronously"""
        if not trade.is_open:
            return False
        
        try:
            # Get current position status (sync operation)
            position_status = self.order_manager.get_position_status(trade)
            if not position_status:
                # Position closed externally
                trade.status = TradeStatus.CLOSED
                trade.exit_timestamp = datetime.now(timezone.utc)
                await self.trade_repo.save_trade(trade)
                return False
            
            # Update trade metrics
            trade.current_pnl = position_status['profit']
            
            # Check if we should manage this trade
            if self._should_manage_trade(trade):
                # Get management decision (could be made async)
                decision = await self._get_management_decision_async(trade, position_status)
                
                if decision:
                    success = self._execute_management_decision(trade, decision)
                    if success:
                        await self.trade_repo.save_trade(trade)
            
            return True
            
        except Exception as e:
            logger.error(f"Trade management failed for {trade.id}: {e}")
            return False
    
    async def _check_portfolio_risk(self, signal: TradingSignal) -> dict:
        """Check portfolio risk constraints asynchronously"""
        # This would be implemented based on portfolio risk manager
        return {'allowed': True, 'reason': None}
    
    async def _update_memory_async(self, trade: Trade):
        """Update memory service asynchronously"""
        # Memory service could be made async
        try:
            self.memory_service.add_trade_case(trade)
        except Exception as e:
            logger.error(f"Failed to update memory: {e}")
    
    async def _get_management_decision_async(self, trade: Trade, position_status: dict):
        """Get trade management decision asynchronously"""
        # This could call GPT asynchronously
        # For now, returning None to skip management
        return None
    
    def _should_manage_trade(self, trade: Trade) -> bool:
        """Check if trade should be actively managed"""
        # Implement logic to determine if trade needs management
        return trade.duration_minutes and trade.duration_minutes > 30
    
    def _execute_management_decision(self, trade: Trade, decision: dict) -> bool:
        """Execute management decision (sync operation)"""
        # This remains sync as MT5 operations are sync
        return True