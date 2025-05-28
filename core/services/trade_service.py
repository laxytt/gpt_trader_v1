"""
Trade service for managing trade lifecycle and execution.
Handles trade execution, monitoring, and management decisions.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone, timedelta

from core.domain.models import (
    RiskClass, Trade, TradingSignal, TradeManagementDecision, TradeStatus, 
    TradeResult, ManagementDecision
)
from core.domain.exceptions import (
    TradeExecutionError, ErrorContext,
    RiskManagementError
)
from core.infrastructure.mt5.order_manager import MT5OrderManager
from core.infrastructure.gpt.client import GPTClient
from core.infrastructure.database.repositories import TradeRepository
from core.infrastructure.gpt.reflection_generator import GPTReflectionGenerator
from core.services.news_service import NewsService
from core.services.memory_service import MemoryService
from core.utils.validation import TradeValidator
from config.settings import TradingSettings


logger = logging.getLogger(__name__)


class TradeManagementAnalyzer:
    """Analyzes trades for management decisions using GPT"""
    
    def __init__(self, gpt_client: GPTClient, prompts_dir: str = "config/prompts"):
        self.gpt_client = gpt_client
        self.management_prompt = self._load_management_prompt(prompts_dir)
    
    def _load_management_prompt(self, prompts_dir: str) -> str:
        """Load trade management prompt from file"""
        try:
            prompt_path = Path(prompts_dir) / "management_prompt.txt"
            
            if not prompt_path.exists():
                return self._get_default_management_prompt()
            
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Error loading management prompt: {e}, using default")
            return self._get_default_management_prompt()
    
    def _get_default_management_prompt(self) -> str:
        """Get default management prompt if file not available"""
        return """
You are an expert trade manager analyzing an open position using strict VSA and FTMO risk management rules.

**Your Analysis:**
- Review H4 trend/background and H1 recent structure
- Assess current price action vs original trade thesis
- Consider upcoming news events and market volatility
- Evaluate risk/reward progress and position health

**Management Options:**
- HOLD: Keep position open (trend/logic intact, no immediate threats)
- MOVE_SL: Adjust stop-loss (move to breakeven after +1R, or trail stops)
- CLOSE_NOW: Close immediately (trend reversal, adverse conditions, news risk)

**Decision Criteria:**
- HOLD only if original thesis remains valid and no immediate threats
- MOVE_SL if in profit and structure supports risk reduction
- CLOSE_NOW if trend reversing, major news within 2 min, or position timed out

Always provide clear reasoning referencing H4/H1 context, volume, and VSA principles.

**Response Format (JSON only):**
{
  "decision": "HOLD" | "MOVE_SL" | "CLOSE_NOW",
  "reason": "Brief explanation with H4/H1 context, volume, news considerations",
  "risk_class": "A" | "B" | "C",
  "new_sl": float or null,
  "new_tp": float or null
}
"""
    
    async def analyze_trade_management(
        self,
        trade: Trade,
        market_data: Dict[str, Any],
        news_events: List,
        current_context: Dict[str, Any]
    ) -> TradeManagementDecision:
        """
        Analyze trade for management decision using GPT.
        
        Args:
            trade: Open trade to analyze
            market_data: Current H1/H4 market data
            news_events: Upcoming news events
            current_context: Current market context
            
        Returns:
            TradeManagementDecision object
        """
        with ErrorContext("Trade management analysis", symbol=trade.symbol) as ctx:
            ctx.add_detail("trade_id", trade.id)
            ctx.add_detail("minutes_open", trade.duration_minutes)
            
            # Prepare analysis data
            analysis_data = self._prepare_management_data(
                trade, market_data, news_events, current_context
            )
            
            # Build GPT messages
            messages = self._build_management_messages(analysis_data)
            
            # Call GPT API
            gpt_response = await self.gpt_client.chat_completion(
                messages=messages,
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=600
            )
            
            # Parse response
            decision_dict = self.gpt_client.parse_json_response(gpt_response['content'])
            
            # Validate required keys
            required_keys = ['decision', 'reason', 'risk_class']
            self.gpt_client.validate_response_schema(decision_dict, required_keys)
            
            # Create management decision
            return self._create_management_decision(decision_dict)
    
    def _prepare_management_data(
        self,
        trade: Trade,
        market_data: Dict[str, Any],
        news_events: List,
        current_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare data for management analysis"""
        
        # Calculate trade metrics
        duration_minutes = trade.duration_minutes or 0
        
        return {
            'trade_summary': {
                'symbol': trade.symbol,
                'side': trade.side.value,
                'entry_price': trade.entry_price,
                'stop_loss': trade.stop_loss,
                'take_profit': trade.take_profit,
                'current_pnl': trade.current_pnl,
                'max_drawdown_pips': trade.max_drawdown_pips,
                'duration_minutes': duration_minutes,
                'risk_reward_ratio': trade.risk_reward_ratio
            },
            'market_data': market_data,
            'upcoming_news': news_events[:5],  # Next 5 events
            'market_context': current_context,
            'management_history': trade.management_history[-3:]  # Last 3 actions
        }
    
    def _build_management_messages(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build messages for GPT management analysis"""
        
        trade_summary = analysis_data['trade_summary']
        
        context_text = f"""
Trade Management Analysis:

**Current Position:**
- {trade_summary['symbol']} {trade_summary['side']} 
- Entry: {trade_summary['entry_price']:.5f}
- SL: {trade_summary['stop_loss']:.5f}
- TP: {trade_summary['take_profit']:.5f}
- Duration: {trade_summary['duration_minutes']:.1f} minutes
- P&L: {trade_summary['current_pnl']:.2f}
- Max DD: {trade_summary['max_drawdown_pips']:.1f} pips

**Market Context:**
- Session: {analysis_data['market_context'].get('session', 'Unknown')}
- Volatility: {analysis_data['market_context'].get('volatility', 'Unknown')}

**News Events (Next 30 min):**
{self._format_news_events(analysis_data['upcoming_news'])}

**Recent Management Actions:**
{len(analysis_data['management_history'])} previous actions

Please analyze this position and recommend the appropriate management action.
"""
        
        # Add market data context
        if analysis_data.get('market_data'):
            h1_data = analysis_data['market_data'].get('h1', {})
            h4_data = analysis_data['market_data'].get('h4', {})
            
            if h1_data.get('latest_candle'):
                context_text += f"""
**H1 Latest:**
- Close: {h1_data['latest_candle'].get('close', 'N/A'):.5f}
- RSI: {h1_data['latest_candle'].get('rsi14', 'N/A')}
- Volume: {h1_data['latest_candle'].get('volume', 'N/A')}
"""
            
            if h4_data.get('latest_candle'):
                context_text += f"""
**H4 Latest:**
- Close: {h4_data['latest_candle'].get('close', 'N/A'):.5f}
- RSI: {h4_data['latest_candle'].get('rsi14', 'N/A')}
"""
        
        return [
            {"role": "system", "content": self.management_prompt},
            {"role": "user", "content": context_text}
        ]
    
    def _format_news_events(self, news_events: List) -> str:
        """Format news events for prompt"""
        if not news_events:
            return "No high-impact news in next 30 minutes"
        
        formatted = []
        for event in news_events:
            time_str = event.get('timestamp', 'Unknown time')
            title = event.get('title', 'Unknown event')
            impact = event.get('impact', 'unknown')
            formatted.append(f"- {time_str}: {title} ({impact})")
        
        return "\n".join(formatted)
    
    def _create_management_decision(self, decision_dict: Dict[str, Any]) -> TradeManagementDecision:
        """Create TradeManagementDecision from GPT response"""
        from core.domain.models import RiskClass
        
        try:
            decision = ManagementDecision(decision_dict['decision'])
            risk_class = RiskClass(decision_dict['risk_class'])
            
            return TradeManagementDecision(
                decision=decision,
                reason=decision_dict['reason'],
                risk_class=risk_class,
                timestamp=datetime.now(timezone.utc),
                new_stop_loss=decision_dict.get('new_sl'),
                new_take_profit=decision_dict.get('new_tp')
            )
            
        except (ValueError, KeyError) as e:
            logger.error(f"Error creating management decision: {e}")
            # Return safe fallback
            return TradeManagementDecision(
                decision=ManagementDecision.HOLD,
                reason=f"Fallback decision due to parsing error: {str(e)}",
                risk_class=RiskClass.C
            )


class TradeService:
    """
    Manages the complete trade lifecycle from execution to closure.
    """
    
    def __init__(
        self,
        order_manager: MT5OrderManager,
        trade_repository: TradeRepository,
        news_service: NewsService,
        memory_service: MemoryService,
        gpt_client: GPTClient,
        trading_config: TradingSettings
    ):
        self.order_manager = order_manager
        self.trade_repository = trade_repository
        self.news_service = news_service
        self.memory_service = memory_service
        self.trading_config = trading_config
        
        # Initialize components
        self.management_analyzer = TradeManagementAnalyzer(gpt_client)
        self.reflection_generator = GPTReflectionGenerator(gpt_client)  # Add this
        self.validator = TradeValidator()
        
        # Trade timeout configuration
        self.max_trade_duration = timedelta(hours=self.trading_config.max_open_trades or 24)
    
    async def execute_signal(
        self, 
        signal: TradingSignal,
        risk_amount_usd: Optional[float] = None
    ) -> Optional[Trade]:
        """
        Execute a trading signal by opening a position.
        
        Args:
            signal: TradingSignal to execute
            risk_amount_usd: Optional risk amount override
            
        Returns:
            Trade object if successful, None otherwise
        """
        if not signal.is_actionable:
            logger.info(f"Cannot execute WAIT signal for {signal.symbol}")
            return None
        
        with ErrorContext("Trade execution", symbol=signal.symbol) as ctx:
            ctx.add_detail("signal_type", signal.signal.value)
            ctx.add_detail("risk_class", signal.risk_class.value)
            
            try:
                # Validate signal before execution
                self.validator.validate_signal(signal)
                
                # Check current open trades limit
                open_trades = await self.get_open_trades()
                if len(open_trades) >= self.trading_config.max_open_trades:
                    raise RiskManagementError(
                        f"Maximum open trades limit reached: {len(open_trades)}/{self.trading_config.max_open_trades}"
                    )
                
                # Execute trade through order manager
                trade = self.order_manager.execute_signal(signal, risk_amount_usd)
                
                if trade:
                    # Validate and save trade
                    self.validator.validate_trade(trade)
                    self.trade_repository.save(trade)
                    
                    logger.info(f"Trade executed successfully: {trade.id}")
                    return trade
                else:
                    logger.error(f"Order manager returned None for {signal.symbol}")
                    return None
                    
            except Exception as e:
                logger.error(f"Trade execution failed for {signal.symbol}: {e}")
                raise TradeExecutionError(f"Failed to execute trade: {str(e)}")
    
    async def manage_trade(self, trade: Trade) -> bool:
        """
        Manage an active trade by analyzing current conditions.
        
        Args:
            trade: Trade to manage
            
        Returns:
            True if trade is still open, False if closed
        """
        if trade.status != TradeStatus.OPEN:
            logger.warning(f"Attempting to manage non-open trade: {trade.id}")
            return False
        
        with ErrorContext("Trade management", symbol=trade.symbol) as ctx:
            ctx.add_detail("trade_id", trade.id)
            
            try:
                # Check if position still exists in MT5
                if not self.order_manager.is_position_open(trade):
                    logger.info(f"Position closed externally: {trade.id}")
                    await self._handle_external_closure(trade)
                    return False
                
                # Update current P&L
                current_pnl = self.order_manager.calculate_current_pnl(trade)
                if current_pnl is not None:
                    trade.current_pnl = current_pnl
                
                # Check for timeout
                if await self._check_trade_timeout(trade):
                    return False
                
                # Get management decision from GPT
                decision = await self._get_management_decision(trade)
                
                # Execute management decision
                return await self._execute_management_decision(trade, decision)
                
            except Exception as e:
                logger.error(f"Trade management failed for {trade.id}: {e}")
                # Don't close trade on management errors, just log
                return True
    
    async def _check_trade_timeout(self, trade: Trade) -> bool:
        """Check if trade has exceeded maximum duration"""
        if not trade.duration_minutes:
            return False
        
        if trade.duration_minutes > (self.max_trade_duration.total_seconds() / 60):
            logger.info(f"Trade {trade.id} exceeded maximum duration, closing")
            
            success = self.order_manager.close_position(trade)
            if success:
                trade.result = TradeResult.TIMEOUT_CLOSE
                await self._finalize_trade(trade)
            
            return success
        
        return False
    
    async def _get_management_decision(self, trade: Trade) -> TradeManagementDecision:
        """Get management decision from GPT analyzer"""
        try:
            # Gather current market data (simplified for management)
            market_data = {}  # Would fetch current H1/H4 data
            
            # Get upcoming news
            news_events = await self.news_service.get_upcoming_news(
                symbol=trade.symbol,
                within_minutes=30
            )
            
            # Current market context
            current_context = {
                'session': 'unknown',  # Would determine current session
                'volatility': 'medium'  # Would calculate from recent data
            }
            
            # Get GPT analysis
            decision = await self.management_analyzer.analyze_trade_management(
                trade=trade,
                market_data=market_data,
                news_events=news_events,
                current_context=current_context
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Failed to get management decision for {trade.id}: {e}")
            # Return safe fallback
            return TradeManagementDecision(
                decision=ManagementDecision.HOLD,
                reason=f"Management analysis failed: {str(e)}",
                risk_class=trade.original_signal.risk_class if trade.original_signal else RiskClass.C
            )
    
    async def _execute_management_decision(
        self, 
        trade: Trade, 
        decision: TradeManagementDecision
    ) -> bool:
        """Execute the management decision"""
        
        # Record management action
        trade.add_management_action(decision.decision, {
            'reason': decision.reason,
            'risk_class': decision.risk_class.value,
            'new_sl': decision.new_stop_loss,
            'new_tp': decision.new_take_profit
        })
        
        if decision.decision == ManagementDecision.HOLD:
            logger.debug(f"Holding position: {trade.id}")
            self.trade_repository.save(trade)
            return True
        
        elif decision.decision == ManagementDecision.MOVE_SL:
            success = self.order_manager.modify_position(
                trade=trade,
                new_stop_loss=decision.new_stop_loss,
                new_take_profit=decision.new_take_profit
            )
            
            if success:
                logger.info(f"Position modified: {trade.id}")
                self.trade_repository.save(trade)
            
            return True
        
        elif decision.decision == ManagementDecision.CLOSE_NOW:
            success = self.order_manager.close_position(trade)
            
            if success:
                trade.result = TradeResult.GPT_CLOSE
                await self._finalize_trade(trade)
                logger.info(f"Position closed by management: {trade.id}")
                return False
            else:
                logger.error(f"Failed to close position: {trade.id}")
                return True
        
        return True
    
    async def _handle_external_closure(self, trade: Trade):
        """Handle trade that was closed externally (hit SL/TP)"""
        # Determine result based on exit conditions
        if trade.current_pnl > 0:
            trade.result = TradeResult.WIN
        elif trade.current_pnl < 0:
            trade.result = TradeResult.LOSS
        else:
            trade.result = TradeResult.BREAKEVEN
        
        await self._finalize_trade(trade)
    
    # Update the _finalize_trade method
    async def _finalize_trade(self, trade: Trade):
        """Finalize a completed trade"""
        trade.status = TradeStatus.CLOSED
        trade.exit_timestamp = datetime.now(timezone.utc)
        
        # Generate reflection for the trade
        try:
            reflection = await self.reflection_generator.generate_reflection(
                trade=trade,
                h1_data=None,  # Could fetch current market data if needed
                h4_data=None,
                additional_context={
                    'trade_duration': trade.duration_minutes,
                    'market_conditions': 'current'  # Could add more context
                }
            )
            trade.reflection = reflection
        except Exception as e:
            logger.warning(f"Failed to generate reflection for trade {trade.id}: {e}")
            trade.reflection = None
        
        # Save to repository
        self.trade_repository.save(trade)
        
        # Add to memory for future learning
        if trade.original_signal and trade.result:
            await self.memory_service.add_trade_case(trade)
        
        logger.info(f"Trade finalized: {trade.id} - {trade.result.value if trade.result else 'unknown'}")
    
    async def get_open_trades(self) -> List[Trade]:
        """Get all currently open trades"""
        try:
            return self.trade_repository.find_open_trades()
        except Exception as e:
            logger.error(f"Failed to get open trades: {e}")
            return []
    
    async def get_trade_by_symbol(self, symbol: str) -> Optional[Trade]:
        """Get open trade for a specific symbol"""
        try:
            trades = self.trade_repository.find_by_symbol(symbol, TradeStatus.OPEN)
            return trades[0] if trades else None
        except Exception as e:
            logger.error(f"Failed to get trade for {symbol}: {e}")
            return None
    
    async def close_all_trades(self, reason: str = "Manual close") -> int:
        """
        Close all open trades.
        
        Args:
            reason: Reason for closing trades
            
        Returns:
            Number of trades closed
        """
        open_trades = await self.get_open_trades()
        closed_count = 0
        
        for trade in open_trades:
            try:
                success = self.order_manager.close_position(trade)
                if success:
                    trade.result = TradeResult.MANUAL_CLOSE
                    await self._finalize_trade(trade)
                    closed_count += 1
            except Exception as e:
                logger.error(f"Failed to close trade {trade.id}: {e}")
        
        logger.info(f"Closed {closed_count}/{len(open_trades)} trades")
        return closed_count


# Export main service
__all__ = ['TradeService', 'TradeManagementAnalyzer']