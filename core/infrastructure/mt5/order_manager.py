"""
MT5 order manager for trade execution and position management.
Handles order placement, modification, and position tracking.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

from core.infrastructure.mt5.client import MT5Client
from core.domain.models import Trade, TradingSignal, TradeStatus, SignalType
from core.domain.exceptions import (
    MT5OrderError, TradeExecutionError, RiskManagementError,
    ErrorContext, ErrorMessages
)
from core.domain.enums import (
    OrderType, TradeAction, ReturnCode, TradingConstants,
    SymbolConfig, SIGNAL_TO_ORDER_TYPE
)
from config.settings import TradingSettings


logger = logging.getLogger(__name__)


class MT5OrderManager:
    """
    Manages trade execution and position management through MT5.
    Handles order placement, modification, and risk calculations.
    """
    
    def __init__(self, mt5_client: MT5Client, trading_config: TradingSettings):
        self.mt5_client = mt5_client
        self.trading_config = trading_config
        self.magic_number = TradingConstants.MAGIC_NUMBER
    
    def execute_signal(
        self, 
        signal: TradingSignal, 
        risk_amount_usd: Optional[float] = None
    ) -> Optional[Trade]:
        """
        Execute a trading signal by placing an order.
        
        Args:
            signal: Trading signal to execute
            risk_amount_usd: Risk amount in USD, calculated if not provided
            
        Returns:
            Trade object if successful, None otherwise
            
        Raises:
            TradeExecutionError: If trade execution fails
        """
        if not signal.is_actionable:
            logger.warning(f"Cannot execute WAIT signal for {signal.symbol}")
            return None
        
        with ErrorContext("Trade execution", symbol=signal.symbol) as ctx:
            ctx.add_detail("signal_type", signal.signal.value)
            ctx.add_detail("entry_price", signal.entry)
            
            # Validate signal parameters
            self._validate_signal(signal)
            
            # Calculate position size
            if risk_amount_usd is None:
                risk_amount_usd = self._calculate_risk_amount(signal.risk_class.value)
            
            lot_size = self._calculate_lot_size(
                signal.symbol,
                signal.entry,
                signal.stop_loss,
                risk_amount_usd
            )
            
            if lot_size <= 0:
                raise TradeExecutionError("Invalid lot size calculated")
            
            # Prepare order request
            order_request = self._build_order_request(signal, lot_size)
            
            # Execute order
            result = self.mt5_client.send_order(order_request)
            
            if not self.mt5_client.check_order_result(result):
                error_msg = f"Order failed: {result.get('comment', 'Unknown error')}"
                raise MT5OrderError(error_msg, retcode=result.get('retcode'))
            
            # Create trade object
            trade = self._create_trade_from_result(signal, result, lot_size, risk_amount_usd)
            
            logger.info(f"Trade executed successfully: {trade.symbol} {trade.side.value} "
                       f"at {trade.entry_price}, ticket {trade.ticket}")
            
            return trade
    
    def _validate_signal(self, signal: TradingSignal):
        """Validate signal parameters before execution"""
        if not signal.entry or not signal.stop_loss or not signal.take_profit:
            raise TradeExecutionError("Signal missing required price levels")
        
        if signal.entry <= 0 or signal.stop_loss <= 0 or signal.take_profit <= 0:
            raise TradeExecutionError("Invalid price levels in signal")
        
        # Validate risk/reward ratio
        if signal.risk_reward and signal.risk_reward < TradingConstants.MIN_RISK_REWARD_RATIO:
            raise RiskManagementError(
                f"Risk/reward ratio {signal.risk_reward} below minimum {TradingConstants.MIN_RISK_REWARD_RATIO}"
            )
        
        # Check if market is open
        if not self.mt5_client.is_market_open(signal.symbol):
            raise TradeExecutionError(f"Market closed for {signal.symbol}")
    
    def _calculate_risk_amount(self, risk_class: str) -> float:
        """Calculate risk amount based on risk class and account balance"""
        account_info = self.mt5_client.get_account_info()
        if not account_info:
            raise TradeExecutionError("Cannot get account information")
        
        balance = account_info.get('balance', 0)
        if balance <= 0:
            raise InsufficientFundsError("Account balance is zero or negative")
        
        # Get risk percentage based on class
        from core.domain.enums import RiskTiers
        risk_multiplier = RiskTiers.TIER_RISK_MAPPING.get(risk_class, 1.0)
        base_risk_percent = self.trading_config.risk_per_trade_percent
        
        risk_amount = balance * (base_risk_percent * risk_multiplier) / 100
        
        logger.debug(f"Risk calculation: Balance={balance}, Class={risk_class}, "
                    f"Multiplier={risk_multiplier}, Risk=${risk_amount:.2f}")
        
        return risk_amount
    
    def _calculate_lot_size(
        self, 
        symbol: str, 
        entry: float, 
        stop_loss: float, 
        risk_amount_usd: float
    ) -> float:
        """Calculate appropriate lot size based on risk parameters"""
        
        # Get symbol information
        symbol_info = self.mt5_client.get_symbol_info(symbol)
        if not symbol_info:
            raise TradeExecutionError(f"Cannot get symbol info for {symbol}")
        
        # Get symbol configuration
        symbol_config = SymbolConfig.SYMBOL_SPECIFICATIONS.get(symbol, {})
        
        # Calculate risk per lot
        point = symbol_info['point']
        contract_size = symbol_info['trade_contract_size']
        
        sl_distance = abs(entry - stop_loss)
        if sl_distance < point * 2:
            raise RiskManagementError("Stop loss too close to entry price")
        
        # Risk per lot in account currency
        risk_per_lot = (sl_distance / point) * (contract_size * point)
        
        # Add spread and commission costs
        spread_pips = self.mt5_client.get_spread(symbol) or 1.0
        commission_per_lot = symbol_config.get('commission_per_lot', 0)
        
        spread_cost = spread_pips * (contract_size * point * 10)  # Convert pips to points
        total_cost_per_lot = risk_per_lot + spread_cost + commission_per_lot
        
        # Calculate lot size
        lot_size = risk_amount_usd / total_cost_per_lot
        
        # Apply symbol constraints
        min_lot = symbol_info.get('volume_min', TradingConstants.MIN_LOT_SIZE)
        max_lot = symbol_info.get('volume_max', TradingConstants.MAX_LOT_SIZE)
        lot_step = symbol_info.get('volume_step', TradingConstants.LOT_STEP)
        
        # Round to valid lot size
        lot_size = max(min_lot, round(lot_size / lot_step) * lot_step)
        lot_size = min(lot_size, max_lot)
        
        logger.debug(f"Lot size calculation for {symbol}: "
                    f"Risk=${risk_amount_usd:.2f}, SL distance={sl_distance:.5f}, "
                    f"Calculated lots={lot_size:.2f}")
        
        return lot_size
    
    def _build_order_request(self, signal: TradingSignal, lot_size: float) -> Dict[str, Any]:
        """Build MT5 order request from signal"""
        
        # Get current prices
        tick = self.mt5_client.get_symbol_tick(signal.symbol)
        if not tick:
            raise TradeExecutionError(f"Cannot get current price for {signal.symbol}")
        
        # Determine order type and price
        if signal.signal == SignalType.BUY:
            order_type = OrderType.BUY
            price = tick['ask']
        else:
            order_type = OrderType.SELL
            price = tick['bid']
        
        # Build request
        request = {
            'action': TradeAction.DEAL,
            'symbol': signal.symbol,
            'volume': lot_size,
            'type': order_type,
            'price': price,
            'sl': signal.stop_loss,
            'tp': signal.take_profit,
            'deviation': 10,  # Price deviation in points
            'magic': self.magic_number,
            'comment': f"GPT Signal {signal.risk_class.value}",
            'type_time': 0,  # GTC (Good Till Cancelled)
            'type_filling': 1  # IOC (Immediate Or Cancel)
        }
        
        return request
    
    def _create_trade_from_result(
        self, 
        signal: TradingSignal, 
        result: Dict[str, Any], 
        lot_size: float,
        risk_amount_usd: float
    ) -> Trade:
        """Create Trade object from successful order result"""
        
        from core.domain.models import create_trade_id
        
        trade_id = create_trade_id(signal.symbol, datetime.now(timezone.utc))
        
        trade = Trade(
            id=trade_id,
            symbol=signal.symbol,
            side=signal.signal,
            entry_price=result.get('price', signal.entry),
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            status=TradeStatus.OPEN,
            timestamp=datetime.now(timezone.utc),
            ticket=result.get('order'),
            lot_size=lot_size,
            risk_reward_ratio=signal.risk_reward,
            risk_amount_usd=risk_amount_usd,
            original_signal=signal
        )
        
        return trade
    
    def modify_position(
        self, 
        trade: Trade, 
        new_stop_loss: Optional[float] = None,
        new_take_profit: Optional[float] = None
    ) -> bool:
        """
        Modify an existing position's stop loss or take profit.
        
        Args:
            trade: Trade to modify
            new_stop_loss: New stop loss level
            new_take_profit: New take profit level
            
        Returns:
            bool: True if modification successful
        """
        if not trade.ticket:
            logger.error("Cannot modify trade without ticket number")
            return False
        
        with ErrorContext("Position modification", symbol=trade.symbol) as ctx:
            ctx.add_detail("ticket", trade.ticket)
            ctx.add_detail("new_sl", new_stop_loss)
            ctx.add_detail("new_tp", new_take_profit)
        
            # Prepare modification request
            request = {
                'action': TradeAction.SLTP,
                'position': trade.ticket,
                'symbol': trade.symbol,
                'sl': new_stop_loss or trade.stop_loss,
                'tp': new_take_profit or trade.take_profit,
                'magic': self.magic_number,
                'comment': "GPT Modification"
            }
            
            # Send modification request
            result = self.mt5_client.send_order(request)
            
            if self.mt5_client.check_order_result(result):
                # Update trade object
                if new_stop_loss:
                    trade.stop_loss = new_stop_loss
                if new_take_profit:
                    trade.take_profit = new_take_profit
                
                logger.info(f"Position modified: {trade.symbol} ticket {trade.ticket}")
                return True
            else:
                logger.error(f"Position modification failed: {result}")
                return False
    
    def close_position(self, trade: Trade) -> bool:
        """
        Close an existing position.
        
        Args:
            trade: Trade to close
            
        Returns:
            bool: True if position closed successfully
        """
        if not trade.ticket:
            logger.error("Cannot close trade without ticket number")
            return False
        
        with ErrorContext("Position closing", symbol=trade.symbol) as ctx:
            ctx.add_detail("ticket", trade.ticket)
        
            # Get current position info
            positions = self.mt5_client.get_positions(symbol=trade.symbol)
            position = None
            
            for pos in positions:
                if pos['ticket'] == trade.ticket:
                    position = pos
                    break
            
            if not position:
                logger.warning(f"Position {trade.ticket} not found, may already be closed")
                return True
            
            # Get current price for closing
            tick = self.mt5_client.get_symbol_tick(trade.symbol)
            if not tick:
                raise TradeExecutionError(f"Cannot get current price for {trade.symbol}")
            
            # Determine close parameters
            if trade.side == SignalType.BUY:
                close_type = OrderType.SELL
                close_price = tick['bid']
            else:
                close_type = OrderType.BUY
                close_price = tick['ask']
            
            # Prepare close request
            request = {
                'action': TradeAction.DEAL,
                'symbol': trade.symbol,
                'volume': position['volume'],
                'type': close_type,
                'position': trade.ticket,
                'price': close_price,
                'deviation': 10,
                'magic': self.magic_number,
                'comment': "GPT Close",
                'type_time': 0,
                'type_filling': 1
            }
            
            # Execute close order
            result = self.mt5_client.send_order(request)
            
            if self.mt5_client.check_order_result(result):
                # Update trade status
                trade.status = TradeStatus.CLOSED
                trade.exit_price = close_price
                trade.exit_timestamp = datetime.now(timezone.utc)
                
                logger.info(f"Position closed: {trade.symbol} ticket {trade.ticket} at {close_price}")
                return True
            else:
                logger.error(f"Position close failed: {result}")
                return False
    
    def get_position_status(self, trade: Trade) -> Optional[Dict[str, Any]]:
        """
        Get current status of a position.
        
        Args:
            trade: Trade to check
            
        Returns:
            Dict with position status or None if not found
        """
        if not trade.ticket:
            return None
        
        positions = self.mt5_client.get_positions(symbol=trade.symbol)
        
        for position in positions:
            if position['ticket'] == trade.ticket:
                return {
                    'ticket': position['ticket'],
                    'volume': position['volume'],
                    'price_open': position['price_open'],
                    'price_current': position['price_current'],
                    'profit': position['profit'],
                    'swap': position['swap'],
                    'sl': position['sl'],
                    'tp': position['tp']
                }
        
        return None
    
    def calculate_current_pnl(self, trade: Trade) -> Optional[float]:
        """
        Calculate current P&L for a trade.
        
        Args:
            trade: Trade to calculate P&L for
            
        Returns:
            Current P&L in account currency or None if failed
        """
        position_status = self.get_position_status(trade)
        if not position_status:
            return None
        
        return position_status.get('profit', 0) + position_status.get('swap', 0)
    
    def is_position_open(self, trade: Trade) -> bool:
        """
        Check if a position is still open.
        
        Args:
            trade: Trade to check
            
        Returns:
            bool: True if position is open
        """
        return self.get_position_status(trade) is not None


# Export main class
__all__ = ['MT5OrderManager']