    def close_position(self, trade: Trade) -> bool:
        """
        Close an existing position with exponential backoff retry.
        
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
            
            def close_with_checks():
                # Get current position info using MT5 directly
                positions = mt5.positions_get(ticket=trade.ticket)
                
                if not positions:
                    # Position might already be closed
                    logger.warning(f"Position {trade.ticket} not found, may already be closed")
                    return {'success': True, 'already_closed': True}
                        
                position = positions[0]
                
                # Get current price for closing
                tick = mt5.symbol_info_tick(trade.symbol)
                if not tick:
                    raise TradeExecutionError(f"Cannot get current price for {trade.symbol}")
                
                # Determine close parameters based on position type
                if position.type == 0:  # BUY position
                    close_type = mt5.ORDER_TYPE_SELL
                    close_price = tick.bid
                else:  # SELL position
                    close_type = mt5.ORDER_TYPE_BUY
                    close_price = tick.ask
                
                # Prepare close request with current position volume
                request = {
                    'action': mt5.TRADE_ACTION_DEAL,
                    'symbol': trade.symbol,
                    'volume': position.volume,  # Use actual position volume
                    'type': close_type,
                    'position': trade.ticket,
                    'price': close_price,
                    'deviation': 20,  # Increased for fast markets
                    'magic': self.magic_number,
                    'comment': "GPT Close",
                    'type_time': mt5.ORDER_TIME_GTC,
                    'type_filling': mt5.ORDER_FILLING_IOC
                }
                
                # Execute close order
                return self.mt5_client.send_order(request)
            
            # Custom retry config for closing positions
            close_retry_config = RetryConfig(
                max_attempts=5,
                initial_delay=0.2,
                max_delay=5.0,
                exponential_base=2.0,
                jitter=True
            )
            
            # Execute with retry logic
            result = retry_mt5_operation(
                close_with_checks,
                config=close_retry_config,
                check_result=lambda r: (
                    r.get('already_closed', False) or 
                    self.mt5_client.check_order_result(r)
                )
            )
            
            if result.get('already_closed'):
                logger.info(f"Position {trade.ticket} was already closed")
                trade.status = TradeStatus.CLOSED
                trade.exit_timestamp = datetime.now(timezone.utc)
                return True
            
            if self.mt5_client.check_order_result(result):
                # Update trade status
                trade.status = TradeStatus.CLOSED
                trade.exit_price = result.get('price', 0)
                trade.exit_timestamp = datetime.now(timezone.utc)
                
                logger.info(f"Position closed: {trade.symbol} ticket {trade.ticket} at {trade.exit_price}")
                return True
            
            logger.error(f"Position close failed after retries: {result}")
            return False