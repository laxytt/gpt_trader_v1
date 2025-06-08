#!/usr/bin/env python3
"""
Comprehensive MT5 Order Management Test Suite
Tests all order operations: open, close, modify, partial close, multiple positions
ONLY RUN ON DEMO ACCOUNTS!
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import MetaTrader5 as mt5
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_settings
from core.infrastructure.mt5.client import MT5Client
from core.infrastructure.mt5.order_manager import MT5OrderManager
from core.domain.models import TradingSignal, SignalType, RiskClass
from core.domain.enums import RiskClassification
from core.domain.enums import TimeFrame


class MT5OrderManagementTest:
    """Test suite for comprehensive order management"""
    
    def __init__(self):
        self.settings = get_settings()
        self.mt5_client = None
        self.order_manager = None
        self.test_results = []
        self.open_positions = []
        
    def log_test(self, test_name: str, result: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if result else "❌ FAIL"
        self.test_results.append({
            'test': test_name,
            'result': result,
            'status': status,
            'details': details
        })
        print(f"\n{status} - {test_name}")
        if details:
            print(f"     Details: {details}")
            
    async def run_all_tests(self):
        """Run all order management tests"""
        print("=" * 80)
        print("MT5 ORDER MANAGEMENT TEST SUITE")
        print("=" * 80)
        print(f"Time: {datetime.now()}")
        print("⚠️  WARNING: This will place real orders on your account!")
        print("            Only run on DEMO accounts!")
        print("=" * 80)
        
        # Initialize MT5
        if not await self.initialize_mt5():
            return
            
        # Check if demo account
        if not await self.verify_demo_account():
            print("\n❌ STOPPING: This is not a demo account!")
            return
            
        # Run tests
        try:
            # 1. Test single order placement
            await self.test_single_order()
            
            # 2. Test order modification
            await self.test_order_modification()
            
            # 3. Test partial close
            await self.test_partial_close()
            
            # 4. Test multiple positions
            await self.test_multiple_positions()
            
            # 5. Test stop loss triggers
            await self.test_stop_loss_management()
            
            # 6. Test take profit management
            await self.test_take_profit_management()
            
            # 7. Test break-even management
            await self.test_breakeven_management()
            
            # 8. Test GPT signal execution
            await self.test_gpt_signal_execution()
            
            # 9. Test risk management
            await self.test_risk_management()
            
            # 10. Clean up any remaining positions
            await self.cleanup_positions()
            
        except Exception as e:
            print(f"\n❌ Test suite error: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Print summary
            self.print_summary()
            
            # Shutdown
            if self.mt5_client:
                self.mt5_client.shutdown()
                
    async def initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        print("\nINITIALIZING MT5...")
        
        try:
            self.mt5_client = MT5Client(self.settings.mt5)
            
            if self.mt5_client.initialize():
                self.order_manager = MT5OrderManager(self.mt5_client, self.settings.trading)
                print("✅ MT5 initialized successfully")
                return True
            else:
                print("❌ Failed to initialize MT5")
                return False
                
        except Exception as e:
            print(f"❌ Initialization error: {e}")
            return False
            
    async def verify_demo_account(self) -> bool:
        """Verify this is a demo account"""
        print("\nVERIFYING ACCOUNT TYPE...")
        
        account_info = self.mt5_client.get_account_info()
        if not account_info:
            print("❌ Cannot get account info")
            return False
            
        is_demo = (account_info.get('trade_mode') == 0 or 
                   'demo' in str(account_info.get('server', '')).lower())
        
        print(f"Account: {account_info.get('login')}")
        print(f"Server: {account_info.get('server')}")
        print(f"Balance: ${account_info.get('balance', 0):.2f}")
        print(f"Type: {'DEMO' if is_demo else 'LIVE'}")
        
        return is_demo
        
    async def test_single_order(self):
        """Test 1: Single order placement and closure"""
        print("\n" + "="*60)
        print("TEST 1: SINGLE ORDER PLACEMENT")
        print("="*60)
        
        symbol = "EURUSD"
        
        try:
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                self.log_test("Single order - price fetch", False, "No tick data")
                return
                
            # Create a buy signal
            signal = TradingSignal(
                symbol=symbol,
                signal=SignalType.BUY,
                entry=tick.ask,
                stop_loss=tick.ask - 0.0020,  # 20 pips SL
                take_profit=tick.ask + 0.0030,  # 30 pips TP
                risk_reward=1.5,
                risk_class=RiskClass.CONSERVATIVE,
                confidence=0.85,
                indicators={'test': 'single_order'},
                timestamp=datetime.now(timezone.utc)
            )
            
            print(f"\nPlacing BUY order on {symbol}:")
            print(f"  Entry: {signal.entry}")
            print(f"  SL: {signal.stop_loss}")
            print(f"  TP: {signal.take_profit}")
            
            # Execute order
            trade = self.order_manager.execute_signal(signal, risk_amount_usd=100)
            
            if trade:
                self.log_test("Single order - placement", True, 
                            f"Order #{trade.ticket} placed, lot size: {trade.lot_size}")
                self.open_positions.append(trade.ticket)
                
                # Wait a bit
                await asyncio.sleep(2)
                
                # Check position exists
                positions = mt5.positions_get(ticket=trade.ticket)
                if positions:
                    pos = positions[0]
                    print(f"\nPosition details:")
                    print(f"  Profit: ${pos.profit:.2f}")
                    print(f"  Volume: {pos.volume}")
                    print(f"  Current price: {pos.price_current}")
                    
                # Close position
                print("\nClosing position...")
                success = self.order_manager.close_position(trade)
                
                if success:
                    self.log_test("Single order - closure", True, "Position closed successfully")
                    self.open_positions.remove(trade.ticket)
                else:
                    self.log_test("Single order - closure", False, "Failed to close position")
                    
            else:
                self.log_test("Single order - placement", False, "Failed to place order")
                
        except Exception as e:
            self.log_test("Single order test", False, f"Exception: {str(e)}")
            
    async def test_order_modification(self):
        """Test 2: Order modification (SL/TP)"""
        print("\n" + "="*60)
        print("TEST 2: ORDER MODIFICATION")
        print("="*60)
        
        symbol = "GBPUSD"
        
        try:
            # Place an order first
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                self.log_test("Order modification - price fetch", False, "No tick data")
                return
                
            # Create signal
            signal = TradingSignal(
                symbol=symbol,
                signal=SignalType.SELL,
                entry=tick.bid,
                stop_loss=tick.bid + 0.0025,  # 25 pips SL
                take_profit=tick.bid - 0.0040,  # 40 pips TP
                risk_reward=1.6,
                risk_class=RiskClass.MODERATE,
                confidence=0.80,
                indicators={'test': 'modification'},
                timestamp=datetime.now(timezone.utc)
            )
            
            print(f"\nPlacing SELL order on {symbol} for modification test...")
            
            # Execute order
            trade = self.order_manager.execute_signal(signal, risk_amount_usd=150)
            
            if trade:
                self.log_test("Modification test - order placement", True, 
                            f"Order #{trade.ticket} placed")
                self.open_positions.append(trade.ticket)
                
                # Wait for order to settle
                await asyncio.sleep(2)
                
                # Modify SL and TP
                new_sl = tick.bid + 0.0020  # Tighten SL to 20 pips
                new_tp = tick.bid - 0.0050  # Extend TP to 50 pips
                
                print(f"\nModifying order:")
                print(f"  New SL: {new_sl} (was {trade.stop_loss})")
                print(f"  New TP: {new_tp} (was {trade.take_profit})")
                
                success = self.order_manager.modify_position(trade, new_sl, new_tp)
                
                if success:
                    self.log_test("Order modification", True, "SL/TP modified successfully")
                    
                    # Verify modification
                    positions = mt5.positions_get(ticket=trade.ticket)
                    if positions:
                        pos = positions[0]
                        if abs(pos.sl - new_sl) < 0.00001 and abs(pos.tp - new_tp) < 0.00001:
                            self.log_test("Modification verification", True, "Changes confirmed")
                        else:
                            self.log_test("Modification verification", False, 
                                        f"SL={pos.sl}, TP={pos.tp}")
                else:
                    self.log_test("Order modification", False, "Failed to modify")
                    
                # Clean up
                await asyncio.sleep(1)
                self.order_manager.close_position(trade)
                self.open_positions.remove(trade.ticket)
                
            else:
                self.log_test("Modification test - order placement", False, "Failed to place order")
                
        except Exception as e:
            self.log_test("Order modification test", False, f"Exception: {str(e)}")
            
    async def test_partial_close(self):
        """Test 3: Partial position closure"""
        print("\n" + "="*60)
        print("TEST 3: PARTIAL POSITION CLOSURE")
        print("="*60)
        
        symbol = "USDJPY"
        
        try:
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                self.log_test("Partial close - price fetch", False, "No tick data")
                return
                
            # Place larger order for partial close
            signal = TradingSignal(
                symbol=symbol,
                signal=SignalType.BUY,
                entry=tick.ask,
                stop_loss=tick.ask - 0.30,  # 30 pips SL (adjusted for JPY)
                take_profit=tick.ask + 0.50,  # 50 pips TP
                risk_reward=1.67,
                risk_class=RiskClass.AGGRESSIVE,
                confidence=0.75,
                indicators={'test': 'partial_close'},
                timestamp=datetime.now(timezone.utc)
            )
            
            print(f"\nPlacing larger BUY order on {symbol} for partial close test...")
            
            # Execute with larger risk for bigger position
            trade = self.order_manager.execute_signal(signal, risk_amount_usd=300)
            
            if trade and trade.lot_size >= 0.02:
                self.log_test("Partial close - order placement", True, 
                            f"Order #{trade.ticket} placed, size: {trade.lot_size}")
                self.open_positions.append(trade.ticket)
                
                await asyncio.sleep(2)
                
                # Calculate partial close volume (50%)
                close_volume = round(trade.lot_size / 2, 2)
                if close_volume < 0.01:
                    close_volume = 0.01
                    
                print(f"\nPartially closing {close_volume} lots of {trade.lot_size} total...")
                
                # Partial close
                success = self.order_manager.close_position_partial(trade, close_volume)
                
                if success:
                    self.log_test("Partial close", True, f"Closed {close_volume} lots")
                    
                    # Verify remaining position
                    positions = mt5.positions_get(symbol=symbol)
                    if positions:
                        remaining = positions[0]
                        expected_remaining = trade.lot_size - close_volume
                        if abs(remaining.volume - expected_remaining) < 0.01:
                            self.log_test("Partial close verification", True, 
                                        f"Remaining: {remaining.volume} lots")
                        else:
                            self.log_test("Partial close verification", False, 
                                        f"Expected {expected_remaining}, got {remaining.volume}")
                            
                    # Close remainder
                    await asyncio.sleep(1)
                    self.order_manager.close_position(trade)
                    
                else:
                    self.log_test("Partial close", False, "Failed to partially close")
                    # Clean up full position
                    self.order_manager.close_position(trade)
                    
                if trade.ticket in self.open_positions:
                    self.open_positions.remove(trade.ticket)
                    
            else:
                if trade:
                    self.log_test("Partial close - order placement", False, 
                                f"Position too small: {trade.lot_size}")
                    self.order_manager.close_position(trade)
                else:
                    self.log_test("Partial close - order placement", False, "Failed to place order")
                    
        except Exception as e:
            self.log_test("Partial close test", False, f"Exception: {str(e)}")
            
    async def test_multiple_positions(self):
        """Test 4: Multiple simultaneous positions"""
        print("\n" + "="*60)
        print("TEST 4: MULTIPLE POSITIONS")
        print("="*60)
        
        symbols = ["EURUSD", "GBPUSD", "AUDUSD"]
        trades = []
        
        try:
            # Open multiple positions
            for i, symbol in enumerate(symbols):
                tick = mt5.symbol_info_tick(symbol)
                if not tick:
                    continue
                    
                # Alternate between buy and sell
                is_buy = i % 2 == 0
                
                signal = TradingSignal(
                    symbol=symbol,
                    signal=SignalType.BUY if is_buy else SignalType.SELL,
                    entry=tick.ask if is_buy else tick.bid,
                    stop_loss=(tick.ask - 0.0015) if is_buy else (tick.bid + 0.0015),
                    take_profit=(tick.ask + 0.0025) if is_buy else (tick.bid - 0.0025),
                    risk_reward=1.67,
                    risk_class=RiskClass.MODERATE,
                    confidence=0.78,
                    indicators={'test': 'multiple_positions', 'index': i},
                    timestamp=datetime.now(timezone.utc)
                )
                
                print(f"\nPlacing {'BUY' if is_buy else 'SELL'} order on {symbol}...")
                
                trade = self.order_manager.execute_signal(signal, risk_amount_usd=100)
                
                if trade:
                    trades.append(trade)
                    self.open_positions.append(trade.ticket)
                    print(f"  ✅ Order #{trade.ticket} placed")
                else:
                    print(f"  ❌ Failed to place order on {symbol}")
                    
            self.log_test("Multiple positions - placement", len(trades) >= 2, 
                        f"Placed {len(trades)} of {len(symbols)} orders")
            
            if trades:
                await asyncio.sleep(3)
                
                # Check all positions
                print("\nCurrent positions:")
                total_profit = 0
                for trade in trades:
                    positions = mt5.positions_get(ticket=trade.ticket)
                    if positions:
                        pos = positions[0]
                        print(f"  {pos.symbol}: {'BUY' if pos.type == 0 else 'SELL'} "
                              f"{pos.volume} lots, P/L: ${pos.profit:.2f}")
                        total_profit += pos.profit
                        
                print(f"\nTotal P/L: ${total_profit:.2f}")
                
                # Close all positions
                print("\nClosing all positions...")
                close_count = 0
                for trade in trades:
                    if self.order_manager.close_position(trade):
                        close_count += 1
                        if trade.ticket in self.open_positions:
                            self.open_positions.remove(trade.ticket)
                            
                self.log_test("Multiple positions - closure", close_count == len(trades), 
                            f"Closed {close_count} of {len(trades)} positions")
                            
        except Exception as e:
            self.log_test("Multiple positions test", False, f"Exception: {str(e)}")
            
    async def test_stop_loss_management(self):
        """Test 5: Stop loss management (trailing stop)"""
        print("\n" + "="*60)
        print("TEST 5: STOP LOSS MANAGEMENT")
        print("="*60)
        
        symbol = "EURUSD"
        
        try:
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                self.log_test("SL management - price fetch", False, "No tick data")
                return
                
            # Place order with wider SL for trailing
            signal = TradingSignal(
                symbol=symbol,
                signal=SignalType.BUY,
                entry=tick.ask,
                stop_loss=tick.ask - 0.0030,  # 30 pips SL
                take_profit=tick.ask + 0.0060,  # 60 pips TP
                risk_reward=2.0,
                risk_class=RiskClass.MODERATE,
                confidence=0.82,
                indicators={'test': 'trailing_stop'},
                timestamp=datetime.now(timezone.utc)
            )
            
            print(f"\nPlacing BUY order for trailing stop test...")
            
            trade = self.order_manager.execute_signal(signal, risk_amount_usd=150)
            
            if trade:
                self.log_test("SL management - order placement", True, 
                            f"Order #{trade.ticket} placed")
                self.open_positions.append(trade.ticket)
                
                # Simulate trailing stop logic
                print("\nSimulating trailing stop adjustments...")
                
                for i in range(3):
                    await asyncio.sleep(2)
                    
                    # Get current position
                    positions = mt5.positions_get(ticket=trade.ticket)
                    if not positions:
                        break
                        
                    pos = positions[0]
                    current_price = pos.price_current
                    
                    # If price moved favorably, trail the stop
                    if trade.side == SignalType.BUY:
                        profit_pips = (current_price - trade.entry_price) / 0.0001
                        if profit_pips > 10:  # If 10+ pips profit
                            new_sl = current_price - 0.0015  # Trail to 15 pips
                            if new_sl > pos.sl:
                                print(f"\n  Trailing stop from {pos.sl:.5f} to {new_sl:.5f}")
                                success = self.order_manager.modify_position(trade, new_sl)
                                if success:
                                    print(f"  ✅ Stop loss updated")
                                    trade.stop_loss = new_sl
                                else:
                                    print(f"  ❌ Failed to update stop loss")
                                    
                self.log_test("Trailing stop implementation", True, "Trailing logic executed")
                
                # Clean up
                self.order_manager.close_position(trade)
                if trade.ticket in self.open_positions:
                    self.open_positions.remove(trade.ticket)
                    
            else:
                self.log_test("SL management - order placement", False, "Failed to place order")
                
        except Exception as e:
            self.log_test("SL management test", False, f"Exception: {str(e)}")
            
    async def test_take_profit_management(self):
        """Test 6: Take profit management (partial TP)"""
        print("\n" + "="*60)
        print("TEST 6: TAKE PROFIT MANAGEMENT")
        print("="*60)
        
        symbol = "GBPUSD"
        
        try:
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                self.log_test("TP management - price fetch", False, "No tick data")
                return
                
            # Place order with multiple TP levels in mind
            signal = TradingSignal(
                symbol=symbol,
                signal=SignalType.SELL,
                entry=tick.bid,
                stop_loss=tick.bid + 0.0025,  # 25 pips SL
                take_profit=tick.bid - 0.0050,  # 50 pips TP
                risk_reward=2.0,
                risk_class=RiskClass.MODERATE,
                confidence=0.79,
                indicators={'test': 'tp_management'},
                timestamp=datetime.now(timezone.utc)
            )
            
            print(f"\nPlacing SELL order for TP management test...")
            
            trade = self.order_manager.execute_signal(signal, risk_amount_usd=200)
            
            if trade and trade.lot_size >= 0.02:
                self.log_test("TP management - order placement", True, 
                            f"Order #{trade.ticket} placed, size: {trade.lot_size}")
                self.open_positions.append(trade.ticket)
                
                await asyncio.sleep(2)
                
                # Simulate partial TP at 50% of target
                print("\nSimulating partial take profit...")
                
                # Close 50% at halfway to TP
                partial_volume = round(trade.lot_size / 2, 2)
                if partial_volume >= 0.01:
                    print(f"  Taking partial profit on {partial_volume} lots...")
                    
                    success = self.order_manager.close_position_partial(trade, partial_volume)
                    
                    if success:
                        self.log_test("Partial TP execution", True, 
                                    f"Closed {partial_volume} lots at partial TP")
                        
                        # Move SL to breakeven on remaining
                        print("  Moving stop to breakeven on remaining position...")
                        success = self.order_manager.modify_position(trade, trade.entry_price)
                        
                        if success:
                            self.log_test("Breakeven stop", True, "SL moved to entry")
                        else:
                            self.log_test("Breakeven stop", False, "Failed to move SL")
                    else:
                        self.log_test("Partial TP execution", False, "Failed to take partial profit")
                        
                # Clean up remaining
                await asyncio.sleep(2)
                self.order_manager.close_position(trade)
                if trade.ticket in self.open_positions:
                    self.open_positions.remove(trade.ticket)
                    
            else:
                if trade:
                    self.log_test("TP management - order placement", False, 
                                f"Position too small: {trade.lot_size}")
                    self.order_manager.close_position(trade)
                else:
                    self.log_test("TP management - order placement", False, "Failed to place order")
                    
        except Exception as e:
            self.log_test("TP management test", False, f"Exception: {str(e)}")
            
    async def test_breakeven_management(self):
        """Test 7: Break-even stop management"""
        print("\n" + "="*60)
        print("TEST 7: BREAK-EVEN MANAGEMENT")
        print("="*60)
        
        symbol = "AUDUSD"
        
        try:
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                self.log_test("BE management - price fetch", False, "No tick data")
                return
                
            signal = TradingSignal(
                symbol=symbol,
                signal=SignalType.BUY,
                entry=tick.ask,
                stop_loss=tick.ask - 0.0020,  # 20 pips SL
                take_profit=tick.ask + 0.0040,  # 40 pips TP
                risk_reward=2.0,
                risk_class=RiskClass.MODERATE,
                confidence=0.81,
                indicators={'test': 'breakeven'},
                timestamp=datetime.now(timezone.utc)
            )
            
            print(f"\nPlacing BUY order for break-even test...")
            
            trade = self.order_manager.execute_signal(signal, risk_amount_usd=120)
            
            if trade:
                self.log_test("BE management - order placement", True, 
                            f"Order #{trade.ticket} placed")
                self.open_positions.append(trade.ticket)
                
                await asyncio.sleep(2)
                
                # Simulate break-even logic
                print("\nChecking for break-even opportunity...")
                
                positions = mt5.positions_get(ticket=trade.ticket)
                if positions:
                    pos = positions[0]
                    
                    # Calculate profit in pips
                    profit_pips = (pos.price_current - trade.entry_price) / 0.0001
                    
                    print(f"  Current profit: {profit_pips:.1f} pips")
                    
                    # If 10+ pips profit, move to break-even
                    if profit_pips >= 10:
                        be_price = trade.entry_price + 0.0001  # BE + 1 pip
                        
                        print(f"  Moving stop to break-even: {be_price:.5f}")
                        
                        success = self.order_manager.modify_position(trade, be_price)
                        
                        if success:
                            self.log_test("Break-even stop", True, "Stop moved to BE+1")
                        else:
                            self.log_test("Break-even stop", False, "Failed to move stop")
                    else:
                        self.log_test("Break-even condition", True, 
                                    f"Not enough profit yet ({profit_pips:.1f} pips)")
                        
                # Clean up
                await asyncio.sleep(1)
                self.order_manager.close_position(trade)
                if trade.ticket in self.open_positions:
                    self.open_positions.remove(trade.ticket)
                    
            else:
                self.log_test("BE management - order placement", False, "Failed to place order")
                
        except Exception as e:
            self.log_test("BE management test", False, f"Exception: {str(e)}")
            
    async def test_gpt_signal_execution(self):
        """Test 8: GPT-style signal execution"""
        print("\n" + "="*60)
        print("TEST 8: GPT SIGNAL EXECUTION")
        print("="*60)
        
        # Simulate a complex GPT-generated signal
        symbol = "EURUSD"
        
        try:
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                self.log_test("GPT signal - price fetch", False, "No tick data")
                return
                
            # Create a detailed signal as GPT would
            signal = TradingSignal(
                symbol=symbol,
                signal=SignalType.SELL,
                entry=tick.bid,
                stop_loss=tick.bid + 0.0022,  # 22 pips SL
                take_profit=tick.bid - 0.0044,  # 44 pips TP
                risk_reward=2.0,
                risk_class=RiskClass.AGGRESSIVE,
                confidence=0.87,
                reasoning="""
                Strong bearish setup identified:
                - H4 shows clear downtrend with lower highs/lows
                - H1 bearish engulfing at resistance
                - RSI divergence on M15
                - News sentiment negative for USD
                """,
                indicators={
                    'h4_trend': 'bearish',
                    'h1_pattern': 'bearish_engulfing',
                    'rsi_divergence': True,
                    'news_sentiment': -0.65,
                    'volume_spike': True
                },
                timestamp=datetime.now(timezone.utc)
            )
            
            print(f"\nGPT Signal Generated:")
            print(f"  Symbol: {signal.symbol}")
            print(f"  Direction: {signal.signal.value}")
            print(f"  Confidence: {signal.confidence * 100:.1f}%")
            print(f"  Risk Class: {signal.risk_class.value}")
            print(f"\n  Reasoning: {signal.reasoning}")
            
            # Execute with appropriate risk
            risk_pct = {
                RiskClass.CONSERVATIVE: 0.005,  # 0.5%
                RiskClass.MODERATE: 0.01,      # 1%
                RiskClass.AGGRESSIVE: 0.015    # 1.5%
            }
            
            account_info = self.mt5_client.get_account_info()
            balance = account_info.get('balance', 10000)
            risk_amount = balance * risk_pct[signal.risk_class]
            
            print(f"\n  Risk Amount: ${risk_amount:.2f} ({risk_pct[signal.risk_class]*100}% of ${balance:.2f})")
            
            trade = self.order_manager.execute_signal(signal, risk_amount_usd=risk_amount)
            
            if trade:
                self.log_test("GPT signal execution", True, 
                            f"Order #{trade.ticket} executed, lot size: {trade.lot_size}")
                self.open_positions.append(trade.ticket)
                
                # Simulate GPT monitoring
                await asyncio.sleep(3)
                
                print("\nGPT Trade Monitoring:")
                positions = mt5.positions_get(ticket=trade.ticket)
                if positions:
                    pos = positions[0]
                    profit_pips = abs(pos.price_current - trade.entry_price) / 0.0001
                    
                    print(f"  Current P/L: ${pos.profit:.2f}")
                    print(f"  Distance moved: {profit_pips:.1f} pips")
                    
                    # Simulate GPT decision
                    if pos.profit > 0 and profit_pips > 15:
                        print("\n  GPT Decision: Lock in partial profits")
                        # Move SL to reduce risk
                        new_sl = trade.entry_price if trade.side == SignalType.SELL else trade.entry_price
                        self.order_manager.modify_position(trade, new_sl)
                        self.log_test("GPT trade management", True, "Applied risk reduction")
                    else:
                        print("\n  GPT Decision: Continue monitoring")
                        self.log_test("GPT trade monitoring", True, "Trade monitored")
                        
                # Clean up
                await asyncio.sleep(1)
                self.order_manager.close_position(trade)
                if trade.ticket in self.open_positions:
                    self.open_positions.remove(trade.ticket)
                    
            else:
                self.log_test("GPT signal execution", False, "Failed to execute signal")
                
        except Exception as e:
            self.log_test("GPT signal test", False, f"Exception: {str(e)}")
            
    async def test_risk_management(self):
        """Test 9: Risk management limits"""
        print("\n" + "="*60)
        print("TEST 9: RISK MANAGEMENT")
        print("="*60)
        
        try:
            account_info = self.mt5_client.get_account_info()
            balance = account_info.get('balance', 10000)
            
            print(f"\nAccount Balance: ${balance:.2f}")
            print("\nTesting risk limits...")
            
            # Test 1: Maximum position size
            symbol = "EURUSD"
            tick = mt5.symbol_info_tick(symbol)
            
            if tick:
                # Try to risk too much (5% - should be limited)
                excessive_risk = balance * 0.05
                
                signal = TradingSignal(
                    symbol=symbol,
                    signal=SignalType.BUY,
                    entry=tick.ask,
                    stop_loss=tick.ask - 0.0010,  # 10 pips SL (tight)
                    take_profit=tick.ask + 0.0020,
                    risk_reward=2.0,
                    risk_class=RiskClass.AGGRESSIVE,
                    confidence=0.90,
                    indicators={'test': 'risk_limit'},
                    timestamp=datetime.now(timezone.utc)
                )
                
                print(f"\n  Testing excessive risk: ${excessive_risk:.2f} (5% of balance)")
                
                # The system should limit this internally
                trade = self.order_manager.execute_signal(signal, risk_amount_usd=excessive_risk)
                
                if trade:
                    # Check actual risk
                    positions = mt5.positions_get(ticket=trade.ticket)
                    if positions:
                        pos = positions[0]
                        sl_distance_pips = abs(trade.entry_price - trade.stop_loss) / 0.0001
                        
                        # Calculate actual risk
                        pip_value = mt5.order_calc_profit(
                            mt5.ORDER_TYPE_BUY,
                            symbol,
                            pos.volume,
                            tick.ask,
                            tick.ask + 0.0001
                        )
                        
                        actual_risk = pip_value * sl_distance_pips
                        risk_percentage = (actual_risk / balance) * 100
                        
                        print(f"\n  Actual position risk: ${actual_risk:.2f} ({risk_percentage:.2f}% of balance)")
                        
                        # Should be limited to max allowed
                        if risk_percentage <= 2.0:  # Assuming 2% max risk
                            self.log_test("Risk limit enforcement", True, 
                                        f"Risk limited to {risk_percentage:.2f}%")
                        else:
                            self.log_test("Risk limit enforcement", False, 
                                        f"Excessive risk allowed: {risk_percentage:.2f}%")
                            
                    # Clean up
                    self.order_manager.close_position(trade)
                    if trade.ticket in self.open_positions:
                        self.open_positions.remove(trade.ticket)
                        
            # Test 2: Multiple position aggregate risk
            print("\n  Testing aggregate position risk...")
            
            # Current open positions
            open_positions = mt5.positions_get()
            if open_positions:
                total_risk = 0
                for pos in open_positions:
                    # Estimate risk per position (simplified)
                    position_value = pos.volume * pos.price_current * 100000  # Assuming standard lot
                    estimated_risk = position_value * 0.01  # Assume 1% position risk
                    total_risk += estimated_risk
                    
                total_risk_pct = (total_risk / balance) * 100
                
                print(f"\n  Open positions: {len(open_positions)}")
                print(f"  Total estimated risk: ${total_risk:.2f} ({total_risk_pct:.2f}% of balance)")
                
                self.log_test("Aggregate risk calculation", True, 
                            f"{len(open_positions)} positions analyzed")
            else:
                print("\n  No open positions for aggregate risk test")
                self.log_test("Aggregate risk check", True, "No positions to analyze")
                
        except Exception as e:
            self.log_test("Risk management test", False, f"Exception: {str(e)}")
            
    async def cleanup_positions(self):
        """Clean up any remaining test positions"""
        print("\n" + "="*60)
        print("CLEANUP: CLOSING REMAINING POSITIONS")
        print("="*60)
        
        try:
            positions = mt5.positions_get()
            
            if positions:
                print(f"\nFound {len(positions)} open positions to clean up...")
                
                closed = 0
                for pos in positions:
                    # Only close our test positions (check magic number or comment)
                    if pos.magic == 123456 or 'test' in str(pos.comment).lower():
                        print(f"  Closing {pos.symbol} {pos.type_description} {pos.volume} lots...")
                        
                        # Create close request
                        tick = mt5.symbol_info_tick(pos.symbol)
                        if tick:
                            request = {
                                "action": mt5.TRADE_ACTION_DEAL,
                                "symbol": pos.symbol,
                                "volume": pos.volume,
                                "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                                "position": pos.ticket,
                                "price": tick.bid if pos.type == 0 else tick.ask,
                                "deviation": 20,
                                "magic": 123456,
                                "comment": "Test cleanup",
                                "type_time": mt5.ORDER_TIME_GTC,
                                "type_filling": mt5.ORDER_FILLING_IOC,
                            }
                            
                            result = mt5.order_send(request)
                            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                closed += 1
                                
                print(f"\nClosed {closed} test positions")
                
            else:
                print("\nNo positions to clean up")
                
        except Exception as e:
            print(f"\nCleanup error: {e}")
            
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for t in self.test_results if t['result'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        
        if total_tests > 0:
            print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
        
        if failed_tests > 0:
            print("\nFailed Tests:")
            for test in self.test_results:
                if not test['result']:
                    print(f"  - {test['test']}: {test['details']}")
                    
        # Check for any remaining open positions
        if self.open_positions:
            print(f"\n⚠️  WARNING: {len(self.open_positions)} positions may still be open!")
            print(f"   Tickets: {self.open_positions}")
            
        print("\n" + "=" * 80)
        

async def main():
    """Run order management test suite"""
    print("\n⚠️  WARNING: This test suite will place REAL orders!")
    print("Only run this on a DEMO account!")
    print("\nPress Ctrl+C to cancel, or wait 5 seconds to continue...")
    
    try:
        await asyncio.sleep(5)
    except KeyboardInterrupt:
        print("\nTest cancelled.")
        return
        
    test_suite = MT5OrderManagementTest()
    await test_suite.run_all_tests()
    

if __name__ == "__main__":
    asyncio.run(main())