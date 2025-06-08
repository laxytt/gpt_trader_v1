#!/usr/bin/env python3
"""
Comprehensive MT5 Connection and Order Management Test Suite
Tests all MT5 functionality to ensure proper setup before live trading
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import MetaTrader5 as mt5

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_settings
from core.infrastructure.mt5.client import MT5Client
from core.infrastructure.mt5.data_provider import MT5DataProvider
from core.infrastructure.mt5.order_manager import MT5OrderManager
from core.domain.models import MarketData, Trade, TradingSignal, SignalType
from core.domain.enums import TimeFrame, OrderType
from core.utils.chart_utils import ChartGenerator


class MT5TestSuite:
    """Comprehensive MT5 testing suite"""
    
    def __init__(self):
        self.settings = get_settings()
        self.mt5_client = None
        self.data_provider = None
        self.order_manager = None
        self.test_results = []
        
    def log_test(self, test_name: str, result: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if result else "❌ FAIL"
        self.test_results.append({
            'test': test_name,
            'result': result,
            'status': status,
            'details': details
        })
        print(f"{status} - {test_name}")
        if details:
            print(f"     Details: {details}")
            
    async def run_all_tests(self):
        """Run all MT5 tests"""
        print("=" * 80)
        print("MT5 CONNECTION AND ORDER MANAGEMENT TEST SUITE")
        print("=" * 80)
        print(f"Time: {datetime.now()}")
        print("=" * 80)
        
        # 1. Test MT5 connection
        await self.test_mt5_connection()
        
        if not self.mt5_client:
            print("\n❌ Cannot proceed without MT5 connection")
            self.print_summary()
            return
            
        # 2. Test account information
        await self.test_account_info()
        
        # 3. Test market data
        await self.test_market_data()
        
        # 4. Test symbol information
        await self.test_symbol_info()
        
        # 5. Test order calculations
        await self.test_order_calculations()
        
        # 6. Test order placement (demo only)
        # Check if it's a demo account from account info
        account_info = self.mt5_client.get_account_info() if self.mt5_client else None
        is_demo = False
        if account_info:
            is_demo = (account_info.get('trade_mode') == 0 or 
                      'demo' in str(account_info.get('server', '')).lower())
        
        if is_demo:
            await self.test_order_placement()
        else:
            print("\n⚠️  Skipping order placement tests (not on demo account)")
            
        # 7. Test historical data
        await self.test_historical_data()
        
        # Print summary
        self.print_summary()
        
    async def test_mt5_connection(self):
        """Test 1: MT5 Connection"""
        print("\n1. TESTING MT5 CONNECTION")
        print("-" * 40)
        
        try:
            # Initialize MT5 client
            self.mt5_client = MT5Client(self.settings.mt5)
            
            # Test initialization
            if self.mt5_client.initialize():
                self.log_test("MT5 initialization", True, 
                            "Connected successfully")
                
                # Get terminal info
                terminal_info = mt5.terminal_info()
                if terminal_info:
                    # Build version string from available attributes
                    version_str = ""
                    if hasattr(terminal_info, 'version'):
                        version_str = f"Version: {terminal_info.version}"
                    elif hasattr(terminal_info, 'build'):
                        version_str = f"Build: {terminal_info.build}"
                    
                    self.log_test("Terminal info retrieval", True,
                                f"{version_str}, Company: {terminal_info.company}")
                else:
                    self.log_test("Terminal info retrieval", False, "No terminal info")
                    
                # Initialize other components
                self.data_provider = MT5DataProvider(self.mt5_client, ChartGenerator())
                self.order_manager = MT5OrderManager(self.mt5_client, self.settings.trading)
                
            else:
                error = mt5.last_error()
                self.log_test("MT5 initialization", False, 
                            f"Error code: {error[0]}, Description: {error[1]}")
                self.mt5_client = None
                
        except Exception as e:
            self.log_test("MT5 connection", False, f"Exception: {str(e)}")
            self.mt5_client = None
            
    async def test_account_info(self):
        """Test 2: Account Information"""
        print("\n2. TESTING ACCOUNT INFORMATION")
        print("-" * 40)
        
        try:
            # Get account info
            account_info = self.mt5_client.get_account_info()
            
            if account_info:
                self.log_test("Account info retrieval", True)
                
                # Display account details
                print(f"\n   Account Details:")
                print(f"   - Login: {account_info.get('login', 'N/A')}")
                print(f"   - Server: {account_info.get('server', 'N/A')}")
                print(f"   - Name: {account_info.get('name', 'N/A')}")
                print(f"   - Company: {account_info.get('company', 'N/A')}")
                print(f"   - Balance: ${account_info.get('balance', 0):.2f}")
                print(f"   - Equity: ${account_info.get('equity', 0):.2f}")
                print(f"   - Margin: ${account_info.get('margin', 0):.2f}")
                print(f"   - Free Margin: ${account_info.get('margin_free', 0):.2f}")
                print(f"   - Leverage: 1:{account_info.get('leverage', 'N/A')}")
                print(f"   - Currency: {account_info.get('currency', 'N/A')}")
                print(f"   - Trade Mode: {'DEMO' if account_info.get('trade_mode') == 0 else 'LIVE'}")
                
                # Test balance check
                self.log_test("Balance check", account_info.get('balance', 0) > 0,
                            f"Balance: ${account_info.get('balance', 0):.2f}")
                
                # Test margin check
                self.log_test("Free margin check", account_info.get('margin_free', 0) > 0,
                            f"Free margin: ${account_info.get('margin_free', 0):.2f}")
                
            else:
                self.log_test("Account info retrieval", False, "No account info available")
                
        except Exception as e:
            self.log_test("Account info test", False, f"Exception: {str(e)}")
            
    async def test_market_data(self):
        """Test 3: Market Data Fetching"""
        print("\n3. TESTING MARKET DATA")
        print("-" * 40)
        
        test_symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        
        for symbol in test_symbols:
            try:
                # Test current price
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    self.log_test(f"Current price for {symbol}", True,
                                f"Bid: {tick.bid}, Ask: {tick.ask}, Spread: {(tick.ask-tick.bid)*10000:.1f} pips")
                else:
                    self.log_test(f"Current price for {symbol}", False, "No tick data")
                    
                # Test market data through data provider
                market_data = await self.data_provider.get_market_data(symbol, TimeFrame.H1, 10)
                if market_data and market_data.candles:
                    self.log_test(f"Historical data for {symbol}", True,
                                f"Retrieved {len(market_data.candles)} candles")
                else:
                    self.log_test(f"Historical data for {symbol}", False, "No candle data")
                    
            except Exception as e:
                self.log_test(f"Market data for {symbol}", False, f"Exception: {str(e)}")
                
    async def test_symbol_info(self):
        """Test 4: Symbol Information"""
        print("\n4. TESTING SYMBOL INFORMATION")
        print("-" * 40)
        
        test_symbol = "EURUSD"
        
        try:
            symbol_info = mt5.symbol_info(test_symbol)
            
            if symbol_info:
                self.log_test(f"Symbol info for {test_symbol}", True)
                
                print(f"\n   Symbol Details for {test_symbol}:")
                print(f"   - Digits: {symbol_info.digits}")
                print(f"   - Point: {symbol_info.point}")
                print(f"   - Tick Value: {symbol_info.trade_tick_value}")
                print(f"   - Tick Size: {symbol_info.trade_tick_size}")
                print(f"   - Contract Size: {symbol_info.trade_contract_size}")
                print(f"   - Min Volume: {symbol_info.volume_min}")
                print(f"   - Max Volume: {symbol_info.volume_max}")
                print(f"   - Volume Step: {symbol_info.volume_step}")
                print(f"   - Stops Level: {symbol_info.trade_stops_level}")
                print(f"   - Spread: {symbol_info.spread}")
                
                # Test tradeable status
                self.log_test(f"{test_symbol} tradeable", symbol_info.visible,
                            f"Trade mode: {symbol_info.trade_mode}")
                
            else:
                self.log_test(f"Symbol info for {test_symbol}", False, "Symbol not found")
                
        except Exception as e:
            self.log_test("Symbol info test", False, f"Exception: {str(e)}")
            
    async def test_order_calculations(self):
        """Test 5: Order Calculations"""
        print("\n5. TESTING ORDER CALCULATIONS")
        print("-" * 40)
        
        test_symbol = "EURUSD"
        test_volume = 0.01  # 0.01 lot
        
        try:
            # Get current price
            tick = mt5.symbol_info_tick(test_symbol)
            if not tick:
                self.log_test("Order calculations", False, "No tick data")
                return
                
            # Test profit calculation
            price_change = 0.0010  # 10 pips
            profit = mt5.order_calc_profit(
                mt5.ORDER_TYPE_BUY,
                test_symbol,
                test_volume,
                tick.ask,
                tick.ask + price_change
            )
            
            if profit is not None:
                self.log_test("Profit calculation", True,
                            f"10 pips profit on {test_volume} lot = ${profit:.2f}")
            else:
                self.log_test("Profit calculation", False, "Calculation failed")
                
            # Test margin calculation
            margin = mt5.order_calc_margin(
                mt5.ORDER_TYPE_BUY,
                test_symbol,
                test_volume,
                tick.ask
            )
            
            if margin is not None:
                self.log_test("Margin calculation", True,
                            f"Required margin for {test_volume} lot = ${margin:.2f}")
            else:
                self.log_test("Margin calculation", False, "Calculation failed")
                
            # Test lot size calculation based on risk
            account_info = mt5.account_info()
            if account_info:
                risk_amount = account_info.balance * 0.01  # 1% risk
                sl_distance = 0.0020  # 20 pips
                
                # Calculate lot size for 1% risk
                pip_value = mt5.order_calc_profit(
                    mt5.ORDER_TYPE_BUY,
                    test_symbol,
                    test_volume,
                    tick.ask,
                    tick.ask + 0.0001  # 1 pip
                )
                
                if pip_value:
                    pips_in_sl = sl_distance / 0.0001
                    calculated_lot = risk_amount / (pip_value * pips_in_sl * (1/test_volume))
                    
                    self.log_test("Lot size calculation", True,
                                f"For ${risk_amount:.2f} risk with 20 pip SL = {calculated_lot:.3f} lots")
                                
        except Exception as e:
            self.log_test("Order calculations", False, f"Exception: {str(e)}")
            
    async def test_order_placement(self):
        """Test 6: Order Placement (Demo Only)"""
        print("\n6. TESTING ORDER PLACEMENT (DEMO ONLY)")
        print("-" * 40)
        
        # Check if it's a demo account
        account_info = self.mt5_client.get_account_info()
        is_demo = False
        if account_info:
            # Check if it's demo by trade_mode or server name
            is_demo = (account_info.get('trade_mode') == 0 or 
                      'demo' in str(account_info.get('server', '')).lower())
        
        if not is_demo:
            print("⚠️  Skipping - Not on demo account")
            return
            
        test_symbol = "EURUSD"
        test_volume = 0.01  # Minimum lot size
        
        try:
            # Get current price
            tick = mt5.symbol_info_tick(test_symbol)
            if not tick:
                self.log_test("Order placement", False, "No tick data")
                return
                
            # Prepare order request
            sl_distance = 0.0020  # 20 pips
            tp_distance = 0.0030  # 30 pips
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": test_symbol,
                "volume": test_volume,
                "type": mt5.ORDER_TYPE_BUY,
                "price": tick.ask,
                "sl": tick.ask - sl_distance,
                "tp": tick.ask + tp_distance,
                "deviation": 20,
                "magic": 123456,
                "comment": "MT5 test order",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            print(f"\n   Test Order Details:")
            print(f"   - Symbol: {test_symbol}")
            print(f"   - Type: BUY")
            print(f"   - Volume: {test_volume}")
            print(f"   - Price: {tick.ask}")
            print(f"   - SL: {request['sl']} (-20 pips)")
            print(f"   - TP: {request['tp']} (+30 pips)")
            
            # Check order
            result = mt5.order_check(request)
            # ORDER_CHECK returns 0 for valid orders
            if result and result.retcode == 0:
                self.log_test("Order validation", True, "Order parameters valid")
                
                # Actually place the order
                print("\n   Placing test order...")
                result = mt5.order_send(request)
                
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    self.log_test("Order placement", True, 
                                f"Order #{result.order} placed successfully")
                    
                    # Wait a moment
                    await asyncio.sleep(2)
                    
                    # Test order modification
                    await self.test_order_modification(result.order)
                    
                    # Test position closure
                    await self.test_position_closure(test_symbol)
                    
                else:
                    error_desc = result.comment if result else "Unknown error"
                    self.log_test("Order placement", False, 
                                f"Error: {error_desc}")
            else:
                if result:
                    error_desc = f"Code: {result.retcode}, Comment: {result.comment}"
                else:
                    error_desc = "Unknown error"
                self.log_test("Order validation", False, 
                            f"Invalid order: {error_desc}")
                
        except Exception as e:
            self.log_test("Order placement test", False, f"Exception: {str(e)}")
            
    async def test_order_modification(self, order_ticket: int):
        """Test order modification"""
        print("\n   Testing order modification...")
        
        try:
            # Get position info
            positions = mt5.positions_get()
            position = None
            
            for pos in positions:
                if pos.ticket == order_ticket:
                    position = pos
                    break
                    
            if not position:
                self.log_test("Order modification", False, "Position not found")
                return
                
            # Modify SL/TP
            new_sl = position.price_open - 0.0015  # Tighten SL to 15 pips
            new_tp = position.price_open + 0.0025  # Reduce TP to 25 pips
            
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "position": position.ticket,
                "sl": new_sl,
                "tp": new_tp,
                "magic": 123456,
            }
            
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.log_test("Order modification", True, 
                            "SL/TP modified successfully")
            else:
                error_desc = result.comment if result else "Unknown error"
                self.log_test("Order modification", False, 
                            f"Error: {error_desc}")
                
        except Exception as e:
            self.log_test("Order modification", False, f"Exception: {str(e)}")
            
    async def test_position_closure(self, symbol: str):
        """Test position closure"""
        print("\n   Testing position closure...")
        
        try:
            # Get open positions
            positions = mt5.positions_get(symbol=symbol)
            
            if not positions:
                self.log_test("Position closure", False, "No positions to close")
                return
                
            position = positions[0]  # Close first position
            
            # Prepare close request
            tick = mt5.symbol_info_tick(symbol)
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
                "position": position.ticket,
                "price": tick.bid if position.type == 0 else tick.ask,
                "deviation": 20,
                "magic": 123456,
                "comment": "Close test position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.log_test("Position closure", True, 
                            f"Position #{position.ticket} closed successfully")
            else:
                error_desc = result.comment if result else "Unknown error"
                self.log_test("Position closure", False, 
                            f"Error: {error_desc}")
                
        except Exception as e:
            self.log_test("Position closure", False, f"Exception: {str(e)}")
            
    async def test_historical_data(self):
        """Test 7: Historical Data Retrieval"""
        print("\n7. TESTING HISTORICAL DATA")
        print("-" * 40)
        
        test_symbol = "EURUSD"
        timeframes = [TimeFrame.M5, TimeFrame.H1, TimeFrame.H4]
        
        for tf in timeframes:
            try:
                # Get historical data
                market_data = await self.data_provider.get_market_data(
                    test_symbol, tf, 100
                )
                
                if market_data and market_data.candles:
                    candles = market_data.candles
                    self.log_test(f"Historical data {tf.value}", True,
                                f"Retrieved {len(candles)} candles")
                    
                    # Check data quality
                    if len(candles) > 1:
                        first = candles[0]
                        last = candles[-1]
                        print(f"     First candle: {first.timestamp}")
                        print(f"     Last candle: {last.timestamp}")
                        print(f"     Latest close: {last.close}")
                        
                else:
                    self.log_test(f"Historical data {tf.value}", False, 
                                "No data retrieved")
                    
            except Exception as e:
                self.log_test(f"Historical data {tf.value}", False, 
                            f"Exception: {str(e)}")
                
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
        print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
        
        if failed_tests > 0:
            print("\nFailed Tests:")
            for test in self.test_results:
                if not test['result']:
                    print(f"  - {test['test']}: {test['details']}")
                    
        print("\n" + "=" * 80)
        
        # Shutdown MT5
        if self.mt5_client:
            self.mt5_client.shutdown()
            

async def main():
    """Run MT5 test suite"""
    test_suite = MT5TestSuite()
    await test_suite.run_all_tests()
    

if __name__ == "__main__":
    print("Starting MT5 Connection and Order Management Tests...")
    print("Make sure you are logged into your MT5 terminal (preferably demo account)")
    print("")
    
    asyncio.run(main())