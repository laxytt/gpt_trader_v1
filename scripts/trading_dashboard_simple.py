#!/usr/bin/env python3
"""
Simplified Trading Dashboard
A streamlined version without ML dependencies
"""

import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional

# Add project root to Python path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import only essential components
from config.settings import get_settings
from core.domain.models import Trade, TradingSignal
from core.infrastructure.database.repositories import TradeRepository, SignalRepository
from core.infrastructure.mt5.client import MT5Client
from scripts.auth_utils import DashboardAuth

# Initialize authentication
auth = DashboardAuth("Trading Dashboard")

st.set_page_config(
    page_title="Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Protect the app - this will show login form if not authenticated
auth.protect_app()

class SimpleTradingDashboard:
    """Simplified trading dashboard without ML dependencies"""
    
    def __init__(self):
        self.settings = get_settings()
        self.trade_repo = TradeRepository(self.settings.database.db_path)
        self.signal_repo = SignalRepository(self.settings.database.db_path)
        self.mt5_client = None
        
    def run(self):
        """Run the dashboard"""
        st.title("🚀 Trading System Dashboard")
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.radio(
            "Select Page",
            ["Overview", "Open Trades", "Trade History", "Signals", "Performance", "GPT Flow"]
        )
        
        # Initialize MT5 if needed
        if page in ["Overview", "Open Trades"]:
            if not self.mt5_client:
                self.mt5_client = MT5Client(self.settings.mt5)
                if not self.mt5_client.initialize():
                    st.error("Failed to connect to MT5")
                    return
        
        # Route to appropriate page
        if page == "Overview":
            self.show_overview()
        elif page == "Open Trades":
            self.show_open_trades()
        elif page == "Trade History":
            self.show_trade_history()
        elif page == "Signals":
            self.show_signals()
        elif page == "Performance":
            self.show_performance()
        elif page == "GPT Flow":
            self.show_gpt_flow()
    
    def show_overview(self):
        """Show system overview"""
        st.header("System Overview")
        
        # Get current metrics
        try:
            # Account info from MT5
            account_info = self._get_account_info()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Account Balance", f"${account_info['balance']:,.2f}")
            
            with col2:
                st.metric("Equity", f"${account_info['equity']:,.2f}")
            
            with col3:
                st.metric("Open P&L", f"${account_info['profit']:,.2f}", 
                         delta=f"{account_info['profit']:.2f}")
            
            with col4:
                st.metric("Open Trades", account_info['positions'])
            
            # Recent activity
            st.subheader("Recent Activity")
            
            # Get recent trades from all symbols
            trades = []
            for symbol in self.settings.trading.symbols:
                symbol_trades = self.trade_repo.find_recent_trades(symbol, limit=5)
                trades.extend(symbol_trades)
            # Sort by timestamp and get last 10
            trades = sorted(trades, key=lambda x: x.timestamp, reverse=True)[:10]
            if trades:
                df = pd.DataFrame([{
                    'Symbol': t.symbol,
                    'Type': t.side.value,
                    'Open Time': t.timestamp,
                    'P&L': f"${t.current_pnl:.2f}" if t.current_pnl else "Open",
                    'Status': t.status.value
                } for t in trades])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No recent trades")
                
        except Exception as e:
            st.error(f"Error loading overview: {str(e)}")
    
    def show_open_trades(self):
        """Show open trades with management options"""
        st.header("Open Trades")
        
        try:
            positions = self._get_open_positions()
            
            if not positions:
                st.info("No open trades")
                return
            
            # Display positions
            for pos in positions:
                with st.container():
                    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 2])
                    
                    with col1:
                        st.write(f"**{pos['symbol']}** - {pos['type']}")
                        st.write(f"Ticket: {pos['ticket']}")
                    
                    with col2:
                        st.write(f"Volume: {pos['volume']}")
                        st.write(f"Open: {pos['price_open']:.5f}")
                    
                    with col3:
                        st.write(f"Current: {pos['price_current']:.5f}")
                        pnl_color = "green" if pos['profit'] > 0 else "red"
                        st.markdown(f"<span style='color:{pnl_color}'>P&L: ${pos['profit']:.2f}</span>", 
                                  unsafe_allow_html=True)
                    
                    with col4:
                        st.write(f"S/L: {pos['sl']:.5f}")
                        st.write(f"T/P: {pos['tp']:.5f}")
                    
                    with col5:
                        if st.button(f"Close", key=f"close_{pos['ticket']}"):
                            if self._close_position(pos['ticket']):
                                st.success(f"Closed position {pos['ticket']}")
                                st.rerun()
                            else:
                                st.error("Failed to close position")
                    
                    st.divider()
                    
        except Exception as e:
            st.error(f"Error loading positions: {str(e)}")
    
    def show_trade_history(self):
        """Show historical trades"""
        st.header("Trade History")
        
        # Date range filter
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", 
                                      datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
        
        # Get trades
        trades = self.trade_repo.get_trades_by_date_range(start_date, end_date)
        
        if not trades:
            st.info("No trades found in selected period")
            return
        
        # Convert to dataframe
        df = pd.DataFrame([{
            'ID': t.id,
            'Symbol': t.symbol,
            'Type': t.side.value,
            'Volume': t.lot_size,
            'Open Time': t.timestamp,
            'Close Time': t.exit_timestamp,
            'Open Price': t.entry_price,
            'Close Price': t.exit_price,
            'P&L': t.current_pnl,
            'Status': t.status.value
        } for t in trades])
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        closed_trades = df[df['Status'] == 'CLOSED']
        if not closed_trades.empty:
            total_pnl = closed_trades['P&L'].sum()
            win_trades = closed_trades[closed_trades['P&L'] > 0]
            win_rate = len(win_trades) / len(closed_trades) * 100
            
            with col1:
                st.metric("Total Trades", len(df))
            with col2:
                st.metric("Win Rate", f"{win_rate:.1f}%")
            with col3:
                st.metric("Total P&L", f"${total_pnl:.2f}")
            with col4:
                avg_pnl = total_pnl / len(closed_trades)
                st.metric("Avg P&L", f"${avg_pnl:.2f}")
        
        # Display trades
        st.subheader("Trade Details")
        st.dataframe(df, use_container_width=True)
        
        # P&L Chart
        if not closed_trades.empty:
            st.subheader("P&L Over Time")
            fig = px.line(closed_trades, x='Close Time', y='P&L', 
                         title='Trade P&L Over Time')
            st.plotly_chart(fig, use_container_width=True)
    
    def show_signals(self):
        """Show trading signals"""
        st.header("Trading Signals")
        
        # Get recent signals from all symbols
        signals = []
        for symbol in self.settings.trading.symbols:
            symbol_signals = self.signal_repo.get_recent_signals(symbol, limit=20)
            signals.extend(symbol_signals)
        # Sort by created_at and get last 50
        signals = sorted(signals, key=lambda x: x.get('created_at', ''), reverse=True)[:50]
        
        if not signals:
            st.info("No signals found")
            return
        
        # Convert to dataframe (signals are dictionaries)
        df = pd.DataFrame([{
            'ID': s.get('id', ''),
            'Symbol': s.get('symbol', ''),
            'Type': s.get('signal_type', ''),
            'Created': s.get('created_at', ''),
            'Entry': s.get('entry_price', 0),
            'SL': s.get('stop_loss', 0),
            'TP': s.get('take_profit', 0),
            'Risk Class': s.get('risk_class', ''),
            'Executed': '✓' if s.get('is_executed') else '✗'
        } for s in signals])
        
        # Summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Signals", len(df))
        
        with col2:
            executed = df[df['Executed'] == '✓']
            execution_rate = len(executed) / len(df) * 100
            st.metric("Execution Rate", f"{execution_rate:.1f}%")
        
        with col3:
            recent_signals = df[pd.to_datetime(df['Created']) > datetime.now() - timedelta(days=1)]
            st.metric("Last 24h", len(recent_signals))
        
        # Display signals
        st.subheader("Signal History")
        st.dataframe(df, use_container_width=True)
    
    def show_performance(self):
        """Show performance metrics"""
        st.header("Performance Analysis")
        
        # Get all closed trades from date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # Last 90 days
        trades = self.trade_repo.get_trades_by_date_range(start_date, end_date)
        closed_trades = [t for t in trades if t.status.value == 'CLOSED' and t.current_pnl is not None]
        
        if not closed_trades:
            st.info("No closed trades to analyze")
            return
        
        # Calculate metrics
        df = pd.DataFrame([{
            'Symbol': t.symbol,
            'P&L': t.current_pnl,
            'Close Time': t.exit_timestamp if t.exit_timestamp else t.timestamp,
            'Duration': (t.exit_timestamp - t.timestamp).total_seconds() / 3600 if t.exit_timestamp else 0
        } for t in closed_trades])
        
        # Overall metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_pnl = df['P&L'].sum()
        wins = df[df['P&L'] > 0]
        losses = df[df['P&L'] < 0]
        
        with col1:
            st.metric("Total P&L", f"${total_pnl:.2f}")
        
        with col2:
            win_rate = len(wins) / len(df) * 100
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        with col3:
            profit_factor = wins['P&L'].sum() / abs(losses['P&L'].sum()) if not losses.empty else float('inf')
            st.metric("Profit Factor", f"{profit_factor:.2f}")
        
        with col4:
            avg_duration = df['Duration'].mean()
            st.metric("Avg Duration", f"{avg_duration:.1f}h")
        
        # Charts
        st.subheader("Performance Charts")
        
        # Cumulative P&L
        df_sorted = df.sort_values('Close Time')
        df_sorted['Cumulative P&L'] = df_sorted['P&L'].cumsum()
        
        fig1 = px.line(df_sorted, x='Close Time', y='Cumulative P&L',
                      title='Cumulative P&L Over Time')
        st.plotly_chart(fig1, use_container_width=True)
        
        # P&L by Symbol
        pnl_by_symbol = df.groupby('Symbol')['P&L'].sum().sort_values(ascending=False)
        fig2 = px.bar(x=pnl_by_symbol.index, y=pnl_by_symbol.values,
                     title='P&L by Symbol')
        st.plotly_chart(fig2, use_container_width=True)
    
    def _get_account_info(self) -> Dict[str, Any]:
        """Get account information from MT5"""
        if not self.mt5_client:
            return {'balance': 0, 'equity': 0, 'profit': 0, 'positions': 0}
        
        try:
            import MetaTrader5 as mt5
            account_info = mt5.account_info()
            positions = mt5.positions_total()
            
            return {
                'balance': account_info.balance if account_info else 0,
                'equity': account_info.equity if account_info else 0,
                'profit': account_info.profit if account_info else 0,
                'positions': positions
            }
        except:
            return {'balance': 0, 'equity': 0, 'profit': 0, 'positions': 0}
    
    def _get_open_positions(self) -> List[Dict[str, Any]]:
        """Get open positions from MT5"""
        if not self.mt5_client:
            return []
        
        try:
            import MetaTrader5 as mt5
            positions = mt5.positions_get()
            
            if positions is None:
                return []
            
            return [{
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': 'BUY' if pos.type == 0 else 'SELL',
                'volume': pos.volume,
                'price_open': pos.price_open,
                'price_current': pos.price_current,
                'sl': pos.sl,
                'tp': pos.tp,
                'profit': pos.profit
            } for pos in positions]
        except:
            return []
    
    def _close_position(self, ticket: int) -> bool:
        """Close a position"""
        if not self.mt5_client:
            return False
        
        try:
            import MetaTrader5 as mt5
            position = mt5.positions_get(ticket=ticket)
            
            if not position:
                return False
            
            pos = position[0]
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "magic": self.settings.mt5.magic_number,
                "comment": "Dashboard close"
            }
            
            result = mt5.order_send(request)
            return result.retcode == mt5.TRADE_RETCODE_DONE
        except:
            return False
    
    def show_gpt_flow(self):
        """Show GPT Flow Visualization page"""
        # Import and run the GPT flow dashboard
        try:
            # Import using absolute path to avoid module issues
            sys.path.insert(0, str(Path(__file__).parent))
            from gpt_flow_dashboard import GPTFlowDashboard
            
            gpt_dashboard = GPTFlowDashboard()
            
            # Run the dashboard content (without the full page setup)
            st.header("🤖 GPT Flow Visualization")
            st.markdown("Real-time monitoring of Trading Council operations and GPT requests")
            
            # View selector
            view_mode = st.selectbox(
                "View Mode",
                ["Flow Overview", "Request Timeline", "Agent Performance", 
                 "Token Usage", "Council Decisions", "Cost Analysis"]
            )
            
            # Show content based on view mode
            if view_mode == "Flow Overview":
                gpt_dashboard.show_flow_overview()
            elif view_mode == "Request Timeline":
                gpt_dashboard.show_request_timeline()
            elif view_mode == "Agent Performance":
                gpt_dashboard.show_agent_performance()
            elif view_mode == "Token Usage":
                gpt_dashboard.show_token_usage()
            elif view_mode == "Council Decisions":
                gpt_dashboard.show_council_decisions()
            elif view_mode == "Cost Analysis":
                gpt_dashboard.show_cost_analysis()
            
            # Add simulation button
            if st.button("🎭 Simulate Council Session"):
                gpt_dashboard._simulate_council_session()
                st.success("Simulated council session added!")
                st.rerun()
                
        except ImportError as e:
            st.error("Could not import GPT Flow Dashboard. Make sure gpt_flow_dashboard.py is in the scripts directory.")
            st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"Error loading GPT Flow Visualization: {e}")
            import traceback
            st.error(traceback.format_exc())


def main():
    dashboard = SimpleTradingDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()