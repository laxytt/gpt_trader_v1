#!/usr/bin/env python3
"""
Comprehensive Trading Dashboard
Enhanced dashboard combining ML metrics, live trading data, and trade management
"""

import asyncio
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import sys
import json
from scripts.auth_utils import DashboardAuth

import sqlite3
from typing import Dict, List, Any, Optional, Tuple
import MetaTrader5 as mt5

sys.path.append(str(Path(__file__).parent.parent))

from core.infrastructure.database.repositories import TradeRepository, SignalRepository
from core.infrastructure.database.backtest_repository import BacktestRepository
from core.infrastructure.mt5.client import MT5Client
from core.services.trade_service import TradeService
from core.services.market_service import MarketService
from core.domain.models import Trade, TradingSignal
# ML imports
from scripts.ml_continuous_learning import ContinuousLearningSystem
from scripts.performance_analytics import PerformanceAnalytics
from config.settings import get_settings
from core.domain.exceptions import ErrorContext


class ComprehensiveTradingDashboard:
    """Enhanced trading dashboard with full trade management capabilities"""
    
    def __init__(self):
        self.settings = get_settings()
        self.trade_repo = TradeRepository(self.settings.database.db_path)
        self.signal_repo = SignalRepository(self.settings.database.db_path)
        self.backtest_repo = BacktestRepository(self.settings.database.db_path)
        self.continuous_learning = ContinuousLearningSystem()
        self.analytics = PerformanceAnalytics()
        
        # Initialize MT5 client for live data
        self.mt5_client = MT5Client(self.settings.mt5)
        self.mt5_initialized = False
        
        # Initialize services
        self.trade_service = None
        self.market_service = None
        
    def initialize_mt5(self):
        """Initialize MT5 connection"""
        if not self.mt5_initialized:
            try:
                if self.mt5_client.initialize():
                    self.mt5_initialized = True
                    self.trade_service = TradeService(
                        self.trade_repo,
                        self.mt5_client.order_manager,
                        self.settings.trading
                    )
                    self.market_service = MarketService(
                        self.mt5_client.data_provider,
                        self.settings.trading
                    )
                    return True
            except Exception as e:
                st.error(f"Failed to initialize MT5: {e}")
        return self.mt5_initialized
    
    def run(self):
        """Main dashboard runner"""
        st.set_page_config(
            page_title="Comprehensive Trading Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
# Initialize authentication
auth = DashboardAuth("Comprehensive Trading Dashboard")

# Protect the app - this will show login form if not authenticated
auth.protect_app()

        
        st.title("📊 Comprehensive Trading Dashboard")
        st.markdown("Complete trading system monitoring and management")
        
        # Check MT5 connection
        if not self.initialize_mt5():
            st.error("⚠️ MT5 not connected. Live data features will be limited.")
        else:
            st.success("✅ MT5 Connected")
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigation",
            [
                "Dashboard Overview",
                "Open Trades",
                "Trade Management",
                "Historical Analysis",
                "Council Decisions",
                "Signal History",
                "ML Performance",
                "Performance Metrics",
                "Database Statistics",
                "GPT Flow Visualization"
            ]
        )
        
        # Add refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            st.rerun()
        
        # Auto-refresh option
        auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
        if auto_refresh:
            st.sidebar.info("Dashboard will refresh every 30 seconds")
            st.empty()  # Placeholder for auto-refresh
        
        # Route to appropriate page
        if page == "Dashboard Overview":
            self.show_dashboard_overview()
        elif page == "Open Trades":
            self.show_open_trades()
        elif page == "Trade Management":
            self.show_trade_management()
        elif page == "Historical Analysis":
            self.show_historical_analysis()
        elif page == "Council Decisions":
            self.show_council_decisions()
        elif page == "Signal History":
            self.show_signal_history()
        elif page == "ML Performance":
            self.show_ml_performance()
        elif page == "Performance Metrics":
            self.show_performance_metrics()
        elif page == "Database Statistics":
            self.show_database_statistics()
        elif page == "GPT Flow Visualization":
            self.show_gpt_flow_visualization()
    
    def show_dashboard_overview(self):
        """Show comprehensive dashboard overview"""
        st.header("Dashboard Overview")
        
        # Top metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Get current stats
        open_trades = self._get_open_trades()
        total_profit_today = self._calculate_today_profit()
        active_symbols = len(self.settings.trading.symbols)
        ml_status = "Enabled" if self.settings.ml.enabled else "Disabled"
        
        with col1:
            st.metric(
                "Open Trades", 
                len(open_trades),
                delta=f"${sum(t.get('profit', 0) for t in open_trades):.2f}"
            )
        
        with col2:
            st.metric("Today's P&L", f"${total_profit_today:.2f}")
        
        with col3:
            st.metric("Active Symbols", active_symbols)
        
        with col4:
            st.metric("ML Status", ml_status)
        
        with col5:
            account_info = self._get_account_info()
            if account_info:
                st.metric("Account Balance", f"${account_info.get('balance', 0):.2f}")
            else:
                st.metric("Account Balance", "N/A")
        
        # Quick stats
        st.subheader("Quick Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Recent trades summary
            st.markdown("### Recent Trades (Last 24h)")
            recent_trades = self._get_recent_trades(hours=24)
            if recent_trades:
                summary_df = pd.DataFrame([
                    {"Metric": "Total Trades", "Value": len(recent_trades)},
                    {"Metric": "Winning Trades", "Value": sum(1 for t in recent_trades if t.get('profit_loss', 0) > 0)},
                    {"Metric": "Losing Trades", "Value": sum(1 for t in recent_trades if t.get('profit_loss', 0) < 0)},
                    {"Metric": "Total Profit", "Value": f"${sum(t.get('profit_loss', 0) for t in recent_trades):.2f}"},
                    {"Metric": "Win Rate", "Value": f"{self._calculate_win_rate(recent_trades):.1%}"}
                ])
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
            else:
                st.info("No trades in the last 24 hours")
        
        with col2:
            # Active signals
            st.markdown("### Active Signals")
            active_signals = self._get_active_signals()
            if active_signals:
                signals_df = pd.DataFrame(active_signals)
                signals_df = signals_df[['symbol', 'direction', 'confidence', 'created_at']]
                st.dataframe(signals_df, use_container_width=True)
            else:
                st.info("No active signals")
        
        # P&L Chart
        st.subheader("Profit & Loss Trend")
        self._show_pnl_chart()
        
        # System health
        st.subheader("System Health")
        self._show_system_health()
    
    def show_open_trades(self):
        """Show current open trades with live P&L"""
        st.header("Open Trades")
        
        if not self.mt5_initialized:
            st.error("MT5 not connected. Cannot fetch live trade data.")
            return
        
        # Get open trades from MT5
        open_trades = self._get_open_trades_live()
        
        if not open_trades:
            st.info("No open trades")
            return
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_profit = sum(t['profit'] for t in open_trades)
        total_volume = sum(t['volume'] for t in open_trades)
        
        with col1:
            st.metric("Total Open Trades", len(open_trades))
        
        with col2:
            st.metric(
                "Total P&L", 
                f"${total_profit:.2f}",
                delta=f"{total_profit:.2f}"
            )
        
        with col3:
            st.metric("Total Volume", f"{total_volume:.2f}")
        
        with col4:
            avg_duration = self._calculate_avg_trade_duration(open_trades)
            st.metric("Avg Duration", avg_duration)
        
        # Trades table with live updates
        st.subheader("Trade Details")
        
        # Create DataFrame
        trades_data = []
        for trade in open_trades:
            trades_data.append({
                'Ticket': trade['ticket'],
                'Symbol': trade['symbol'],
                'Type': 'BUY' if trade['type'] == mt5.ORDER_TYPE_BUY else 'SELL',
                'Volume': trade['volume'],
                'Entry Price': trade['price_open'],
                'Current Price': trade['price_current'],
                'S/L': trade['sl'],
                'T/P': trade['tp'],
                'P&L': f"${trade['profit']:.2f}",
                'P&L %': f"{((trade['price_current'] - trade['price_open']) / trade['price_open'] * 100):.2%}",
                'Duration': self._format_duration(datetime.now().timestamp() - trade['time']),
                'Comment': trade.get('comment', '')
            })
        
        df = pd.DataFrame(trades_data)
        
        # Style the dataframe
        def color_pnl(val):
            if isinstance(val, str) and '$' in val:
                num_val = float(val.replace('$', '').replace(',', ''))
                color = 'green' if num_val >= 0 else 'red'
                return f'color: {color}'
            return ''
        
        styled_df = df.style.applymap(color_pnl, subset=['P&L'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Trade actions
        st.subheader("Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Close All Profitable", type="secondary"):
                self._close_profitable_trades()
        
        with col2:
            if st.button("Update All S/L to B/E", type="secondary"):
                self._update_all_sl_breakeven()
        
        with col3:
            if st.button("Export to CSV"):
                self._export_trades_csv(df)
    
    def show_trade_management(self):
        """Show trade management interface"""
        st.header("Trade Management")
        
        if not self.mt5_initialized:
            st.error("MT5 not connected. Trade management unavailable.")
            return
        
        # Get open trades
        open_trades = self._get_open_trades_live()
        
        if not open_trades:
            st.info("No open trades to manage")
            return
        
        # Trade selector
        trade_tickets = [f"{t['ticket']} - {t['symbol']} ({t['profit']:.2f})" for t in open_trades]
        selected_trade_str = st.selectbox("Select Trade", trade_tickets)
        
        if selected_trade_str:
            selected_ticket = int(selected_trade_str.split(' - ')[0])
            selected_trade = next(t for t in open_trades if t['ticket'] == selected_ticket)
            
            # Display trade details
            st.subheader("Trade Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Symbol:** {selected_trade['symbol']}")
                st.write(f"**Type:** {'BUY' if selected_trade['type'] == mt5.ORDER_TYPE_BUY else 'SELL'}")
                st.write(f"**Volume:** {selected_trade['volume']}")
                st.write(f"**Entry Price:** {selected_trade['price_open']}")
            
            with col2:
                st.write(f"**Current Price:** {selected_trade['price_current']}")
                st.write(f"**Current P&L:** ${selected_trade['profit']:.2f}")
                st.write(f"**S/L:** {selected_trade['sl']}")
                st.write(f"**T/P:** {selected_trade['tp']}")
            
            # Management actions
            st.subheader("Modify Trade")
            
            tab1, tab2, tab3 = st.tabs(["Modify S/L & T/P", "Partial Close", "Close Trade"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    new_sl = st.number_input(
                        "New Stop Loss",
                        value=float(selected_trade['sl']) if selected_trade['sl'] > 0 else 0.0,
                        format="%.5f"
                    )
                
                with col2:
                    new_tp = st.number_input(
                        "New Take Profit",
                        value=float(selected_trade['tp']) if selected_trade['tp'] > 0 else 0.0,
                        format="%.5f"
                    )
                
                if st.button("Update S/L & T/P", type="primary"):
                    success = self._modify_trade(selected_ticket, new_sl, new_tp)
                    if success:
                        st.success("Trade modified successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to modify trade")
            
            with tab2:
                close_volume = st.number_input(
                    "Volume to Close",
                    min_value=0.01,
                    max_value=selected_trade['volume'],
                    value=selected_trade['volume'] / 2,
                    step=0.01
                )
                
                if st.button("Partial Close", type="primary"):
                    success = self._partial_close_trade(selected_ticket, close_volume)
                    if success:
                        st.success(f"Partially closed {close_volume} lots")
                        st.rerun()
                    else:
                        st.error("Failed to partially close trade")
            
            with tab3:
                st.warning("⚠️ This will close the entire position")
                
                if st.button("Close Trade", type="primary"):
                    success = self._close_trade(selected_ticket)
                    if success:
                        st.success("Trade closed successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to close trade")
    
    def show_historical_analysis(self):
        """Show historical trades analysis"""
        st.header("Historical Trades Analysis")
        
        # Date range selector
        col1, col2, col3 = st.columns(3)
        
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
        
        with col2:
            end_date = st.date_input("End Date", datetime.now())
        
        with col3:
            symbol_filter = st.selectbox(
                "Symbol Filter",
                ["All"] + self.settings.trading.symbols
            )
        
        # Get historical trades
        trades = self._get_historical_trades(start_date, end_date, symbol_filter)
        
        if not trades:
            st.info("No trades found in selected period")
            return
        
        # Summary metrics
        st.subheader("Summary Metrics")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.get('profit_loss', 0) > 0)
        losing_trades = sum(1 for t in trades if t.get('profit_loss', 0) < 0)
        total_profit = sum(t.get('profit_loss', 0) for t in trades)
        
        with col1:
            st.metric("Total Trades", total_trades)
        
        with col2:
            st.metric("Winning Trades", winning_trades)
        
        with col3:
            st.metric("Losing Trades", losing_trades)
        
        with col4:
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        with col5:
            st.metric("Total P&L", f"${total_profit:.2f}")
        
        with col6:
            avg_profit = total_profit / total_trades if total_trades > 0 else 0
            st.metric("Avg P&L", f"${avg_profit:.2f}")
        
        # Profit distribution chart
        st.subheader("Profit Distribution")
        self._show_profit_distribution(trades)
        
        # Trades table
        st.subheader("Trade History")
        
        trades_df = pd.DataFrame(trades)
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        
        # Select columns to display
        display_columns = [
            'ticket', 'symbol', 'direction', 'volume',
            'entry_price', 'exit_price', 'profit_loss',
            'entry_time', 'exit_time', 'trade_type'
        ]
        
        trades_df = trades_df[display_columns].sort_values('entry_time', ascending=False)
        
        # Style the dataframe
        def style_profit(val):
            if isinstance(val, (int, float)):
                color = 'green' if val >= 0 else 'red'
                return f'color: {color}'
            return ''
        
        styled_df = trades_df.style.applymap(style_profit, subset=['profit_loss'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Monthly performance
        st.subheader("Monthly Performance")
        self._show_monthly_performance(trades_df)
    
    def show_council_decisions(self):
        """Show Trading Council decisions viewer"""
        st.header("Trading Council Decisions")
        
        # Get council decisions from signals
        signals = self._get_council_signals()
        
        if not signals:
            st.info("No council decisions found")
            return
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            days_back = st.slider("Days to Show", 1, 30, 7)
        
        with col2:
            symbol_filter = st.selectbox(
                "Symbol",
                ["All"] + list(set(s['symbol'] for s in signals))
            )
        
        with col3:
            direction_filter = st.selectbox(
                "Direction",
                ["All", "BUY", "SELL", "HOLD"]
            )
        
        # Filter signals
        filtered_signals = self._filter_council_signals(
            signals, days_back, symbol_filter, direction_filter
        )
        
        # Display decisions
        for signal in filtered_signals[:20]:  # Show latest 20
            with st.expander(f"{signal['symbol']} - {signal['direction']} - {signal['created_at']}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Parse and display council analysis
                    if signal.get('analysis'):
                        try:
                            analysis = json.loads(signal['analysis'])
                            
                            # Show each agent's opinion
                            st.markdown("### Agent Opinions")
                            
                            for agent, opinion in analysis.get('agent_opinions', {}).items():
                                st.markdown(f"**{agent}:**")
                                st.write(f"Direction: {opinion.get('direction', 'N/A')}")
                                st.write(f"Confidence: {opinion.get('confidence', 0):.1%}")
                                st.write(f"Reasoning: {opinion.get('reasoning', 'N/A')}")
                                st.markdown("---")
                            
                            # Show consensus
                            if 'consensus' in analysis:
                                st.markdown("### Final Consensus")
                                consensus = analysis['consensus']
                                st.write(f"**Direction:** {consensus.get('direction', 'N/A')}")
                                st.write(f"**Confidence:** {consensus.get('confidence', 0):.1%}")
                                st.write(f"**Key Points:** {consensus.get('key_points', 'N/A')}")
                        except:
                            st.text(signal.get('analysis', 'No analysis available'))
                
                with col2:
                    # Signal metrics
                    st.markdown("### Signal Metrics")
                    st.metric("Confidence", f"{signal.get('confidence', 0):.1%}")
                    st.metric("ML Confidence", f"{signal.get('ml_confidence', 0):.1%}")
                    
                    # Trade result if available
                    trade = self._get_trade_for_signal(signal['id'])
                    if trade:
                        st.markdown("### Trade Result")
                        st.metric("P&L", f"${trade.get('profit_loss', 0):.2f}")
                        st.write(f"Status: {trade.get('status', 'N/A')}")
    
    def show_signal_history(self):
        """Show signal history and performance"""
        st.header("Signal History")
        
        # Date range and filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
        
        with col2:
            end_date = st.date_input("End Date", datetime.now())
        
        with col3:
            symbol_filter = st.selectbox("Symbol", ["All"] + self.settings.trading.symbols)
        
        with col4:
            status_filter = st.selectbox("Status", ["All", "Executed", "Not Executed", "Expired"])
        
        # Get signals
        signals = self._get_signal_history(start_date, end_date, symbol_filter, status_filter)
        
        if not signals:
            st.info("No signals found")
            return
        
        # Summary stats
        st.subheader("Signal Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_signals = len(signals)
        executed_signals = sum(1 for s in signals if s.get('executed'))
        avg_confidence = sum(s.get('confidence', 0) for s in signals) / total_signals if total_signals > 0 else 0
        
        with col1:
            st.metric("Total Signals", total_signals)
        
        with col2:
            st.metric("Executed", executed_signals)
        
        with col3:
            execution_rate = (executed_signals / total_signals * 100) if total_signals > 0 else 0
            st.metric("Execution Rate", f"{execution_rate:.1f}%")
        
        with col4:
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        # Signal performance chart
        st.subheader("Signal Performance Over Time")
        self._show_signal_performance_chart(signals)
        
        # Signals table
        st.subheader("Signal Details")
        
        signals_df = pd.DataFrame(signals)
        
        # Prepare display columns
        display_columns = [
            'id', 'symbol', 'direction', 'confidence', 'ml_confidence',
            'executed', 'created_at', 'trade_result'
        ]
        
        # Add trade result column
        signals_df['trade_result'] = signals_df.apply(
            lambda x: self._get_signal_trade_result(x['id']), axis=1
        )
        
        signals_df = signals_df[display_columns].sort_values('created_at', ascending=False)
        
        # Style the dataframe
        def style_confidence(val):
            if isinstance(val, (int, float)):
                if val >= 0.8:
                    return 'background-color: lightgreen'
                elif val >= 0.6:
                    return 'background-color: lightyellow'
                else:
                    return 'background-color: lightcoral'
            return ''
        
        styled_df = signals_df.style.applymap(
            style_confidence, 
            subset=['confidence', 'ml_confidence']
        )
        
        st.dataframe(styled_df, use_container_width=True)
    
    def show_ml_performance(self):
        """Show ML model performance"""
        st.header("ML Model Performance")
        
        # ML status
        if not self.settings.ml.enabled:
            st.warning("ML is currently disabled in settings")
        
        # Symbol selector
        symbol = st.selectbox("Select Symbol", self.settings.trading.symbols)
        
        if symbol:
            # Time range
            days_back = st.slider("Days to Analyze", 7, 90, 30)
            
            # Get ML performance
            ml_performance = asyncio.run(
                self.continuous_learning.evaluate_model_performance(symbol, days_back)
            )
            
            # Display metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Trades", ml_performance.get('total_trades', 0))
            
            with col2:
                win_rate = ml_performance.get('win_rate', 0)
                st.metric(
                    "Win Rate",
                    f"{win_rate:.1%}",
                    delta=f"{win_rate - 0.5:.1%}"
                )
            
            with col3:
                st.metric("Avg Profit", f"${ml_performance.get('avg_profit', 0):.2f}")
            
            with col4:
                st.metric("Total Profit", f"${ml_performance.get('total_profit', 0):.2f}")
            
            with col5:
                status = "⚠️ Needs Retraining" if ml_performance.get('needs_retraining', False) else "✅ Good"
                st.metric("Model Status", status)
            
            # Model details
            st.subheader("Model Information")
            
            model_info = self._get_model_info(symbol)
            if model_info:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Model File:** {model_info['file']}")
                    st.write(f"**Last Updated:** {model_info['modified']}")
                    st.write(f"**File Size:** {model_info['size']}")
                
                with col2:
                    st.write(f"**Model Version:** {model_info.get('version', 'N/A')}")
                    st.write(f"**Training Samples:** {model_info.get('training_samples', 'N/A')}")
                    st.write(f"**Features:** {model_info.get('features', 'N/A')}")
            
            # Performance trend
            st.subheader("ML Performance Trend")
            self._show_ml_performance_trend(symbol)
            
            # Feature importance
            st.subheader("Feature Importance")
            self._show_feature_importance(symbol)
            
            # Manual controls
            st.subheader("Model Management")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Force Retrain", type="primary"):
                    with st.spinner(f"Retraining {symbol}..."):
                        result = asyncio.run(self.continuous_learning.retrain_model(symbol))
                        if result['status'] == 'success':
                            st.success("Model retrained successfully!")
                            st.json(result['backtest'])
                        else:
                            st.error(f"Retraining failed: {result.get('error')}")
            
            with col2:
                if st.button("Run Backtest"):
                    st.info("Backtest functionality to be implemented")
            
            with col3:
                if st.button("Export Model Stats"):
                    self._export_model_stats(symbol)
    
    def show_performance_metrics(self):
        """Show comprehensive performance metrics"""
        st.header("Performance Metrics")
        
        # Time period selector
        period = st.selectbox(
            "Time Period",
            ["Today", "This Week", "This Month", "Last 30 Days", "Last 90 Days", "YTD", "All Time"]
        )
        
        start_date, end_date = self._get_period_dates(period)
        
        # Get performance data
        performance_data = self._calculate_performance_metrics(start_date, end_date)
        
        # Overall metrics
        st.subheader("Overall Performance")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Net Profit", f"${performance_data['net_profit']:.2f}")
        
        with col2:
            st.metric("Win Rate", f"{performance_data['win_rate']:.1%}")
        
        with col3:
            st.metric("Profit Factor", f"{performance_data['profit_factor']:.2f}")
        
        with col4:
            st.metric("Sharpe Ratio", f"{performance_data['sharpe_ratio']:.2f}")
        
        with col5:
            st.metric("Max Drawdown", f"{performance_data['max_drawdown']:.1%}")
        
        with col6:
            st.metric("Recovery Factor", f"{performance_data['recovery_factor']:.2f}")
        
        # Performance by symbol
        st.subheader("Performance by Symbol")
        
        symbol_performance = performance_data.get('by_symbol', {})
        if symbol_performance:
            symbol_df = pd.DataFrame.from_dict(symbol_performance, orient='index')
            
            # Create heatmap
            fig = px.imshow(
                symbol_df.T,
                labels=dict(x="Symbol", y="Metric", color="Value"),
                title="Symbol Performance Heatmap",
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.dataframe(symbol_df, use_container_width=True)
        
        # Equity curve
        st.subheader("Equity Curve")
        self._show_equity_curve(start_date, end_date)
        
        # Risk metrics
        st.subheader("Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk metrics
            risk_metrics = performance_data.get('risk_metrics', {})
            
            risk_df = pd.DataFrame([
                {"Metric": "Value at Risk (95%)", "Value": f"${risk_metrics.get('var_95', 0):.2f}"},
                {"Metric": "Expected Shortfall", "Value": f"${risk_metrics.get('expected_shortfall', 0):.2f}"},
                {"Metric": "Max Consecutive Losses", "Value": risk_metrics.get('max_consecutive_losses', 0)},
                {"Metric": "Average Loss", "Value": f"${risk_metrics.get('avg_loss', 0):.2f}"},
                {"Metric": "Loss Standard Deviation", "Value": f"${risk_metrics.get('loss_std', 0):.2f}"}
            ])
            
            st.dataframe(risk_df, use_container_width=True, hide_index=True)
        
        with col2:
            # Drawdown chart
            self._show_drawdown_chart(start_date, end_date)
    
    def show_database_statistics(self):
        """Show database statistics and health"""
        st.header("Database Statistics")
        
        # Database file info
        db_path = Path(self.settings.database.db_path)
        if db_path.exists():
            db_size = db_path.stat().st_size / (1024 * 1024)  # MB
            db_modified = datetime.fromtimestamp(db_path.stat().st_mtime)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Database Size", f"{db_size:.2f} MB")
            
            with col2:
                st.metric("Last Modified", db_modified.strftime("%Y-%m-%d %H:%M"))
            
            with col3:
                st.metric("Database Path", str(db_path))
        
        # Table statistics
        st.subheader("Table Statistics")
        
        table_stats = self._get_table_statistics()
        
        stats_df = pd.DataFrame(table_stats)
        st.dataframe(stats_df, use_container_width=True)
        
        # Data growth chart
        st.subheader("Data Growth Over Time")
        self._show_data_growth_chart()
        
        # Recent activity
        st.subheader("Recent Database Activity")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Recent Trades")
            recent_trades = self._get_recent_db_entries('trades', 5)
            if recent_trades:
                for trade in recent_trades:
                    st.write(f"- {trade['symbol']} - {trade['direction']} - {trade['created_at']}")
            else:
                st.info("No recent trades")
        
        with col2:
            st.markdown("### Recent Signals")
            recent_signals = self._get_recent_db_entries('signals', 5)
            if recent_signals:
                for signal in recent_signals:
                    st.write(f"- {signal['symbol']} - {signal['direction']} - {signal['created_at']}")
            else:
                st.info("No recent signals")
        
        # Database maintenance
        st.subheader("Database Maintenance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Analyze Database", type="secondary"):
                self._analyze_database()
        
        with col2:
            if st.button("Backup Database", type="primary"):
                self._backup_database()
        
        with col3:
            if st.button("Optimize Database", type="secondary"):
                self._optimize_database()
    
    # Helper methods
    def _get_open_trades(self) -> List[Dict]:
        """Get open trades from database"""
        try:
            trades = self.trade_repo.get_open_trades()
            return [self._trade_to_dict(t) for t in trades]
        except Exception as e:
            st.error(f"Error fetching open trades: {e}")
            return []
    
    def _get_open_trades_live(self) -> List[Dict]:
        """Get open trades from MT5"""
        if not self.mt5_initialized:
            return []
        
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
            
            trades = []
            for pos in positions:
                trades.append({
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': pos.type,
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'price_current': pos.price_current,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'profit': pos.profit,
                    'time': pos.time,
                    'comment': pos.comment
                })
            
            return trades
        except Exception as e:
            st.error(f"Error fetching live trades: {e}")
            return []
    
    def _calculate_today_profit(self) -> float:
        """Calculate today's total profit"""
        try:
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            trades = self.trade_repo.get_trades_by_date_range(today_start, datetime.now())
            return sum(t.profit_loss for t in trades if t.profit_loss is not None)
        except Exception:
            return 0.0
    
    def _get_account_info(self) -> Optional[Dict]:
        """Get MT5 account information"""
        if not self.mt5_initialized:
            return None
        
        try:
            account_info = mt5.account_info()
            if account_info:
                return {
                    'balance': account_info.balance,
                    'equity': account_info.equity,
                    'margin': account_info.margin,
                    'free_margin': account_info.margin_free,
                    'margin_level': account_info.margin_level
                }
        except Exception:
            pass
        
        return None
    
    def _get_recent_trades(self, hours: int = 24) -> List[Dict]:
        """Get recent trades"""
        try:
            since = datetime.now() - timedelta(hours=hours)
            trades = self.trade_repo.get_trades_by_date_range(since, datetime.now())
            return [self._trade_to_dict(t) for t in trades]
        except Exception:
            return []
    
    def _trade_to_dict(self, trade: Trade) -> Dict:
        """Convert Trade object to dictionary"""
        return {
            'id': trade.id,
            'ticket': trade.ticket,
            'symbol': trade.symbol,
            'direction': trade.direction,
            'volume': trade.volume,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'profit_loss': trade.profit_loss,
            'entry_time': trade.entry_time,
            'exit_time': trade.exit_time,
            'status': trade.status,
            'trade_type': trade.trade_type
        }
    
    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate from trades"""
        if not trades:
            return 0.0
        
        wins = sum(1 for t in trades if t.get('profit_loss', 0) > 0)
        return wins / len(trades) if len(trades) > 0 else 0.0
    
    def _get_active_signals(self) -> List[Dict]:
        """Get active trading signals"""
        try:
            # Get signals from last hour that haven't been executed
            since = datetime.now() - timedelta(hours=1)
            signals = self.signal_repo.get_signals_by_date_range(since, datetime.now())
            
            active_signals = []
            for signal in signals:
                if not signal.executed and signal.direction != 'HOLD':
                    active_signals.append({
                        'id': signal.id,
                        'symbol': signal.symbol,
                        'direction': signal.direction,
                        'confidence': signal.confidence,
                        'created_at': signal.created_at.strftime('%H:%M:%S')
                    })
            
            return active_signals
        except Exception:
            return []
    
    def _show_pnl_chart(self):
        """Show P&L trend chart"""
        try:
            # Get trades for last 30 days
            trades = self._get_recent_trades(hours=24*30)
            
            if not trades:
                st.info("No trade data for P&L chart")
                return
            
            # Calculate cumulative P&L
            trades_sorted = sorted(trades, key=lambda x: x['exit_time'] or x['entry_time'])
            
            dates = []
            cumulative_pnl = []
            running_total = 0
            
            for trade in trades_sorted:
                if trade.get('profit_loss') is not None and trade.get('exit_time'):
                    running_total += trade['profit_loss']
                    dates.append(trade['exit_time'])
                    cumulative_pnl.append(running_total)
            
            if dates:
                # Create chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=cumulative_pnl,
                    mode='lines+markers',
                    name='Cumulative P&L',
                    line=dict(color='green' if running_total >= 0 else 'red', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(0,255,0,0.1)' if running_total >= 0 else 'rgba(255,0,0,0.1)'
                ))
                
                fig.update_layout(
                    title="Cumulative Profit & Loss",
                    xaxis_title="Date",
                    yaxis_title="P&L ($)",
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No completed trades for P&L chart")
                
        except Exception as e:
            st.error(f"Error creating P&L chart: {e}")
    
    def _show_system_health(self):
        """Show system health indicators"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # MT5 connection status
            mt5_status = "🟢 Connected" if self.mt5_initialized else "🔴 Disconnected"
            st.metric("MT5 Status", mt5_status)
        
        with col2:
            # Database status
            db_status = "🟢 Healthy" if Path(self.settings.database.db_path).exists() else "🔴 Error"
            st.metric("Database", db_status)
        
        with col3:
            # ML status
            ml_status = "🟢 Enabled" if self.settings.ml.enabled else "🟡 Disabled"
            st.metric("ML System", ml_status)
        
        with col4:
            # Last signal time
            try:
                latest_signal = self.signal_repo.get_latest_signal()
                if latest_signal:
                    time_diff = datetime.now() - latest_signal.created_at
                    if time_diff.total_seconds() < 3600:  # Less than 1 hour
                        signal_status = "🟢 Active"
                    elif time_diff.total_seconds() < 86400:  # Less than 1 day
                        signal_status = "🟡 Inactive"
                    else:
                        signal_status = "🔴 Stale"
                else:
                    signal_status = "⚪ No Signals"
                
                st.metric("Signal Generation", signal_status)
            except Exception:
                st.metric("Signal Generation", "⚪ Unknown")
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds/60)}m"
        elif seconds < 86400:
            return f"{int(seconds/3600)}h {int((seconds%3600)/60)}m"
        else:
            return f"{int(seconds/86400)}d {int((seconds%86400)/3600)}h"
    
    def _calculate_avg_trade_duration(self, trades: List[Dict]) -> str:
        """Calculate average trade duration"""
        if not trades:
            return "N/A"
        
        durations = []
        for trade in trades:
            if 'time' in trade:
                duration = datetime.now().timestamp() - trade['time']
                durations.append(duration)
        
        if durations:
            avg_duration = sum(durations) / len(durations)
            return self._format_duration(avg_duration)
        
        return "N/A"
    
    def _close_profitable_trades(self):
        """Close all profitable trades"""
        try:
            positions = mt5.positions_get()
            if not positions:
                st.info("No open positions")
                return
            
            closed_count = 0
            for pos in positions:
                if pos.profit > 0:
                    result = self.mt5_client.order_manager.close_position(pos.ticket)
                    if result:
                        closed_count += 1
            
            if closed_count > 0:
                st.success(f"Closed {closed_count} profitable trades")
            else:
                st.info("No profitable trades to close")
                
        except Exception as e:
            st.error(f"Error closing trades: {e}")
    
    def _update_all_sl_breakeven(self):
        """Update all stop losses to breakeven"""
        try:
            positions = mt5.positions_get()
            if not positions:
                st.info("No open positions")
                return
            
            updated_count = 0
            for pos in positions:
                if pos.profit > 0 and pos.sl != pos.price_open:
                    # Update SL to breakeven
                    result = self.mt5_client.order_manager.modify_position(
                        pos.ticket, pos.price_open, pos.tp
                    )
                    if result:
                        updated_count += 1
            
            if updated_count > 0:
                st.success(f"Updated {updated_count} stop losses to breakeven")
            else:
                st.info("No eligible trades for breakeven update")
                
        except Exception as e:
            st.error(f"Error updating stop losses: {e}")
    
    def _export_trades_csv(self, df: pd.DataFrame):
        """Export trades to CSV"""
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"open_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    def _modify_trade(self, ticket: int, sl: float, tp: float) -> bool:
        """Modify trade SL and TP"""
        try:
            return self.mt5_client.order_manager.modify_position(ticket, sl, tp)
        except Exception as e:
            st.error(f"Error modifying trade: {e}")
            return False
    
    def _partial_close_trade(self, ticket: int, volume: float) -> bool:
        """Partially close a trade"""
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return False
            
            pos = position[0]
            
            # Create opposite order for partial close
            order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": volume,
                "type": order_type,
                "position": ticket,
                "deviation": 20,
                "magic": self.settings.mt5.magic_number,
                "comment": "Partial close"
            }
            
            result = mt5.order_send(request)
            return result.retcode == mt5.TRADE_RETCODE_DONE
            
        except Exception as e:
            st.error(f"Error in partial close: {e}")
            return False
    
    def _close_trade(self, ticket: int) -> bool:
        """Close a trade completely"""
        try:
            return self.mt5_client.order_manager.close_position(ticket)
        except Exception as e:
            st.error(f"Error closing trade: {e}")
            return False
    
    def _get_historical_trades(self, start_date, end_date, symbol_filter: str) -> List[Dict]:
        """Get historical trades"""
        try:
            if symbol_filter == "All":
                trades = self.trade_repo.get_trades_by_date_range(start_date, end_date)
            else:
                trades = self.trade_repo.get_trades_by_date_range(start_date, end_date, symbol_filter)
            
            return [self._trade_to_dict(t) for t in trades]
        except Exception:
            return []
    
    def _show_profit_distribution(self, trades: List[Dict]):
        """Show profit distribution histogram"""
        profits = [t.get('profit_loss', 0) for t in trades if t.get('profit_loss') is not None]
        
        if not profits:
            st.info("No profit data available")
            return
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=profits,
            nbinsx=30,
            name='Profit Distribution',
            marker_color='lightblue',
            opacity=0.7
        ))
        
        # Add mean line
        mean_profit = sum(profits) / len(profits)
        fig.add_vline(
            x=mean_profit,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: ${mean_profit:.2f}"
        )
        
        # Add zero line
        fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1)
        
        fig.update_layout(
            title="Trade Profit Distribution",
            xaxis_title="Profit/Loss ($)",
            yaxis_title="Frequency",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_monthly_performance(self, trades_df: pd.DataFrame):
        """Show monthly performance breakdown"""
        if trades_df.empty:
            st.info("No data for monthly performance")
            return
        
        # Group by month
        trades_df['month'] = pd.to_datetime(trades_df['exit_time']).dt.to_period('M')
        
        monthly = trades_df.groupby('month').agg({
            'profit_loss': ['sum', 'mean', 'count'],
            'ticket': 'count'
        }).round(2)
        
        monthly.columns = ['Total P&L', 'Avg P&L', 'Trades', 'Count']
        monthly = monthly.drop('Count', axis=1)
        
        # Calculate win rate per month
        monthly['Win Rate'] = trades_df.groupby('month').apply(
            lambda x: (x['profit_loss'] > 0).sum() / len(x) * 100
        ).round(1)
        
        st.dataframe(monthly, use_container_width=True)
        
        # Monthly P&L chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=monthly.index.astype(str),
            y=monthly['Total P&L'],
            name='Monthly P&L',
            marker_color=['green' if x > 0 else 'red' for x in monthly['Total P&L']]
        ))
        
        fig.update_layout(
            title="Monthly P&L Performance",
            xaxis_title="Month",
            yaxis_title="P&L ($)",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _get_council_signals(self) -> List[Dict]:
        """Get signals with council analysis"""
        try:
            # Get recent signals with analysis
            signals = self.signal_repo.get_all_signals(limit=100)
            
            council_signals = []
            for signal in signals:
                if signal.analysis and 'agent_opinions' in str(signal.analysis):
                    council_signals.append({
                        'id': signal.id,
                        'symbol': signal.symbol,
                        'direction': signal.direction,
                        'confidence': signal.confidence,
                        'ml_confidence': signal.ml_confidence,
                        'analysis': signal.analysis,
                        'created_at': signal.created_at
                    })
            
            return council_signals
        except Exception:
            return []
    
    def _filter_council_signals(self, signals: List[Dict], days_back: int, 
                               symbol_filter: str, direction_filter: str) -> List[Dict]:
        """Filter council signals"""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        filtered = []
        for signal in signals:
            # Date filter
            if signal['created_at'] < cutoff_date:
                continue
            
            # Symbol filter
            if symbol_filter != "All" and signal['symbol'] != symbol_filter:
                continue
            
            # Direction filter
            if direction_filter != "All" and signal['direction'] != direction_filter:
                continue
            
            filtered.append(signal)
        
        return sorted(filtered, key=lambda x: x['created_at'], reverse=True)
    
    def _get_trade_for_signal(self, signal_id: int) -> Optional[Dict]:
        """Get trade associated with a signal"""
        try:
            trades = self.trade_repo.get_trades_by_signal_id(signal_id)
            if trades:
                return self._trade_to_dict(trades[0])
        except Exception:
            pass
        return None
    
    def _get_signal_history(self, start_date, end_date, symbol_filter: str, 
                           status_filter: str) -> List[Dict]:
        """Get signal history with filters"""
        try:
            signals = self.signal_repo.get_signals_by_date_range(start_date, end_date)
            
            filtered_signals = []
            for signal in signals:
                # Symbol filter
                if symbol_filter != "All" and signal.symbol != symbol_filter:
                    continue
                
                # Status filter
                if status_filter == "Executed" and not signal.executed:
                    continue
                elif status_filter == "Not Executed" and signal.executed:
                    continue
                elif status_filter == "Expired" and signal.valid_until > datetime.now():
                    continue
                
                filtered_signals.append({
                    'id': signal.id,
                    'symbol': signal.symbol,
                    'direction': signal.direction,
                    'confidence': signal.confidence,
                    'ml_confidence': signal.ml_confidence,
                    'executed': signal.executed,
                    'created_at': signal.created_at
                })
            
            return filtered_signals
        except Exception:
            return []
    
    def _show_signal_performance_chart(self, signals: List[Dict]):
        """Show signal performance over time"""
        if not signals:
            return
        
        # Group by date
        df = pd.DataFrame(signals)
        df['date'] = pd.to_datetime(df['created_at']).dt.date
        
        daily_stats = df.groupby('date').agg({
            'id': 'count',
            'confidence': 'mean',
            'executed': 'sum'
        }).rename(columns={'id': 'total_signals', 'executed': 'executed_signals'})
        
        # Create chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=daily_stats.index,
            y=daily_stats['total_signals'],
            name='Total Signals',
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            x=daily_stats.index,
            y=daily_stats['executed_signals'],
            name='Executed Signals',
            marker_color='darkblue'
        ))
        
        fig.add_trace(go.Scatter(
            x=daily_stats.index,
            y=daily_stats['confidence'] * 100,
            name='Avg Confidence %',
            yaxis='y2',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title="Signal Generation and Execution",
            xaxis_title="Date",
            yaxis_title="Number of Signals",
            yaxis2=dict(
                title="Avg Confidence %",
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _get_signal_trade_result(self, signal_id: int) -> str:
        """Get trade result for a signal"""
        trade = self._get_trade_for_signal(signal_id)
        if trade:
            profit = trade.get('profit_loss', 0)
            if profit > 0:
                return f"✅ ${profit:.2f}"
            else:
                return f"❌ ${profit:.2f}"
        return "No Trade"
    
    def _get_model_info(self, symbol: str) -> Optional[Dict]:
        """Get ML model information"""
        try:
            model_files = list(Path("models").glob(f"{symbol}_ml_package_*.pkl"))
            if model_files:
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                return {
                    'file': latest_model.name,
                    'modified': datetime.fromtimestamp(latest_model.stat().st_mtime),
                    'size': f"{latest_model.stat().st_size / 1024:.1f} KB",
                    'path': str(latest_model)
                }
        except Exception:
            pass
        return None
    
    def _show_ml_performance_trend(self, symbol: str):
        """Show ML performance trend over time"""
        # This would show the ML model's performance metrics over time
        st.info("ML performance trend visualization to be implemented")
    
    def _show_feature_importance(self, symbol: str):
        """Show ML feature importance"""
        # This would show which features are most important for the model
        st.info("Feature importance visualization to be implemented")
    
    def _export_model_stats(self, symbol: str):
        """Export model statistics"""
        model_info = self._get_model_info(symbol)
        if model_info:
            stats = {
                'symbol': symbol,
                'model_info': model_info,
                'performance': asyncio.run(
                    self.continuous_learning.evaluate_model_performance(symbol)
                )
            }
            
            st.download_button(
                label="Download Model Stats",
                data=json.dumps(stats, indent=2, default=str),
                file_name=f"{symbol}_model_stats_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    def _get_period_dates(self, period: str) -> Tuple[datetime, datetime]:
        """Get start and end dates for period"""
        end_date = datetime.now()
        
        if period == "Today":
            start_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "This Week":
            start_date = end_date - timedelta(days=end_date.weekday())
        elif period == "This Month":
            start_date = end_date.replace(day=1)
        elif period == "Last 30 Days":
            start_date = end_date - timedelta(days=30)
        elif period == "Last 90 Days":
            start_date = end_date - timedelta(days=90)
        elif period == "YTD":
            start_date = end_date.replace(month=1, day=1)
        else:  # All Time
            start_date = datetime(2020, 1, 1)
        
        return start_date, end_date
    
    def _calculate_performance_metrics(self, start_date: datetime, end_date: datetime) -> Dict:
        """Calculate comprehensive performance metrics"""
        try:
            trades = self.trade_repo.get_trades_by_date_range(start_date, end_date)
            
            if not trades:
                return {
                    'net_profit': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'recovery_factor': 0,
                    'by_symbol': {},
                    'risk_metrics': {}
                }
            
            # Calculate metrics
            profits = [t.profit_loss for t in trades if t.profit_loss is not None]
            
            net_profit = sum(profits)
            wins = [p for p in profits if p > 0]
            losses = [abs(p) for p in profits if p < 0]
            
            win_rate = len(wins) / len(profits) if profits else 0
            profit_factor = sum(wins) / sum(losses) if losses else float('inf')
            
            # Calculate Sharpe ratio (simplified)
            if len(profits) > 1:
                returns = pd.Series(profits)
                sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Calculate drawdown
            cumulative = pd.Series(profits).cumsum()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
            
            # Recovery factor
            recovery_factor = net_profit / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # By symbol breakdown
            by_symbol = {}
            for symbol in self.settings.trading.symbols:
                symbol_trades = [t for t in trades if t.symbol == symbol]
                if symbol_trades:
                    symbol_profits = [t.profit_loss for t in symbol_trades if t.profit_loss is not None]
                    if symbol_profits:
                        by_symbol[symbol] = {
                            'trades': len(symbol_trades),
                            'net_profit': sum(symbol_profits),
                            'win_rate': len([p for p in symbol_profits if p > 0]) / len(symbol_profits),
                            'avg_profit': sum(symbol_profits) / len(symbol_profits)
                        }
            
            # Risk metrics
            risk_metrics = {
                'var_95': np.percentile(losses, 95) if losses else 0,
                'expected_shortfall': np.mean([l for l in losses if l > np.percentile(losses, 95)]) if losses else 0,
                'max_consecutive_losses': self._calculate_max_consecutive_losses(profits),
                'avg_loss': np.mean(losses) if losses else 0,
                'loss_std': np.std(losses) if losses else 0
            }
            
            return {
                'net_profit': net_profit,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'recovery_factor': recovery_factor,
                'by_symbol': by_symbol,
                'risk_metrics': risk_metrics
            }
            
        except Exception as e:
            st.error(f"Error calculating metrics: {e}")
            return {}
    
    def _calculate_max_consecutive_losses(self, profits: List[float]) -> int:
        """Calculate maximum consecutive losses"""
        max_losses = 0
        current_losses = 0
        
        for profit in profits:
            if profit < 0:
                current_losses += 1
                max_losses = max(max_losses, current_losses)
            else:
                current_losses = 0
        
        return max_losses
    
    def _show_equity_curve(self, start_date: datetime, end_date: datetime):
        """Show equity curve"""
        try:
            trades = self.trade_repo.get_trades_by_date_range(start_date, end_date)
            
            if not trades:
                st.info("No trades for equity curve")
                return
            
            # Sort by exit time
            trades_sorted = sorted(
                [t for t in trades if t.exit_time is not None],
                key=lambda x: x.exit_time
            )
            
            # Calculate cumulative equity
            dates = []
            equity = []
            running_total = 10000  # Starting equity
            
            for trade in trades_sorted:
                if trade.profit_loss is not None:
                    running_total += trade.profit_loss
                    dates.append(trade.exit_time)
                    equity.append(running_total)
            
            if dates:
                # Create chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=equity,
                    mode='lines',
                    name='Equity',
                    line=dict(color='blue', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(0,100,255,0.1)'
                ))
                
                # Add starting line
                fig.add_hline(y=10000, line_dash="dash", line_color="gray",
                            annotation_text="Starting Equity")
                
                fig.update_layout(
                    title="Equity Curve",
                    xaxis_title="Date",
                    yaxis_title="Equity ($)",
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error creating equity curve: {e}")
    
    def _show_drawdown_chart(self, start_date: datetime, end_date: datetime):
        """Show drawdown chart"""
        try:
            trades = self.trade_repo.get_trades_by_date_range(start_date, end_date)
            
            if not trades:
                return
            
            # Calculate drawdown series
            trades_sorted = sorted(
                [t for t in trades if t.exit_time is not None],
                key=lambda x: x.exit_time
            )
            
            dates = []
            cumulative = []
            running_total = 0
            
            for trade in trades_sorted:
                if trade.profit_loss is not None:
                    running_total += trade.profit_loss
                    dates.append(trade.exit_time)
                    cumulative.append(running_total)
            
            if dates:
                # Calculate drawdown
                cumulative_series = pd.Series(cumulative, index=dates)
                running_max = cumulative_series.expanding().max()
                drawdown = ((cumulative_series - running_max) / running_max * 100)
                
                # Create chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='red', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255,0,0,0.2)'
                ))
                
                fig.update_layout(
                    title="Drawdown %",
                    xaxis_title="Date",
                    yaxis_title="Drawdown (%)",
                    hovermode='x unified',
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error creating drawdown chart: {e}")
    
    def _get_table_statistics(self) -> List[Dict]:
        """Get database table statistics"""
        try:
            conn = sqlite3.connect(self.settings.database.db_path)
            cursor = conn.cursor()
            
            tables = ['trades', 'signals', 'memory_cases', 'backtest_results']
            stats = []
            
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    
                    cursor.execute(f"SELECT MIN(created_at), MAX(created_at) FROM {table}")
                    dates = cursor.fetchone()
                    
                    stats.append({
                        'Table': table,
                        'Records': count,
                        'First Record': dates[0] if dates[0] else 'N/A',
                        'Last Record': dates[1] if dates[1] else 'N/A'
                    })
                except Exception:
                    stats.append({
                        'Table': table,
                        'Records': 'Error',
                        'First Record': 'N/A',
                        'Last Record': 'N/A'
                    })
            
            conn.close()
            return stats
            
        except Exception as e:
            st.error(f"Error getting table statistics: {e}")
            return []
    
    def _show_data_growth_chart(self):
        """Show data growth over time"""
        # This would show how the database has grown over time
        st.info("Data growth visualization to be implemented")
    
    def _get_recent_db_entries(self, table: str, limit: int) -> List[Dict]:
        """Get recent entries from database table"""
        try:
            if table == 'trades':
                trades = self.trade_repo.get_all_trades(limit=limit)
                return [self._trade_to_dict(t) for t in trades]
            elif table == 'signals':
                signals = self.signal_repo.get_all_signals(limit=limit)
                return [{
                    'symbol': s.symbol,
                    'direction': s.direction,
                    'created_at': s.created_at.strftime('%Y-%m-%d %H:%M')
                } for s in signals]
        except Exception:
            return []
    
    def _analyze_database(self):
        """Analyze database for issues"""
        with st.spinner("Analyzing database..."):
            try:
                conn = sqlite3.connect(self.settings.database.db_path)
                cursor = conn.cursor()
                
                # Run PRAGMA checks
                cursor.execute("PRAGMA integrity_check")
                integrity = cursor.fetchone()[0]
                
                if integrity == "ok":
                    st.success("Database integrity check passed")
                else:
                    st.error(f"Database integrity issues: {integrity}")
                
                # Check for orphaned records
                cursor.execute("""
                    SELECT COUNT(*) FROM trades 
                    WHERE signal_id IS NOT NULL 
                    AND signal_id NOT IN (SELECT id FROM signals)
                """)
                orphaned = cursor.fetchone()[0]
                
                if orphaned > 0:
                    st.warning(f"Found {orphaned} orphaned trade records")
                
                conn.close()
                
            except Exception as e:
                st.error(f"Database analysis failed: {e}")
    
    def _backup_database(self):
        """Backup the database"""
        with st.spinner("Creating backup..."):
            try:
                import shutil
                import gzip
                
                db_path = Path(self.settings.database.db_path)
                backup_dir = Path("backups/database")
                backup_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"trades_db_backup_{timestamp}.db"
                backup_path = backup_dir / backup_name
                
                # Copy database
                shutil.copy2(db_path, backup_path)
                
                # Compress
                with open(backup_path, 'rb') as f_in:
                    with gzip.open(f"{backup_path}.gz", 'wb') as f_out:
                        f_out.writelines(f_in)
                
                # Remove uncompressed backup
                backup_path.unlink()
                
                st.success(f"Database backed up to {backup_path}.gz")
                
            except Exception as e:
                st.error(f"Backup failed: {e}")
    
    def _optimize_database(self):
        """Optimize database performance"""
        with st.spinner("Optimizing database..."):
            try:
                conn = sqlite3.connect(self.settings.database.db_path)
                cursor = conn.cursor()
                
                # Run VACUUM
                cursor.execute("VACUUM")
                
                # Analyze tables
                cursor.execute("ANALYZE")
                
                conn.close()
                
                st.success("Database optimization complete")
                
            except Exception as e:
                st.error(f"Optimization failed: {e}")
    
    def show_gpt_flow_visualization(self):
        """Show GPT Flow Visualization page"""
        # Import and run the GPT flow dashboard
        try:
            from scripts.gpt_flow_dashboard import GPTFlowDashboard
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


def main():
    """Run the comprehensive trading dashboard"""
    dashboard = ComprehensiveTradingDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()