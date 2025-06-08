#!/usr/bin/env python3
"""
ML Improvement Dashboard
Interactive monitoring and management of ML model performance
"""

import asyncio
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import sys
import json
from scripts.auth_utils import DashboardAuth

from typing import Dict, List, Any, Optional

sys.path.append(str(Path(__file__).parent.parent))

from core.infrastructure.database.repositories import TradeRepository, SignalRepository
from core.infrastructure.database.backtest_repository import BacktestRepository
from scripts.ml_continuous_learning import ContinuousLearningSystem
from scripts.performance_analytics import PerformanceAnalytics
from config.settings import get_settings


class MLImprovementDashboard:
    """Dashboard for monitoring and managing ML model improvements"""
    
    def __init__(self):
        self.settings = get_settings()
        self.trade_repo = TradeRepository(self.settings.database.db_path)
        self.signal_repo = SignalRepository(self.settings.database.db_path)
        self.backtest_repo = BacktestRepository(self.settings.database.db_path)
        self.continuous_learning = ContinuousLearningSystem()
        self.analytics = PerformanceAnalytics()
        
    def run(self):
        """Main dashboard runner"""
        st.set_page_config(
            page_title="ML Trading Model Dashboard",
            page_icon="🤖",
            layout="wide"
        )
# Initialize authentication
auth = DashboardAuth("ML Trading Model Dashboard")

# Protect the app - this will show login form if not authenticated
auth.protect_app()

        
        st.title("🤖 ML Trading Model Dashboard")
        st.markdown("Monitor and manage machine learning model performance")
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigation",
            ["Overview", "Model Performance", "Improvement History", "Manual Controls", "Analytics"]
        )
        
        if page == "Overview":
            self.show_overview()
        elif page == "Model Performance":
            self.show_model_performance()
        elif page == "Improvement History":
            self.show_improvement_history()
        elif page == "Manual Controls":
            self.show_manual_controls()
        elif page == "Analytics":
            self.show_analytics()
    
    def show_overview(self):
        """Show overview page"""
        st.header("System Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Get current stats
        symbols = self.settings.trading.symbols
        
        with col1:
            st.metric("Active Symbols", len(symbols))
        
        with col2:
            ml_enabled = "Enabled" if self.settings.ml.enabled else "Disabled"
            st.metric("ML Status", ml_enabled)
        
        with col3:
            st.metric("Update Frequency", f"{self.settings.ml.update_frequency_days} days")
        
        with col4:
            st.metric("Confidence Threshold", f"{self.settings.ml.confidence_threshold:.0%}")
        
        # Performance summary across all symbols
        st.subheader("Recent Performance Summary (Last 30 Days)")
        
        performance_data = []
        for symbol in symbols:
            perf = asyncio.run(self.continuous_learning.evaluate_model_performance(symbol))
            performance_data.append({
                'Symbol': symbol,
                'Trades': perf.get('total_trades', 0),
                'Win Rate': f"{perf.get('win_rate', 0):.1%}",
                'Avg Profit': f"${perf.get('avg_profit', 0):.2f}",
                'Total Profit': f"${perf.get('total_profit', 0):.2f}",
                'Needs Retraining': '⚠️' if perf.get('needs_retraining', False) else '✅'
            })
        
        df = pd.DataFrame(performance_data)
        st.dataframe(df, use_container_width=True)
        
        # Quick actions
        st.subheader("Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Run Performance Analysis", type="primary"):
                with st.spinner("Running analysis..."):
                    report = self.analytics.analyze_performance()
                    st.success("Analysis complete!")
                    st.json(report['recommendations'])
        
        with col2:
            if st.button("Check for Model Updates"):
                self._check_model_updates()
        
        with col3:
            if st.button("Generate Report"):
                self.analytics.generate_report()
                st.success("Report generated in reports/performance_analysis.json")
    
    def show_model_performance(self):
        """Show detailed model performance"""
        st.header("Model Performance Details")
        
        symbol = st.selectbox("Select Symbol", self.settings.trading.symbols)
        
        if symbol:
            # Time range selector
            days_back = st.slider("Days to analyze", 7, 90, 30)
            
            # Get performance metrics
            performance = asyncio.run(
                self.continuous_learning.evaluate_model_performance(symbol, days_back)
            )
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades", performance.get('total_trades', 0))
            
            with col2:
                win_rate = performance.get('win_rate', 0)
                st.metric(
                    "Win Rate", 
                    f"{win_rate:.1%}",
                    delta=f"{win_rate - 0.5:.1%}" if win_rate != 0 else None
                )
            
            with col3:
                st.metric("Avg Profit/Trade", f"${performance.get('avg_profit', 0):.2f}")
            
            with col4:
                st.metric("Total Profit", f"${performance.get('total_profit', 0):.2f}")
            
            # Retraining status
            if performance.get('needs_retraining', False):
                st.warning("⚠️ Model needs retraining")
                st.write("Reasons:", performance.get('retrain_reasons', []))
            else:
                st.success("✅ Model performance is satisfactory")
            
            # Performance chart
            st.subheader("Performance Trend")
            self._show_performance_chart(symbol)
            
            # Trade distribution
            st.subheader("Trade Analysis")
            self._show_trade_analysis(symbol, days_back)
    
    def show_improvement_history(self):
        """Show ML improvement history"""
        st.header("ML Improvement History")
        
        history_file = Path("reports/ml_improvements.json")
        
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            if history:
                # Convert to DataFrame for display
                history_data = []
                for entry in history:
                    for symbol, details in entry.get('details', {}).items():
                        history_data.append({
                            'Timestamp': entry.get('timestamp', 'N/A'),
                            'Symbol': symbol,
                            'Action': details.get('action_taken', 'unknown'),
                            'Win Rate': f"{details.get('performance', {}).get('win_rate', 0):.1%}",
                            'Trades': details.get('performance', {}).get('total_trades', 0)
                        })
                
                df = pd.DataFrame(history_data)
                
                # Filter by action type
                action_filter = st.multiselect(
                    "Filter by Action",
                    ['none', 'retrained_and_deployed', 'retrained_but_not_deployed', 'retrain_failed'],
                    default=['retrained_and_deployed', 'retrained_but_not_deployed']
                )
                
                if action_filter:
                    df = df[df['Action'].isin(action_filter)]
                
                st.dataframe(df, use_container_width=True)
                
                # Improvement success rate
                total_retrained = len(df[df['Action'] == 'retrained_and_deployed'])
                total_attempted = len(df[df['Action'].str.contains('retrain')])
                
                if total_attempted > 0:
                    success_rate = total_retrained / total_attempted
                    st.metric("Deployment Success Rate", f"{success_rate:.1%}")
            else:
                st.info("No improvement history available yet")
        else:
            st.info("No improvement history file found")
    
    def show_manual_controls(self):
        """Show manual control panel"""
        st.header("Manual Model Management")
        
        st.warning("⚠️ Manual operations should be used with caution")
        
        # Symbol selector
        symbol = st.selectbox("Select Symbol", self.settings.trading.symbols)
        
        if symbol:
            # Current model info
            st.subheader(f"Current Model: {symbol}")
            
            # Check model status
            model_files = list(Path("models").glob(f"{symbol}_ml_package_*.pkl"))
            if model_files:
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                st.info(f"Model file: {latest_model.name}")
                st.info(f"Last modified: {datetime.fromtimestamp(latest_model.stat().st_mtime)}")
            else:
                st.warning("No model file found")
            
            # Manual actions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Force Retrain", type="primary"):
                    with st.spinner(f"Retraining {symbol}..."):
                        result = asyncio.run(self.continuous_learning.retrain_model(symbol))
                        if result['status'] == 'success':
                            st.success("Retraining completed!")
                            st.json(result['backtest'])
                        else:
                            st.error(f"Retraining failed: {result.get('error')}")
            
            with col2:
                if st.button("Run Backtest"):
                    self._run_model_backtest(symbol)
            
            with col3:
                if st.button("Compare Models"):
                    self._compare_models(symbol)
    
    def show_analytics(self):
        """Show detailed analytics"""
        st.header("Advanced Analytics")
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
        
        if st.button("Run Analysis"):
            with st.spinner("Analyzing performance..."):
                # Get analysis
                analysis = self.analytics.analyze_performance(
                    days_back=(end_date - start_date).days
                )
                
                # Overall metrics
                st.subheader("Overall Performance")
                overall = analysis.get('overall', {})
                
                if overall.get('status') != 'no_trades':
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Trades", overall.get('total_trades', 0))
                    with col2:
                        st.metric("Win Rate", f"{overall.get('overall_win_rate', 0):.1%}")
                    with col3:
                        st.metric("Total Profit", f"${overall.get('total_profit', 0):.2f}")
                    with col4:
                        st.metric("Best Symbol", overall.get('best_symbol', 'N/A'))
                
                # Recommendations
                st.subheader("System Recommendations")
                recommendations = analysis.get('recommendations', [])
                
                if recommendations:
                    for rec in recommendations:
                        st.warning(f"• {rec}")
                else:
                    st.success("No critical issues found")
                
                # Symbol breakdown
                st.subheader("Symbol Performance Breakdown")
                
                symbol_data = []
                for symbol, metrics in analysis.get('symbols', {}).items():
                    if isinstance(metrics, dict) and metrics.get('total_trades', 0) > 0:
                        symbol_data.append({
                            'Symbol': symbol,
                            'Trades': metrics.get('total_trades', 0),
                            'Win Rate': metrics.get('win_rate', 0),
                            'Profit Factor': metrics.get('profit_factor', 0),
                            'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                            'Max Drawdown': metrics.get('drawdown', 0)
                        })
                
                if symbol_data:
                    df = pd.DataFrame(symbol_data)
                    
                    # Create visualizations
                    fig_winrate = px.bar(
                        df, x='Symbol', y='Win Rate',
                        title='Win Rate by Symbol',
                        color='Win Rate',
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig_winrate, use_container_width=True)
                    
                    fig_sharpe = px.bar(
                        df, x='Symbol', y='Sharpe Ratio',
                        title='Sharpe Ratio by Symbol',
                        color='Sharpe Ratio',
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig_sharpe, use_container_width=True)
    
    def _show_performance_chart(self, symbol: str):
        """Show performance trend chart"""
        # Get trade history
        trades = self.trade_repo.get_trades_by_symbol(symbol, limit=100)
        
        if trades:
            # Calculate cumulative profit
            dates = []
            cumulative_profit = []
            running_total = 0
            
            for trade in sorted(trades, key=lambda x: x.entry_time):
                if trade.profit_loss is not None:
                    running_total += trade.profit_loss
                    dates.append(trade.entry_time)
                    cumulative_profit.append(running_total)
            
            # Create chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=cumulative_profit,
                mode='lines+markers',
                name='Cumulative Profit',
                line=dict(color='green' if running_total > 0 else 'red', width=2)
            ))
            
            fig.update_layout(
                title=f"{symbol} Cumulative Profit",
                xaxis_title="Date",
                yaxis_title="Profit ($)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trade data available for chart")
    
    def _show_trade_analysis(self, symbol: str, days_back: int):
        """Show trade distribution analysis"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        trades = self.trade_repo.get_trades_by_date_range(start_date, end_date, symbol)
        
        if trades:
            # Profit distribution
            profits = [t.profit_loss for t in trades if t.profit_loss is not None]
            
            if profits:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=profits,
                    nbinsx=20,
                    name='Profit Distribution',
                    marker_color='lightblue'
                ))
                
                # Add mean line
                mean_profit = sum(profits) / len(profits)
                fig.add_vline(
                    x=mean_profit, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"Mean: ${mean_profit:.2f}"
                )
                
                fig.update_layout(
                    title="Trade Profit Distribution",
                    xaxis_title="Profit ($)",
                    yaxis_title="Count",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Trade statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Largest Win", f"${max(profits):.2f}")
                with col2:
                    st.metric("Largest Loss", f"${min(profits):.2f}")
                with col3:
                    st.metric("Std Deviation", f"${pd.Series(profits).std():.2f}")
        else:
            st.info("No trades in selected period")
    
    def _check_model_updates(self):
        """Check if any models need updating"""
        with st.spinner("Checking models..."):
            symbols_needing_update = []
            
            for symbol in self.settings.trading.symbols:
                performance = asyncio.run(
                    self.continuous_learning.evaluate_model_performance(symbol)
                )
                if performance.get('needs_retraining', False):
                    symbols_needing_update.append(symbol)
            
            if symbols_needing_update:
                st.warning(f"Models needing update: {', '.join(symbols_needing_update)}")
            else:
                st.success("All models are performing well!")
    
    def _run_model_backtest(self, symbol: str):
        """Run backtest for a specific model"""
        with st.spinner(f"Running backtest for {symbol}..."):
            # This would integrate with your backtesting system
            st.info("Backtest functionality to be implemented")
    
    def _compare_models(self, symbol: str):
        """Compare different model versions"""
        st.subheader(f"Model Comparison: {symbol}")
        
        # Find all model files for this symbol
        model_files = list(Path("models").glob(f"{symbol}_ml_package_*.pkl"))
        
        if len(model_files) > 1:
            st.info(f"Found {len(model_files)} model versions")
            
            # Show model files
            for model_file in sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True):
                st.write(f"- {model_file.name} (Modified: {datetime.fromtimestamp(model_file.stat().st_mtime)})")
        else:
            st.info("Only one model version found")


def main():
    """Run the dashboard"""
    dashboard = MLImprovementDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()