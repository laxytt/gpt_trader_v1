#!/usr/bin/env python3
"""
GPT Flow Visualization Dashboard
Real-time visualization of Trading Council flow, GPT requests, and system operations
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import sys
import time
from scripts.auth_utils import DashboardAuth

from typing import Dict, List, Any, Optional, Tuple
import sqlite3
from collections import deque, defaultdict
import threading
import queue

sys.path.append(str(Path(__file__).parent.parent))

from core.infrastructure.gpt.client import GPTClient
from core.infrastructure.gpt.request_logger import get_request_logger
from core.infrastructure.gpt.rate_limiter import get_rate_limiter
from core.infrastructure.database.repositories import SignalRepository
from core.domain.enums import GPTModels
from config.settings import get_settings


class GPTRequestMonitor:
    """Monitor and track GPT requests in real-time"""
    
    def __init__(self):
        self.request_queue = deque(maxlen=100)  # Keep last 100 requests
        self.active_requests = {}
        self.request_stats = defaultdict(lambda: {'count': 0, 'total_tokens': 0, 'total_cost': 0})
        self.agent_stats = defaultdict(lambda: {'count': 0, 'avg_confidence': 0})
        
    def log_request(self, request_id: str, agent_type: str, prompt: str, timestamp: datetime):
        """Log a new request"""
        request = {
            'id': request_id,
            'agent_type': agent_type,
            'prompt': prompt[:500] + '...' if len(prompt) > 500 else prompt,
            'timestamp': timestamp,
            'status': 'pending',
            'duration': None,
            'tokens': None,
            'cost': None,
            'response': None
        }
        self.active_requests[request_id] = request
        self.request_queue.append(request)
        
    def complete_request(self, request_id: str, response: Dict[str, Any], duration: float):
        """Mark request as complete with response details"""
        if request_id in self.active_requests:
            request = self.active_requests[request_id]
            request['status'] = 'completed'
            request['duration'] = duration
            request['tokens'] = response.get('token_usage', {})
            request['cost'] = response.get('estimated_cost', 0)
            request['response'] = response.get('content', '')[:500] + '...' if len(response.get('content', '')) > 500 else response.get('content', '')
            
            # Update stats
            agent = request['agent_type']
            self.request_stats[agent]['count'] += 1
            self.request_stats[agent]['total_tokens'] += request['tokens'].get('total_tokens', 0)
            self.request_stats[agent]['total_cost'] += request['cost']
            
            del self.active_requests[request_id]
            
    def get_recent_requests(self, limit: int = 20) -> List[Dict]:
        """Get recent requests"""
        return list(self.request_queue)[-limit:]
    
    def get_active_requests(self) -> List[Dict]:
        """Get currently active requests"""
        return list(self.active_requests.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics"""
        return dict(self.request_stats)


class GPTFlowDashboard:
    """Dashboard for visualizing GPT request flow and Trading Council operations"""
    
    def __init__(self):
        self.settings = get_settings()
        self.signal_repo = SignalRepository(self.settings.database.db_path)
        self.monitor = GPTRequestMonitor()
        
        # Initialize production components
        self.request_logger = get_request_logger()
        self.rate_limiter = get_rate_limiter()
        
        # Initialize session state
        if 'request_monitor' not in st.session_state:
            st.session_state.request_monitor = self.monitor
        else:
            self.monitor = st.session_state.request_monitor
    
    def run(self):
        """Main dashboard runner"""
        st.set_page_config(
            page_title="GPT Flow Visualization",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
# Initialize authentication
auth = DashboardAuth("GPT Flow Visualization")

# Protect the app - this will show login form if not authenticated
auth.protect_app()

        
        st.title("🤖 GPT Flow Visualization Dashboard")
        st.markdown("Real-time monitoring of Trading Council operations and GPT requests")
        
        # Sidebar controls
        with st.sidebar:
            st.header("Controls")
            
            # Refresh controls
            auto_refresh = st.checkbox("Auto-refresh (5s)", value=False)
            if st.button("🔄 Refresh Now"):
                st.rerun()
            
            # View selector
            view_mode = st.selectbox(
                "View Mode",
                ["Flow Overview", "Request Timeline", "Agent Performance", 
                 "Token Usage", "Council Decisions", "Cost Analysis",
                 "Request Payloads", "Rate Limiter Status"]
            )
            
            # Time range
            time_range = st.selectbox(
                "Time Range",
                ["Last 5 minutes", "Last 30 minutes", "Last hour", "Last 24 hours"]
            )
            
            # Simulate requests toggle
            if st.button("🎭 Simulate Council Session"):
                self._simulate_council_session()
                st.success("Simulated council session added!")
                st.rerun()
        
        # Auto-refresh logic
        if auto_refresh:
            time.sleep(5)
            st.rerun()
        
        # Main content based on view mode
        if view_mode == "Flow Overview":
            self.show_flow_overview()
        elif view_mode == "Request Timeline":
            self.show_request_timeline()
        elif view_mode == "Agent Performance":
            self.show_agent_performance()
        elif view_mode == "Token Usage":
            self.show_token_usage()
        elif view_mode == "Council Decisions":
            self.show_council_decisions()
        elif view_mode == "Cost Analysis":
            self.show_cost_analysis()
        elif view_mode == "Request Payloads":
            self.show_request_payloads()
        elif view_mode == "Rate Limiter Status":
            self.show_rate_limiter_status()
    
    def show_flow_overview(self):
        """Show visual representation of the Trading Council flow"""
        st.header("Trading Council Flow")
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["System Architecture", "Live Flow", "Request Status"])
        
        with tab1:
            self._show_system_architecture()
        
        with tab2:
            self._show_live_flow()
        
        with tab3:
            self._show_request_status()
    
    def _show_system_architecture(self):
        """Show static system architecture diagram"""
        st.subheader("Trading Council Architecture")
        
        # Create Sankey diagram for system flow
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=[
                    "Market Data", "News Feed", "ML Models",  # Sources (0-2)
                    "Technical Analyst", "Fundamental Analyst", "Sentiment Reader",  # Agents (3-5)
                    "Risk Manager", "Momentum Trader", "Contrarian Trader",  # Agents (6-8)
                    "Head Trader",  # Synthesizer (9)
                    "Council Decision",  # Output (10)
                    "Trading Signal", "Risk Assessment", "Execution"  # Final outputs (11-13)
                ],
                color=[
                    "#1f77b4", "#1f77b4", "#1f77b4",  # Blue for sources
                    "#ff7f0e", "#ff7f0e", "#ff7f0e",  # Orange for analysts
                    "#2ca02c", "#2ca02c", "#2ca02c",  # Green for traders
                    "#d62728",  # Red for head trader
                    "#9467bd",  # Purple for decision
                    "#8c564b", "#8c564b", "#8c564b"  # Brown for outputs
                ]
            ),
            link=dict(
                source=[0, 0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10],
                target=[3, 6, 7, 4, 5, 5, 8, 9, 9, 9, 9, 9, 9, 10, 11, 12, 13],
                value=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 1, 1, 1],
                color="rgba(0,0,0,0.2)"
            )
        )])
        
        fig.update_layout(
            title="Trading Council Information Flow",
            font_size=10,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Agent descriptions
        with st.expander("Agent Roles and Responsibilities"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Analysis Agents:**
                - 📊 **Technical Analyst**: Chart patterns, indicators, support/resistance
                - 📰 **Fundamental Analyst**: News impact, economic events, market catalysts
                - 🧠 **Sentiment Reader**: Market psychology, trader positioning, sentiment shifts
                """)
            
            with col2:
                st.markdown("""
                **Trading Agents:**
                - 🛡️ **Risk Manager**: Position sizing, stop loss, risk assessment (has veto power)
                - 🚀 **Momentum Trader**: Trend following, breakout strategies
                - 🔄 **Contrarian Trader**: Reversals, mean reversion, fade moves
                - 👔 **Head Trader**: Synthesizes decisions, moderates debates
                """)
    
    def _show_live_flow(self):
        """Show live request flow visualization"""
        st.subheader("Live Request Flow")
        
        # Get active and recent requests
        active_requests = self.monitor.get_active_requests()
        recent_requests = self.monitor.get_recent_requests(10)
        
        # Active requests indicator
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Active Requests", len(active_requests))
        
        with col2:
            total_requests = len(self.monitor.request_queue)
            st.metric("Total Requests (Session)", total_requests)
        
        with col3:
            if recent_requests:
                avg_duration = sum(r['duration'] or 0 for r in recent_requests if r['duration']) / len([r for r in recent_requests if r['duration']])
                st.metric("Avg Response Time", f"{avg_duration:.2f}s")
            else:
                st.metric("Avg Response Time", "N/A")
        
        # Visual flow representation
        if active_requests or recent_requests:
            # Create timeline visualization
            fig = go.Figure()
            
            # Add active requests
            for i, req in enumerate(active_requests):
                fig.add_trace(go.Scatter(
                    x=[req['timestamp'], datetime.now()],
                    y=[i, i],
                    mode='lines+markers',
                    name=f"{req['agent_type']} (Active)",
                    line=dict(color='orange', width=3),
                    marker=dict(size=10),
                    hovertemplate=f"Agent: {req['agent_type']}<br>Status: Active<br>Started: %{{x}}<extra></extra>"
                ))
            
            # Add completed requests
            offset = len(active_requests)
            for i, req in enumerate(recent_requests[-5:]):
                if req['status'] == 'completed' and req['duration']:
                    end_time = req['timestamp'] + timedelta(seconds=req['duration'])
                    fig.add_trace(go.Scatter(
                        x=[req['timestamp'], end_time],
                        y=[i + offset, i + offset],
                        mode='lines+markers',
                        name=f"{req['agent_type']}",
                        line=dict(color='green', width=2),
                        marker=dict(size=8),
                        hovertemplate=f"Agent: {req['agent_type']}<br>Duration: {req['duration']:.2f}s<br>Tokens: {req['tokens'].get('total_tokens', 0) if req['tokens'] else 0}<extra></extra>"
                    ))
            
            fig.update_layout(
                title="Request Timeline",
                xaxis_title="Time",
                yaxis_title="Request",
                showlegend=False,
                height=400,
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No requests to display. Simulate a council session to see the flow.")
    
    def _show_request_status(self):
        """Show detailed request status"""
        st.subheader("Request Details")
        
        recent_requests = self.monitor.get_recent_requests(20)
        
        if recent_requests:
            # Convert to DataFrame for display
            df_data = []
            for req in recent_requests:
                df_data.append({
                    'Timestamp': req['timestamp'].strftime('%H:%M:%S'),
                    'Agent': req['agent_type'],
                    'Status': req['status'],
                    'Duration (s)': f"{req['duration']:.2f}" if req['duration'] else 'N/A',
                    'Tokens': req['tokens'].get('total_tokens', 0) if req['tokens'] else 0,
                    'Cost ($)': f"{req['cost']:.4f}" if req['cost'] else 'N/A'
                })
            
            df = pd.DataFrame(df_data)
            
            # Style the dataframe
            def style_status(val):
                if val == 'completed':
                    return 'color: green'
                elif val == 'pending':
                    return 'color: orange'
                else:
                    return 'color: red'
            
            styled_df = df.style.applymap(style_status, subset=['Status'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Show request/response details
            st.subheader("Request/Response Details")
            
            selected_idx = st.selectbox(
                "Select request to view details",
                range(len(recent_requests)),
                format_func=lambda x: f"{recent_requests[x]['timestamp'].strftime('%H:%M:%S')} - {recent_requests[x]['agent_type']}"
            )
            
            if selected_idx is not None:
                req = recent_requests[selected_idx]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Request:**")
                    st.code(req['prompt'], language='text')
                
                with col2:
                    st.markdown("**Response:**")
                    if req['response']:
                        st.code(req['response'], language='text')
                    else:
                        st.info("No response yet")
        else:
            st.info("No requests logged yet")
    
    def show_request_timeline(self):
        """Show detailed request timeline"""
        st.header("Request Timeline")
        
        # Get all requests
        all_requests = list(self.monitor.request_queue)
        
        if not all_requests:
            st.info("No requests to display")
            return
        
        # Create Gantt chart
        fig = go.Figure()
        
        # Group by agent type
        agent_requests = defaultdict(list)
        for req in all_requests:
            agent_requests[req['agent_type']].append(req)
        
        # Color map for agents
        colors = px.colors.qualitative.Set3
        agent_colors = {agent: colors[i % len(colors)] for i, agent in enumerate(agent_requests.keys())}
        
        # Add traces for each agent
        y_pos = 0
        for agent, requests in agent_requests.items():
            for req in requests:
                if req['status'] == 'completed' and req['duration']:
                    start = req['timestamp']
                    end = start + timedelta(seconds=req['duration'])
                    
                    fig.add_trace(go.Scatter(
                        x=[start, end, end, start, start],
                        y=[y_pos-0.4, y_pos-0.4, y_pos+0.4, y_pos+0.4, y_pos-0.4],
                        fill='toself',
                        fillcolor=agent_colors[agent],
                        line=dict(color=agent_colors[agent]),
                        name=agent,
                        text=f"{agent}<br>Tokens: {req['tokens'].get('total_tokens', 0) if req['tokens'] else 0}<br>Cost: ${req['cost']:.4f}" if req['cost'] else "No cost data",
                        hoverinfo='text',
                        showlegend=False
                    ))
            y_pos += 1
        
        # Update layout
        fig.update_layout(
            title="Agent Request Timeline",
            xaxis_title="Time",
            yaxis=dict(
                ticktext=list(agent_requests.keys()),
                tickvals=list(range(len(agent_requests))),
                title="Agent"
            ),
            height=max(400, len(agent_requests) * 80),
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("Timeline Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        completed_requests = [r for r in all_requests if r['status'] == 'completed']
        
        with col1:
            st.metric("Total Requests", len(all_requests))
        
        with col2:
            st.metric("Completed", len(completed_requests))
        
        with col3:
            if completed_requests:
                total_duration = sum(r['duration'] for r in completed_requests if r['duration'])
                st.metric("Total Time", f"{total_duration:.1f}s")
            else:
                st.metric("Total Time", "0s")
        
        with col4:
            if completed_requests:
                total_cost = sum(r['cost'] for r in completed_requests if r['cost'])
                st.metric("Total Cost", f"${total_cost:.4f}")
            else:
                st.metric("Total Cost", "$0")
    
    def show_agent_performance(self):
        """Show agent performance metrics"""
        st.header("Agent Performance Analysis")
        
        stats = self.monitor.get_stats()
        
        if not stats:
            st.info("No performance data available")
            return
        
        # Convert to DataFrame
        df_data = []
        for agent, data in stats.items():
            if data['count'] > 0:
                df_data.append({
                    'Agent': agent,
                    'Requests': data['count'],
                    'Total Tokens': data['total_tokens'],
                    'Avg Tokens': data['total_tokens'] / data['count'],
                    'Total Cost': data['total_cost'],
                    'Avg Cost': data['total_cost'] / data['count']
                })
        
        if df_data:
            df = pd.DataFrame(df_data)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Request Count by Agent', 'Average Tokens per Request',
                              'Total Cost by Agent', 'Cost Efficiency (Tokens per Dollar)'),
                specs=[[{'type': 'bar'}, {'type': 'bar'}],
                       [{'type': 'pie'}, {'type': 'scatter'}]]
            )
            
            # Request count
            fig.add_trace(
                go.Bar(x=df['Agent'], y=df['Requests'], name='Requests'),
                row=1, col=1
            )
            
            # Average tokens
            fig.add_trace(
                go.Bar(x=df['Agent'], y=df['Avg Tokens'], name='Avg Tokens'),
                row=1, col=2
            )
            
            # Cost distribution
            fig.add_trace(
                go.Pie(labels=df['Agent'], values=df['Total Cost'], name='Cost Share'),
                row=2, col=1
            )
            
            # Cost efficiency
            df['Efficiency'] = df['Total Tokens'] / (df['Total Cost'] + 0.0001)  # Avoid division by zero
            fig.add_trace(
                go.Scatter(x=df['Agent'], y=df['Efficiency'], mode='markers+lines', 
                          marker=dict(size=10), name='Efficiency'),
                row=2, col=2
            )
            
            fig.update_layout(height=800, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.subheader("Detailed Agent Metrics")
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No agent performance data available")
    
    def show_token_usage(self):
        """Show token usage analysis"""
        st.header("Token Usage Analysis")
        
        all_requests = list(self.monitor.request_queue)
        completed_requests = [r for r in all_requests if r['status'] == 'completed' and r['tokens']]
        
        if not completed_requests:
            st.info("No token usage data available")
            return
        
        # Token usage over time
        st.subheader("Token Usage Over Time")
        
        # Prepare data
        times = []
        prompt_tokens = []
        completion_tokens = []
        total_tokens = []
        
        for req in completed_requests:
            times.append(req['timestamp'])
            tokens = req['tokens']
            prompt_tokens.append(tokens.get('prompt_tokens', 0))
            completion_tokens.append(tokens.get('completion_tokens', 0))
            total_tokens.append(tokens.get('total_tokens', 0))
        
        # Create line chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=times, y=prompt_tokens,
            mode='lines+markers',
            name='Prompt Tokens',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=times, y=completion_tokens,
            mode='lines+markers',
            name='Completion Tokens',
            line=dict(color='green')
        ))
        
        fig.add_trace(go.Scatter(
            x=times, y=total_tokens,
            mode='lines+markers',
            name='Total Tokens',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title="Token Usage Timeline",
            xaxis_title="Time",
            yaxis_title="Tokens",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Token distribution by agent
        st.subheader("Token Distribution by Agent")
        
        agent_tokens = defaultdict(lambda: {'prompt': 0, 'completion': 0, 'total': 0, 'count': 0})
        
        for req in completed_requests:
            agent = req['agent_type']
            tokens = req['tokens']
            agent_tokens[agent]['prompt'] += tokens.get('prompt_tokens', 0)
            agent_tokens[agent]['completion'] += tokens.get('completion_tokens', 0)
            agent_tokens[agent]['total'] += tokens.get('total_tokens', 0)
            agent_tokens[agent]['count'] += 1
        
        # Create stacked bar chart
        agents = list(agent_tokens.keys())
        prompt_vals = [agent_tokens[a]['prompt'] for a in agents]
        completion_vals = [agent_tokens[a]['completion'] for a in agents]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Prompt Tokens',
            x=agents,
            y=prompt_vals,
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Completion Tokens',
            x=agents,
            y=completion_vals,
            marker_color='lightgreen'
        ))
        
        fig.update_layout(
            barmode='stack',
            title="Token Usage by Agent",
            xaxis_title="Agent",
            yaxis_title="Total Tokens"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("Token Usage Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_prompt = sum(prompt_tokens)
        total_completion = sum(completion_tokens)
        total_all = sum(total_tokens)
        
        with col1:
            st.metric("Total Prompt Tokens", f"{total_prompt:,}")
        
        with col2:
            st.metric("Total Completion Tokens", f"{total_completion:,}")
        
        with col3:
            st.metric("Total Tokens", f"{total_all:,}")
        
        with col4:
            ratio = total_completion / total_prompt if total_prompt > 0 else 0
            st.metric("Completion/Prompt Ratio", f"{ratio:.2f}")
    
    def show_council_decisions(self):
        """Show Trading Council decisions with debate flow"""
        st.header("Trading Council Decisions")
        
        # Get recent signals with council analysis
        try:
            signals = self.signal_repo.get_all_signals(limit=20)
            council_signals = []
            
            for signal in signals:
                if signal.get('analysis') and 'agent_opinions' in str(signal.get('analysis', '')):
                    # Parse created_at timestamp
                    created_at_str = signal.get('created_at', '')
                    try:
                        if 'T' in created_at_str:
                            created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                        else:
                            created_at = datetime.strptime(created_at_str, '%Y-%m-%d %H:%M:%S')
                    except:
                        created_at = datetime.now()
                    
                    council_signals.append({
                        'id': signal.get('id'),
                        'symbol': signal.get('symbol'),
                        'direction': signal.get('direction'),
                        'confidence': signal.get('confidence', 0),
                        'ml_confidence': signal.get('ml_confidence', 0),
                        'analysis': signal.get('analysis'),
                        'created_at': created_at
                    })
            
            if not council_signals:
                st.info("No council decisions found. Simulate a council session to see decisions.")
                return
            
            # Decision selector
            selected_idx = st.selectbox(
                "Select Council Decision",
                range(len(council_signals)),
                format_func=lambda x: f"{council_signals[x]['created_at'].strftime('%Y-%m-%d %H:%M')} - {council_signals[x]['symbol']} - {council_signals[x]['direction']}"
            )
            
            if selected_idx is not None:
                decision = council_signals[selected_idx]
                
                # Display decision details
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Symbol", decision['symbol'])
                
                with col2:
                    st.metric("Direction", decision['direction'])
                
                with col3:
                    st.metric("Confidence", f"{decision['confidence']:.1%}")
                
                # Parse and display council analysis
                try:
                    analysis = json.loads(decision['analysis']) if isinstance(decision['analysis'], str) else decision['analysis']
                    
                    # Visualize debate flow
                    st.subheader("Council Debate Flow")
                    
                    if 'agent_opinions' in analysis:
                        # Create Sankey diagram for decision flow
                        agents = list(analysis['agent_opinions'].keys())
                        directions = ['BUY', 'SELL', 'WAIT']
                        
                        # Build links
                        source = []
                        target = []
                        value = []
                        link_colors = []
                        
                        agent_indices = {agent: i for i, agent in enumerate(agents)}
                        direction_indices = {d: i + len(agents) for i, d in enumerate(directions)}
                        
                        for agent, opinion in analysis['agent_opinions'].items():
                            agent_idx = agent_indices[agent]
                            direction = opinion.get('direction', 'WAIT')
                            if direction in direction_indices:
                                direction_idx = direction_indices[direction]
                                confidence = opinion.get('confidence', 0.5)
                                
                                source.append(agent_idx)
                                target.append(direction_idx)
                                value.append(confidence)
                                
                                # Color based on confidence
                                if confidence >= 0.8:
                                    link_colors.append('rgba(0,255,0,0.4)')
                                elif confidence >= 0.6:
                                    link_colors.append('rgba(255,255,0,0.4)')
                                else:
                                    link_colors.append('rgba(255,0,0,0.4)')
                        
                        # Create Sankey
                        fig = go.Figure(data=[go.Sankey(
                            node=dict(
                                pad=15,
                                thickness=20,
                                line=dict(color="black", width=0.5),
                                label=agents + directions,
                                color=['#1f77b4'] * len(agents) + ['#2ca02c', '#d62728', '#ff7f0e']
                            ),
                            link=dict(
                                source=source,
                                target=target,
                                value=value,
                                color=link_colors
                            )
                        )])
                        
                        fig.update_layout(
                            title="Agent Vote Distribution",
                            font_size=10,
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show individual agent analyses
                    st.subheader("Agent Analyses")
                    
                    for agent, opinion in analysis.get('agent_opinions', {}).items():
                        with st.expander(f"{agent} - {opinion.get('direction', 'N/A')} ({opinion.get('confidence', 0):.1%})"):
                            st.write(f"**Reasoning:** {opinion.get('reasoning', 'No reasoning provided')}")
                            
                            if 'key_points' in opinion:
                                st.write("**Key Points:**")
                                for point in opinion['key_points']:
                                    st.write(f"- {point}")
                    
                    # Show consensus details
                    if 'consensus' in analysis:
                        st.subheader("Final Consensus")
                        consensus = analysis['consensus']
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Direction:** {consensus.get('direction', 'N/A')}")
                            st.write(f"**Confidence:** {consensus.get('confidence', 0):.1%}")
                        
                        with col2:
                            st.write(f"**Key Points:** {consensus.get('key_points', 'N/A')}")
                            
                            if 'dissenting_views' in consensus:
                                st.write("**Dissenting Views:**")
                                for view in consensus['dissenting_views']:
                                    st.write(f"- {view}")
                    
                except Exception as e:
                    st.error(f"Error parsing council analysis: {e}")
                    st.text(str(decision['analysis']))
                    
        except Exception as e:
            st.error(f"Error loading council decisions: {e}")
    
    def show_cost_analysis(self):
        """Show cost analysis and projections"""
        st.header("Cost Analysis")
        
        all_requests = list(self.monitor.request_queue)
        completed_requests = [r for r in all_requests if r['status'] == 'completed' and r['cost']]
        
        if not completed_requests:
            st.info("No cost data available")
            return
        
        # Cost over time
        st.subheader("Cumulative Cost")
        
        times = []
        costs = []
        cumulative_cost = 0
        
        for req in sorted(completed_requests, key=lambda x: x['timestamp']):
            cumulative_cost += req['cost']
            times.append(req['timestamp'])
            costs.append(cumulative_cost)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=times,
            y=costs,
            mode='lines+markers',
            name='Cumulative Cost',
            line=dict(color='green', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,255,0,0.1)'
        ))
        
        fig.update_layout(
            title="Cumulative API Cost",
            xaxis_title="Time",
            yaxis_title="Cost ($)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Cost breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            # Cost by agent
            st.subheader("Cost by Agent")
            
            agent_costs = defaultdict(float)
            for req in completed_requests:
                agent_costs[req['agent_type']] += req['cost']
            
            fig = go.Figure(data=[go.Pie(
                labels=list(agent_costs.keys()),
                values=list(agent_costs.values()),
                hole=0.3
            )])
            
            fig.update_layout(title="Cost Distribution by Agent")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cost projections
            st.subheader("Cost Projections")
            
            # Calculate hourly rate
            if len(completed_requests) > 1:
                time_span = (completed_requests[-1]['timestamp'] - completed_requests[0]['timestamp']).total_seconds() / 3600
                if time_span > 0:
                    hourly_rate = cumulative_cost / time_span
                    
                    projections = {
                        'Per Hour': hourly_rate,
                        'Per Day': hourly_rate * 24,
                        'Per Week': hourly_rate * 24 * 7,
                        'Per Month': hourly_rate * 24 * 30
                    }
                    
                    projection_df = pd.DataFrame(
                        list(projections.items()),
                        columns=['Period', 'Projected Cost ($)']
                    )
                    
                    projection_df['Projected Cost ($)'] = projection_df['Projected Cost ($)'].apply(lambda x: f"${x:.2f}")
                    
                    st.dataframe(projection_df, use_container_width=True)
        
        # Summary metrics
        st.subheader("Cost Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_cost = sum(r['cost'] for r in completed_requests)
        avg_cost = total_cost / len(completed_requests) if completed_requests else 0
        
        with col1:
            st.metric("Total Cost", f"${total_cost:.4f}")
        
        with col2:
            st.metric("Average Cost/Request", f"${avg_cost:.4f}")
        
        with col3:
            st.metric("Total Requests", len(completed_requests))
        
        with col4:
            # Model cost breakdown
            model_costs = defaultdict(float)
            for req in completed_requests:
                # Assume model from config (you'd track this per request in production)
                model = self.settings.gpt.model
                model_costs[model] += req['cost']
            
            most_expensive_model = max(model_costs.items(), key=lambda x: x[1])[0] if model_costs else "N/A"
            st.metric("Primary Model", most_expensive_model)
    
    def show_request_payloads(self):
        """Show detailed GPT request payloads from the database"""
        st.header("📋 GPT Request Payloads")
        
        # Time filter
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            hours_back = st.slider("Hours to look back", 1, 24, 4)
        
        with col2:
            agent_filter = st.selectbox(
                "Filter by Agent",
                ["All"] + ["Technical Analyst", "Fundamental Analyst", "Sentiment Reader",
                          "Risk Manager", "Momentum Trader", "Contrarian Trader", "Head Trader"]
            )
        
        with col3:
            include_errors = st.checkbox("Include Errors", value=True)
        
        # Get requests from database
        try:
            requests = self.request_logger.get_recent_requests(
                limit=100,
                hours_back=hours_back,
                agent_type=None if agent_filter == "All" else agent_filter,
                include_errors=include_errors
            )
            
            if not requests:
                st.info("No requests found in the selected time range")
                return
            
            # Display summary stats
            stats = self.request_logger.get_usage_stats(hours=hours_back)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Requests", stats['overall'].get('total_requests', 0))
            
            with col2:
                st.metric("Total Tokens", f"{stats['overall'].get('total_tokens', 0):,}")
            
            with col3:
                st.metric("Total Cost", f"${stats['overall'].get('total_cost', 0):.4f}")
            
            with col4:
                error_rate = (stats['overall'].get('error_count', 0) / 
                             stats['overall'].get('total_requests', 1)) * 100
                st.metric("Error Rate", f"{error_rate:.1f}%")
            
            # Show individual requests
            st.subheader("Recent Requests")
            
            for req in requests:
                # Parse timestamp and handle timezone
                timestamp_str = req['timestamp']
                try:
                    # Parse ISO timestamp
                    if 'T' in timestamp_str:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    
                    # Convert to local time for display
                    if timestamp.tzinfo is None:
                        # Assume UTC if no timezone info
                        from datetime import timezone as tz
                        timestamp = timestamp.replace(tzinfo=tz.utc)
                    local_ts = timestamp.astimezone()
                    time_str = local_ts.strftime('%H:%M:%S')
                except:
                    # Fallback to simple extraction
                    time_str = timestamp_str.split('T')[1][:8] if 'T' in timestamp_str else timestamp_str[11:19]
                
                # Create expandable section for each request
                with st.expander(
                    f"{req['agent_type'] or 'Unknown'} - {time_str} "
                    f"({req['total_tokens'] or 0} tokens) "
                    f"{'❌' if req['error'] else '✅'}"
                ):
                    # Request details
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.write("**Request ID:**")
                        st.write("**Model:**")
                        st.write("**Symbol:**")
                        st.write("**Type:**")
                        st.write("**Duration:**")
                        st.write("**Cost:**")
                        if req['error']:
                            st.write("**Error:**")
                    
                    with col2:
                        st.write(f"`{req['id'][:8]}...`")
                        st.write(req['model'])
                        st.write(req['symbol'] or "N/A")
                        st.write(req['request_type'])
                        st.write(f"{req['duration_ms'] or 0}ms")
                        st.write(f"${req['cost'] or 0:.6f}")
                        if req['error']:
                            st.error(req['error'])
                    
                    # Messages
                    st.subheader("Messages")
                    messages = req['messages']
                    
                    for i, msg in enumerate(messages):
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')
                        
                        if role == 'system':
                            st.info(f"**System:** {content[:500]}...")
                        elif role == 'user':
                            st.write(f"**User:** {content[:500]}...")
                        elif role == 'assistant':
                            st.success(f"**Assistant:** {content[:500]}...")
                    
                    # Response
                    if req['response_text']:
                        st.subheader("Response")
                        st.code(req['response_text'][:1000] + "..." if len(req['response_text']) > 1000 else req['response_text'])
                    
                    # Token breakdown
                    if req['total_tokens']:
                        st.subheader("Token Usage")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Prompt Tokens", req['prompt_tokens'])
                        
                        with col2:
                            st.metric("Completion Tokens", req['completion_tokens'])
                        
                        with col3:
                            st.metric("Total Tokens", req['total_tokens'])
            
        except Exception as e:
            st.error(f"Error loading request payloads: {e}")
            st.info("Make sure the trading system has been running with the new logging infrastructure.")
    
    def show_rate_limiter_status(self):
        """Show rate limiter status and statistics"""
        st.header("🚦 Rate Limiter Status")
        
        # Get rate limiter stats
        stats = self.rate_limiter.get_stats()
        
        # Overall stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("OpenAI Tier", stats['tier'].upper())
        
        with col2:
            st.metric("Safety Margin", f"{stats['safety_margin'] * 100:.0f}%")
        
        with col3:
            st.metric("Requests Allowed", stats['requests_allowed'])
        
        with col4:
            st.metric("Requests Throttled", stats['requests_throttled'])
        
        # Rate limit status per model
        st.subheader("Available Capacity by Model")
        
        bucket_data = []
        for model, buckets in stats['current_buckets'].items():
            bucket_data.append({
                'Model': model,
                'Requests Available': f"{buckets['requests_available']:.0f}",
                'Tokens Available': f"{buckets['tokens_available']:,.0f}",
                'Requests/Min Limit': self.rate_limiter.request_buckets[model].capacity,
                'Tokens/Min Limit': self.rate_limiter.token_buckets[model].capacity
            })
        
        if bucket_data:
            df = pd.DataFrame(bucket_data)
            st.dataframe(df, use_container_width=True)
        
        # Visualize bucket levels
        st.subheader("Rate Limit Bucket Levels")
        
        models = list(stats['current_buckets'].keys())
        
        if models:
            # Create subplot for each model
            fig = make_subplots(
                rows=len(models), 
                cols=2,
                subplot_titles=[title for model in models for title in [f"{model} - Requests", f"{model} - Tokens"]],
                specs=[[{"type": "indicator"}, {"type": "indicator"}] for _ in models]
            )
            
            for i, model in enumerate(models):
                buckets = stats['current_buckets'][model]
                request_bucket = self.rate_limiter.request_buckets[model]
                token_bucket = self.rate_limiter.token_buckets[model]
                
                # Requests gauge
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=buckets['requests_available'],
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Requests"},
                        gauge={
                            'axis': {'range': [0, request_bucket.capacity]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, request_bucket.capacity * 0.2], 'color': "red"},
                                {'range': [request_bucket.capacity * 0.2, request_bucket.capacity * 0.5], 'color': "yellow"},
                                {'range': [request_bucket.capacity * 0.5, request_bucket.capacity], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': request_bucket.capacity * 0.8
                            }
                        }
                    ),
                    row=i+1, col=1
                )
                
                # Tokens gauge
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=buckets['tokens_available'],
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Tokens"},
                        gauge={
                            'axis': {'range': [0, token_bucket.capacity]},
                            'bar': {'color': "darkgreen"},
                            'steps': [
                                {'range': [0, token_bucket.capacity * 0.2], 'color': "red"},
                                {'range': [token_bucket.capacity * 0.2, token_bucket.capacity * 0.5], 'color': "yellow"},
                                {'range': [token_bucket.capacity * 0.5, token_bucket.capacity], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': token_bucket.capacity * 0.8
                            }
                        }
                    ),
                    row=i+1, col=2
                )
            
            fig.update_layout(height=300 * len(models), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Rate limit recommendations
        st.subheader("Recommendations")
        
        if stats['requests_throttled'] > stats['requests_allowed'] * 0.1:
            st.warning("""
            ⚠️ **High throttling rate detected!**
            
            Consider:
            - Reducing the number of trading symbols
            - Increasing delays between agent calls
            - Upgrading your OpenAI tier
            - Enabling quick mode to skip debates
            """)
        else:
            st.success("✅ Rate limiting is working well with current settings.")
        
        # Show current configuration
        with st.expander("Current Rate Limiting Configuration"):
            st.json({
                "tier": stats['tier'],
                "safety_margin": stats['safety_margin'],
                "agent_delay": self.settings.trading.council_agent_delay,
                "quick_mode": self.settings.trading.council_quick_mode,
                "debate_rounds": self.settings.trading.council_debate_rounds,
                "symbols": self.settings.trading.symbols
            })
    
    def _simulate_council_session(self):
        """Simulate a Trading Council session for demonstration"""
        import uuid
        from datetime import datetime
        import random
        
        agents = [
            "Technical Analyst",
            "Fundamental Analyst", 
            "Sentiment Reader",
            "Risk Manager",
            "Momentum Trader",
            "Contrarian Trader",
            "Head Trader"
        ]
        
        # Simulate individual analysis phase
        for agent in agents[:-1]:  # All except Head Trader
            request_id = str(uuid.uuid4())
            prompt = f"Analyze EURUSD market conditions from {agent} perspective..."
            
            self.monitor.log_request(request_id, agent, prompt, datetime.now())
            
            # Simulate completion after random delay
            duration = random.uniform(0.5, 2.5)
            tokens = {
                'prompt_tokens': random.randint(200, 500),
                'completion_tokens': random.randint(150, 400),
                'total_tokens': 0
            }
            tokens['total_tokens'] = tokens['prompt_tokens'] + tokens['completion_tokens']
            
            response = {
                'content': f"{agent} analysis: Market shows {'bullish' if random.random() > 0.5 else 'bearish'} signals...",
                'token_usage': tokens,
                'estimated_cost': tokens['total_tokens'] * 0.00002  # Rough estimate
            }
            
            self.monitor.complete_request(request_id, response, duration)
        
        # Simulate debate rounds
        for round_num in range(1, 4):
            for agent in agents[:-1]:
                request_id = str(uuid.uuid4())
                prompt = f"Debate round {round_num}: {agent} responds to other agents..."
                
                self.monitor.log_request(request_id, agent, prompt, datetime.now())
                
                duration = random.uniform(0.3, 1.5)
                tokens = {
                    'prompt_tokens': random.randint(300, 600),
                    'completion_tokens': random.randint(100, 250),
                    'total_tokens': 0
                }
                tokens['total_tokens'] = tokens['prompt_tokens'] + tokens['completion_tokens']
                
                response = {
                    'content': f"Round {round_num} - {agent}: I {'maintain' if random.random() > 0.3 else 'adjust'} my position...",
                    'token_usage': tokens,
                    'estimated_cost': tokens['total_tokens'] * 0.00002
                }
                
                self.monitor.complete_request(request_id, response, duration)
        
        # Head Trader synthesis
        request_id = str(uuid.uuid4())
        prompt = "Head Trader: Synthesize all agent inputs and debates into final decision..."
        
        self.monitor.log_request(request_id, "Head Trader", prompt, datetime.now())
        
        duration = random.uniform(1.0, 3.0)
        tokens = {
            'prompt_tokens': random.randint(800, 1200),
            'completion_tokens': random.randint(200, 400),
            'total_tokens': 0
        }
        tokens['total_tokens'] = tokens['prompt_tokens'] + tokens['completion_tokens']
        
        response = {
            'content': "Final decision: BUY EURUSD with 78% confidence based on technical breakout and positive sentiment...",
            'token_usage': tokens,
            'estimated_cost': tokens['total_tokens'] * 0.00002
        }
        
        self.monitor.complete_request(request_id, response, duration)


def main():
    """Run the GPT Flow Dashboard"""
    dashboard = GPTFlowDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()