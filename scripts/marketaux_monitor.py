"""
MarketAux Integration Monitoring Dashboard
Real-time monitoring of API usage, cache performance, and sentiment impact
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from core.infrastructure.marketaux import MarketAuxClient

# Page config
st.set_page_config(
    page_title="MarketAux Monitor",
    page_icon="📰",
    layout="wide"
)

st.title("📰 MarketAux Integration Monitor")


@st.cache_resource
def get_database_connection():
    """Get database connection"""
    settings = get_settings()
    cache_db_path = settings.paths.data_dir / "marketaux_cache.db"
    trades_db_path = settings.database.db_path
    return str(cache_db_path), str(trades_db_path)


def load_api_usage_data(db_path: str, days: int = 7):
    """Load API usage data"""
    conn = sqlite3.connect(db_path)
    
    # API usage over time
    query = """
    SELECT 
        date(request_time) as date,
        COUNT(*) as total_requests,
        SUM(CASE WHEN response_code < 400 THEN 1 ELSE 0 END) as successful,
        SUM(CASE WHEN response_code >= 400 THEN 1 ELSE 0 END) as failed,
        SUM(articles_returned) as articles_fetched
    FROM marketaux_api_usage
    WHERE request_time >= datetime('now', '-{} days')
    GROUP BY date(request_time)
    ORDER BY date DESC
    """.format(days)
    
    df_usage = pd.read_sql_query(query, conn)
    
    # Hourly distribution
    query_hourly = """
    SELECT 
        strftime('%H', request_time) as hour,
        COUNT(*) as requests
    FROM marketaux_api_usage
    WHERE request_time >= datetime('now', '-{} days')
    AND response_code < 400
    GROUP BY hour
    ORDER BY hour
    """.format(days)
    
    df_hourly = pd.read_sql_query(query_hourly, conn)
    
    # Error analysis
    query_errors = """
    SELECT 
        error_message,
        COUNT(*) as count,
        MAX(request_time) as last_occurrence
    FROM marketaux_api_usage
    WHERE error_message IS NOT NULL
    AND request_time >= datetime('now', '-{} days')
    GROUP BY error_message
    ORDER BY count DESC
    LIMIT 10
    """.format(days)
    
    df_errors = pd.read_sql_query(query_errors, conn)
    
    conn.close()
    
    return df_usage, df_hourly, df_errors


def load_cache_data(db_path: str):
    """Load cache statistics"""
    conn = sqlite3.connect(db_path)
    
    # Cache statistics
    query = """
    SELECT 
        COUNT(*) as total_articles,
        SUM(CASE WHEN expires_at > datetime('now') THEN 1 ELSE 0 END) as valid_articles,
        SUM(CASE WHEN is_high_impact = 1 THEN 1 ELSE 0 END) as high_impact_articles,
        AVG(relevance_score) as avg_relevance
    FROM marketaux_articles
    """
    
    cache_stats = pd.read_sql_query(query, conn).iloc[0].to_dict()
    
    # Articles by source
    query_sources = """
    SELECT 
        source,
        COUNT(*) as count
    FROM marketaux_articles
    WHERE expires_at > datetime('now')
    GROUP BY source
    ORDER BY count DESC
    LIMIT 10
    """
    
    df_sources = pd.read_sql_query(query_sources, conn)
    
    # Sentiment distribution
    query_sentiment = """
    SELECT 
        json_extract(sentiment_data, '$.overall') as sentiment,
        COUNT(*) as count
    FROM marketaux_articles
    WHERE sentiment_data IS NOT NULL
    AND expires_at > datetime('now')
    GROUP BY sentiment
    """
    
    df_sentiment = pd.read_sql_query(query_sentiment, conn)
    
    conn.close()
    
    return cache_stats, df_sources, df_sentiment


def load_trading_impact(trades_db_path: str, days: int = 7):
    """Load trading impact data"""
    try:
        conn = sqlite3.connect(trades_db_path)
        
        # Trades with sentiment data
        query = """
        SELECT 
            t.symbol,
            t.side,
            t.result,
            tns.sentiment_score,
            tns.sentiment_label,
            tns.article_count
        FROM trades t
        LEFT JOIN trade_news_sentiment tns ON t.id = tns.trade_id
        WHERE t.timestamp >= datetime('now', '-{} days')
        """.format(days)
        
        df_trades = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df_trades.empty:
            # Calculate win rate by sentiment
            sentiment_impact = df_trades.groupby('sentiment_label').agg({
                'result': lambda x: (x == 'WIN').mean() * 100,
                'symbol': 'count'
            }).rename(columns={'result': 'win_rate', 'symbol': 'trade_count'})
            
            return df_trades, sentiment_impact
        else:
            return pd.DataFrame(), pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error loading trading impact: {e}")
        return pd.DataFrame(), pd.DataFrame()


def create_usage_chart(df_usage):
    """Create API usage chart"""
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Bar(
        x=df_usage['date'],
        y=df_usage['total_requests'],
        name='Total Requests',
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        x=df_usage['date'],
        y=df_usage['failed'],
        name='Failed Requests',
        marker_color='red'
    ))
    
    # Add daily limit line
    settings = get_settings()
    fig.add_hline(
        y=settings.marketaux.daily_limit,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Daily Limit ({settings.marketaux.daily_limit})"
    )
    
    fig.update_layout(
        title="API Usage History",
        xaxis_title="Date",
        yaxis_title="Requests",
        barmode='stack',
        height=400
    )
    
    return fig


def create_hourly_distribution(df_hourly):
    """Create hourly distribution chart"""
    # Fill missing hours with 0
    all_hours = pd.DataFrame({'hour': [f"{i:02d}" for i in range(24)]})
    df_hourly = all_hours.merge(df_hourly, on='hour', how='left').fillna(0)
    
    fig = px.bar(
        df_hourly,
        x='hour',
        y='requests',
        title="Hourly Request Distribution",
        labels={'requests': 'Number of Requests', 'hour': 'Hour (UTC)'}
    )
    
    fig.update_layout(height=300)
    return fig


def create_sentiment_pie(df_sentiment):
    """Create sentiment distribution pie chart"""
    if df_sentiment.empty:
        return go.Figure().add_annotation(text="No sentiment data available")
    
    # Define colors for sentiments
    colors = {
        'very_positive': '#1f77b4',
        'positive': '#7fcdbb',
        'neutral': '#fee08b',
        'negative': '#fdae61',
        'very_negative': '#d73027'
    }
    
    df_sentiment['color'] = df_sentiment['sentiment'].map(colors)
    
    fig = px.pie(
        df_sentiment,
        values='count',
        names='sentiment',
        title="Article Sentiment Distribution",
        color='sentiment',
        color_discrete_map=colors
    )
    
    fig.update_layout(height=350)
    return fig


def main():
    """Main dashboard"""
    settings = get_settings()
    cache_db_path, trades_db_path = get_database_connection()
    
    # Check if MarketAux is enabled
    if not settings.marketaux.enabled:
        st.warning("MarketAux integration is not enabled. Update MARKETAUX_ENABLED=true in .env")
        return
    
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        
        days_lookback = st.slider("Days to analyze", 1, 30, 7)
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
        
        if auto_refresh:
            st.experimental_rerun()
        
        st.divider()
        
        # Quick stats
        st.subheader("Configuration")
        st.text(f"Enabled: {settings.marketaux.enabled}")
        st.text(f"Daily Limit: {settings.marketaux.daily_limit}")
        st.text(f"Cache TTL: {settings.marketaux.cache_ttl_hours}h")
        st.text(f"Sentiment Weight: {settings.marketaux.sentiment_weight}")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Usage", "💾 Cache", "📈 Impact", "⚠️ Alerts"])
    
    with tab1:
        st.header("API Usage Analytics")
        
        # Load usage data
        df_usage, df_hourly, df_errors = load_api_usage_data(cache_db_path, days_lookback)
        
        # Today's usage
        if not df_usage.empty:
            today_usage = df_usage[df_usage['date'] == datetime.now().strftime('%Y-%m-%d')]
            if not today_usage.empty:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    requests_today = today_usage['total_requests'].iloc[0]
                    remaining = settings.marketaux.daily_limit - requests_today
                    st.metric(
                        "Today's Requests",
                        requests_today,
                        f"{remaining} remaining",
                        delta_color="inverse"
                    )
                
                with col2:
                    success_rate = (today_usage['successful'].iloc[0] / today_usage['total_requests'].iloc[0] * 100) if today_usage['total_requests'].iloc[0] > 0 else 0
                    st.metric(
                        "Success Rate",
                        f"{success_rate:.1f}%"
                    )
                
                with col3:
                    articles_today = today_usage['articles_fetched'].iloc[0] or 0
                    st.metric(
                        "Articles Fetched",
                        int(articles_today)
                    )
                
                with col4:
                    avg_articles = articles_today / requests_today if requests_today > 0 else 0
                    st.metric(
                        "Avg Articles/Request",
                        f"{avg_articles:.1f}"
                    )
        
        # Usage chart
        st.plotly_chart(create_usage_chart(df_usage), use_container_width=True)
        
        # Hourly distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_hourly_distribution(df_hourly), use_container_width=True)
        
        with col2:
            st.subheader("Recent Errors")
            if not df_errors.empty:
                st.dataframe(
                    df_errors[['error_message', 'count', 'last_occurrence']],
                    use_container_width=True
                )
            else:
                st.success("No errors in the selected period!")
    
    with tab2:
        st.header("Cache Performance")
        
        # Load cache data
        cache_stats, df_sources, df_sentiment = load_cache_data(cache_db_path)
        
        # Cache metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Articles",
                cache_stats.get('total_articles', 0)
            )
        
        with col2:
            valid_articles = cache_stats.get('valid_articles', 0)
            cache_hit_rate = (valid_articles / cache_stats.get('total_articles', 1)) * 100
            st.metric(
                "Valid Articles",
                valid_articles,
                f"{cache_hit_rate:.1f}% hit rate"
            )
        
        with col3:
            st.metric(
                "High Impact",
                cache_stats.get('high_impact_articles', 0)
            )
        
        with col4:
            avg_relevance = cache_stats.get('avg_relevance', 0)
            st.metric(
                "Avg Relevance",
                f"{avg_relevance:.3f}"
            )
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top News Sources")
            if not df_sources.empty:
                fig_sources = px.bar(
                    df_sources.head(10),
                    x='count',
                    y='source',
                    orientation='h',
                    title="Articles by Source"
                )
                fig_sources.update_layout(height=400)
                st.plotly_chart(fig_sources, use_container_width=True)
        
        with col2:
            st.plotly_chart(create_sentiment_pie(df_sentiment), use_container_width=True)
    
    with tab3:
        st.header("Trading Impact Analysis")
        
        # Load trading impact
        df_trades, sentiment_impact = load_trading_impact(trades_db_path, days_lookback)
        
        if not df_trades.empty and not sentiment_impact.empty:
            # Sentiment impact on win rate
            st.subheader("Win Rate by Market Sentiment")
            
            fig_impact = px.bar(
                sentiment_impact.reset_index(),
                x='sentiment_label',
                y='win_rate',
                text='trade_count',
                title="Trading Performance by Sentiment",
                labels={'win_rate': 'Win Rate (%)', 'sentiment_label': 'Market Sentiment', 'trade_count': 'Trades'}
            )
            
            fig_impact.update_traces(texttemplate='%{text} trades', textposition='outside')
            fig_impact.update_layout(height=400)
            st.plotly_chart(fig_impact, use_container_width=True)
            
            # Detailed stats
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sentiment vs Direction")
                sentiment_direction = pd.crosstab(
                    df_trades['sentiment_label'],
                    df_trades['side'],
                    normalize='index'
                ) * 100
                st.dataframe(sentiment_direction.round(1))
            
            with col2:
                st.subheader("Symbol Performance")
                symbol_sentiment = df_trades.groupby(['symbol', 'sentiment_label']).size().unstack(fill_value=0)
                st.dataframe(symbol_sentiment)
        else:
            st.info("No trading data with sentiment analysis found in the selected period")
    
    with tab4:
        st.header("System Alerts")
        
        alerts = []
        
        # Check daily limit
        if not df_usage.empty:
            today_usage = df_usage[df_usage['date'] == datetime.now().strftime('%Y-%m-%d')]
            if not today_usage.empty:
                usage_pct = (today_usage['total_requests'].iloc[0] / settings.marketaux.daily_limit) * 100
                if usage_pct > 90:
                    alerts.append(("🔴 Critical", f"Daily API limit nearly exhausted ({usage_pct:.0f}%)"))
                elif usage_pct > 70:
                    alerts.append(("🟡 Warning", f"High API usage today ({usage_pct:.0f}%)"))
        
        # Check error rate
        if not df_usage.empty:
            recent_usage = df_usage.head(3)
            total_recent = recent_usage['total_requests'].sum()
            failed_recent = recent_usage['failed'].sum()
            if total_recent > 0:
                error_rate = (failed_recent / total_recent) * 100
                if error_rate > 10:
                    alerts.append(("🔴 Critical", f"High error rate: {error_rate:.1f}%"))
        
        # Check cache effectiveness
        if cache_stats.get('total_articles', 0) > 0:
            cache_effectiveness = (cache_stats.get('valid_articles', 0) / cache_stats['total_articles']) * 100
            if cache_effectiveness < 50:
                alerts.append(("🟡 Warning", f"Low cache hit rate: {cache_effectiveness:.1f}%"))
        
        # Display alerts
        if alerts:
            for level, message in alerts:
                if "Critical" in level:
                    st.error(f"{level}: {message}")
                else:
                    st.warning(f"{level}: {message}")
        else:
            st.success("✅ All systems operational")
        
        # Recommendations
        st.subheader("Optimization Recommendations")
        
        recommendations = []
        
        # Based on usage patterns
        if not df_hourly.empty:
            peak_hour = df_hourly.loc[df_hourly['requests'].idxmax(), 'hour']
            recommendations.append(f"• Peak usage at {peak_hour}:00 UTC - consider spreading requests")
        
        # Based on cache
        if cache_stats.get('avg_relevance', 0) < settings.marketaux.min_relevance_score:
            recommendations.append(f"• Average relevance ({cache_stats['avg_relevance']:.3f}) below threshold - consider adjusting filters")
        
        # Based on errors
        if not df_errors.empty and df_errors['count'].sum() > 10:
            recommendations.append("• Multiple API errors detected - check API token and rate limits")
        
        for rec in recommendations:
            st.text(rec)


if __name__ == "__main__":
    main()