#!/usr/bin/env python3
"""
ML Performance Monitor - Terminal and HTML-based monitoring for ML models
Provides comprehensive visualizations without streamlit dependencies
"""

import argparse
import logging
import sqlite3
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import joblib
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLPerformanceMonitor:
    """Monitor and visualize ML model performance"""
    
    def __init__(self, db_path: str = "data/trades.db", models_dir: str = "models"):
        self.db_path = Path(db_path)
        self.models_dir = Path(models_dir)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        
    def close(self):
        """Close database connection"""
        self.conn.close()
        
    def get_model_metadata(self) -> pd.DataFrame:
        """Fetch all model metadata from database"""
        query = """
        SELECT 
            model_id,
            model_type,
            version,
            created_at,
            training_metrics,
            hyperparameters,
            is_active,
            model_path
        FROM model_metadata
        ORDER BY created_at DESC
        """
        
        df = pd.read_sql_query(query, self.conn)
        
        # Parse JSON fields
        df['training_metrics'] = df['training_metrics'].apply(lambda x: json.loads(x) if x else {})
        df['hyperparameters'] = df['hyperparameters'].apply(lambda x: json.loads(x) if x else {})
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        return df
    
    def get_ml_predictions(self, days: int = 30) -> pd.DataFrame:
        """Fetch ML predictions from the last N days"""
        query = """
        SELECT 
            id,
            created_at,
            symbol,
            predicted_signal,
            ml_confidence,
            actual_signal,
            was_correct,
            model_version,
            features_used
        FROM ml_predictions
        WHERE created_at >= datetime('now', ?)
        ORDER BY created_at DESC
        """
        
        try:
            df = pd.read_sql_query(query, self.conn, params=(f'-{days} days',))
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['features_used'] = df['features_used'].apply(lambda x: json.loads(x) if x else {})
            return df
        except sqlite3.OperationalError:
            logger.warning("ml_predictions table not found. Using sample data.")
            return self._generate_sample_predictions(days)
    
    def _generate_sample_predictions(self, days: int) -> pd.DataFrame:
        """Generate sample predictions for demonstration"""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=days*24, freq='H')
        
        data = []
        for date in dates:
            for symbol in ['EURUSD', 'GBPUSD', 'USDJPY']:
                predicted = np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.3, 0.3, 0.4])
                confidence = np.random.uniform(0.5, 0.95)
                actual = predicted if np.random.random() > 0.3 else np.random.choice(['BUY', 'SELL', 'HOLD'])
                
                data.append({
                    'id': len(data),
                    'created_at': date,
                    'symbol': symbol,
                    'predicted_signal': predicted,
                    'ml_confidence': confidence,
                    'actual_signal': actual,
                    'was_correct': predicted == actual,
                    'model_version': '1.0.0',
                    'features_used': {
                        'rsi': np.random.uniform(20, 80),
                        'ma_cross': np.random.choice([0, 1]),
                        'volume_ratio': np.random.uniform(0.5, 2.0)
                    }
                })
        
        return pd.DataFrame(data)
    
    def print_summary_statistics(self, predictions_df: pd.DataFrame, models_df: pd.DataFrame):
        """Print summary statistics to terminal"""
        print("\n" + "="*80)
        print("ML PERFORMANCE MONITOR - SUMMARY STATISTICS")
        print("="*80)
        
        # Active models
        print("\n📊 ACTIVE MODELS:")
        active_models = models_df[models_df['is_active'] == 1]
        if not active_models.empty:
            for _, model in active_models.iterrows():
                metrics = model['training_metrics']
                print(f"\n  Model: {model['model_type']} (v{model['version']})")
                print(f"  Created: {model['created_at'].strftime('%Y-%m-%d %H:%M')}")
                print(f"  Accuracy: {metrics.get('accuracy', 0):.3f}")
                print(f"  Precision: {metrics.get('precision', 0):.3f}")
                print(f"  F1 Score: {metrics.get('f1_score', 0):.3f}")
        else:
            print("  No active models found")
        
        # Prediction statistics
        if not predictions_df.empty:
            print("\n📈 PREDICTION PERFORMANCE (Last 30 days):")
            
            # Overall accuracy
            overall_accuracy = predictions_df['was_correct'].mean()
            print(f"\n  Overall Accuracy: {overall_accuracy:.1%}")
            
            # By symbol
            print("\n  Accuracy by Symbol:")
            symbol_accuracy = predictions_df.groupby('symbol')['was_correct'].agg(['mean', 'count'])
            for symbol, row in symbol_accuracy.iterrows():
                print(f"    {symbol}: {row['mean']:.1%} ({row['count']} predictions)")
            
            # By signal type
            print("\n  Accuracy by Signal Type:")
            signal_accuracy = predictions_df.groupby('predicted_signal')['was_correct'].agg(['mean', 'count'])
            for signal, row in signal_accuracy.iterrows():
                print(f"    {signal}: {row['mean']:.1%} ({row['count']} predictions)")
            
            # Confidence analysis
            print("\n  Confidence Analysis:")
            high_conf = predictions_df[predictions_df['ml_confidence'] >= 0.8]
            med_conf = predictions_df[(predictions_df['ml_confidence'] >= 0.6) & (predictions_df['ml_confidence'] < 0.8)]
            low_conf = predictions_df[predictions_df['ml_confidence'] < 0.6]
            
            print(f"    High Confidence (≥80%): {high_conf['was_correct'].mean():.1%} accuracy ({len(high_conf)} predictions)")
            print(f"    Medium Confidence (60-80%): {med_conf['was_correct'].mean():.1%} accuracy ({len(med_conf)} predictions)")
            print(f"    Low Confidence (<60%): {low_conf['was_correct'].mean():.1%} accuracy ({len(low_conf)} predictions)")
        
        print("\n" + "="*80)
    
    def create_performance_plots(self, predictions_df: pd.DataFrame, models_df: pd.DataFrame) -> Dict[str, go.Figure]:
        """Create interactive Plotly visualizations"""
        figures = {}
        
        # 1. Model Performance Timeline
        fig_timeline = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Daily Accuracy Trend', 'Prediction Volume'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        if not predictions_df.empty:
            daily_stats = predictions_df.groupby([pd.Grouper(key='created_at', freq='D'), 'symbol']).agg({
                'was_correct': ['mean', 'count']
            }).reset_index()
            daily_stats.columns = ['date', 'symbol', 'accuracy', 'count']
            
            for symbol in daily_stats['symbol'].unique():
                symbol_data = daily_stats[daily_stats['symbol'] == symbol]
                
                # Accuracy trend
                fig_timeline.add_trace(
                    go.Scatter(
                        x=symbol_data['date'],
                        y=symbol_data['accuracy'] * 100,
                        name=f'{symbol} Accuracy',
                        mode='lines+markers',
                        line=dict(width=2)
                    ),
                    row=1, col=1
                )
                
                # Prediction count
                fig_timeline.add_trace(
                    go.Bar(
                        x=symbol_data['date'],
                        y=symbol_data['count'],
                        name=f'{symbol} Count',
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        fig_timeline.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
        fig_timeline.update_yaxes(title_text="# Predictions", row=2, col=1)
        fig_timeline.update_xaxes(title_text="Date", row=2, col=1)
        fig_timeline.update_layout(height=700, title="Model Performance Over Time")
        figures['timeline'] = fig_timeline
        
        # 2. Accuracy by Confidence Level
        if not predictions_df.empty:
            # Create confidence bins
            predictions_df['confidence_bin'] = pd.cut(
                predictions_df['ml_confidence'],
                bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                labels=['<50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
            )
            
            conf_accuracy = predictions_df.groupby(['confidence_bin', 'symbol']).agg({
                'was_correct': ['mean', 'count']
            }).reset_index()
            conf_accuracy.columns = ['confidence_bin', 'symbol', 'accuracy', 'count']
            
            fig_confidence = px.bar(
                conf_accuracy,
                x='confidence_bin',
                y='accuracy',
                color='symbol',
                title='Accuracy by Confidence Level',
                labels={'accuracy': 'Accuracy', 'confidence_bin': 'Confidence Level'},
                barmode='group'
            )
            fig_confidence.update_yaxis(tickformat='.0%')
            figures['confidence'] = fig_confidence
        
        # 3. Confusion Matrix Heatmap
        if not predictions_df.empty:
            # Create confusion matrix
            signals = ['BUY', 'SELL', 'HOLD']
            confusion_data = []
            
            for actual in signals:
                row = []
                for predicted in signals:
                    count = len(predictions_df[
                        (predictions_df['actual_signal'] == actual) & 
                        (predictions_df['predicted_signal'] == predicted)
                    ])
                    row.append(count)
                confusion_data.append(row)
            
            fig_confusion = go.Figure(data=go.Heatmap(
                z=confusion_data,
                x=signals,
                y=signals,
                text=confusion_data,
                texttemplate="%{text}",
                colorscale='Blues'
            ))
            
            fig_confusion.update_layout(
                title='Prediction Confusion Matrix',
                xaxis_title='Predicted Signal',
                yaxis_title='Actual Signal',
                height=500
            )
            figures['confusion'] = fig_confusion
        
        # 4. Feature Importance (if available)
        if not models_df.empty and not predictions_df.empty:
            # Extract feature importance from the most recent model
            latest_model = models_df.iloc[0]
            
            # For demonstration, use average feature values as proxy for importance
            feature_stats = {}
            for _, pred in predictions_df.iterrows():
                features = pred['features_used']
                for feat, val in features.items():
                    if feat not in feature_stats:
                        feature_stats[feat] = []
                    feature_stats[feat].append(val)
            
            if feature_stats:
                feature_importance = pd.DataFrame([
                    {'feature': feat, 'importance': np.std(vals)}
                    for feat, vals in feature_stats.items()
                ]).sort_values('importance', ascending=True)
                
                fig_features = go.Figure(go.Bar(
                    x=feature_importance['importance'],
                    y=feature_importance['feature'],
                    orientation='h'
                ))
                
                fig_features.update_layout(
                    title='Feature Importance (Std Dev as Proxy)',
                    xaxis_title='Importance',
                    yaxis_title='Feature',
                    height=400
                )
                figures['features'] = fig_features
        
        # 5. Model Comparison
        if len(models_df) > 1:
            model_comparison = []
            for _, model in models_df.iterrows():
                metrics = model['training_metrics']
                model_comparison.append({
                    'model': f"{model['model_type']} v{model['version']}",
                    'accuracy': metrics.get('accuracy', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1_score': metrics.get('f1_score', 0)
                })
            
            comp_df = pd.DataFrame(model_comparison)
            
            fig_comparison = go.Figure()
            
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
            for metric in metrics_to_plot:
                fig_comparison.add_trace(go.Bar(
                    name=metric.replace('_', ' ').title(),
                    x=comp_df['model'],
                    y=comp_df[metric]
                ))
            
            fig_comparison.update_layout(
                title='Model Performance Comparison',
                xaxis_title='Model',
                yaxis_title='Score',
                barmode='group',
                height=500
            )
            figures['comparison'] = fig_comparison
        
        return figures
    
    def generate_html_report(self, figures: Dict[str, go.Figure], predictions_df: pd.DataFrame, models_df: pd.DataFrame):
        """Generate comprehensive HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML Performance Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1, h2 {{
                    color: #333;
                }}
                .metric-box {{
                    display: inline-block;
                    padding: 20px;
                    margin: 10px;
                    background-color: #f8f9fa;
                    border-radius: 5px;
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #007bff;
                }}
                .metric-label {{
                    color: #666;
                    margin-top: 5px;
                }}
                .plot-container {{
                    margin: 30px 0;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 10px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f8f9fa;
                    font-weight: bold;
                }}
                .timestamp {{
                    color: #666;
                    font-style: italic;
                    margin-top: 30px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 ML Performance Monitoring Report</h1>
                <p class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
                
                <h2>📊 Key Metrics</h2>
                <div class="metrics-container">
        """
        
        # Add key metrics
        if not predictions_df.empty:
            overall_accuracy = predictions_df['was_correct'].mean()
            total_predictions = len(predictions_df)
            avg_confidence = predictions_df['ml_confidence'].mean()
            
            html_content += f"""
                    <div class="metric-box">
                        <div class="metric-value">{overall_accuracy:.1%}</div>
                        <div class="metric-label">Overall Accuracy</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{total_predictions:,}</div>
                        <div class="metric-label">Total Predictions</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{avg_confidence:.1%}</div>
                        <div class="metric-label">Avg Confidence</div>
                    </div>
            """
        
        html_content += """
                </div>
                
                <h2>📈 Performance Visualizations</h2>
        """
        
        # Add plots
        for name, fig in figures.items():
            html_content += f"""
                <div class="plot-container">
                    <div id="plot-{name}"></div>
                </div>
            """
        
        # Add model summary table
        if not models_df.empty:
            html_content += """
                <h2>🎯 Model Summary</h2>
                <table>
                    <tr>
                        <th>Model Type</th>
                        <th>Version</th>
                        <th>Created</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>F1 Score</th>
                        <th>Status</th>
                    </tr>
            """
            
            for _, model in models_df.head(10).iterrows():
                metrics = model['training_metrics']
                status = "🟢 Active" if model['is_active'] else "⚪ Inactive"
                html_content += f"""
                    <tr>
                        <td>{model['model_type']}</td>
                        <td>{model['version']}</td>
                        <td>{model['created_at'].strftime('%Y-%m-%d %H:%M')}</td>
                        <td>{metrics.get('accuracy', 0):.3f}</td>
                        <td>{metrics.get('precision', 0):.3f}</td>
                        <td>{metrics.get('f1_score', 0):.3f}</td>
                        <td>{status}</td>
                    </tr>
                """
            
            html_content += "</table>"
        
        # Add recent predictions sample
        if not predictions_df.empty:
            html_content += """
                <h2>📋 Recent Predictions Sample</h2>
                <table>
                    <tr>
                        <th>Time</th>
                        <th>Symbol</th>
                        <th>Predicted</th>
                        <th>Actual</th>
                        <th>Confidence</th>
                        <th>Result</th>
                    </tr>
            """
            
            for _, pred in predictions_df.head(20).iterrows():
                result = "✅" if pred['was_correct'] else "❌"
                html_content += f"""
                    <tr>
                        <td>{pred['created_at'].strftime('%Y-%m-%d %H:%M')}</td>
                        <td>{pred['symbol']}</td>
                        <td>{pred['predicted_signal']}</td>
                        <td>{pred['actual_signal']}</td>
                        <td>{pred['ml_confidence']:.1%}</td>
                        <td>{result}</td>
                    </tr>
                """
            
            html_content += "</table>"
        
        # Add JavaScript for plots
        html_content += """
            </div>
            <script>
        """
        
        for name, fig in figures.items():
            html_content += f"""
                Plotly.newPlot('plot-{name}', {fig.to_json()});
            """
        
        html_content += """
            </script>
        </body>
        </html>
        """
        
        # Save report
        report_path = Path("reports/ml_performance_report.html")
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to: {report_path}")
        return report_path
    
    def generate_matplotlib_plots(self, predictions_df: pd.DataFrame, models_df: pd.DataFrame):
        """Generate static matplotlib plots for terminal viewing"""
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ML Performance Overview', fontsize=16)
        
        # 1. Accuracy over time
        if not predictions_df.empty:
            daily_accuracy = predictions_df.groupby(pd.Grouper(key='created_at', freq='D'))['was_correct'].mean()
            
            ax = axes[0, 0]
            daily_accuracy.plot(ax=ax, marker='o', linestyle='-', color='blue')
            ax.set_title('Daily Accuracy Trend')
            ax.set_xlabel('Date')
            ax.set_ylabel('Accuracy')
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
            ax.grid(True, alpha=0.3)
        
        # 2. Accuracy by symbol
        if not predictions_df.empty:
            symbol_accuracy = predictions_df.groupby('symbol')['was_correct'].mean().sort_values(ascending=True)
            
            ax = axes[0, 1]
            symbol_accuracy.plot(kind='barh', ax=ax, color='green')
            ax.set_title('Accuracy by Symbol')
            ax.set_xlabel('Accuracy')
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
        
        # 3. Confidence distribution
        if not predictions_df.empty:
            ax = axes[1, 0]
            predictions_df['ml_confidence'].hist(bins=20, ax=ax, color='orange', edgecolor='black')
            ax.set_title('Confidence Distribution')
            ax.set_xlabel('Confidence Level')
            ax.set_ylabel('Frequency')
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
        
        # 4. Model metrics comparison
        if not models_df.empty:
            ax = axes[1, 1]
            
            # Get latest models by type
            latest_models = models_df.groupby('model_type').first().head(5)
            
            metrics_data = []
            for idx, model in latest_models.iterrows():
                metrics = model['training_metrics']
                metrics_data.append({
                    'model': idx,
                    'accuracy': metrics.get('accuracy', 0),
                    'precision': metrics.get('precision', 0),
                    'f1_score': metrics.get('f1_score', 0)
                })
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                metrics_df.set_index('model')[['accuracy', 'precision', 'f1_score']].plot(
                    kind='bar', ax=ax, rot=45
                )
                ax.set_title('Model Performance Metrics')
                ax.set_ylabel('Score')
                ax.legend(loc='best')
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path("reports/ml_performance_plots.png")
        plot_path.parent.mkdir(exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Matplotlib plots saved to: {plot_path}")
        
        return plot_path


def main():
    parser = argparse.ArgumentParser(description='ML Performance Monitor')
    parser.add_argument('--db-path', type=str, default='data/trades.db',
                        help='Path to the database')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory containing ML models')
    parser.add_argument('--days', type=int, default=30,
                        help='Number of days to analyze')
    parser.add_argument('--html', action='store_true',
                        help='Generate HTML report')
    parser.add_argument('--plot', action='store_true',
                        help='Generate matplotlib plots')
    parser.add_argument('--terminal', action='store_true',
                        help='Show summary in terminal (default)')
    
    args = parser.parse_args()
    
    # Default to terminal if no output specified
    if not args.html and not args.plot:
        args.terminal = True
    
    try:
        monitor = MLPerformanceMonitor(args.db_path, args.models_dir)
        
        # Fetch data
        logger.info("Fetching model metadata...")
        models_df = monitor.get_model_metadata()
        
        logger.info(f"Fetching predictions from last {args.days} days...")
        predictions_df = monitor.get_ml_predictions(args.days)
        
        # Terminal output
        if args.terminal:
            monitor.print_summary_statistics(predictions_df, models_df)
        
        # Generate matplotlib plots
        if args.plot:
            logger.info("Generating matplotlib plots...")
            plot_path = monitor.generate_matplotlib_plots(predictions_df, models_df)
            print(f"\n📊 Plots saved to: {plot_path}")
        
        # Generate HTML report
        if args.html:
            logger.info("Creating interactive visualizations...")
            figures = monitor.create_performance_plots(predictions_df, models_df)
            
            logger.info("Generating HTML report...")
            report_path = monitor.generate_html_report(figures, predictions_df, models_df)
            print(f"\n📄 HTML report saved to: {report_path}")
            print(f"   Open in browser: file:///{report_path.absolute()}")
        
        monitor.close()
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())