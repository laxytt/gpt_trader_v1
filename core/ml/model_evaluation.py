# core/ml/model_evaluation.py
"""
Model evaluation utilities for trading strategies.
Provides comprehensive metrics for model performance assessment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class ModelEvaluator:
    """
    Comprehensive model evaluation for trading strategies.
    Combines ML metrics with trading-specific performance measures.
    """
    
    def __init__(self, evaluation_config: Optional[Dict] = None):
        self.config = evaluation_config or self._get_default_config()
        self.evaluation_history = []
    
    def _get_default_config(self) -> Dict:
        """Get default evaluation configuration"""
        return {
            'min_trades': 30,  # Minimum trades for valid evaluation
            'confidence_level': 0.95,  # For confidence intervals
            'risk_free_rate': 0.02,  # Annual risk-free rate
            'trading_days_per_year': 252,
            'cost_per_trade': 0.0002,  # 2 bps per trade
            'slippage': 0.0001,  # 1 bp slippage
        }
    
    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        backtest_results: Optional['BacktestResults'] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation combining ML and trading metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            backtest_results: Backtest results for trading metrics
            
        Returns:
            Dictionary with evaluation metrics
        """
        evaluation = {
            'ml_metrics': self._calculate_ml_metrics(y_true, y_pred, y_prob),
            'trading_metrics': self._calculate_trading_metrics(backtest_results) if backtest_results else {},
            'statistical_tests': self._perform_statistical_tests(y_true, y_pred),
            'timestamp': datetime.now().isoformat()
        }
        
        # Calculate combined score
        evaluation['combined_score'] = self._calculate_combined_score(evaluation)
        
        # Store in history
        self.evaluation_history.append(evaluation)
        
        return evaluation
    
    def _calculate_ml_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate machine learning metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        # Add probabilistic metrics if available
        if y_prob is not None:
            if len(np.unique(y_true)) == 2:  # Binary classification
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])
                
                # Calculate profit-based threshold
                optimal_threshold = self._find_optimal_threshold(y_true, y_prob[:, 1])
                metrics['optimal_threshold'] = optimal_threshold
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Class-specific metrics
        class_report = classification_report(y_true, y_pred, output_dict=True)
        metrics['class_metrics'] = class_report
        
        return metrics
    
    def _calculate_trading_metrics(self, backtest_results: 'BacktestResults') -> Dict[str, float]:
        """Calculate trading-specific metrics"""
        if not backtest_results or backtest_results.total_trades < self.config['min_trades']:
            return {'error': 'Insufficient trades for evaluation'}
        
        metrics = {
            # Basic performance
            'total_return': backtest_results.total_return,
            'annualized_return': self._annualize_return(
                backtest_results.total_return,
                backtest_results.config.start_date,
                backtest_results.config.end_date
            ),
            'win_rate': backtest_results.win_rate,
            'profit_factor': backtest_results.profit_factor,
            
            # Risk metrics
            'sharpe_ratio': backtest_results.sharpe_ratio,
            'sortino_ratio': backtest_results.sortino_ratio,
            'calmar_ratio': backtest_results.calmar_ratio,
            'max_drawdown': backtest_results.max_drawdown,
            
            # Trade statistics
            'avg_win': backtest_results.average_win,
            'avg_loss': backtest_results.average_loss,
            'avg_trade_duration': backtest_results.average_bars_held,
            'expectancy': backtest_results.expectancy,
            
            # Risk-adjusted metrics
            'risk_adjusted_return': self._calculate_risk_adjusted_return(backtest_results),
            'information_ratio': self._calculate_information_ratio(backtest_results),
            'omega_ratio': self._calculate_omega_ratio(backtest_results),
        }
        
        # Add confidence intervals
        metrics['confidence_intervals'] = self._calculate_confidence_intervals(backtest_results)
        
        return metrics
    
    def _perform_statistical_tests(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Perform statistical significance tests"""
        from scipy import stats
        
        tests = {}
        
        # McNemar's test for paired predictions
        if len(y_true) > 100:  # Need sufficient samples
            # Create contingency table
            correct_baseline = np.zeros_like(y_true)  # Assume baseline always predicts 0
            
            # Count disagreements
            n01 = np.sum((correct_baseline == y_true) & (y_pred != y_true))
            n10 = np.sum((correct_baseline != y_true) & (y_pred == y_true))
            
            # Perform McNemar's test
            if n01 + n10 > 0:
                statistic = (abs(n01 - n10) - 1)**2 / (n01 + n10)
                p_value = 1 - stats.chi2.cdf(statistic, df=1)
                
                tests['mcnemar'] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        # Binomial test for win rate
        if hasattr(self, 'backtest_results') and self.backtest_results:
            wins = self.backtest_results.winning_trades
            total = self.backtest_results.total_trades
            
            if total > 0:
                # Test if win rate is significantly different from 50%
                p_value = stats.binom_test(wins, total, p=0.5)
                
                tests['binomial_win_rate'] = {
                    'win_rate': wins / total,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return tests
    
    def _find_optimal_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Find optimal probability threshold for trading"""
        thresholds = np.linspace(0.3, 0.7, 41)
        best_profit = -np.inf
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred_thresh = (y_prob >= threshold).astype(int)
            
            # Simulate simple profit calculation
            # True positives gain, false positives lose
            tp = np.sum((y_pred_thresh == 1) & (y_true == 1))
            fp = np.sum((y_pred_thresh == 1) & (y_true == 0))
            
            # Assume 2:1 risk-reward ratio
            profit = tp * 2 - fp * 1
            
            if profit > best_profit:
                best_profit = profit
                best_threshold = threshold
        
        return best_threshold
    
    def _annualize_return(self, total_return: float, start_date: datetime, end_date: datetime) -> float:
        """Annualize returns"""
        days = (end_date - start_date).days
        if days <= 0:
            return 0.0
        
        years = days / 365.25
        annualized = (1 + total_return / 100) ** (1 / years) - 1
        return annualized * 100
    
    def _calculate_risk_adjusted_return(self, results: 'BacktestResults') -> float:
        """Calculate risk-adjusted return (return per unit of risk)"""
        if results.max_drawdown == 0:
            return 0.0
        
        return results.total_return / results.max_drawdown
    
    def _calculate_information_ratio(self, results: 'BacktestResults') -> float:
        """Calculate information ratio"""
        if not results.daily_returns:
            return 0.0
        
        # Assume benchmark return is 0 (cash)
        excess_returns = np.array(results.daily_returns)
        
        if len(excess_returns) < 2:
            return 0.0
        
        tracking_error = np.std(excess_returns)
        
        if tracking_error == 0:
            return 0.0
        
        return np.sqrt(252) * np.mean(excess_returns) / tracking_error
    
    def _calculate_omega_ratio(self, results: 'BacktestResults', threshold: float = 0.0) -> float:
        """Calculate Omega ratio"""
        if not results.daily_returns:
            return 0.0
        
        returns = np.array(results.daily_returns)
        
        # Calculate probability-weighted gains and losses
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        if len(losses) == 0 or np.sum(losses) == 0:
            return np.inf
        
        return np.sum(gains) / np.sum(losses)
    
    def _calculate_confidence_intervals(self, results: 'BacktestResults') -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for key metrics"""
        if results.total_trades < self.config['min_trades']:
            return {}
        
        from scipy import stats
        
        confidence_level = self.config['confidence_level']
        
        # Win rate confidence interval (Wilson score interval)
        wins = results.winning_trades
        n = results.total_trades
        
        if n > 0:
            z = stats.norm.ppf((1 + confidence_level) / 2)
            p_hat = wins / n
            
            denominator = 1 + z**2 / n
            centre = (p_hat + z**2 / (2 * n)) / denominator
            margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denominator
            
            win_rate_ci = (max(0, centre - margin), min(1, centre + margin))
        else:
            win_rate_ci = (0, 1)
        
        # Sharpe ratio confidence interval (bootstrap would be better)
        if results.daily_returns and len(results.daily_returns) > 30:
            returns = np.array(results.daily_returns)
            se_sharpe = np.sqrt((1 + 0.5 * results.sharpe_ratio**2) / len(returns))
            sharpe_ci = (
                results.sharpe_ratio - 1.96 * se_sharpe,
                results.sharpe_ratio + 1.96 * se_sharpe
            )
        else:
            sharpe_ci = (results.sharpe_ratio, results.sharpe_ratio)
        
        return {
            'win_rate': win_rate_ci,
            'sharpe_ratio': sharpe_ci
        }
    
    def _calculate_combined_score(self, evaluation: Dict[str, Any]) -> float:
        """Calculate combined score from ML and trading metrics"""
        score = 0.0
        weights = {
            'ml_accuracy': 0.1,
            'ml_precision': 0.1,
            'sharpe_ratio': 0.3,
            'win_rate': 0.2,
            'profit_factor': 0.2,
            'max_drawdown': 0.1
        }
        
        # ML metrics contribution
        if 'ml_metrics' in evaluation:
            ml = evaluation['ml_metrics']
            score += weights['ml_accuracy'] * ml.get('accuracy', 0)
            score += weights['ml_precision'] * ml.get('precision', 0)
        
        # Trading metrics contribution
        if 'trading_metrics' in evaluation:
            tm = evaluation['trading_metrics']
            
            # Normalize metrics to 0-1 range
            sharpe_normalized = np.clip(tm.get('sharpe_ratio', 0) / 3, 0, 1)
            win_rate = tm.get('win_rate', 0.5)
            profit_factor_normalized = np.clip(tm.get('profit_factor', 1) / 3, 0, 1)
            drawdown_normalized = 1 - np.clip(tm.get('max_drawdown', 0) / 50, 0, 1)
            
            score += weights['sharpe_ratio'] * sharpe_normalized
            score += weights['win_rate'] * win_rate
            score += weights['profit_factor'] * profit_factor_normalized
            score += weights['max_drawdown'] * drawdown_normalized
        
        return score
    
    def generate_evaluation_report(
        self,
        evaluation: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """Generate comprehensive evaluation report"""
        from datetime import datetime
        
        report = []
        report.append("=" * 80)
        report.append("MODEL EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now()}")
        report.append("")
        
        # ML Metrics
        if 'ml_metrics' in evaluation:
            report.append("Machine Learning Metrics:")
            report.append("-" * 40)
            ml = evaluation['ml_metrics']
            report.append(f"Accuracy: {ml.get('accuracy', 0):.3f}")
            report.append(f"Precision: {ml.get('precision', 0):.3f}")
            report.append(f"Recall: {ml.get('recall', 0):.3f}")
            report.append(f"F1 Score: {ml.get('f1_score', 0):.3f}")
            if 'auc_roc' in ml:
                report.append(f"AUC-ROC: {ml['auc_roc']:.3f}")
            report.append("")
        
        # Trading Metrics
        if 'trading_metrics' in evaluation:
            report.append("Trading Performance Metrics:")
            report.append("-" * 40)
            tm = evaluation['trading_metrics']
            report.append(f"Total Return: {tm.get('total_return', 0):.2f}%")
            report.append(f"Annualized Return: {tm.get('annualized_return', 0):.2f}%")
            report.append(f"Win Rate: {tm.get('win_rate', 0):.1%}")
            report.append(f"Sharpe Ratio: {tm.get('sharpe_ratio', 0):.2f}")
            report.append(f"Max Drawdown: {tm.get('max_drawdown', 0):.2f}%")
            report.append(f"Profit Factor: {tm.get('profit_factor', 0):.2f}")
            report.append("")
            
            # Confidence Intervals
            if 'confidence_intervals' in tm:
                report.append("Confidence Intervals (95%):")
                ci = tm['confidence_intervals']
                if 'win_rate' in ci:
                    report.append(f"Win Rate: [{ci['win_rate'][0]:.1%}, {ci['win_rate'][1]:.1%}]")
                if 'sharpe_ratio' in ci:
                    report.append(f"Sharpe Ratio: [{ci['sharpe_ratio'][0]:.2f}, {ci['sharpe_ratio'][1]:.2f}]")
                report.append("")
        
        # Statistical Tests
        if 'statistical_tests' in evaluation:
            report.append("Statistical Significance Tests:")
            report.append("-" * 40)
            tests = evaluation['statistical_tests']
            
            if 'mcnemar' in tests:
                mcn = tests['mcnemar']
                report.append(f"McNemar's Test: p-value = {mcn['p_value']:.4f} "
                            f"({'Significant' if mcn['significant'] else 'Not significant'})")
            
            if 'binomial_win_rate' in tests:
                bwr = tests['binomial_win_rate']
                report.append(f"Binomial Test (Win Rate): p-value = {bwr['p_value']:.4f} "
                            f"({'Significant' if bwr['significant'] else 'Not significant'})")
            report.append("")
        
        # Combined Score
        report.append(f"Combined Score: {evaluation.get('combined_score', 0):.3f}")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # Save if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def plot_evaluation_results(
        self,
        evaluation: Dict[str, Any],
        save_path: Optional[str] = None
    ):
        """Create visualization plots for evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix
        if 'ml_metrics' in evaluation and 'confusion_matrix' in evaluation['ml_metrics']:
            cm = np.array(evaluation['ml_metrics']['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
            axes[0, 0].set_title('Confusion Matrix')
            axes[0, 0].set_xlabel('Predicted')
            axes[0, 0].set_ylabel('Actual')
        
        # 2. Feature Importance (if available)
        if hasattr(self, 'feature_importance') and self.feature_importance is not None:
            top_features = self.feature_importance.head(15)
            axes[0, 1].barh(top_features['feature'], top_features['importance'])
            axes[0, 1].set_xlabel('Importance')
            axes[0, 1].set_title('Top 15 Feature Importances')
            axes[0, 1].invert_yaxis()
        
        # 3. Model Performance Over Time
        if self.evaluation_history:
            scores = [e.get('combined_score', 0) for e in self.evaluation_history]
            axes[1, 0].plot(scores, marker='o')
            axes[1, 0].set_xlabel('Evaluation')
            axes[1, 0].set_ylabel('Combined Score')
            axes[1, 0].set_title('Model Performance Over Time')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Risk-Return Scatter
        if 'trading_metrics' in evaluation:
            tm = evaluation['trading_metrics']
            returns = [tm.get('annualized_return', 0)]
            risks = [tm.get('max_drawdown', 0)]
            sharpes = [tm.get('sharpe_ratio', 0)]
            
            scatter = axes[1, 1].scatter(risks, returns, s=100, c=sharpes, 
                                        cmap='RdYlGn', vmin=-1, vmax=3)
            axes[1, 1].set_xlabel('Max Drawdown (%)')
            axes[1, 1].set_ylabel('Annualized Return (%)')
            axes[1, 1].set_title('Risk-Return Profile')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=axes[1, 1])
            cbar.set_label('Sharpe Ratio')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        plt.close()  # Close figure to prevent memory leak
    
    def compare_models(
        self,
        models: Dict[str, Any],
        test_data: pd.DataFrame,
        test_labels: pd.Series
    ) -> pd.DataFrame:
        """Compare multiple models on the same test data"""
        results = []
        
        for model_name, model_info in models.items():
            model = model_info['model']
            feature_engineer = model_info.get('feature_engineer')
            
            # Get predictions
            if feature_engineer:
                features = feature_engineer.transform_features(test_data)
            else:
                features = test_data
            
            predictions = model.predict(features)
            probabilities = model.predict_proba(features) if hasattr(model, 'predict_proba') else None
            
            # Evaluate
            evaluation = self.evaluate_model(
                test_labels.values,
                predictions,
                probabilities
            )
            
            # Extract key metrics
            results.append({
                'model': model_name,
                'accuracy': evaluation['ml_metrics']['accuracy'],
                'precision': evaluation['ml_metrics']['precision'],
                'recall': evaluation['ml_metrics']['recall'],
                'f1_score': evaluation['ml_metrics']['f1_score'],
                'auc_roc': evaluation['ml_metrics'].get('auc_roc', None),
                'combined_score': evaluation['combined_score']
            })
        
        return pd.DataFrame(results).sort_values('combined_score', ascending=False)