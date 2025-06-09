#!/usr/bin/env python3
"""
Production ML Training System
Learns from actual trading patterns and market conditions
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import pickle
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, make_scorer
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import lightgbm as lgb
import joblib

# Project imports
from core.infrastructure.data.unified_data_provider import UnifiedDataProvider, DataRequest
from core.infrastructure.database.repositories import TradeRepository, SignalRepository
from core.infrastructure.database.backtest_repository import BacktestRepository
from core.services.model_management_service import ModelManagementService, ModelRepository
from core.infrastructure.mt5.data_provider import MT5DataProvider
from core.infrastructure.mt5.client import MT5Client
from core.domain.enums.mt5_enums import TimeFrame
from core.domain.models import TradeResult
from core.utils.chart_utils import ChartGenerator
from config.settings import get_settings
from config.symbols import get_symbols_by_group

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TradingExample:
    """Represents a historical trading example"""
    timestamp: datetime
    symbol: str
    features: Dict[str, float]
    signal: str  # BUY/SELL/WAIT
    outcome: Optional[TradeResult] = None
    profit_pips: Optional[float] = None
    risk_reward: Optional[float] = None


class ProductionFeatureEngineer(BaseEstimator, TransformerMixin):
    """Production-grade feature engineering focused on actual trading patterns"""
    
    def __init__(self, lookback_periods: List[int] = [5, 10, 20, 50, 100]):
        self.lookback_periods = lookback_periods
        self.feature_names = []
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform raw OHLCV data into trading features"""
        features = pd.DataFrame(index=df.index)
        
        # Price Action Features
        features['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        features['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
        features['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / (df['high'] - df['low'] + 1e-10)
        features['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        # Trend Features
        for period in self.lookback_periods:
            # Price momentum
            features[f'momentum_{period}'] = df['close'].pct_change(period)
            
            # Moving averages
            sma = df['close'].rolling(period).mean()
            features[f'close_to_sma_{period}'] = (df['close'] - sma) / sma
            
            # Volatility
            features[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std()
            
            # Volume features
            features[f'volume_ratio_{period}'] = df['volume'] / df['volume'].rolling(period).mean()
            
            # High/Low channels
            high_max = df['high'].rolling(period).max()
            low_min = df['low'].rolling(period).min()
            features[f'channel_position_{period}'] = (df['close'] - low_min) / (high_max - low_min + 1e-10)
            
            # Support/Resistance
            features[f'distance_to_high_{period}'] = (high_max - df['close']) / df['close']
            features[f'distance_to_low_{period}'] = (df['close'] - low_min) / df['close']
        
        # Technical Indicators
        # RSI
        for period in [7, 14, 21]:
            features[f'rsi_{period}'] = self._calculate_rsi(df['close'], period)
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = macd - signal
        
        # Bollinger Bands
        for period in [20, 50]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            features[f'bb_upper_{period}'] = (sma + 2*std - df['close']) / df['close']
            features[f'bb_lower_{period}'] = (df['close'] - (sma - 2*std)) / df['close']
            features[f'bb_width_{period}'] = 4*std / sma
        
        # ATR
        features['atr_14'] = self._calculate_atr(df, 14) / df['close']
        features['atr_20'] = self._calculate_atr(df, 20) / df['close']
        
        # Pattern Recognition
        features['bullish_engulfing'] = self._bullish_engulfing(df)
        features['bearish_engulfing'] = self._bearish_engulfing(df)
        features['morning_star'] = self._morning_star(df)
        features['evening_star'] = self._evening_star(df)
        features['hammer'] = self._hammer(df)
        features['inverted_hammer'] = self._inverted_hammer(df)
        
        # Market Microstructure
        features['spread_ratio'] = (df['high'] - df['low']) / df['close']
        features['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # Time features
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        
        # Volume Profile
        features['volume_trend'] = df['volume'].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        features['volume_acceleration'] = features['volume_trend'].diff()
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        # Handle NaN and inf values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        
        return features.values
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi / 100  # Normalize to 0-1
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(period).mean()
    
    def _bullish_engulfing(self, df: pd.DataFrame) -> pd.Series:
        """Detect bullish engulfing pattern"""
        prev_bearish = df['close'].shift(1) < df['open'].shift(1)
        curr_bullish = df['close'] > df['open']
        engulfs = (df['open'] <= df['close'].shift(1)) & (df['close'] >= df['open'].shift(1))
        return (prev_bearish & curr_bullish & engulfs).astype(int)
    
    def _bearish_engulfing(self, df: pd.DataFrame) -> pd.Series:
        """Detect bearish engulfing pattern"""
        prev_bullish = df['close'].shift(1) > df['open'].shift(1)
        curr_bearish = df['close'] < df['open']
        engulfs = (df['open'] >= df['close'].shift(1)) & (df['close'] <= df['open'].shift(1))
        return (prev_bullish & curr_bearish & engulfs).astype(int)
    
    def _morning_star(self, df: pd.DataFrame) -> pd.Series:
        """Detect morning star pattern"""
        # Simplified version
        first_bearish = df['close'].shift(2) < df['open'].shift(2)
        second_small = abs(df['close'].shift(1) - df['open'].shift(1)) < (df['high'].shift(1) - df['low'].shift(1)) * 0.3
        third_bullish = df['close'] > df['open']
        third_closes_high = df['close'] > (df['open'].shift(2) + df['close'].shift(2)) / 2
        return (first_bearish & second_small & third_bullish & third_closes_high).astype(int)
    
    def _evening_star(self, df: pd.DataFrame) -> pd.Series:
        """Detect evening star pattern"""
        # Simplified version
        first_bullish = df['close'].shift(2) > df['open'].shift(2)
        second_small = abs(df['close'].shift(1) - df['open'].shift(1)) < (df['high'].shift(1) - df['low'].shift(1)) * 0.3
        third_bearish = df['close'] < df['open']
        third_closes_low = df['close'] < (df['open'].shift(2) + df['close'].shift(2)) / 2
        return (first_bullish & second_small & third_bearish & third_closes_low).astype(int)
    
    def _hammer(self, df: pd.DataFrame) -> pd.Series:
        """Detect hammer pattern"""
        body = abs(df['close'] - df['open'])
        lower_shadow = np.minimum(df['open'], df['close']) - df['low']
        upper_shadow = df['high'] - np.maximum(df['open'], df['close'])
        return ((lower_shadow > 2 * body) & (upper_shadow < body * 0.3)).astype(int)
    
    def _inverted_hammer(self, df: pd.DataFrame) -> pd.Series:
        """Detect inverted hammer pattern"""
        body = abs(df['close'] - df['open'])
        upper_shadow = df['high'] - np.maximum(df['open'], df['close'])
        lower_shadow = np.minimum(df['open'], df['close']) - df['low']
        return ((upper_shadow > 2 * body) & (lower_shadow < body * 0.3)).astype(int)


class TradingPatternLearner:
    """Learns from successful trading patterns in historical data"""
    
    def __init__(self):
        self.settings = get_settings()
        self.setup_infrastructure()
        self.feature_engineer = ProductionFeatureEngineer()
        
    def setup_infrastructure(self):
        """Initialize all required services"""
        db_path = self.settings.database.db_path
        self.trade_repository = TradeRepository(db_path)
        self.signal_repository = SignalRepository(db_path)
        self.backtest_repository = BacktestRepository(db_path)
        self.model_repository = ModelRepository(db_path)
        
        self.mt5_client = MT5Client(self.settings.mt5)
        self.data_provider = UnifiedDataProvider(mt5_client=self.mt5_client)
        
        self.model_management = ModelManagementService(
            models_dir=Path("models"),
            model_repository=self.model_repository
        )
    
    async def learn_from_patterns(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Learn from successful trading patterns"""
        
        logger.info(f"Learning trading patterns for {symbol}")
        
        try:
            # 1. Fetch comprehensive market data
            df = await self._fetch_comprehensive_data(symbol, start_date, end_date)
            if df is None or len(df) < 500:
                return {'status': 'error', 'error': 'Insufficient data'}
            
            # 2. Create sophisticated labels based on tradeable opportunities
            labels, metadata = self._create_trading_labels(df)
            
            logger.info(f"Created labels with distribution: {labels.value_counts().to_dict()}")
            logger.info(f"Metadata: {metadata}")
            
            # 3. Engineer features
            features = self.feature_engineer.transform(df)
            feature_names = self.feature_engineer.feature_names
            
            # 4. Align data
            min_length = min(len(features), len(labels))
            features = features[-min_length:]
            labels = labels.iloc[-min_length:]
            
            # 5. Remove samples with NaN labels
            valid_indices = ~labels.isna()
            features = features[valid_indices]
            labels = labels[valid_indices]
            
            X = pd.DataFrame(features, columns=feature_names)
            y = labels.astype(int)
            
            logger.info(f"Training on {len(X)} samples with {len(feature_names)} features")
            
            # 6. Train advanced models
            best_model, performance = await self._train_advanced_models(X, y, symbol)
            
            # 7. Save if performance meets criteria
            if self._meets_deployment_criteria(performance):
                model_metadata = await self._deploy_model(
                    model=best_model,
                    symbol=symbol,
                    performance=performance,
                    feature_names=feature_names
                )
                
                return {
                    'status': 'success',
                    'model_id': model_metadata['model_id'],
                    'performance': performance,
                    'label_metadata': metadata
                }
            else:
                return {
                    'status': 'rejected',
                    'performance': performance,
                    'reason': 'Does not meet deployment criteria'
                }
                
        except Exception as e:
            logger.error(f"Error learning patterns for {symbol}: {e}", exc_info=True)
            return {'status': 'error', 'error': str(e)}
    
    async def _fetch_comprehensive_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Fetch comprehensive market data"""
        
        request = DataRequest(
            symbol=symbol,
            timeframe=TimeFrame.H1,
            start_date=start_date,
            end_date=end_date
        )
        
        market_data = await self.data_provider.get_data(request)
        if not market_data:
            return None
        
        # Convert to DataFrame
        data = []
        for candle in market_data.candles:
            data.append({
                'timestamp': candle.timestamp,
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        # Add spread estimation (important for trading)
        df['spread'] = (df['high'] - df['low']) * 0.1  # Rough estimate
        
        return df
    
    def _create_trading_labels(self, df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """Create labels based on actual tradeable opportunities"""
        
        labels = pd.Series(index=df.index, dtype=float)
        
        # Calculate key levels
        atr = self.feature_engineer._calculate_atr(df, 14)
        
        # Look for profitable trading opportunities
        for i in range(100, len(df) - 50):  # Need history and future
            current_idx = df.index[i]
            
            # Skip if ATR is too low (no volatility)
            if atr.iloc[i] < df['close'].iloc[i] * 0.0003:  # Reduced to 3 pips minimum
                labels.iloc[i] = 0
                continue
            
            # Check future price movement
            future_prices = df['close'].iloc[i+1:i+25]  # Next 24 hours
            entry_price = df['close'].iloc[i]
            
            # More flexible profit targets for better balance
            # Option 1: Conservative targets (1.5:1 RR)
            long_target_1 = entry_price + (atr.iloc[i] * 1.5)
            long_stop_1 = entry_price - (atr.iloc[i] * 1)
            
            # Option 2: Moderate targets (1:1 RR) 
            long_target_2 = entry_price + (atr.iloc[i] * 1)
            long_stop_2 = entry_price - (atr.iloc[i] * 1)
            
            # Option 3: Scalping targets (0.5 ATR profit)
            long_target_3 = entry_price + (atr.iloc[i] * 0.5)
            long_stop_3 = entry_price - (atr.iloc[i] * 0.5)
            
            short_target_1 = entry_price - (atr.iloc[i] * 1.5)
            short_stop_1 = entry_price + (atr.iloc[i] * 1)
            
            short_target_2 = entry_price - (atr.iloc[i] * 1)
            short_stop_2 = entry_price + (atr.iloc[i] * 1)
            
            short_target_3 = entry_price - (atr.iloc[i] * 0.5)
            short_stop_3 = entry_price + (atr.iloc[i] * 0.5)
            
            # Check multiple profit scenarios
            long_profitable = False
            short_profitable = False
            
            # Check each target level
            for target, stop in [(long_target_1, long_stop_1), 
                                (long_target_2, long_stop_2),
                                (long_target_3, long_stop_3)]:
                hit_target = any(future_prices >= target)
                hit_stop = any(future_prices <= stop)
                
                if hit_target and hit_stop:
                    target_idx = future_prices[future_prices >= target].index[0]
                    stop_idx = future_prices[future_prices <= stop].index[0]
                    if target_idx < stop_idx:
                        long_profitable = True
                        break
                elif hit_target and not hit_stop:
                    long_profitable = True
                    break
            
            for target, stop in [(short_target_1, short_stop_1),
                                (short_target_2, short_stop_2), 
                                (short_target_3, short_stop_3)]:
                hit_target = any(future_prices <= target)
                hit_stop = any(future_prices >= stop)
                
                if hit_target and hit_stop:
                    target_idx = future_prices[future_prices <= target].index[0]
                    stop_idx = future_prices[future_prices >= stop].index[0]
                    if target_idx < stop_idx:
                        short_profitable = True
                        break
                elif hit_target and not hit_stop:
                    short_profitable = True
                    break
            
            # Relaxed filters for better balance
            # 1. Trend alignment (more flexible)
            sma_20 = df['close'].rolling(20).mean().iloc[i]
            sma_50 = df['close'].rolling(50).mean().iloc[i]
            
            # Allow trades in ranging markets too
            trend_up = entry_price > sma_20
            trend_down = entry_price < sma_20
            strong_trend_up = entry_price > sma_20 > sma_50
            strong_trend_down = entry_price < sma_20 < sma_50
            
            # 2. Recent momentum (less strict)
            momentum_5 = (entry_price - df['close'].iloc[i-5]) / df['close'].iloc[i-5]
            momentum_10 = (entry_price - df['close'].iloc[i-10]) / df['close'].iloc[i-10]
            
            # 3. Volatility condition
            recent_volatility = df['close'].iloc[i-20:i].pct_change().std()
            # Calculate historical volatility over past 50 periods
            historical_vol = df['close'].iloc[max(0, i-50):i].pct_change().std()
            is_volatile = recent_volatility > historical_vol * 1.2  # 20% above historical average
            
            # Label assignment with relaxed filters
            if long_profitable:
                # Strong setup
                if strong_trend_up and momentum_5 > 0.001:
                    labels.iloc[i] = 1
                # Moderate setup
                elif trend_up and (momentum_5 > 0 or momentum_10 > 0.002):
                    labels.iloc[i] = 1
                # Volatility breakout setup
                elif is_volatile and momentum_5 > 0.0005:
                    labels.iloc[i] = 1
                else:
                    labels.iloc[i] = 0
                    
            elif short_profitable:
                # Strong setup
                if strong_trend_down and momentum_5 < -0.001:
                    labels.iloc[i] = 2
                # Moderate setup  
                elif trend_down and (momentum_5 < 0 or momentum_10 < -0.002):
                    labels.iloc[i] = 2
                # Volatility breakout setup
                elif is_volatile and momentum_5 < -0.0005:
                    labels.iloc[i] = 2
                else:
                    labels.iloc[i] = 0
            else:
                labels.iloc[i] = 0
        
        # Convert to binary for initial training (1 = trade, 0 = no trade)
        binary_labels = labels.copy()
        binary_labels[binary_labels > 0] = 1
        
        metadata = {
            'total_samples': len(labels),
            'buy_signals': (labels == 1).sum(),
            'sell_signals': (labels == 2).sum(),
            'no_signals': (labels == 0).sum(),
            'trade_ratio': (labels > 0).sum() / len(labels)
        }
        
        return binary_labels, metadata
    
    async def _train_advanced_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        symbol: str
    ) -> Tuple[Any, Dict]:
        """Train advanced models with proper methodology"""
        
        # Time series split
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Calculate class weights for imbalanced data
        n_positive = (y == 1).sum()
        n_negative = (y == 0).sum()
        scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0
        
        logger.info(f"Class distribution - Positive: {n_positive}, Negative: {n_negative}")
        logger.info(f"Scale positive weight: {scale_pos_weight:.2f}")
        
        # Models to evaluate
        models = {
            'lgb': lgb.LGBMClassifier(
                n_estimators=200,  # Reduced to avoid overfitting
                learning_rate=0.1,  # Increased for faster learning
                num_leaves=15,  # Reduced for simpler trees
                max_depth=5,  # Reduced depth
                min_child_samples=50,  # Increased to avoid overfitting
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,  # Increased regularization
                reg_lambda=0.5,  # Increased regularization
                random_state=42,
                n_jobs=-1,
                importance_type='gain',
                scale_pos_weight=scale_pos_weight,  # Use calculated weight
                verbosity=-1  # Suppress warnings
            ),
            'rf': RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.8,
                random_state=42
            )
        }
        
        # Create pipeline with scaling
        best_score = 0
        best_model = None
        best_model_name = None
        
        logger.info(f"Training models for {symbol} with {n_splits}-fold time series CV...")
        
        for name, model in models.items():
            pipeline = Pipeline([
                ('scaler', RobustScaler()),
                ('classifier', model)
            ])
            
            # Custom scoring for trading
            def trading_score(y_true, y_pred):
                # Prioritize precision over recall for trading
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                # Custom score that values precision more
                return 0.7 * precision + 0.3 * f1
            
            scorer = make_scorer(trading_score)
            
            # Cross-validation
            cv_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Skip if validation set has no positive samples
                if y_val.sum() == 0:
                    continue
                
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_val)
                score = trading_score(y_val, y_pred)
                cv_scores.append(score)
            
            avg_score = np.mean(cv_scores) if cv_scores else 0
            logger.info(f"{name} CV score: {avg_score:.3f}")
            
            if avg_score > best_score:
                best_score = avg_score
                best_model = pipeline
                best_model_name = name
        
        # Train final model on all data
        logger.info(f"Training final {best_model_name} model on full dataset...")
        best_model.fit(X, y)
        
        # Get feature importances
        if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
            importances = best_model.named_steps['classifier'].feature_importances_
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            logger.info("Top 10 features:")
            for idx, row in feature_importance.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.3f}")
        
        # Final evaluation on last 20% of data
        split_idx = int(len(X) * 0.8)
        X_final_test = X.iloc[split_idx:]
        y_final_test = y.iloc[split_idx:]
        
        y_pred = best_model.predict(X_final_test)
        y_prob = best_model.predict_proba(X_final_test)[:, 1]
        
        performance = {
            'model_type': best_model_name,
            'cv_score': best_score,
            'precision': precision_score(y_final_test, y_pred, zero_division=0),
            'recall': recall_score(y_final_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_final_test, y_pred, zero_division=0),
            'total_trades_predicted': y_pred.sum(),
            'total_trades_actual': y_final_test.sum()
        }
        
        # Add AUC if we have positive samples
        if y_final_test.sum() > 0 and len(np.unique(y_final_test)) > 1:
            performance['auc'] = roc_auc_score(y_final_test, y_prob)
        
        return best_model, performance
    
    def _meets_deployment_criteria(self, performance: Dict) -> bool:
        """Check if model meets deployment criteria"""
        # More realistic criteria for trading
        min_precision = 0.55  # At least 55% of predicted trades should be correct
        min_cv_score = 0.45   # Decent cross-validation score
        min_f1 = 0.50        # Minimum F1 score
        
        # Log the performance metrics for debugging
        logger.info(f"Model performance metrics:")
        logger.info(f"  Precision: {performance.get('precision', 0):.3f}")
        logger.info(f"  CV Score: {performance.get('cv_score', 0):.3f}")
        logger.info(f"  F1 Score: {performance.get('f1_score', 0):.3f}")
        logger.info(f"  Trades predicted: {performance.get('total_trades_predicted', 0)}")
        
        return (
            performance.get('precision', 0) >= min_precision and
            performance.get('cv_score', 0) >= min_cv_score and
            performance.get('f1_score', 0) >= min_f1 and
            performance.get('total_trades_predicted', 0) > 10  # Must predict some trades
        )
    
    async def _deploy_model(
        self,
        model: Any,
        symbol: str,
        performance: Dict,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Deploy model for production use"""
        
        # Create comprehensive model package
        model_package = {
            'pipeline': model,
            'feature_engineer': self.feature_engineer,
            'feature_names': feature_names,
            'symbol': symbol,
            'training_date': datetime.now().isoformat(),
            'performance': performance,
            'model_type': performance['model_type'],
            'deployment_criteria': {
                'min_precision': 0.6,
                'min_cv_score': 0.5
            }
        }
        
        # Save complete package
        package_path = Path("models") / f"{symbol}_ml_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        package_path.parent.mkdir(exist_ok=True)
        
        with open(package_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        # Save via model management service
        model_id = await self.model_management.save_model(
            model=model_package,
            model_type=f"pattern_trader_{symbol}",
            version="4.0.0",
            training_metrics=performance,
            feature_names=feature_names,
            hyperparameters={
                'model_type': performance['model_type'],
                'training_approach': 'pattern_learning',
                'label_strategy': 'profitable_trades'
            },
            training_data_info={
                'symbol': symbol,
                'features_count': len(feature_names),
                'samples_used': performance.get('total_trades_actual', 0)
            }
        )
        
        # Deploy
        await self.model_management.deploy_model(model_id)
        
        logger.info(f"Model deployed successfully: {model_id}")
        logger.info(f"Package saved to: {package_path}")
        
        return {'model_id': model_id, 'package_path': str(package_path)}


async def main():
    """Main training pipeline"""
    learner = TradingPatternLearner()
    
    # Initialize MT5
    if not learner.mt5_client.initialize():
        logger.error("Failed to initialize MT5")
        return
    
    try:
        # Configuration
        symbols = get_symbols_by_group('conservative')
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year of data for better patterns
        
        logger.info("="*80)
        logger.info("PRODUCTION ML TRAINING SYSTEM")
        logger.info("="*80)
        logger.info(f"Training period: {start_date.date()} to {end_date.date()}")
        logger.info(f"Symbols: {symbols}")
        logger.info("="*80)
        
        results = {}
        
        for symbol in symbols:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {symbol}")
            logger.info(f"{'='*60}")
            
            result = await learner.learn_from_patterns(symbol, start_date, end_date)
            results[symbol] = result
            
            if result['status'] == 'success':
                perf = result['performance']
                logger.info(f"✓ SUCCESS - Model deployed for {symbol}")
                logger.info(f"  Precision: {perf['precision']:.3f}")
                logger.info(f"  CV Score: {perf['cv_score']:.3f}")
                logger.info(f"  Trades found: {perf['total_trades_actual']}")
            else:
                logger.warning(f"✗ FAILED - {symbol}: {result.get('reason', 'Unknown error')}")
        
        # Summary
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        
        successful = sum(1 for r in results.values() if r['status'] == 'success')
        print(f"\nSuccessfully deployed: {successful}/{len(symbols)} models")
        
        for symbol, result in results.items():
            print(f"\n{symbol}:")
            print(f"  Status: {result['status']}")
            if result['status'] == 'success':
                perf = result['performance']
                print(f"  Model Type: {perf['model_type']}")
                print(f"  Precision: {perf['precision']:.3f}")
                print(f"  Recall: {perf['recall']:.3f}")
                print(f"  F1 Score: {perf['f1_score']:.3f}")
                print(f"  CV Score: {perf['cv_score']:.3f}")
                print(f"  Model ID: {result['model_id']}")
        
        print("\n" + "="*80)
        print("ML models are now ready for use in trading!")
        print("To enable: Set ML_ENABLED=true in your .env file")
        print("="*80)
        
    finally:
        learner.mt5_client.shutdown()


if __name__ == "__main__":
    # Check for LightGBM
    try:
        import lightgbm
        asyncio.run(main())
    except ImportError:
        print("Please install LightGBM first: pip install lightgbm")
        sys.exit(1)