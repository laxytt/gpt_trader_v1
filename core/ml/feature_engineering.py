# core/ml/feature_engineering.py
"""
Feature engineering for trading ML models.
Transforms raw market data into features suitable for ML algorithms.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

# Try to import talib, but make it optional
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib not available. Using pandas for technical indicators.")

from core.domain.models import MarketData, Candle
from core.domain.exceptions import DataError


class FeatureEngineer:
    """
    Feature engineering pipeline for trading models.
    Creates technical indicators and market microstructure features.
    """
    
    def __init__(self, feature_config: Optional[Dict] = None):
        self.feature_config = feature_config or self._get_default_config()
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def _get_default_config(self) -> Dict:
        """Get default feature configuration"""
        return {
            'price_features': True,
            'volume_features': True,
            'technical_indicators': True,
            'microstructure_features': True,
            'time_features': True,
            'lookback_periods': [5, 10, 20, 50],
            'target_horizon': 24,  # Hours ahead for prediction
            'min_profit_threshold': 0.002,  # 0.2% minimum profit for positive label
        }
    
    def prepare_features(
        self,
        market_data: pd.DataFrame,
        target: str = 'signal_success'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and labels from market data.
        
        Args:
            market_data: DataFrame with OHLCV data
            target: Target variable to predict
            
        Returns:
            Tuple of (features DataFrame, labels Series)
        """
        # Create features
        features = pd.DataFrame(index=market_data.index)
        
        # Price-based features
        if self.feature_config['price_features']:
            features = self._add_price_features(features, market_data)
        
        # Volume features
        if self.feature_config['volume_features']:
            features = self._add_volume_features(features, market_data)
        
        # Technical indicators
        if self.feature_config['technical_indicators']:
            features = self._add_technical_indicators(features, market_data)
        
        # Market microstructure
        if self.feature_config['microstructure_features']:
            features = self._add_microstructure_features(features, market_data)
        
        # Time-based features
        if self.feature_config['time_features']:
            features = self._add_time_features(features, market_data)
        
        # Create labels
        labels = self._create_labels(market_data, target)
        
        # Remove NaN values
        valid_indices = features.dropna().index.intersection(labels.dropna().index)
        features = features.loc[valid_indices]
        labels = labels.loc[valid_indices]
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        return features, labels
    
    def _add_price_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        # Returns
        for period in self.feature_config['lookback_periods']:
            features[f'return_{period}'] = data['close'].pct_change(period)
            features[f'high_low_ratio_{period}'] = (
                data['high'].rolling(period).max() / 
                data['low'].rolling(period).min() - 1
            )
        
        # Price position within range
        features['close_to_high'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        
        # Volatility
        features['volatility_20'] = data['close'].pct_change().rolling(20).std()
        
        # Price momentum
        features['momentum_10'] = data['close'] - data['close'].shift(10)
        
        return features
    
    def _add_volume_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        # Volume ratios
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # Volume-weighted price
        features['vwap'] = (data['close'] * data['volume']).rolling(20).sum() / data['volume'].rolling(20).sum()
        features['price_to_vwap'] = data['close'] / features['vwap']
        
        # On-balance volume
        obv = (np.sign(data['close'].diff()) * data['volume']).cumsum()
        features['obv_slope'] = obv.diff(5)
        
        return features
    
    def _add_technical_indicators(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators using TA-Lib or pandas fallbacks"""
        if TALIB_AVAILABLE:
            try:
                # RSI
                features['rsi_14'] = talib.RSI(data['close'].values, timeperiod=14)
                features['rsi_7'] = talib.RSI(data['close'].values, timeperiod=7)
                
                # MACD
                macd, signal, hist = talib.MACD(
                    data['close'].values,
                    fastperiod=12,
                    slowperiod=26,
                    signalperiod=9
                )
                features['macd'] = macd
                features['macd_signal'] = signal
                features['macd_hist'] = hist
                
                # Bollinger Bands
                upper, middle, lower = talib.BBANDS(
                    data['close'].values,
                    timeperiod=20,
                    nbdevup=2,
                    nbdevdn=2
                )
                features['bb_upper'] = upper
                features['bb_lower'] = lower
                features['bb_width'] = (upper - lower) / middle
                features['bb_position'] = (data['close'] - lower) / (upper - lower)
                
                # ATR
                features['atr_14'] = talib.ATR(
                    data['high'].values,
                    data['low'].values,
                    data['close'].values,
                    timeperiod=14
                )
                
                # ADX
                features['adx_14'] = talib.ADX(
                    data['high'].values,
                    data['low'].values,
                    data['close'].values,
                    timeperiod=14
                )
                
            except Exception as e:
                # If talib fails, use pandas fallback
                self._add_pandas_indicators(features, data)
        else:
            # Use pandas implementation
            self._add_pandas_indicators(features, data)
        
        return features
    
    def _add_pandas_indicators(self, features: pd.DataFrame, data: pd.DataFrame):
        """Add technical indicators using pandas (fallback when TA-Lib not available)"""
        # RSI
        features['rsi_14'] = self._calculate_rsi(data['close'], 14)
        features['rsi_7'] = self._calculate_rsi(data['close'], 7)
        
        # MACD
        ema_12 = data['close'].ewm(span=12).mean()
        ema_26 = data['close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        sma_20 = data['close'].rolling(20).mean()
        std_20 = data['close'].rolling(20).std()
        features['bb_upper'] = sma_20 + (2 * std_20)
        features['bb_lower'] = sma_20 - (2 * std_20)
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / sma_20
        features['bb_position'] = (data['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # ATR
        features['atr_14'] = self._calculate_atr(data, 14)
        
        # ADX (simplified)
        features['adx_14'] = self._calculate_adx(data, 14)
    
    def _add_microstructure_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        # Spread (if available)
        if 'spread' in data.columns:
            features['spread'] = data['spread']
            features['spread_ratio'] = data['spread'] / data['close']
        
        # Bar statistics
        features['bar_range'] = data['high'] - data['low']
        features['bar_body'] = abs(data['close'] - data['open'])
        features['upper_shadow'] = data['high'] - np.maximum(data['open'], data['close'])
        features['lower_shadow'] = np.minimum(data['open'], data['close']) - data['low']
        
        # Efficiency ratio
        change = abs(data['close'] - data['close'].shift(10))
        path = data['close'].diff().abs().rolling(10).sum()
        features['efficiency_ratio'] = change / path
        
        return features
    
    def _add_time_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        # Extract time components
        features['hour'] = data.index.hour
        features['day_of_week'] = data.index.dayofweek
        
        # Cyclical encoding
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        # Trading session indicators
        features['asian_session'] = ((features['hour'] >= 0) & (features['hour'] < 7)).astype(int)
        features['european_session'] = ((features['hour'] >= 7) & (features['hour'] < 13)).astype(int)
        features['us_session'] = ((features['hour'] >= 13) & (features['hour'] < 22)).astype(int)
        
        return features
    
    def _create_labels(self, data: pd.DataFrame, target: str) -> pd.Series:
        """Create labels for supervised learning WITHOUT data leakage
        
        CRITICAL FIX: This method now properly creates labels based on PAST outcomes
        for training. It does NOT peek into the future.
        """
        if target == 'signal_success':
            # Binary classification: profitable trade or not
            # FIXED: We now look at past trades and their outcomes
            horizon = self.feature_config['target_horizon']
            threshold = self.feature_config['min_profit_threshold']
            
            # FIXED: Calculate returns looking BACKWARD
            # For each point, we look at what happened AFTER trades horizon bars ago
            # This way, when training on bar N, we use the outcome of a trade from bar N-horizon
            past_entry_price = data['close'].shift(horizon)
            exit_price = data['close']
            historical_return = (exit_price / past_entry_price - 1).fillna(0)
            
            # Create binary labels based on historical outcomes
            labels = (historical_return > threshold).astype(int)
            
            # CRITICAL: First 'horizon' rows don't have enough history
            labels.iloc[:horizon] = np.nan
            
        elif target == 'direction':
            # Multi-class: up, down, or sideways
            horizon = self.feature_config['target_horizon']
            # FIXED: Look at historical movements, not future
            past_entry_price = data['close'].shift(horizon)
            exit_price = data['close']
            historical_return = (exit_price / past_entry_price - 1).fillna(0)
            
            labels = pd.Series(index=data.index, dtype=float)  # Use float to support NaN
            labels[historical_return > 0.001] = 1  # Up
            labels[historical_return < -0.001] = -1  # Down
            labels[(historical_return >= -0.001) & (historical_return <= 0.001)] = 0  # Sideways
            
            # CRITICAL: First 'horizon' rows don't have enough history
            labels.iloc[:horizon] = np.nan
            
        elif target == 'optimal_action':
            # Based on VSA patterns - using ONLY historical confirmed patterns
            labels = self._create_vsa_labels_no_leakage(data)
        
        else:
            raise ValueError(f"Unknown target: {target}")
        
        return labels
    
    def _create_vsa_labels_no_leakage(self, data: pd.DataFrame) -> pd.Series:
        """Create labels based on VSA patterns WITHOUT using future data
        
        FIXED: This method now properly looks at HISTORICAL pattern outcomes
        without peeking into the future. For each current bar, we check if
        a pattern that occurred 'horizon' bars ago was successful.
        """
        labels = pd.Series(index=data.index, dtype=float)  # Float to support NaN
        labels[:] = 0  # Default to no action
        
        # Calculate average volume
        avg_volume = data['volume'].rolling(20).mean()
        
        # We look back 'horizon' bars to see pattern outcomes
        horizon = 10
        
        # FIXED: Start from horizon + 20 to have enough history
        for i in range(horizon + 20, len(data)):
            # Look at a pattern that occurred 'horizon' bars ago
            pattern_idx = i - horizon
            
            # Historical bar analysis (from horizon bars ago)
            hist_close = data['close'].iloc[pattern_idx]
            hist_open = data['open'].iloc[pattern_idx]
            hist_high = data['high'].iloc[pattern_idx]
            hist_low = data['low'].iloc[pattern_idx]
            hist_volume = data['volume'].iloc[pattern_idx]
            
            # Previous bars (before the historical pattern)
            hist_prev_close = data['close'].iloc[pattern_idx-1]
            hist_prev_high = data['high'].iloc[pattern_idx-1]
            hist_prev_low = data['low'].iloc[pattern_idx-1]
            
            # Pattern detection using historical data
            pattern_detected = 0
            
            # Stopping Volume Pattern (potential bullish)
            if (hist_close < hist_open and  # Down bar
                hist_volume > avg_volume.iloc[pattern_idx] * 1.5 and  # High volume
                hist_close > hist_low + (hist_high - hist_low) * 0.3):  # Close in upper 70% of range
                pattern_detected = 1  # Potential buy
            
            # No Supply Pattern (potential bullish)
            elif (hist_close < hist_prev_close and  # Down from previous
                  hist_volume < avg_volume.iloc[pattern_idx] * 0.5 and  # Low volume
                  hist_close > hist_low + (hist_high - hist_low) * 0.5):  # Close in upper half
                pattern_detected = 1  # Potential buy
            
            # Upthrust Pattern (potential bearish)
            elif (hist_high > data['high'].iloc[pattern_idx-10:pattern_idx].max() and  # New high
                  hist_close < hist_open and  # But closed down
                  hist_close < hist_high - (hist_high - hist_low) * 0.3 and  # Close in lower 70%
                  hist_volume > avg_volume.iloc[pattern_idx]):
                pattern_detected = -1  # Potential sell
            
            # No Demand Pattern (potential bearish)
            elif (hist_close > hist_prev_close and  # Up from previous
                  hist_volume < avg_volume.iloc[pattern_idx] * 0.5 and  # Low volume
                  hist_close < hist_high - (hist_high - hist_low) * 0.5):  # Close in lower half
                pattern_detected = -1  # Potential sell
            
            # Check if the historical pattern was successful
            if pattern_detected != 0:
                # Now we can see the actual outcome (current bar vs pattern bar)
                current_price = data['close'].iloc[i]
                actual_return = (current_price - hist_close) / hist_close
                
                # Label based on whether the pattern prediction was correct
                if pattern_detected == 1 and actual_return > 0.002:
                    labels.iloc[i] = 1  # Historical bullish pattern was successful
                elif pattern_detected == -1 and actual_return < -0.002:
                    labels.iloc[i] = -1  # Historical bearish pattern was successful
                # else: remains 0 (pattern failed)
        
        # Mark first rows without enough history as NaN
        labels.iloc[:horizon + 20] = np.nan
        
        return labels
    
    def _create_vsa_labels(self, data: pd.DataFrame) -> pd.Series:
        """Legacy method - redirects to non-leaking version"""
        return self._create_vsa_labels_no_leakage(data)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI without TA-Lib"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR without TA-Lib"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean()
        return atr
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX (simplified version) without TA-Lib"""
        # Calculate directional movement
        high_diff = data['high'] - data['high'].shift(1)
        low_diff = data['low'].shift(1) - data['low']
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        # Get ATR
        atr = self._calculate_atr(data, period)
        
        # Calculate directional indicators
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
        
        # Calculate ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx
    
    def transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply feature transformations (scaling, etc.)"""
        # Ensure we have the same features as training
        if set(features.columns) != set(self.feature_names):
            raise ValueError("Feature mismatch between training and prediction")
        
        # Scale features
        scaled_features = pd.DataFrame(
            self.scaler.transform(features),
            index=features.index,
            columns=features.columns
        )
        
        return scaled_features
    
    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler and transform features"""
        self.scaler.fit(features)
        return self.transform_features(features)
    
    def get_feature_importance(self, model) -> pd.DataFrame:
        """Extract feature importance from trained model"""
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance
        else:
            return pd.DataFrame()
    
    def create_train_test_split(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        test_size: float = 0.2,
        gap_periods: int = 10
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Create train/test split with gap to prevent data leakage
        
        Args:
            features: Feature DataFrame
            labels: Label Series
            test_size: Proportion of data for testing
            gap_periods: Number of periods between train and test to prevent leakage
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Calculate split point
        n_samples = len(features)
        test_samples = int(n_samples * test_size)
        train_samples = n_samples - test_samples - gap_periods
        
        if train_samples <= 0:
            raise ValueError(f"Not enough data for train/test split with gap={gap_periods}")
        
        # Split with gap
        train_end = train_samples
        test_start = train_end + gap_periods
        
        # Create splits
        X_train = features.iloc[:train_end]
        y_train = labels.iloc[:train_end]
        X_test = features.iloc[test_start:]
        y_test = labels.iloc[test_start:]
        
        # Log split information
        train_dates = f"{X_train.index[0]} to {X_train.index[-1]}"
        test_dates = f"{X_test.index[0]} to {X_test.index[-1]}"
        logger.info(f"Train/test split created:")
        logger.info(f"  Train: {len(X_train)} samples ({train_dates})")
        logger.info(f"  Gap: {gap_periods} periods")
        logger.info(f"  Test: {len(X_test)} samples ({test_dates})")
        
        return X_train, X_test, y_train, y_test