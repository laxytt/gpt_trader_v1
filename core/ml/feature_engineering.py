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
        """Create labels for supervised learning"""
        if target == 'signal_success':
            # Binary classification: profitable trade or not
            horizon = self.feature_config['target_horizon']
            threshold = self.feature_config['min_profit_threshold']
            
            # Calculate future return
            future_return = data['close'].shift(-horizon) / data['close'] - 1
            
            # Create binary labels
            labels = (future_return > threshold).astype(int)
            
        elif target == 'direction':
            # Multi-class: up, down, or sideways
            horizon = self.feature_config['target_horizon']
            future_return = data['close'].shift(-horizon) / data['close'] - 1
            
            labels = pd.Series(index=data.index, dtype=int)
            labels[future_return > 0.001] = 1  # Up
            labels[future_return < -0.001] = -1  # Down
            labels[(future_return >= -0.001) & (future_return <= 0.001)] = 0  # Sideways
            
        elif target == 'optimal_action':
            # Based on VSA patterns and momentum
            labels = self._create_vsa_labels(data)
        
        else:
            raise ValueError(f"Unknown target: {target}")
        
        return labels
    
    def _create_vsa_labels(self, data: pd.DataFrame) -> pd.Series:
        """Create labels based on VSA (Volume Spread Analysis) patterns"""
        labels = pd.Series(index=data.index, dtype=int)
        
        # Calculate average volume
        avg_volume = data['volume'].rolling(20).mean()
        
        # Identify patterns
        for i in range(20, len(data)):
            # Current bar
            close = data['close'].iloc[i]
            open_price = data['open'].iloc[i]
            high = data['high'].iloc[i]
            low = data['low'].iloc[i]
            volume = data['volume'].iloc[i]
            
            # Previous bar
            prev_close = data['close'].iloc[i-1]
            prev_volume = data['volume'].iloc[i-1]
            
            # Future price (for labeling)
            if i + 10 < len(data):
                future_price = data['close'].iloc[i+10]
                future_return = (future_price - close) / close
                
                # Stopping Volume Pattern (Bullish)
                if (close < open_price and  # Down bar
                    volume > avg_volume.iloc[i] * 1.5 and  # High volume
                    close > low and  # Close off lows
                    future_return > 0.002):  # Profitable
                    labels.iloc[i] = 1
                
                # No Supply Pattern (Bullish)
                elif (close < open_price and  # Down bar
                      volume < avg_volume.iloc[i] * 0.5 and  # Low volume
                      future_return > 0.002):
                    labels.iloc[i] = 1
                
                
                elif (high > data['high'].iloc[i-10:i].max() and  # New high
                      close < open_price and  # But closed down
                      volume > avg_volume.iloc[i] and
                      future_return < -0.002):
                    labels.iloc[i] = -1
                
                # No Demand Pattern (Bearish)
                elif (close > open_price and  # Up bar
                      volume < avg_volume.iloc[i] * 0.5 and  # Low volume
                      future_return < -0.002):
                    labels.iloc[i] = -1
                
                else:
                    labels.iloc[i] = 0  # No clear pattern
        
        return labels
    
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