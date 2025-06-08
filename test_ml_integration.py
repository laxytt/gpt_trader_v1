#!/usr/bin/env python3
"""
Test ML integration to verify it's working properly
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.ml.ml_predictor import MLPredictor
from core.domain.models import MarketData, Candle
from core.domain.enums import TimeFrame

def create_test_market_data(symbol: str) -> MarketData:
    """Create sample market data for testing"""
    candles = []
    base_price = 1.0800 if symbol == "EURUSD" else 1.2500
    
    # Create 200 candles with some price movement
    for i in range(200):
        time = datetime.now(timezone.utc) - timedelta(hours=200-i)
        open_price = base_price + (i % 10) * 0.0001
        high = open_price + 0.0005
        low = open_price - 0.0003
        close = open_price + 0.0002
        volume = 1000 + i * 10
        
        candles.append(Candle(
            time=time,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume
        ))
    
    return MarketData(
        symbol=symbol,
        timeframe=TimeFrame.H1,
        candles=candles
    )

async def test_ml_predictor():
    """Test the ML predictor"""
    print("ML Integration Test")
    print("=" * 60)
    
    # Initialize ML predictor
    print("\n1. Initializing ML Predictor...")
    predictor = MLPredictor()
    
    # Check loaded models
    print(f"\nLoaded models for symbols: {list(predictor.loaded_models.keys())}")
    
    if not predictor.loaded_models:
        print("❌ No models loaded! Please ensure models exist in the 'models' directory.")
        return
    
    # Test each symbol
    for symbol in ["EURUSD", "GBPUSD", "AUDUSD", "USDCAD"]:
        print(f"\n2. Testing {symbol}...")
        
        if not predictor.has_model(symbol):
            print(f"   ⚠️ No model found for {symbol}")
            continue
            
        # Get model info
        model_info = predictor.get_model_info(symbol)
        if model_info:
            print(f"   Model type: {model_info['model_type']}")
            print(f"   Training date: {model_info['training_date']}")
            print(f"   Feature count: {model_info['feature_count']}")
            performance = model_info['performance']
            if performance:
                print(f"   Performance: Precision={performance.get('precision', 'N/A')}, "
                      f"Recall={performance.get('recall', 'N/A')}")
        
        # Create test data
        print(f"\n3. Creating test market data for {symbol}...")
        market_data = {
            'h1': create_test_market_data(symbol),
            'h4': create_test_market_data(symbol)  # Simplified for testing
        }
        
        # Get prediction
        print(f"\n4. Getting ML prediction for {symbol}...")
        try:
            result = await predictor.get_ml_prediction(symbol, market_data)
            
            print(f"\n   Results:")
            print(f"   - ML Enabled: {result['ml_enabled']}")
            print(f"   - ML Signal: {result['ml_signal']}")
            print(f"   - ML Confidence: {result['ml_confidence']:.1f}%" if result['ml_confidence'] else "N/A")
            
            metadata = result.get('ml_metadata', {})
            if metadata:
                print(f"   - Model Type: {metadata.get('model_type', 'Unknown')}")
                print(f"   - Raw Prediction: {metadata.get('prediction_raw', 'N/A')}")
                
                # Show top features if available
                top_features = metadata.get('feature_importances', [])
                if top_features:
                    print(f"   - Top Features:")
                    for feat in top_features[:3]:
                        print(f"     * {feat['name']}: {feat['importance']:.3f}")
                        
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test complete!")

def main():
    """Run the test"""
    asyncio.run(test_ml_predictor())

if __name__ == "__main__":
    main()