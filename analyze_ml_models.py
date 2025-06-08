#!/usr/bin/env python3
"""
Analyze ML model structure to understand format and requirements
"""

import pickle
from pathlib import Path
import json

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    print("Note: joblib not available, using pickle only")

def analyze_model_file(model_path):
    """Analyze a single model file"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_path.name}")
    print(f"{'='*60}")
    
    try:
        # Try loading with joblib first if available
        if HAS_JOBLIB:
            try:
                model_data = joblib.load(model_path)
                print("✓ Loaded with joblib")
            except:
                # Fall back to pickle
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                print("✓ Loaded with pickle")
        else:
            # Only use pickle
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            print("✓ Loaded with pickle")
        
        # Analyze structure
        print(f"\nType: {type(model_data)}")
        
        if isinstance(model_data, dict):
            print(f"\nKeys in model package:")
            for key in model_data.keys():
                print(f"  - {key}: {type(model_data[key])}")
                
            # Detailed analysis of common keys
            if 'model' in model_data:
                model = model_data['model']
                print(f"\nModel type: {type(model)}")
                if hasattr(model, 'feature_importances_'):
                    print(f"Number of features: {len(model.feature_importances_)}")
                if hasattr(model, 'n_features_in_'):
                    print(f"Expected features: {model.n_features_in_}")
                    
            if 'feature_names' in model_data:
                features = model_data['feature_names']
                print(f"\nFeature names ({len(features)} total):")
                for i, feat in enumerate(features[:10]):  # Show first 10
                    print(f"  {i+1}. {feat}")
                if len(features) > 10:
                    print(f"  ... and {len(features) - 10} more")
                    
            if 'metadata' in model_data:
                metadata = model_data['metadata']
                print(f"\nMetadata:")
                if isinstance(metadata, dict):
                    for k, v in metadata.items():
                        if isinstance(v, (dict, list)) and len(str(v)) > 100:
                            print(f"  - {k}: {type(v)} with {len(v)} items")
                        else:
                            print(f"  - {k}: {v}")
                            
            if 'scaler' in model_data:
                scaler = model_data['scaler']
                print(f"\nScaler type: {type(scaler)}")
                if hasattr(scaler, 'mean_'):
                    print(f"Scaler fitted on {len(scaler.mean_)} features")
                    
            if 'performance' in model_data:
                perf = model_data['performance']
                print(f"\nModel performance:")
                if isinstance(perf, dict):
                    for metric, value in perf.items():
                        if isinstance(value, float):
                            print(f"  - {metric}: {value:.4f}")
                        else:
                            print(f"  - {metric}: {value}")
                            
            if 'label_encoder' in model_data:
                le = model_data['label_encoder']
                print(f"\nLabel encoder classes: {le.classes_}")
                
        else:
            # Not a dictionary, might be a direct model
            print(f"\nDirect model object: {type(model_data)}")
            if hasattr(model_data, 'predict'):
                print("✓ Has predict method")
            if hasattr(model_data, 'predict_proba'):
                print("✓ Has predict_proba method")
                
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()

def analyze_pattern_trader_dir(dir_path):
    """Analyze pattern trader directory structure"""
    print(f"\n{'='*60}")
    print(f"Analyzing directory: {dir_path.name}")
    print(f"{'='*60}")
    
    # List contents
    contents = list(dir_path.iterdir())
    print(f"\nContents ({len(contents)} items):")
    for item in sorted(contents):
        print(f"  - {item.name}")
        
    # Look for version directories
    version_dirs = [d for d in contents if d.is_dir() and d.name.replace('.', '').isdigit()]
    if version_dirs:
        latest_version = sorted(version_dirs, key=lambda x: x.name)[-1]
        print(f"\nLatest version: {latest_version.name}")
        
        # Check version contents
        version_contents = list(latest_version.iterdir())
        print(f"\nVersion contents:")
        for item in version_contents:
            print(f"  - {item.name}")
            
        # Try to load model from version
        model_file = latest_version / "model.pkl"
        if model_file.exists():
            analyze_model_file(model_file)

def main():
    """Main analysis function"""
    models_dir = Path("models")
    
    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        return
        
    print("ML Model Structure Analysis")
    print("="*60)
    
    # Analyze individual model files
    model_files = sorted(models_dir.glob("*_ml_package_*.pkl"))
    print(f"\nFound {len(model_files)} model package files")
    
    # Analyze one model per symbol to understand structure
    symbols_analyzed = set()
    for model_file in model_files:
        symbol = model_file.name.split('_')[0]
        if symbol not in symbols_analyzed:
            analyze_model_file(model_file)
            symbols_analyzed.add(symbol)
            
    # Analyze pattern trader directories
    pattern_dirs = sorted(models_dir.glob("pattern_trader_*"))
    print(f"\n\nFound {len(pattern_dirs)} pattern trader directories")
    
    for pattern_dir in pattern_dirs[:1]:  # Analyze just one to understand structure
        analyze_pattern_trader_dir(pattern_dir)
        
    print("\n" + "="*60)
    print("Analysis complete!")

if __name__ == "__main__":
    main()