# Secure ML Model Loading Guide

This guide explains the secure ML model loading system implemented in the GPT Trading System to protect against malicious models and ensure model integrity.

## Overview

The secure model loading system provides:
- **Checksum verification** to detect tampering
- **Digital signatures** for authenticity
- **Restricted deserialization** to prevent code execution
- **Access control** via allowed model lists
- **Audit logging** for all model operations

## Security Features

### 1. Checksum Verification
Every model file has associated SHA256 and SHA512 checksums stored in metadata:
```python
# Automatic checksum verification on load
model = secure_loader.load_model("model.pkl", verify_checksum=True)
```

### 2. Digital Signatures (HMAC)
Models can be signed with a secret key for authenticity:
```bash
# Set secret key for signatures
export MODEL_SECRET_KEY="your-secret-key-here"
```

### 3. Restricted Unpickler
The `RestrictedUnpickler` only allows safe classes during deserialization:
- NumPy arrays and data types
- Pandas DataFrames and Series
- Scikit-learn models and transformers
- Basic Python types (dict, list, etc.)

### 4. Allowed Models List
A whitelist of approved model checksums:
```json
{
  "EURUSD_pattern_trader_v20250108.1.0_20250108_120000.joblib": "a1b2c3d4..."
}
```

## Usage

### Training Models Securely

Use the secure training script:
```bash
# Train models with security features
python scripts/train_ml_secure.py --symbols EURUSD GBPUSD --authorized-by "John Doe"
```

### Migrating Existing Models

Convert existing models to secure format:
```bash
# Set secret key
export MODEL_SECRET_KEY="your-secret-key-here"

# Migrate all models
python scripts/migrate_models_to_secure.py --models-dir models

# Verify migration
python scripts/migrate_models_to_secure.py --verify-only
```

### Loading Models in Code

Replace `MLPredictor` with `SecureMLPredictor`:

```python
from core.ml.secure_ml_predictor import create_secure_ml_predictor

# Create secure predictor
predictor = create_secure_ml_predictor(
    models_dir="models",
    db_path="data/trades.db",
    secret_key=os.environ.get('MODEL_SECRET_KEY')
)

# Use normally
ml_confidence, details = predictor.predict_signal(
    symbol="EURUSD",
    market_data=market_data
)
```

### Manual Model Verification

Verify a model without loading it:
```python
from core.ml.secure_model_loader import SecureModelLoader

loader = SecureModelLoader("models")
verification = loader.verify_model("model.pkl")

print(f"Checksum valid: {verification['checksum_valid']}")
print(f"Signature valid: {verification['signature_valid']}")
print(f"SHA256: {verification['checksum_sha256']}")
```

## Model Metadata Structure

Each model has an associated metadata file in `.metadata/`:
```json
{
  "model_name": "EURUSD_pattern_trader",
  "version": "20250108.1.0",
  "created_at": "2025-01-08T12:00:00Z",
  "checksum_sha256": "a1b2c3d4...",
  "checksum_sha512": "e5f6g7h8...",
  "file_size": 1048576,
  "model_type": "joblib",
  "feature_names": ["rsi_14", "macd_signal", ...],
  "performance_metrics": {
    "accuracy": 0.75,
    "f1_score": 0.72
  },
  "authorized_by": "John Doe",
  "signature": "i9j0k1l2..."
}
```

## Security Best Practices

1. **Always use checksums**: Set `verify_checksum=True` when loading models
2. **Enable signatures**: Set `MODEL_SECRET_KEY` environment variable
3. **Maintain allowed list**: Keep `allowed_models.json` up to date
4. **Regular verification**: Periodically verify all models
5. **Backup before migration**: Always backup models before converting
6. **Audit model sources**: Only load models from trusted sources
7. **Monitor model operations**: Check logs for security violations

## Environment Variables

- `MODEL_SECRET_KEY`: Secret key for HMAC signatures (required for signatures)
- `ML_MODELS_DIR`: Directory containing ML models (default: "models")
- `ML_VERIFY_CHECKSUMS`: Enable checksum verification (default: "true")
- `ML_VERIFY_SIGNATURES`: Enable signature verification (default: "true")

## Troubleshooting

### "SecurityError: Model checksum mismatch"
The model file has been modified. Verify the model source and retrain if necessary.

### "SecurityError: Model signature verification failed"
The signature doesn't match. Ensure you're using the correct `MODEL_SECRET_KEY`.

### "SecurityError: Attempted to load unsafe module"
The model contains unsafe classes. This model should not be loaded.

### Migration fails with "Not a valid model package"
The model file doesn't have the expected structure. Check if it's a raw model or model package.

## API Reference

### SecureModelLoader
```python
class SecureModelLoader:
    def __init__(self, models_dir, secret_key=None, enable_signature_verification=True)
    def save_model(self, model, model_name, version, **kwargs) -> Path
    def load_model(self, model_path, verify_checksum=True, verify_signature=None) -> Any
    def verify_model(self, model_path) -> Dict[str, Any]
    def list_models(self) -> List[Dict[str, Any]]
```

### SecureMLPredictor
```python
class SecureMLPredictor:
    def __init__(self, models_dir, db_path, secret_key=None)
    def predict_signal(self, symbol, market_data, news_sentiment=0.0, vsa_signal=0.0)
    def get_model_info(self, symbol) -> Optional[Dict[str, Any]]
    def verify_all_models(self) -> Dict[str, Dict[str, Any]]
    def reload_models(self)
```

## Migration Checklist

- [ ] Set `MODEL_SECRET_KEY` environment variable
- [ ] Backup existing models
- [ ] Run migration script
- [ ] Verify all models migrated successfully
- [ ] Update code to use `SecureMLPredictor`
- [ ] Test model loading and predictions
- [ ] Remove backup after confirming everything works
- [ ] Update production deployment with new environment variables

## Security Considerations

1. **Secret Key Management**: Store `MODEL_SECRET_KEY` securely (e.g., environment variables, secrets manager)
2. **Model Distribution**: Only share model files with their metadata files
3. **Access Control**: Restrict write access to models directory
4. **Regular Audits**: Periodically verify all models and check for unauthorized changes
5. **Incident Response**: Have a plan for handling compromised models

By following this guide, you can ensure that your ML models are loaded securely and protected against tampering and malicious code execution.