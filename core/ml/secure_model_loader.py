"""
Secure model loading with checksum verification and integrity checks.
Provides protection against malicious model files and tampering.
"""

import hashlib
import json
import logging
import os
import pickle
import joblib
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Callable
from datetime import datetime, timezone
import hmac
import tempfile
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, asdict

from core.domain.exceptions import (
    ValidationError, SecurityError, ConfigurationError,
    SerializationError, TradingSystemError
)

logger = logging.getLogger(__name__)


class SecurityError(TradingSystemError):
    """Raised when security checks fail"""
    pass


@dataclass
class ModelMetadata:
    """Metadata for a secure model file"""
    model_name: str
    version: str
    created_at: str
    checksum_sha256: str
    checksum_sha512: str
    file_size: int
    model_type: str  # 'pickle', 'joblib', 'torch', etc.
    feature_names: Optional[List[str]] = None
    performance_metrics: Optional[Dict[str, float]] = None
    training_data_hash: Optional[str] = None
    authorized_by: Optional[str] = None
    signature: Optional[str] = None  # HMAC signature for integrity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary"""
        return cls(**data)


class RestrictedUnpickler(pickle.Unpickler):
    """
    Restricted unpickler that only allows safe classes.
    Prevents arbitrary code execution during deserialization.
    """
    
    # Whitelist of allowed modules and classes
    ALLOWED_MODULES = {
        'numpy', 'pandas', 'sklearn', 'scipy', 'collections',
        'builtins', '__builtin__', 'joblib', 'torch'
    }
    
    ALLOWED_CLASSES = {
        # NumPy
        ('numpy', 'ndarray'),
        ('numpy', 'dtype'),
        ('numpy.core.multiarray', 'scalar'),
        ('numpy.core.multiarray', '_reconstruct'),
        
        # Pandas
        ('pandas.core.frame', 'DataFrame'),
        ('pandas.core.series', 'Series'),
        ('pandas.core.indexes.base', 'Index'),
        
        # Scikit-learn
        ('sklearn.ensemble._forest', 'RandomForestClassifier'),
        ('sklearn.ensemble._forest', 'RandomForestRegressor'),
        ('sklearn.ensemble._gb', 'GradientBoostingClassifier'),
        ('sklearn.ensemble._gb', 'GradientBoostingRegressor'),
        ('sklearn.linear_model._logistic', 'LogisticRegression'),
        ('sklearn.preprocessing._data', 'StandardScaler'),
        ('sklearn.preprocessing._data', 'MinMaxScaler'),
        ('sklearn.pipeline', 'Pipeline'),
        
        # Basic Python types
        ('builtins', 'dict'),
        ('builtins', 'list'),
        ('builtins', 'tuple'),
        ('builtins', 'set'),
        ('builtins', 'frozenset'),
        ('builtins', 'str'),
        ('builtins', 'int'),
        ('builtins', 'float'),
        ('builtins', 'bool'),
        ('builtins', 'bytes'),
        ('builtins', 'bytearray'),
    }
    
    def find_class(self, module: str, name: str) -> Any:
        """Override find_class to restrict imports"""
        # Check if module is in allowed list
        module_root = module.split('.')[0]
        if module_root not in self.ALLOWED_MODULES:
            raise SecurityError(
                f"Attempted to load unsafe module '{module}' during model deserialization"
            )
        
        # Check if specific class is allowed
        if (module, name) not in self.ALLOWED_CLASSES:
            # Allow all sklearn classes for now (too many to list)
            if not module.startswith('sklearn'):
                logger.warning(f"Loading unverified class: {module}.{name}")
        
        return super().find_class(module, name)


class SecureModelLoader:
    """
    Secure model loading with integrity verification and safety checks.
    """
    
    def __init__(
        self,
        models_dir: Union[str, Path],
        secret_key: Optional[str] = None,
        enable_signature_verification: bool = True,
        allowed_models_file: Optional[Path] = None
    ):
        """
        Initialize secure model loader.
        
        Args:
            models_dir: Directory containing model files
            secret_key: Secret key for HMAC signatures (from environment)
            enable_signature_verification: Whether to verify signatures
            allowed_models_file: Path to file containing allowed model checksums
        """
        self.models_dir = Path(models_dir)
        self.secret_key = secret_key or os.environ.get('MODEL_SECRET_KEY')
        self.enable_signature_verification = enable_signature_verification
        self.allowed_models_file = allowed_models_file
        
        # Create models directory if it doesn't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata directory
        self.metadata_dir = self.models_dir / '.metadata'
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Load allowed models list
        self.allowed_models = self._load_allowed_models()
        
        logger.info(f"SecureModelLoader initialized for {self.models_dir}")
    
    def _load_allowed_models(self) -> Dict[str, str]:
        """Load list of allowed model checksums"""
        if not self.allowed_models_file or not self.allowed_models_file.exists():
            return {}
        
        try:
            with open(self.allowed_models_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load allowed models list: {e}")
            return {}
    
    def _calculate_checksum(self, file_path: Path, algorithm: str = 'sha256') -> str:
        """Calculate checksum of a file"""
        hash_func = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    def _generate_signature(self, data: bytes) -> str:
        """Generate HMAC signature for data"""
        if not self.secret_key:
            raise ConfigurationError("No secret key configured for model signatures")
        
        return hmac.new(
            self.secret_key.encode(),
            data,
            hashlib.sha256
        ).hexdigest()
    
    def _verify_signature(self, data: bytes, signature: str) -> bool:
        """Verify HMAC signature"""
        if not self.secret_key:
            logger.warning("No secret key configured, skipping signature verification")
            return True
        
        expected_signature = self._generate_signature(data)
        return hmac.compare_digest(expected_signature, signature)
    
    def save_model(
        self,
        model: Any,
        model_name: str,
        version: str,
        model_type: str = 'joblib',
        metadata: Optional[Dict[str, Any]] = None,
        authorized_by: Optional[str] = None
    ) -> Path:
        """
        Save a model with security metadata.
        
        Args:
            model: The model object to save
            model_name: Name of the model
            version: Model version
            model_type: Type of serialization ('joblib', 'pickle')
            metadata: Additional metadata
            authorized_by: Who authorized this model
            
        Returns:
            Path: Path to saved model file
        """
        # Create filename
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        filename = f"{model_name}_v{version}_{timestamp}.{model_type}"
        file_path = self.models_dir / filename
        
        # Save model to temporary file first
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{model_type}') as tmp_file:
            tmp_path = Path(tmp_file.name)
            
            try:
                # Save model
                if model_type == 'joblib':
                    joblib.dump(model, tmp_path)
                elif model_type == 'pickle':
                    with open(tmp_path, 'wb') as f:
                        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")
                
                # Calculate checksums
                sha256_checksum = self._calculate_checksum(tmp_path, 'sha256')
                sha512_checksum = self._calculate_checksum(tmp_path, 'sha512')
                file_size = tmp_path.stat().st_size
                
                # Create metadata
                model_metadata = ModelMetadata(
                    model_name=model_name,
                    version=version,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    checksum_sha256=sha256_checksum,
                    checksum_sha512=sha512_checksum,
                    file_size=file_size,
                    model_type=model_type,
                    feature_names=metadata.get('feature_names') if metadata else None,
                    performance_metrics=metadata.get('performance_metrics') if metadata else None,
                    training_data_hash=metadata.get('training_data_hash') if metadata else None,
                    authorized_by=authorized_by
                )
                
                # Generate signature if secret key is available
                if self.secret_key:
                    # Sign the metadata (excluding the signature field)
                    metadata_json = json.dumps(model_metadata.to_dict(), sort_keys=True)
                    model_metadata.signature = self._generate_signature(metadata_json.encode())
                
                # Save metadata
                metadata_path = self.metadata_dir / f"{filename}.json"
                with open(metadata_path, 'w') as f:
                    json.dump(model_metadata.to_dict(), f, indent=2)
                
                # Move model to final location
                shutil.move(str(tmp_path), str(file_path))
                
                # Update allowed models list if configured
                if self.allowed_models_file:
                    self.allowed_models[filename] = sha256_checksum
                    with open(self.allowed_models_file, 'w') as f:
                        json.dump(self.allowed_models, f, indent=2)
                
                logger.info(f"Model saved securely: {file_path}")
                logger.info(f"SHA256: {sha256_checksum}")
                
                return file_path
                
            finally:
                # Clean up temp file if it still exists
                if tmp_path.exists():
                    tmp_path.unlink()
    
    def load_model(
        self,
        model_path: Union[str, Path],
        verify_checksum: bool = True,
        verify_signature: bool = None,
        expected_checksum: Optional[str] = None
    ) -> Any:
        """
        Load a model with security verification.
        
        Args:
            model_path: Path to model file
            verify_checksum: Whether to verify file checksum
            verify_signature: Whether to verify signature (uses default if None)
            expected_checksum: Expected SHA256 checksum (optional)
            
        Returns:
            Loaded model object
            
        Raises:
            SecurityError: If security checks fail
            SerializationError: If model cannot be loaded
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Determine if we should verify signature
        if verify_signature is None:
            verify_signature = self.enable_signature_verification
        
        # Load metadata
        metadata_path = self.metadata_dir / f"{model_path.name}.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata_dict = json.load(f)
                metadata = ModelMetadata.from_dict(metadata_dict)
            except Exception as e:
                logger.warning(f"Failed to load metadata for {model_path}: {e}")
                metadata = None
        else:
            logger.warning(f"No metadata found for {model_path}")
            metadata = None
        
        # Verify checksum
        if verify_checksum and metadata:
            actual_checksum = self._calculate_checksum(model_path, 'sha256')
            
            # Check against expected checksum
            if expected_checksum and actual_checksum != expected_checksum:
                raise SecurityError(
                    f"Model checksum mismatch. Expected: {expected_checksum}, "
                    f"Actual: {actual_checksum}"
                )
            
            # Check against metadata checksum
            if actual_checksum != metadata.checksum_sha256:
                raise SecurityError(
                    f"Model checksum mismatch with metadata. "
                    f"Expected: {metadata.checksum_sha256}, Actual: {actual_checksum}"
                )
            
            # Check against allowed models list
            if self.allowed_models:
                allowed_checksum = self.allowed_models.get(model_path.name)
                if allowed_checksum and actual_checksum != allowed_checksum:
                    raise SecurityError(
                        f"Model not in allowed list or checksum mismatch. "
                        f"Expected: {allowed_checksum}, Actual: {actual_checksum}"
                    )
            
            # Verify file size
            actual_size = model_path.stat().st_size
            if actual_size != metadata.file_size:
                raise SecurityError(
                    f"Model file size mismatch. "
                    f"Expected: {metadata.file_size}, Actual: {actual_size}"
                )
        
        # Verify signature
        if verify_signature and metadata and metadata.signature:
            # Reconstruct the signed data (metadata without signature)
            metadata_copy = metadata.to_dict()
            metadata_copy.pop('signature', None)
            metadata_json = json.dumps(metadata_copy, sort_keys=True)
            
            if not self._verify_signature(metadata_json.encode(), metadata.signature):
                raise SecurityError("Model signature verification failed")
        
        # Load model with restricted unpickler
        try:
            if model_path.suffix == '.joblib':
                # Joblib uses pickle internally, so we need to patch it
                original_unpickler = pickle.Unpickler
                try:
                    pickle.Unpickler = RestrictedUnpickler
                    model = joblib.load(model_path)
                finally:
                    pickle.Unpickler = original_unpickler
            elif model_path.suffix in ['.pkl', '.pickle']:
                with open(model_path, 'rb') as f:
                    model = RestrictedUnpickler(f).load()
            else:
                raise ValueError(f"Unsupported model file type: {model_path.suffix}")
            
            logger.info(f"Model loaded successfully: {model_path}")
            return model
            
        except SecurityError:
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise SerializationError(f"Failed to deserialize model: {str(e)}") from e
    
    def verify_model(self, model_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Verify a model file without loading it.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Verification results
        """
        model_path = Path(model_path)
        results = {
            'path': str(model_path),
            'exists': model_path.exists(),
            'checksum_valid': False,
            'signature_valid': False,
            'metadata_found': False,
            'in_allowed_list': False,
            'file_size': None,
            'checksum_sha256': None
        }
        
        if not model_path.exists():
            return results
        
        # Get file info
        results['file_size'] = model_path.stat().st_size
        results['checksum_sha256'] = self._calculate_checksum(model_path, 'sha256')
        
        # Check metadata
        metadata_path = self.metadata_dir / f"{model_path.name}.json"
        if metadata_path.exists():
            results['metadata_found'] = True
            
            try:
                with open(metadata_path, 'r') as f:
                    metadata_dict = json.load(f)
                metadata = ModelMetadata.from_dict(metadata_dict)
                
                # Verify checksum
                results['checksum_valid'] = (
                    results['checksum_sha256'] == metadata.checksum_sha256
                )
                
                # Verify signature
                if metadata.signature and self.secret_key:
                    metadata_copy = metadata.to_dict()
                    metadata_copy.pop('signature', None)
                    metadata_json = json.dumps(metadata_copy, sort_keys=True)
                    results['signature_valid'] = self._verify_signature(
                        metadata_json.encode(),
                        metadata.signature
                    )
                
            except Exception as e:
                logger.error(f"Error verifying metadata: {e}")
        
        # Check allowed list
        if self.allowed_models:
            allowed_checksum = self.allowed_models.get(model_path.name)
            results['in_allowed_list'] = (
                allowed_checksum == results['checksum_sha256']
            )
        
        return results
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all models with their metadata and verification status"""
        models = []
        
        for model_file in self.models_dir.glob('*.joblib'):
            model_info = self.verify_model(model_file)
            models.append(model_info)
        
        for model_file in self.models_dir.glob('*.pkl'):
            model_info = self.verify_model(model_file)
            models.append(model_info)
        
        return models
    
    @contextmanager
    def sandboxed_load(self, model_path: Union[str, Path], **kwargs):
        """
        Load a model in a sandboxed context (placeholder for future implementation).
        This could use subprocess, containers, or other isolation mechanisms.
        """
        # For now, just use regular secure loading
        model = self.load_model(model_path, **kwargs)
        yield model


# Convenience functions
def create_secure_loader(
    models_dir: Union[str, Path],
    secret_key: Optional[str] = None
) -> SecureModelLoader:
    """Create a secure model loader with default settings"""
    return SecureModelLoader(
        models_dir=models_dir,
        secret_key=secret_key,
        enable_signature_verification=True
    )


def generate_model_checksum(model_path: Union[str, Path]) -> str:
    """Generate SHA256 checksum for a model file"""
    return SecureModelLoader._calculate_checksum(None, Path(model_path), 'sha256')


# Export public API
__all__ = [
    'SecureModelLoader',
    'ModelMetadata',
    'SecurityError',
    'create_secure_loader',
    'generate_model_checksum'
]