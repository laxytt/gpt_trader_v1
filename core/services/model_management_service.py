"""
Model Management Service for ML model lifecycle management.
Handles model loading, saving, deployment, versioning, and rollback.
"""

import logging
import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import joblib
import numpy as np

from core.domain.exceptions import ServiceError, ErrorContext
from core.infrastructure.database.repositories import BaseRepository
from core.ml.model_evaluation import ModelEvaluator
from core.utils.error_handling import with_error_recovery

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a trained model"""
    model_id: str
    model_type: str
    version: str
    created_at: datetime
    training_metrics: Dict[str, float]
    feature_names: List[str]
    hyperparameters: Dict[str, Any]
    training_data_info: Dict[str, Any]
    is_active: bool = False
    deployment_history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.deployment_history is None:
            self.deployment_history = []


class ModelRepository(BaseRepository):
    """Repository for model metadata storage"""
    
    def __init__(self, db_path: str):
        super().__init__(db_path)
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """Create model metadata table if it doesn't exist"""
        with self._get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS model_metadata (
                    model_id TEXT PRIMARY KEY,
                    model_type TEXT NOT NULL,
                    version TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    training_metrics TEXT,
                    feature_names TEXT,
                    hyperparameters TEXT,
                    training_data_info TEXT,
                    is_active BOOLEAN DEFAULT FALSE,
                    deployment_history TEXT,
                    model_path TEXT,
                    UNIQUE(model_type, version)
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_active_models 
                ON model_metadata(model_type, is_active)
            ''')
    
    def save_metadata(self, metadata: ModelMetadata, model_path: str):
        """Save model metadata to database"""
        with self._get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO model_metadata 
                (model_id, model_type, version, created_at, training_metrics,
                 feature_names, hyperparameters, training_data_info, is_active,
                 deployment_history, model_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.model_id,
                metadata.model_type,
                metadata.version,
                metadata.created_at,
                json.dumps(metadata.training_metrics),
                json.dumps(metadata.feature_names),
                json.dumps(metadata.hyperparameters),
                json.dumps(metadata.training_data_info),
                metadata.is_active,
                json.dumps(metadata.deployment_history),
                model_path
            ))
    
    def get_active_model(self, model_type: str) -> Optional[Tuple[ModelMetadata, str]]:
        """Get the currently active model of a given type"""
        with self._get_connection() as conn:
            row = conn.execute('''
                SELECT * FROM model_metadata 
                WHERE model_type = ? AND is_active = TRUE
                ORDER BY created_at DESC
                LIMIT 1
            ''', (model_type,)).fetchone()
            
            if row:
                return self._row_to_metadata(row), row['model_path']
            return None
    
    def get_model_by_id(self, model_id: str) -> Optional[Tuple[ModelMetadata, str]]:
        """Get model metadata by ID"""
        with self._get_connection() as conn:
            row = conn.execute('''
                SELECT * FROM model_metadata WHERE model_id = ?
            ''', (model_id,)).fetchone()
            
            if row:
                return self._row_to_metadata(row), row['model_path']
            return None
    
    def set_active_model(self, model_id: str, model_type: str):
        """Set a model as active (deactivating others of same type)"""
        with self._get_connection() as conn:
            # Deactivate all models of this type
            conn.execute('''
                UPDATE model_metadata 
                SET is_active = FALSE 
                WHERE model_type = ?
            ''', (model_type,))
            
            # Activate the specified model
            conn.execute('''
                UPDATE model_metadata 
                SET is_active = TRUE 
                WHERE model_id = ?
            ''', (model_id,))
    
    def _row_to_metadata(self, row) -> ModelMetadata:
        """Convert database row to ModelMetadata"""
        return ModelMetadata(
            model_id=row['model_id'],
            model_type=row['model_type'],
            version=row['version'],
            created_at=datetime.fromisoformat(row['created_at']),
            training_metrics=json.loads(row['training_metrics']),
            feature_names=json.loads(row['feature_names']),
            hyperparameters=json.loads(row['hyperparameters']),
            training_data_info=json.loads(row['training_data_info']),
            is_active=bool(row['is_active']),
            deployment_history=json.loads(row['deployment_history'])
        )


class ModelManagementService:
    """
    Service for managing ML model lifecycle including:
    - Model persistence (save/load)
    - Model versioning
    - Model deployment and rollback
    - Model performance tracking
    """
    
    def __init__(self, models_dir: Path, model_repository: ModelRepository):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.repository = model_repository
        self._loaded_models: Dict[str, Any] = {}
        
        logger.info(f"Model Management Service initialized - Models directory: {self.models_dir}")
    
    @with_error_recovery
    async def save_model(
        self,
        model: Any,
        model_type: str,
        version: str,
        training_metrics: Dict[str, float],
        feature_names: List[str],
        hyperparameters: Dict[str, Any],
        training_data_info: Dict[str, Any]
    ) -> str:
        """
        Save a trained model with metadata.
        
        Returns:
            model_id: Unique identifier for the saved model
        """
        with ErrorContext("Saving model", model_type=model_type, version=version):
            # Generate unique model ID
            model_id = self._generate_model_id(model_type, version)
            
            # Create model directory
            model_dir = self.models_dir / model_type / version
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model file
            model_path = model_dir / f"{model_id}.pkl"
            joblib.dump(model, model_path)
            
            # Create metadata
            metadata = ModelMetadata(
                model_id=model_id,
                model_type=model_type,
                version=version,
                created_at=datetime.now(timezone.utc),
                training_metrics=training_metrics,
                feature_names=feature_names,
                hyperparameters=hyperparameters,
                training_data_info=training_data_info
            )
            
            # Save metadata to database
            self.repository.save_metadata(metadata, str(model_path))
            
            logger.info(f"Model saved successfully - ID: {model_id}, Path: {model_path}")
            return model_id
    
    @with_error_recovery
    async def load_model(self, model_id: str) -> Tuple[Any, ModelMetadata]:
        """
        Load a model by ID.
        
        Returns:
            Tuple of (model, metadata)
        """
        with ErrorContext("Loading model", model_id=model_id):
            # Check cache first
            if model_id in self._loaded_models:
                logger.debug(f"Model {model_id} loaded from cache")
                return self._loaded_models[model_id]
            
            # Get metadata from repository
            result = self.repository.get_model_by_id(model_id)
            if not result:
                raise ServiceError(f"Model not found: {model_id}")
            
            metadata, model_path = result
            
            # Load model from file
            model = joblib.load(model_path)
            
            # Cache the loaded model
            self._loaded_models[model_id] = (model, metadata)
            
            logger.info(f"Model loaded successfully - ID: {model_id}")
            return model, metadata
    
    @with_error_recovery
    async def get_active_model(self, model_type: str) -> Optional[Tuple[Any, ModelMetadata]]:
        """
        Get the currently active model for a given type.
        
        Returns:
            Tuple of (model, metadata) or None if no active model
        """
        with ErrorContext("Getting active model", model_type=model_type):
            result = self.repository.get_active_model(model_type)
            if not result:
                logger.warning(f"No active model found for type: {model_type}")
                return None
            
            metadata, _ = result
            return await self.load_model(metadata.model_id)
    
    @with_error_recovery
    async def deploy_model(self, model_id: str) -> bool:
        """
        Deploy a model (make it active).
        
        Returns:
            True if deployment successful
        """
        with ErrorContext("Deploying model", model_id=model_id):
            # Load model to validate it exists
            model, metadata = await self.load_model(model_id)
            
            # Run validation checks
            if not await self._validate_model_for_deployment(model, metadata):
                raise ServiceError(f"Model {model_id} failed deployment validation")
            
            # Record deployment
            deployment_record = {
                'deployed_at': datetime.now(timezone.utc).isoformat(),
                'deployed_by': 'system',
                'action': 'deploy'
            }
            metadata.deployment_history.append(deployment_record)
            
            # Set as active model
            self.repository.set_active_model(model_id, metadata.model_type)
            
            # Update metadata with deployment history
            self.repository.save_metadata(
                metadata, 
                str(self.models_dir / metadata.model_type / metadata.version / f"{model_id}.pkl")
            )
            
            logger.info(f"Model deployed successfully - ID: {model_id}, Type: {metadata.model_type}")
            return True
    
    @with_error_recovery
    async def rollback_model(self, model_type: str, target_version: Optional[str] = None) -> bool:
        """
        Rollback to a previous model version.
        
        Args:
            model_type: Type of model to rollback
            target_version: Specific version to rollback to (or previous if None)
            
        Returns:
            True if rollback successful
        """
        with ErrorContext("Rolling back model", model_type=model_type):
            # Get deployment history
            current_result = self.repository.get_active_model(model_type)
            if not current_result:
                raise ServiceError(f"No active model to rollback for type: {model_type}")
            
            current_metadata, _ = current_result
            
            # Find target model
            if target_version:
                # Find specific version
                target_model = self._find_model_by_version(model_type, target_version)
                if not target_model:
                    raise ServiceError(f"Target version not found: {target_version}")
                target_id = target_model.model_id
            else:
                # Find previous version
                # This would require additional repository methods to list models
                raise NotImplementedError("Automatic previous version detection not yet implemented")
            
            # Deploy the target model
            success = await self.deploy_model(target_id)
            
            if success:
                # Record rollback in current model's history
                rollback_record = {
                    'rolled_back_at': datetime.now(timezone.utc).isoformat(),
                    'rolled_back_to': target_id,
                    'action': 'rollback'
                }
                current_metadata.deployment_history.append(rollback_record)
                self.repository.save_metadata(
                    current_metadata,
                    str(self.models_dir / current_metadata.model_type / current_metadata.version / f"{current_metadata.model_id}.pkl")
                )
            
            logger.info(f"Model rolled back successfully - Type: {model_type}, Target: {target_id}")
            return success
    
    async def get_model_performance_history(self, model_id: str) -> Dict[str, Any]:
        """Get historical performance metrics for a model"""
        # This would integrate with performance monitoring
        # For now, return training metrics
        _, metadata = await self.load_model(model_id)
        return {
            'model_id': model_id,
            'training_metrics': metadata.training_metrics,
            'deployment_history': metadata.deployment_history
        }
    
    def _generate_model_id(self, model_type: str, version: str) -> str:
        """Generate unique model ID"""
        timestamp = datetime.now(timezone.utc).isoformat()
        content = f"{model_type}:{version}:{timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    async def _validate_model_for_deployment(self, model: Any, metadata: ModelMetadata) -> bool:
        """Validate model is ready for deployment"""
        try:
            # Check model has predict method
            if not hasattr(model, 'predict'):
                logger.error("Model missing predict method")
                return False
            
            # Check feature names are defined
            if not metadata.feature_names:
                logger.error("Model missing feature names")
                return False
            
            # Test prediction with dummy data
            dummy_features = np.zeros((1, len(metadata.feature_names)))
            _ = model.predict(dummy_features)
            
            # Check minimum performance thresholds
            metrics = metadata.training_metrics
            if metrics.get('accuracy', 0) < 0.5:  # Example threshold
                logger.error(f"Model accuracy too low: {metrics.get('accuracy', 0)}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
    
    def _find_model_by_version(self, model_type: str, version: str) -> Optional[ModelMetadata]:
        """Find model by type and version"""
        # This would require additional repository method
        # For now, return None
        return None