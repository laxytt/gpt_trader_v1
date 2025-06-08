"""
Script to migrate existing ML models to secure format with checksums and signatures.
This script will:
1. Load existing models
2. Verify they are safe
3. Save them with security metadata
4. Generate an allowed models list
"""

import os
import sys
import json
import pickle
import joblib
import hashlib
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.ml.secure_model_loader import SecureModelLoader, ModelMetadata
from core.ml.feature_engineering import FeatureEngineer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelMigrator:
    """Migrates existing models to secure format"""
    
    def __init__(self, models_dir: str, secret_key: str):
        self.models_dir = Path(models_dir)
        self.secure_loader = SecureModelLoader(
            models_dir=self.models_dir,
            secret_key=secret_key,
            enable_signature_verification=True
        )
        self.migration_report: List[Dict[str, Any]] = []
    
    def migrate_all_models(self, backup: bool = True) -> Dict[str, Any]:
        """
        Migrate all models in the directory to secure format.
        
        Args:
            backup: Whether to backup original models
            
        Returns:
            Migration summary
        """
        # Create backup directory if requested
        if backup:
            backup_dir = self.models_dir / 'backup_unsecure'
            backup_dir.mkdir(exist_ok=True)
        
        # Find all model files
        model_files = list(self.models_dir.glob("*.pkl")) + list(self.models_dir.glob("*.joblib"))
        
        # Skip already secure models (those with metadata)
        model_files = [
            f for f in model_files 
            if not (self.models_dir / '.metadata' / f"{f.name}.json").exists()
        ]
        
        logger.info(f"Found {len(model_files)} models to migrate")
        
        successful = 0
        failed = 0
        
        for model_file in model_files:
            try:
                logger.info(f"Migrating {model_file.name}...")
                
                # Load the model (unsafe for now)
                if model_file.suffix == '.pkl':
                    with open(model_file, 'rb') as f:
                        model_data = pickle.load(f)
                    model_type = 'pickle'
                else:
                    model_data = joblib.load(model_file)
                    model_type = 'joblib'
                
                # Validate model structure
                if isinstance(model_data, dict):
                    # It's a model package
                    if 'pipeline' not in model_data:
                        logger.warning(f"Skipping {model_file.name}: Not a valid model package")
                        continue
                    
                    # Extract metadata
                    metadata = self._extract_metadata(model_data)
                    model_name = metadata['model_name']
                    version = metadata['version']
                else:
                    # It's a raw model
                    model_name = model_file.stem
                    version = "1.0.0"
                    metadata = {'model_name': model_name}
                
                # Backup original if requested
                if backup:
                    backup_path = backup_dir / model_file.name
                    import shutil
                    shutil.copy2(model_file, backup_path)
                    logger.info(f"Backed up to {backup_path}")
                
                # Save in secure format
                new_path = self.secure_loader.save_model(
                    model=model_data,
                    model_name=model_name,
                    version=version,
                    model_type=model_type,
                    metadata=metadata,
                    authorized_by="migration_script"
                )
                
                # Remove original file
                model_file.unlink()
                
                # Record migration
                self.migration_report.append({
                    'original_file': model_file.name,
                    'new_file': new_path.name,
                    'model_name': model_name,
                    'version': version,
                    'status': 'success'
                })
                
                successful += 1
                logger.info(f"Successfully migrated {model_file.name} -> {new_path.name}")
                
            except Exception as e:
                logger.error(f"Failed to migrate {model_file.name}: {e}")
                self.migration_report.append({
                    'original_file': model_file.name,
                    'status': 'failed',
                    'error': str(e)
                })
                failed += 1
        
        # Generate allowed models list
        allowed_models_file = self.models_dir / 'allowed_models.json'
        self._generate_allowed_models_list(allowed_models_file)
        
        # Save migration report
        report_file = self.models_dir / f'migration_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w') as f:
            json.dump(self.migration_report, f, indent=2)
        
        summary = {
            'total_models': len(model_files),
            'successful': successful,
            'failed': failed,
            'report_file': str(report_file),
            'allowed_models_file': str(allowed_models_file)
        }
        
        logger.info(f"Migration complete: {successful} successful, {failed} failed")
        return summary
    
    def _extract_metadata(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from model package"""
        metadata = model_data.get('metadata', {})
        
        # Try to determine model name from metadata or features
        if 'symbol' in metadata:
            model_name = f"{metadata['symbol']}_pattern_trader"
        elif 'feature_names' in model_data:
            # Try to infer from feature names
            model_name = "pattern_trader"
        else:
            model_name = "ml_model"
        
        # Extract version
        version = metadata.get('version', '1.0.0')
        
        # Extract performance metrics
        performance = model_data.get('performance', {})
        
        return {
            'model_name': model_name,
            'version': version,
            'feature_names': model_data.get('feature_names', []),
            'performance_metrics': performance,
            'training_date': metadata.get('training_date'),
            'training_data_hash': metadata.get('data_hash')
        }
    
    def _generate_allowed_models_list(self, output_file: Path):
        """Generate allowed models list with checksums"""
        allowed_models = {}
        
        # Get all secure models
        metadata_dir = self.models_dir / '.metadata'
        if metadata_dir.exists():
            for metadata_file in metadata_dir.glob("*.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    model_name = metadata_file.stem
                    allowed_models[model_name] = metadata['checksum_sha256']
                    
                except Exception as e:
                    logger.error(f"Error reading metadata {metadata_file}: {e}")
        
        # Save allowed models list
        with open(output_file, 'w') as f:
            json.dump(allowed_models, f, indent=2)
        
        logger.info(f"Generated allowed models list with {len(allowed_models)} models")
    
    def verify_migration(self) -> List[Dict[str, Any]]:
        """Verify all migrated models"""
        verification_results = []
        
        # Verify all models using secure loader
        all_models = self.secure_loader.list_models()
        
        for model_info in all_models:
            verification_results.append({
                'model': model_info['path'],
                'checksum_valid': model_info['checksum_valid'],
                'signature_valid': model_info['signature_valid'],
                'metadata_found': model_info['metadata_found'],
                'status': 'secure' if all([
                    model_info['checksum_valid'],
                    model_info['metadata_found']
                ]) else 'insecure'
            })
        
        return verification_results


def main():
    """Main migration function"""
    parser = argparse.ArgumentParser(
        description="Migrate ML models to secure format with checksums and signatures"
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models',
        help='Directory containing models to migrate'
    )
    parser.add_argument(
        '--secret-key',
        type=str,
        help='Secret key for model signatures (or set MODEL_SECRET_KEY env var)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not backup original models'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing models without migration'
    )
    
    args = parser.parse_args()
    
    # Get secret key
    secret_key = args.secret_key or os.environ.get('MODEL_SECRET_KEY')
    if not secret_key:
        logger.error("Secret key required. Set MODEL_SECRET_KEY environment variable or use --secret-key")
        sys.exit(1)
    
    # Initialize migrator
    migrator = ModelMigrator(args.models_dir, secret_key)
    
    if args.verify_only:
        # Just verify existing models
        logger.info("Verifying models...")
        results = migrator.verify_migration()
        
        secure_count = sum(1 for r in results if r['status'] == 'secure')
        insecure_count = len(results) - secure_count
        
        print(f"\nVerification Results:")
        print(f"Total models: {len(results)}")
        print(f"Secure models: {secure_count}")
        print(f"Insecure models: {insecure_count}")
        
        if insecure_count > 0:
            print("\nInsecure models:")
            for r in results:
                if r['status'] == 'insecure':
                    print(f"  - {r['model']}")
    else:
        # Perform migration
        logger.info(f"Starting model migration in {args.models_dir}")
        summary = migrator.migrate_all_models(backup=not args.no_backup)
        
        print(f"\nMigration Summary:")
        print(f"Total models: {summary['total_models']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Report saved to: {summary['report_file']}")
        print(f"Allowed models list: {summary['allowed_models_file']}")
        
        # Verify migration
        print("\nVerifying migrated models...")
        verification = migrator.verify_migration()
        secure_count = sum(1 for r in verification if r['status'] == 'secure')
        print(f"Verification complete: {secure_count}/{len(verification)} models are secure")


if __name__ == "__main__":
    main()