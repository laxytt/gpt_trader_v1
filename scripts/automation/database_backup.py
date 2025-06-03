"""
Database backup automation for GPT Trading System.
"""

import shutil
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import gzip
import json

from config.settings import get_settings

logger = logging.getLogger(__name__)


class DatabaseBackup:
    """Automated database backup with rotation and compression"""
    
    def __init__(self):
        self.settings = get_settings()
        self.db_path = Path(self.settings.database.db_path)
        self.backup_dir = Path("backups/database")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup retention settings
        self.daily_backups_to_keep = 7
        self.weekly_backups_to_keep = 4
        self.monthly_backups_to_keep = 3
    
    async def create_backup(
        self, 
        backup_type: str = "daily",
        compress: bool = True
    ) -> Dict[str, Any]:
        """Create database backup"""
        logger.info(f"Creating {backup_type} database backup")
        
        try:
            # Generate backup filename
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_name = f"trades_db_{backup_type}_{timestamp}"
            
            if compress:
                backup_path = self.backup_dir / f"{backup_name}.db.gz"
            else:
                backup_path = self.backup_dir / f"{backup_name}.db"
            
            # Perform backup
            if compress:
                await self._create_compressed_backup(backup_path)
            else:
                await self._create_standard_backup(backup_path)
            
            # Verify backup
            is_valid = await self._verify_backup(backup_path)
            
            if not is_valid:
                raise Exception("Backup verification failed")
            
            # Clean old backups
            await self._cleanup_old_backups(backup_type)
            
            # Get backup stats
            backup_size = backup_path.stat().st_size / (1024 * 1024)  # MB
            
            result = {
                'success': True,
                'backup_path': str(backup_path),
                'backup_type': backup_type,
                'size_mb': round(backup_size, 2),
                'timestamp': timestamp,
                'compressed': compress
            }
            
            # Save backup metadata
            await self._save_backup_metadata(result)
            
            logger.info(f"Backup completed: {backup_path} ({backup_size:.2f}MB)")
            return result
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _create_standard_backup(self, backup_path: Path):
        """Create standard SQLite backup"""
        # Use SQLite backup API for consistency
        source_conn = sqlite3.connect(str(self.db_path))
        backup_conn = sqlite3.connect(str(backup_path))
        
        try:
            with source_conn:
                source_conn.backup(backup_conn)
            logger.info("Standard backup completed")
        finally:
            source_conn.close()
            backup_conn.close()
    
    async def _create_compressed_backup(self, backup_path: Path):
        """Create compressed backup"""
        # First create temporary uncompressed backup
        temp_backup = backup_path.with_suffix('')
        await self._create_standard_backup(temp_backup)
        
        # Compress the backup
        with open(temp_backup, 'rb') as f_in:
            with gzip.open(backup_path, 'wb', compresslevel=9) as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove temporary file
        temp_backup.unlink()
        logger.info("Compressed backup completed")
    
    async def _verify_backup(self, backup_path: Path) -> bool:
        """Verify backup integrity"""
        try:
            if backup_path.suffix == '.gz':
                # Verify compressed backup
                with gzip.open(backup_path, 'rb') as f:
                    # Try to read first few bytes
                    data = f.read(100)
                    if not data:
                        return False
            else:
                # Verify SQLite backup
                conn = sqlite3.connect(str(backup_path))
                try:
                    # Check if we can query the database
                    cursor = conn.execute("SELECT COUNT(*) FROM sqlite_master")
                    count = cursor.fetchone()[0]
                    return count > 0
                finally:
                    conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False
    
    async def _cleanup_old_backups(self, backup_type: str):
        """Clean up old backups based on retention policy"""
        # Get all backups of this type
        pattern = f"trades_db_{backup_type}_*.db*"
        backups = sorted(self.backup_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
        
        # Determine how many to keep
        if backup_type == "daily":
            to_keep = self.daily_backups_to_keep
        elif backup_type == "weekly":
            to_keep = self.weekly_backups_to_keep
        elif backup_type == "monthly":
            to_keep = self.monthly_backups_to_keep
        else:
            to_keep = 7  # Default
        
        # Remove old backups
        if len(backups) > to_keep:
            for backup in backups[:-to_keep]:
                logger.info(f"Removing old backup: {backup}")
                backup.unlink()
    
    async def _save_backup_metadata(self, backup_info: Dict[str, Any]):
        """Save backup metadata for tracking"""
        metadata_file = self.backup_dir / "backup_metadata.json"
        
        # Load existing metadata
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {'backups': []}
        
        # Add new backup info
        metadata['backups'].append(backup_info)
        
        # Keep only last 100 entries
        metadata['backups'] = metadata['backups'][-100:]
        
        # Save updated metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    async def restore_backup(self, backup_path: str) -> Dict[str, Any]:
        """Restore database from backup"""
        logger.warning(f"Restoring database from {backup_path}")
        
        try:
            backup_file = Path(backup_path)
            
            if not backup_file.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_path}")
            
            # Create safety backup of current database
            safety_backup = self.backup_dir / f"pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            shutil.copy2(self.db_path, safety_backup)
            logger.info(f"Created safety backup: {safety_backup}")
            
            # Restore based on backup type
            if backup_file.suffix == '.gz':
                # Decompress and restore
                with gzip.open(backup_file, 'rb') as f_in:
                    with open(self.db_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                # Direct copy
                shutil.copy2(backup_file, self.db_path)
            
            # Verify restored database
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.execute("SELECT COUNT(*) FROM trades")
                trade_count = cursor.fetchone()[0]
                logger.info(f"Restore completed. Trade count: {trade_count}")
            finally:
                conn.close()
            
            return {
                'success': True,
                'restored_from': backup_path,
                'safety_backup': str(safety_backup)
            }
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_backup_info(self) -> Dict[str, Any]:
        """Get information about existing backups"""
        backups = []
        
        for backup_file in self.backup_dir.glob("trades_db_*.db*"):
            stat = backup_file.stat()
            backups.append({
                'filename': backup_file.name,
                'path': str(backup_file),
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'created': datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                'compressed': backup_file.suffix == '.gz'
            })
        
        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x['created'], reverse=True)
        
        # Calculate total backup size
        total_size_mb = sum(b['size_mb'] for b in backups)
        
        return {
            'backup_count': len(backups),
            'total_size_mb': round(total_size_mb, 2),
            'backups': backups,
            'retention_policy': {
                'daily': self.daily_backups_to_keep,
                'weekly': self.weekly_backups_to_keep,
                'monthly': self.monthly_backups_to_keep
            }
        }


async def main():
    """Run backup standalone"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Database Backup Tool')
    parser.add_argument('--action', choices=['backup', 'restore', 'info'],
                      default='backup', help='Action to perform')
    parser.add_argument('--type', choices=['daily', 'weekly', 'monthly'],
                      default='daily', help='Backup type')
    parser.add_argument('--restore-from', help='Path to backup file for restore')
    parser.add_argument('--no-compress', action='store_true',
                      help='Skip compression')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    
    backup_tool = DatabaseBackup()
    
    if args.action == 'backup':
        result = await backup_tool.create_backup(
            backup_type=args.type,
            compress=not args.no_compress
        )
        
        if result['success']:
            print(f"✅ Backup created successfully!")
            print(f"   Path: {result['backup_path']}")
            print(f"   Size: {result['size_mb']}MB")
        else:
            print(f"❌ Backup failed: {result['error']}")
    
    elif args.action == 'restore':
        if not args.restore_from:
            print("❌ Please specify --restore-from with backup path")
            return
        
        result = await backup_tool.restore_backup(args.restore_from)
        
        if result['success']:
            print(f"✅ Database restored successfully!")
            print(f"   Safety backup: {result['safety_backup']}")
        else:
            print(f"❌ Restore failed: {result['error']}")
    
    elif args.action == 'info':
        info = await backup_tool.get_backup_info()
        
        print(f"📊 Backup Information:")
        print(f"   Total backups: {info['backup_count']}")
        print(f"   Total size: {info['total_size_mb']}MB")
        print(f"\n   Recent backups:")
        
        for backup in info['backups'][:10]:  # Show last 10
            print(f"     - {backup['filename']} ({backup['size_mb']}MB)")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())