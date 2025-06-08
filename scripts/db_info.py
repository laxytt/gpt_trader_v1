#!/usr/bin/env python
"""
Simple database information script without dependencies.

Security measures implemented:
1. Table names are fetched from sqlite_master (trusted source)
2. Table names are validated to be alphanumeric (with underscores)
3. Table names are checked against a whitelist of expected tables
4. No user input is accepted - this is a read-only utility script

Note: SQLite doesn't support parameterized queries for table/column names,
so we must use string formatting, but with proper validation.
"""

import sqlite3
import os
from pathlib import Path


def format_size(bytes):
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"


def main():
    """Show database information"""
    
    # Whitelist of expected tables for additional security
    EXPECTED_TABLES = {
        'trades', 'signals', 'memory_cases', 'ml_predictions', 
        'migrations', 'model_registry', 'gpt_requests', 'backtest_trades',
        'backtest_signals', 'marketaux_cache'
    }
    
    # Determine database path
    script_dir = Path(__file__).parent.parent
    db_path = script_dir / "data" / "trades.db"
    
    print(f"\n{'='*70}")
    print("GPT Trading System - Database Information")
    print(f"{'='*70}\n")
    
    print(f"Database path: {db_path}")
    
    if not db_path.exists():
        print("✗ Database file not found!")
        return 1
    
    # Show file size
    size = os.path.getsize(db_path)
    print(f"Database size: {format_size(size)}")
    
    try:
        with sqlite3.connect(str(db_path)) as conn:
            # Get all tables
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)
            
            tables = [row[0] for row in cursor.fetchall()]
            
            print(f"\nFound {len(tables)} tables:")
            print("-" * 70)
            
            for table_name in tables:
                # Validate table name (already from sqlite_master, but double-check)
                if not table_name.replace('_', '').isalnum():
                    print(f"\nSkipping table with invalid name: {table_name}")
                    continue
                
                # Additional security: warn if table not in expected list
                if table_name not in EXPECTED_TABLES:
                    print(f"\n⚠️  Warning: Unexpected table '{table_name}' found")
                
                # Get row count - using parameterized query where possible
                # For table names, we can't use parameters, but we've validated above
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                
                # Get column info - PRAGMA is safe with validated table names
                cursor = conn.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                print(f"\n{table_name}: {row_count} rows, {len(columns)} columns")
                
                # Show columns
                for col in columns[:8]:  # Show first 8 columns
                    col_name = col[1]
                    col_type = col[2]
                    is_pk = " (PK)" if col[5] else ""
                    print(f"  - {col_name}: {col_type}{is_pk}")
                
                if len(columns) > 8:
                    print(f"  ... and {len(columns) - 8} more columns")
            
            # Check for ml_predictions table specifically
            print("\n" + "="*70)
            if 'ml_predictions' in tables:
                print("✓ ml_predictions table EXISTS")
                
                # Show ml_predictions details
                cursor = conn.execute("SELECT COUNT(*) FROM ml_predictions")
                count = cursor.fetchone()[0]
                print(f"  Records: {count}")
                
                if count > 0:
                    # Show sample data
                    cursor = conn.execute("""
                        SELECT symbol, predicted_signal, ml_confidence, was_correct, model_version
                        FROM ml_predictions
                        ORDER BY created_at DESC
                        LIMIT 5
                    """)
                    
                    print("\n  Recent predictions:")
                    for row in cursor:
                        symbol, signal, conf, correct, version = row
                        correct_str = "✓" if correct else "✗" if correct is not None else "?"
                        print(f"    {symbol}: {signal} (conf: {conf:.2f}, {correct_str}, v{version})")
            else:
                print("✗ ml_predictions table DOES NOT EXIST")
                print("  Run: python scripts/migrate_ml_predictions.py")
            
            # Check migrations table
            if 'migrations' in tables:
                cursor = conn.execute("SELECT MAX(version) FROM migrations")
                version = cursor.fetchone()[0]
                print(f"\nDatabase version: {version}")
            
        print("\n" + "="*70 + "\n")
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())