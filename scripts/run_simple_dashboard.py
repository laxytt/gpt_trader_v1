#!/usr/bin/env python3
"""
Launch script for the Simple Trading Dashboard (without ML dependencies)
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the simple trading dashboard"""
    print("🚀 Launching Simple Trading Dashboard...")
    print("=" * 50)
    print("ℹ️  This dashboard does not include ML features")
    print("=" * 50)
    
    # Get the dashboard script path
    dashboard_script = Path(__file__).parent / "trading_dashboard_simple.py"
    
    # Launch with streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_script),
            "--server.port", "8502",  # Different port to avoid conflicts
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n✅ Dashboard closed successfully")
    except Exception as e:
        print(f"\n❌ Error launching dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()