#!/usr/bin/env python3
"""
Launch script for the Comprehensive Trading Dashboard
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the comprehensive trading dashboard"""
    print("🚀 Launching Comprehensive Trading Dashboard...")
    print("=" * 50)
    
    # Get the dashboard script path
    dashboard_script = Path(__file__).parent / "comprehensive_trading_dashboard.py"
    
    # Launch with streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_script),
            "--server.port", "8501",
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