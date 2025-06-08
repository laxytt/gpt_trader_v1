#!/usr/bin/env python3
"""
Launch script for the Comprehensive Trading Dashboard with torch compatibility fix
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the comprehensive trading dashboard with torch compatibility fixes"""
    print("🚀 Launching Comprehensive Trading Dashboard (with compatibility fixes)...")
    print("=" * 50)
    
    # Set environment variables to handle torch/streamlit compatibility
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # Get the dashboard script path
    dashboard_script = Path(__file__).parent / "comprehensive_trading_dashboard.py"
    
    # Create a wrapper script that imports torch before streamlit
    wrapper_content = """
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# Pre-import potential problem libraries
try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass  # torch not installed, which is fine

try:
    import faiss
except ImportError:
    pass  # faiss not installed, which is fine

# Now run the actual dashboard
exec(open(r'{}').read())
""".format(str(dashboard_script).replace('\\', '\\\\'))
    
    # Create temporary wrapper file
    wrapper_path = Path(__file__).parent / "_temp_dashboard_wrapper.py"
    wrapper_path.write_text(wrapper_content)
    
    try:
        # Launch with streamlit using the wrapper
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(wrapper_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n✅ Dashboard closed successfully")
    except Exception as e:
        print(f"\n❌ Error launching dashboard: {e}")
        print("\n💡 Try using the simple dashboard instead:")
        print("   python scripts/run_simple_dashboard.py")
    finally:
        # Clean up wrapper file
        if wrapper_path.exists():
            wrapper_path.unlink()

if __name__ == "__main__":
    main()