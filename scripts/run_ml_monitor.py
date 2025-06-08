#!/usr/bin/env python3
"""
Simple launcher for ML Performance Monitor
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.ml_performance_monitor import main

if __name__ == "__main__":
    # Run with all outputs by default
    if len(sys.argv) == 1:
        sys.argv.extend(['--terminal', '--html', '--plot'])
    
    main()