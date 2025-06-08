#!/usr/bin/env python3
"""
Graceful stop utility for Windows
Creates a stop signal file that the trading system checks
"""

import os
from pathlib import Path

STOP_FILE = Path("STOP_TRADING")

def request_stop():
    """Create stop signal file"""
    print("🛑 Requesting graceful shutdown...")
    STOP_FILE.touch()
    print(f"✅ Stop signal created: {STOP_FILE.absolute()}")
    print("The trading system will stop after completing the current operation.")

def clear_stop():
    """Remove stop signal file"""
    if STOP_FILE.exists():
        STOP_FILE.unlink()
        print("✅ Stop signal cleared")
    else:
        print("No stop signal to clear")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "clear":
        clear_stop()
    else:
        request_stop()