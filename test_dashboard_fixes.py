#!/usr/bin/env python3
"""
Test script to verify dashboard fixes
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from core.infrastructure.database.repositories import SignalRepository
from core.infrastructure.gpt.request_logger import get_request_logger
from config.settings import get_settings

def test_signal_repository():
    """Test that get_all_signals method exists and works"""
    print("Testing SignalRepository.get_all_signals()...")
    
    settings = get_settings()
    repo = SignalRepository(settings.database.db_path)
    
    try:
        signals = repo.get_all_signals(limit=5)
        print(f"✓ Successfully retrieved {len(signals)} signals")
        
        if signals:
            print(f"  First signal: {signals[0].get('symbol')} - {signals[0].get('direction')}")
            print(f"  Has analysis: {'analysis' in signals[0]}")
            print(f"  Has confidence: {'confidence' in signals[0]}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    return True

def test_request_logger():
    """Test that get_recent_requests works with hours_back parameter"""
    print("\nTesting GPTRequestLogger.get_recent_requests()...")
    
    logger = get_request_logger()
    
    try:
        # Test with hours_back parameter
        requests = logger.get_recent_requests(limit=10, hours_back=2)
        print(f"✓ Successfully retrieved {len(requests)} requests from last 2 hours")
        
        if requests:
            first_req = requests[0]
            print(f"  First request: {first_req.get('agent_type')} at {first_req.get('timestamp')}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    return True

def test_timestamp_parsing():
    """Test timestamp parsing logic"""
    print("\nTesting timestamp parsing...")
    
    from datetime import datetime, timezone as tz
    
    test_timestamps = [
        "2024-12-06T15:00:17.123456",
        "2024-12-06T15:00:17Z",
        "2024-12-06 15:00:17",
        "2024-12-06T15:00:17+00:00"
    ]
    
    for ts_str in test_timestamps:
        try:
            # Parse ISO timestamp
            if 'T' in ts_str:
                timestamp = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            else:
                timestamp = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
            
            # Convert to local time for display
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=tz.utc)
            local_ts = timestamp.astimezone()
            time_str = local_ts.strftime('%H:%M:%S')
            
            print(f"✓ {ts_str} -> {time_str}")
        except Exception as e:
            print(f"✗ Failed to parse {ts_str}: {e}")

def main():
    """Run all tests"""
    print("Dashboard Fixes Test")
    print("=" * 50)
    
    all_passed = True
    
    if not test_signal_repository():
        all_passed = False
    
    if not test_request_logger():
        all_passed = False
    
    test_timestamp_parsing()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All tests passed! Dashboard should work correctly.")
    else:
        print("✗ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()