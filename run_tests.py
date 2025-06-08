#!/usr/bin/env python3
"""
Test runner for the GPT Trading System.
Runs unit and integration tests with coverage reporting.
"""

import sys
import subprocess
from pathlib import Path

def run_tests():
    """Run all tests with pytest."""
    project_root = Path(__file__).parent
    
    # Base pytest command
    cmd = [
        sys.executable, "-m", "pytest",
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--color=yes",  # Colored output
    ]
    
    # Add coverage if available
    try:
        import coverage
        cmd.extend([
            "--cov=core",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "--cov-config=.coveragerc"
        ])
        print("Running tests with coverage...")
    except ImportError:
        print("Coverage not installed. Running tests without coverage.")
        print("Install with: pip install pytest-cov")
    
    # Add test directory
    test_dir = project_root / "tests"
    if test_dir.exists():
        cmd.append(str(test_dir))
    else:
        print(f"Test directory not found: {test_dir}")
        return 1
    
    # Run tests
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(project_root))
    
    return result.returncode


def run_integration_tests_only():
    """Run only integration tests."""
    project_root = Path(__file__).parent
    
    cmd = [
        sys.executable, "-m", "pytest",
        "-v",
        "--tb=short",
        "--color=yes",
        str(project_root / "tests" / "integration")
    ]
    
    print("Running integration tests only...")
    result = subprocess.run(cmd, cwd=str(project_root))
    
    return result.returncode


def run_specific_test(test_path):
    """Run a specific test file or test case."""
    project_root = Path(__file__).parent
    
    cmd = [
        sys.executable, "-m", "pytest",
        "-v",
        "--tb=short",
        "--color=yes",
        test_path
    ]
    
    print(f"Running specific test: {test_path}")
    result = subprocess.run(cmd, cwd=str(project_root))
    
    return result.returncode


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "integration":
            exit_code = run_integration_tests_only()
        else:
            # Run specific test
            exit_code = run_specific_test(sys.argv[1])
    else:
        # Run all tests
        exit_code = run_tests()
    
    sys.exit(exit_code)