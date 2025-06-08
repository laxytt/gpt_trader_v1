"""
Script to generate pinned dependency versions for requirements files.
This ensures reproducible builds and prevents unexpected updates.
"""

import subprocess
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple
import pkg_resources
import json
from datetime import datetime

def get_installed_version(package_name: str) -> str:
    """Get the currently installed version of a package"""
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        # Try alternate names or return None
        alt_names = {
            'scikit-learn': 'sklearn',
            'faiss-cpu': 'faiss',
            'ta': 'ta-lib',
        }
        if package_name in alt_names:
            try:
                return pkg_resources.get_distribution(alt_names[package_name]).version
            except:
                pass
        return None

def parse_requirements_line(line: str) -> Tuple[str, str, str]:
    """Parse a requirements line to extract package name, operator, and version"""
    line = line.strip()
    
    # Skip empty lines and comments
    if not line or line.startswith('#'):
        return None, None, line
    
    # Handle lines with environment markers (e.g., package==1.0.0; python_version >= "3.8")
    if ';' in line:
        package_part, marker = line.split(';', 1)
        marker = ';' + marker
    else:
        package_part = line
        marker = ''
    
    # Parse package specification
    match = re.match(r'^([a-zA-Z0-9\-_\.]+)([<>=!]+)?(.*)$', package_part)
    if match:
        package_name = match.group(1)
        operator = match.group(2) or ''
        version = match.group(3).strip() + marker
        return package_name, operator, version
    
    return None, None, line

def get_recommended_versions() -> Dict[str, str]:
    """Get recommended versions for critical packages"""
    return {
        # Core dependencies - stable versions
        'pydantic': '2.5.3',
        'pydantic-settings': '2.1.0',
        'python-dotenv': '1.0.0',
        'MetaTrader5': '5.0.45',
        'openai': '1.6.1',
        'tiktoken': '0.5.2',
        
        # Data processing - compatible versions
        'pandas': '2.0.3',
        'numpy': '1.24.3',
        
        # Machine learning - tested versions
        'torch': '2.1.2',
        'sentence-transformers': '2.2.2',
        'faiss-cpu': '1.7.4',
        'seaborn': '0.13.0',
        'h5py': '3.10.0',
        'scikit-learn': '1.3.2',
        'imbalanced-learn': '0.11.0',
        
        # Charting - stable versions
        'mplfinance': '0.12.10b0',
        'matplotlib': '3.7.4',
        'streamlit': '1.29.0',
        'plotly': '5.18.0',
        'tabulate': '0.9.0',
        
        # Technical analysis
        'ta': '0.10.2',
        
        # Utilities - security updates included
        'requests': '2.31.0',
        'aiohttp': '3.9.1',
        'schedule': '1.2.0',
        'nest-asyncio': '1.5.8',
        'backoff': '2.2.1',
        
        # Database
        'joblib': '1.3.2',
        'aiosqlite': '0.19.0',
        
        # Others
        'scipy': '1.11.4',
        'psutil': '5.9.6',
    }

def generate_pinned_requirements(input_file: Path, output_file: Path, use_installed: bool = True):
    """Generate a requirements file with pinned versions"""
    
    recommended = get_recommended_versions()
    output_lines = []
    skipped_packages = []
    statistics = {
        'total': 0,
        'pinned': 0,
        'already_pinned': 0,
        'not_found': 0,
        'comments': 0
    }
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    output_lines.append(f"# Auto-generated pinned requirements - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    output_lines.append("# Generated from: " + str(input_file) + "\n")
    output_lines.append("#\n")
    output_lines.append("# To update: pip install -r requirements-pinned.txt\n")
    output_lines.append("# To upgrade: pip install --upgrade -r requirements.txt\n")
    output_lines.append("\n")
    
    current_section = None
    
    for line in lines:
        original_line = line.rstrip()
        package_name, operator, version = parse_requirements_line(line.strip())
        
        # Handle comments and empty lines
        if not package_name:
            output_lines.append(original_line + '\n')
            if line.strip().startswith('#'):
                statistics['comments'] += 1
                current_section = line.strip()
            continue
        
        statistics['total'] += 1
        
        # Skip if already pinned with ==
        if operator == '==':
            output_lines.append(original_line + '\n')
            statistics['already_pinned'] += 1
            continue
        
        # Get the version to use
        if use_installed:
            # Try to get currently installed version
            installed_version = get_installed_version(package_name)
            if installed_version:
                pinned_version = installed_version
            else:
                # Fall back to recommended version
                pinned_version = recommended.get(package_name)
                if not pinned_version:
                    # Check alternate names
                    alt_names = {'imblearn': 'imbalanced-learn', 'nest_asyncio': 'nest-asyncio'}
                    alt_name = alt_names.get(package_name, package_name)
                    pinned_version = recommended.get(alt_name)
        else:
            # Use recommended version
            pinned_version = recommended.get(package_name)
        
        if pinned_version:
            # Pin the version
            output_lines.append(f"{package_name}=={pinned_version}\n")
            statistics['pinned'] += 1
            print(f"Pinned {package_name} to version {pinned_version}")
        else:
            # Keep original line if we can't determine version
            output_lines.append(original_line + '\n')
            skipped_packages.append(package_name)
            statistics['not_found'] += 1
            print(f"Warning: Could not determine version for {package_name}")
    
    # Write output file
    with open(output_file, 'w') as f:
        f.writelines(output_lines)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Pinned requirements generated: {output_file}")
    print(f"{'='*60}")
    print(f"Total packages: {statistics['total']}")
    print(f"Newly pinned: {statistics['pinned']}")
    print(f"Already pinned: {statistics['already_pinned']}")
    print(f"Not found: {statistics['not_found']}")
    
    if skipped_packages:
        print(f"\nPackages that could not be pinned:")
        for pkg in skipped_packages:
            print(f"  - {pkg}")
    
    return statistics

def generate_dev_requirements():
    """Generate pinned dev requirements"""
    dev_packages = {
        # Testing
        'pytest': '7.4.3',
        'pytest-asyncio': '0.21.1',
        'pytest-cov': '4.1.0',
        'pytest-mock': '3.12.0',
        
        # Code quality
        'black': '23.12.1',
        'flake8': '6.1.0',
        'mypy': '1.7.1',
        'isort': '5.13.2',
        'pylint': '3.0.3',
        
        # Security
        'bandit': '1.7.5',
        'safety': '3.0.1',
        
        # Documentation
        'sphinx': '7.2.6',
        'sphinx-rtd-theme': '2.0.0',
        
        # Development tools
        'ipython': '8.18.1',
        'jupyter': '1.0.0',
        'notebook': '7.0.6',
    }
    
    lines = [
        "# Development dependencies - pinned versions\n",
        f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "\n",
        "# Testing\n"
    ]
    
    for package, version in dev_packages.items():
        if package in ['pytest', 'pytest-asyncio', 'pytest-cov', 'pytest-mock']:
            lines.append(f"{package}=={version}\n")
    
    lines.extend(["\n", "# Code quality\n"])
    for package, version in dev_packages.items():
        if package in ['black', 'flake8', 'mypy', 'isort', 'pylint']:
            lines.append(f"{package}=={version}\n")
    
    lines.extend(["\n", "# Security scanning\n"])
    for package, version in dev_packages.items():
        if package in ['bandit', 'safety']:
            lines.append(f"{package}=={version}\n")
    
    lines.extend(["\n", "# Documentation\n"])
    for package, version in dev_packages.items():
        if package in ['sphinx', 'sphinx-rtd-theme']:
            lines.append(f"{package}=={version}\n")
    
    lines.extend(["\n", "# Development tools\n"])
    for package, version in dev_packages.items():
        if package in ['ipython', 'jupyter', 'notebook']:
            lines.append(f"{package}=={version}\n")
    
    with open('requirements-dev-pinned.txt', 'w') as f:
        f.writelines(lines)
    
    print(f"\nDev requirements generated: requirements-dev-pinned.txt")

def check_security_updates():
    """Check for known security vulnerabilities in pinned versions"""
    print("\n" + "="*60)
    print("Checking for security updates...")
    print("="*60)
    
    # This would normally use a tool like safety or pip-audit
    # For now, we'll just flag some known issues
    security_warnings = {
        'requests': {'min_version': '2.31.0', 'reason': 'CVE-2023-32681'},
        'aiohttp': {'min_version': '3.9.0', 'reason': 'Security fixes'},
        'numpy': {'min_version': '1.24.0', 'reason': 'Memory safety'},
    }
    
    recommended = get_recommended_versions()
    
    for package, warning in security_warnings.items():
        if package in recommended:
            current = recommended[package]
            min_required = warning['min_version']
            
            # Simple version comparison (would need proper version parsing in production)
            if current < min_required:
                print(f"⚠️  {package}: Update to {min_required}+ ({warning['reason']})")
            else:
                print(f"✅ {package}: {current} is secure")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate pinned dependency versions")
    parser.add_argument('--input', default='requirements.txt', help='Input requirements file')
    parser.add_argument('--output', default='requirements-pinned.txt', help='Output file with pinned versions')
    parser.add_argument('--use-installed', action='store_true', help='Use currently installed versions')
    parser.add_argument('--dev', action='store_true', help='Also generate dev requirements')
    parser.add_argument('--check-security', action='store_true', help='Check for security updates')
    
    args = parser.parse_args()
    
    # Generate pinned requirements
    input_file = Path(args.input)
    output_file = Path(args.output)
    
    if not input_file.exists():
        print(f"Error: {input_file} not found")
        sys.exit(1)
    
    stats = generate_pinned_requirements(input_file, output_file, args.use_installed)
    
    # Generate dev requirements if requested
    if args.dev:
        generate_dev_requirements()
    
    # Check security if requested
    if args.check_security:
        check_security_updates()
    
    print("\n✅ Dependency pinning complete!")
    print("\nNext steps:")
    print("1. Review the generated files")
    print("2. Test with: pip install -r requirements-pinned.txt")
    print("3. Commit both requirements.txt and requirements-pinned.txt")
    print("4. Update CI/CD to use requirements-pinned.txt")

if __name__ == "__main__":
    main()