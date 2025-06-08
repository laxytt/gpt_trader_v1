"""
Check dependencies for known security vulnerabilities and updates.
This script helps maintain secure dependencies by checking multiple sources.
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import requests
import pkg_resources

def check_with_pip_audit():
    """Check dependencies using pip-audit"""
    print("🔍 Checking with pip-audit...")
    try:
        result = subprocess.run(
            ["pip-audit", "--format", "json"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            vulnerabilities = json.loads(result.stdout)
            if not vulnerabilities:
                print("✅ No vulnerabilities found by pip-audit")
                return []
            else:
                print(f"⚠️  Found {len(vulnerabilities)} vulnerabilities:")
                for vuln in vulnerabilities:
                    print(f"   - {vuln['name']} {vuln['version']}: {vuln['vuln_id']}")
                return vulnerabilities
        else:
            print("❌ pip-audit check failed")
            return []
    except FileNotFoundError:
        print("ℹ️  pip-audit not installed. Install with: pip install pip-audit")
        return []
    except Exception as e:
        print(f"❌ Error running pip-audit: {e}")
        return []

def check_with_safety():
    """Check dependencies using safety"""
    print("\n🔍 Checking with safety...")
    try:
        result = subprocess.run(
            ["safety", "check", "--json"],
            capture_output=True,
            text=True
        )
        
        vulnerabilities = json.loads(result.stdout)
        if not vulnerabilities:
            print("✅ No vulnerabilities found by safety")
            return []
        else:
            print(f"⚠️  Found {len(vulnerabilities)} vulnerabilities:")
            for vuln in vulnerabilities:
                print(f"   - {vuln['package_name']} {vuln['installed_version']}: {vuln['vulnerability_id']}")
            return vulnerabilities
    except FileNotFoundError:
        print("ℹ️  safety not installed. Install with: pip install safety")
        return []
    except Exception as e:
        print(f"❌ Error running safety: {e}")
        return []

def check_outdated_packages():
    """Check for outdated packages"""
    print("\n📦 Checking for outdated packages...")
    try:
        result = subprocess.run(
            ["pip", "list", "--outdated", "--format", "json"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            outdated = json.loads(result.stdout)
            if not outdated:
                print("✅ All packages are up to date")
                return []
            else:
                print(f"📋 Found {len(outdated)} outdated packages:")
                
                # Categorize by update type
                major_updates = []
                minor_updates = []
                patch_updates = []
                
                for pkg in outdated:
                    current = pkg['version'].split('.')
                    latest = pkg['latest_version'].split('.')
                    
                    try:
                        if len(current) >= 1 and len(latest) >= 1:
                            if current[0] != latest[0]:
                                major_updates.append(pkg)
                            elif len(current) >= 2 and len(latest) >= 2 and current[1] != latest[1]:
                                minor_updates.append(pkg)
                            else:
                                patch_updates.append(pkg)
                    except:
                        patch_updates.append(pkg)
                
                if major_updates:
                    print("\n   🔴 Major updates available:")
                    for pkg in major_updates:
                        print(f"      - {pkg['name']}: {pkg['version']} → {pkg['latest_version']}")
                
                if minor_updates:
                    print("\n   🟡 Minor updates available:")
                    for pkg in minor_updates[:5]:  # Show first 5
                        print(f"      - {pkg['name']}: {pkg['version']} → {pkg['latest_version']}")
                    if len(minor_updates) > 5:
                        print(f"      ... and {len(minor_updates) - 5} more")
                
                if patch_updates:
                    print("\n   🟢 Patch updates available:")
                    for pkg in patch_updates[:5]:  # Show first 5
                        print(f"      - {pkg['name']}: {pkg['version']} → {pkg['latest_version']}")
                    if len(patch_updates) > 5:
                        print(f"      ... and {len(patch_updates) - 5} more")
                
                return outdated
        else:
            print("❌ Failed to check for outdated packages")
            return []
    except Exception as e:
        print(f"❌ Error checking outdated packages: {e}")
        return []

def check_critical_packages():
    """Check specific critical packages for security updates"""
    print("\n🔒 Checking critical packages...")
    
    critical_packages = {
        'requests': {'min_version': '2.31.0', 'cve': 'CVE-2023-32681'},
        'aiohttp': {'min_version': '3.9.0', 'cve': 'Multiple security fixes'},
        'urllib3': {'min_version': '1.26.18', 'cve': 'CVE-2023-45803'},
        'cryptography': {'min_version': '41.0.7', 'cve': 'Multiple CVEs'},
        'numpy': {'min_version': '1.24.0', 'cve': 'Memory safety'},
        'pillow': {'min_version': '10.0.1', 'cve': 'CVE-2023-44271'},
    }
    
    issues_found = False
    
    for package, info in critical_packages.items():
        try:
            current_version = pkg_resources.get_distribution(package).version
            min_version = info['min_version']
            
            # Simple version comparison (would need proper parsing in production)
            if current_version < min_version:
                print(f"   ⚠️  {package} {current_version} < {min_version} ({info['cve']})")
                issues_found = True
            else:
                print(f"   ✅ {package} {current_version} is secure")
        except pkg_resources.DistributionNotFound:
            # Package not installed, skip
            pass
        except Exception as e:
            print(f"   ❌ Error checking {package}: {e}")
    
    if not issues_found:
        print("   ✅ All critical packages are up to date")

def generate_security_report():
    """Generate a comprehensive security report"""
    print("\n" + "="*60)
    print("🛡️  DEPENDENCY SECURITY REPORT")
    print("="*60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run all checks
    pip_audit_vulns = check_with_pip_audit()
    safety_vulns = check_with_safety()
    outdated = check_outdated_packages()
    check_critical_packages()
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    total_vulns = len(pip_audit_vulns) + len(safety_vulns)
    if total_vulns == 0:
        print("✅ No security vulnerabilities found!")
    else:
        print(f"⚠️  Found {total_vulns} total vulnerabilities")
        print("   Run 'pip install --upgrade' for affected packages")
    
    if outdated:
        major_count = sum(1 for pkg in outdated 
                         if pkg['version'].split('.')[0] != pkg['latest_version'].split('.')[0])
        if major_count > 0:
            print(f"📦 {major_count} packages have major updates available")
    
    # Recommendations
    print("\n📋 RECOMMENDATIONS:")
    if total_vulns > 0:
        print("1. Update vulnerable packages immediately")
        print("2. Test thoroughly after updates")
        print("3. Review changelog for breaking changes")
    else:
        print("1. Continue regular security scans")
        print("2. Enable automated dependency updates")
        print("3. Monitor security advisories")
    
    # Save report
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'pip_audit_vulnerabilities': pip_audit_vulns,
        'safety_vulnerabilities': safety_vulns,
        'outdated_packages': outdated,
        'total_vulnerabilities': total_vulns
    }
    
    report_file = Path(f"security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n💾 Full report saved to: {report_file}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check dependencies for security issues")
    parser.add_argument('--install-tools', action='store_true', help='Install security scanning tools')
    parser.add_argument('--fix', action='store_true', help='Attempt to fix issues automatically')
    
    args = parser.parse_args()
    
    if args.install_tools:
        print("Installing security tools...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pip-audit", "safety"])
        print("✅ Security tools installed")
        return
    
    # Generate security report
    generate_security_report()
    
    if args.fix:
        print("\n🔧 Attempting to fix issues...")
        print("This feature is not yet implemented.")
        print("Please manually update packages using:")
        print("  pip install --upgrade package-name")

if __name__ == "__main__":
    main()