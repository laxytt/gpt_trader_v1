#!/usr/bin/env python3
"""
Script to add authentication to all Streamlit dashboards
"""

import os
from pathlib import Path
import re


def add_auth_to_dashboard(file_path: Path) -> bool:
    """Add authentication to a dashboard file"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already has authentication
    if 'from scripts.auth_utils import DashboardAuth' in content:
        print(f"✓ {file_path.name} already has authentication")
        return False
    
    # Find the imports section - handle both sys.path.insert and sys.path.append
    imports_pattern = r'(import sys\s*\n\s*sys\.path\.(insert|append).*?\n)'
    match = re.search(imports_pattern, content, re.DOTALL)
    
    if not match:
        # Try to find just the sys import
        sys_import_pattern = r'(import sys\s*\n)'
        sys_match = re.search(sys_import_pattern, content)
        if sys_match:
            # Find the next import or from statement
            next_import = re.search(r'\n(from|import)', content[sys_match.end():])
            if next_import:
                insert_pos = sys_match.end() + next_import.start()
            else:
                insert_pos = sys_match.end()
        else:
            print(f"⚠️  {file_path.name} - couldn't find import section")
            return False
    else:
        # Add auth import after the sys.path manipulation
        insert_pos = match.end()
    
    # Find where st.set_page_config is called
    config_pattern = r'(st\.set_page_config\([^)]+\))'
    config_match = re.search(config_pattern, content)
    
    if not config_match:
        print(f"⚠️  {file_path.name} - couldn't find st.set_page_config")
        return False
    
    # Extract app name from the file or page title
    app_name = "Dashboard"
    title_match = re.search(r'page_title\s*=\s*["\']([^"\']+)["\']', config_match.group(1))
    if title_match:
        app_name = title_match.group(1)
    
    # Build the auth additions
    auth_import = "\nfrom scripts.auth_utils import DashboardAuth\n"
    auth_init = f'\n# Initialize authentication\nauth = DashboardAuth("{app_name}")\n'
    auth_protect = '\n# Protect the app - this will show login form if not authenticated\nauth.protect_app()\n'
    
    # Insert auth import after sys.path
    modified_content = content[:insert_pos] + auth_import + content[insert_pos:]
    
    # Find position after st.set_page_config
    config_end = config_match.end()
    # Adjust for the auth_import we just added
    adjusted_config_end = modified_content.find(config_match.group(1)) + len(config_match.group(1))
    
    # Insert auth initialization and protection
    final_content = (
        modified_content[:adjusted_config_end] + 
        auth_init + 
        auth_protect +
        modified_content[adjusted_config_end:]
    )
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(final_content)
    
    print(f"✅ {file_path.name} - authentication added")
    return True


def main():
    """Add authentication to all dashboard files"""
    
    scripts_dir = Path(__file__).parent
    
    # List of dashboard files to secure
    dashboard_files = [
        'comprehensive_trading_dashboard.py',
        'trading_dashboard_simple.py',
        'ml_improvement_dashboard.py',
        'gpt_flow_dashboard.py',
        'ml_performance_monitor.py'
    ]
    
    print("Adding authentication to Streamlit dashboards...\n")
    
    modified_count = 0
    for dashboard_file in dashboard_files:
        file_path = scripts_dir / dashboard_file
        if file_path.exists():
            if add_auth_to_dashboard(file_path):
                modified_count += 1
        else:
            print(f"⚠️  {dashboard_file} not found")
    
    print(f"\n✅ Modified {modified_count} dashboards")
    print("\n⚠️  Default credentials: admin/admin123")
    print("Please change the admin password immediately using:")
    print("  python scripts/manage_dashboard_users.py passwd admin")
    
    # Create auth management documentation
    doc_content = """# Dashboard Authentication Guide

## Default Credentials
- Username: `admin`
- Password: `admin123`

**⚠️ CHANGE THE DEFAULT PASSWORD IMMEDIATELY!**

## Managing Users

### Change Admin Password
```bash
python scripts/manage_dashboard_users.py passwd admin
```

### Add a New User
```bash
python scripts/manage_dashboard_users.py add <username> --role viewer
python scripts/manage_dashboard_users.py add <username> --role admin
```

### List All Users
```bash
python scripts/manage_dashboard_users.py list
```

### Remove a User
```bash
python scripts/manage_dashboard_users.py remove <username>
```

## Security Notes

1. The authentication file is stored at `config/.dashboard_auth.json`
2. Passwords are hashed using SHA-256
3. Sessions expire after 8 hours of inactivity
4. The auth file has restricted permissions (600) on Unix systems

## Accessing Dashboards

All dashboards now require authentication. When you access any dashboard:
1. You'll see a login screen
2. Enter your username and password
3. Click "Login"
4. Your session will remain active for 8 hours

## Troubleshooting

If you're locked out:
1. Delete `config/.dashboard_auth.json` to reset to defaults
2. Or manually edit the file to add/modify users
"""
    
    doc_path = scripts_dir / "DASHBOARD_AUTH_GUIDE.md"
    with open(doc_path, 'w') as f:
        f.write(doc_content)
    
    print(f"\n📚 Documentation created: {doc_path}")


if __name__ == "__main__":
    main()