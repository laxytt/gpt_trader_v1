#!/usr/bin/env python3
"""
Command-line utility to manage dashboard users
"""

import argparse
import getpass
from pathlib import Path
import sys

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.auth_utils import add_user, change_password, remove_user, DashboardAuth


def main():
    parser = argparse.ArgumentParser(description="Manage dashboard users")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Add user command
    add_parser = subparsers.add_parser('add', help='Add a new user')
    add_parser.add_argument('username', help='Username')
    add_parser.add_argument('--role', default='viewer', choices=['admin', 'viewer'], 
                           help='User role (default: viewer)')
    
    # Change password command
    passwd_parser = subparsers.add_parser('passwd', help='Change password')
    passwd_parser.add_argument('username', help='Username')
    
    # Remove user command
    remove_parser = subparsers.add_parser('remove', help='Remove a user')
    remove_parser.add_argument('username', help='Username')
    
    # List users command
    list_parser = subparsers.add_parser('list', help='List all users')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    auth = DashboardAuth()
    
    if args.command == 'add':
        password = getpass.getpass(f"Password for {args.username}: ")
        confirm = getpass.getpass("Confirm password: ")
        
        if password != confirm:
            print("❌ Passwords do not match")
            return 1
        
        if len(password) < 6:
            print("❌ Password must be at least 6 characters")
            return 1
        
        success, message = add_user(args.username, password, args.role)
        if success:
            print(f"✅ {message}")
        else:
            print(f"❌ {message}")
            return 1
    
    elif args.command == 'passwd':
        if args.username == 'admin':
            # For admin, allow password reset without old password
            print("Resetting admin password...")
            new_password = getpass.getpass("New password: ")
            confirm = getpass.getpass("Confirm password: ")
            
            if new_password != confirm:
                print("❌ Passwords do not match")
                return 1
            
            if len(new_password) < 6:
                print("❌ Password must be at least 6 characters")
                return 1
            
            # Direct password update for admin
            auth_data = auth._load_auth_data()
            auth_data['users']['admin']['password_hash'] = auth._hash_password(new_password)
            if 'note' in auth_data['users']['admin']:
                del auth_data['users']['admin']['note']
            auth._save_auth_data(auth_data)
            print("✅ Admin password updated successfully")
        else:
            old_password = getpass.getpass("Current password: ")
            new_password = getpass.getpass("New password: ")
            confirm = getpass.getpass("Confirm password: ")
            
            if new_password != confirm:
                print("❌ Passwords do not match")
                return 1
            
            if len(new_password) < 6:
                print("❌ Password must be at least 6 characters")
                return 1
            
            success, message = change_password(args.username, old_password, new_password)
            if success:
                print(f"✅ {message}")
            else:
                print(f"❌ {message}")
                return 1
    
    elif args.command == 'remove':
        confirm = input(f"Are you sure you want to remove user '{args.username}'? (yes/no): ")
        if confirm.lower() == 'yes':
            success, message = remove_user(args.username)
            if success:
                print(f"✅ {message}")
            else:
                print(f"❌ {message}")
                return 1
        else:
            print("Cancelled")
    
    elif args.command == 'list':
        auth_data = auth._load_auth_data()
        users = auth_data.get('users', {})
        
        if not users:
            print("No users found")
        else:
            print("\nDashboard Users:")
            print("-" * 50)
            for username, user_data in users.items():
                role = user_data.get('role', 'viewer')
                created = user_data.get('created_at', 'Unknown')[:10]
                note = user_data.get('note', '')
                if note:
                    print(f"{username:20} {role:10} (created: {created}) ⚠️  {note}")
                else:
                    print(f"{username:20} {role:10} (created: {created})")
            print("-" * 50)
            
            # Show active sessions
            sessions = auth_data.get('sessions', {})
            active_count = len(sessions)
            if active_count > 0:
                print(f"\nActive sessions: {active_count}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())