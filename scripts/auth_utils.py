#!/usr/bin/env python3
"""
Authentication utilities for Streamlit dashboards.
Provides simple password-based authentication with session management.
"""

import streamlit as st
import hashlib
import os
from typing import Optional, Callable, Dict, Any
from functools import wraps
import json
from pathlib import Path
from datetime import datetime, timedelta
import secrets


class DashboardAuth:
    """Simple authentication system for Streamlit dashboards"""
    
    def __init__(self, app_name: str = "Trading Dashboard"):
        self.app_name = app_name
        self.auth_file = Path(__file__).parent.parent / "config" / ".dashboard_auth.json"
        self._ensure_auth_file()
        
    def _ensure_auth_file(self):
        """Ensure authentication file exists with default credentials"""
        if not self.auth_file.exists():
            # Create default auth file
            default_auth = {
                "users": {
                    "admin": {
                        "password_hash": self._hash_password("admin123"),
                        "created_at": datetime.now().isoformat(),
                        "role": "admin",
                        "note": "DEFAULT PASSWORD - CHANGE IMMEDIATELY"
                    }
                },
                "sessions": {},
                "settings": {
                    "session_timeout_minutes": 480,  # 8 hours
                    "max_login_attempts": 5,
                    "lockout_duration_minutes": 15
                }
            }
            
            self.auth_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.auth_file, 'w') as f:
                json.dump(default_auth, f, indent=2)
            
            # Set restrictive permissions (Unix-like systems)
            try:
                os.chmod(self.auth_file, 0o600)
            except:
                pass  # Windows doesn't support chmod
    
    def _load_auth_data(self) -> Dict[str, Any]:
        """Load authentication data from file"""
        try:
            with open(self.auth_file, 'r') as f:
                return json.load(f)
        except:
            self._ensure_auth_file()
            with open(self.auth_file, 'r') as f:
                return json.load(f)
    
    def _save_auth_data(self, data: Dict[str, Any]):
        """Save authentication data to file"""
        with open(self.auth_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _generate_session_token(self) -> str:
        """Generate secure session token"""
        return secrets.token_urlsafe(32)
    
    def check_authentication(self) -> bool:
        """Check if user is authenticated"""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        
        if 'session_token' in st.session_state:
            # Verify session token
            auth_data = self._load_auth_data()
            sessions = auth_data.get('sessions', {})
            token = st.session_state.session_token
            
            if token in sessions:
                session_data = sessions[token]
                expiry = datetime.fromisoformat(session_data['expires_at'])
                
                if datetime.now() < expiry:
                    st.session_state.authenticated = True
                    st.session_state.username = session_data['username']
                    return True
                else:
                    # Session expired
                    del sessions[token]
                    auth_data['sessions'] = sessions
                    self._save_auth_data(auth_data)
        
        return st.session_state.authenticated
    
    def login_form(self):
        """Display login form"""
        st.markdown(f"### 🔒 {self.app_name} Login")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if self.authenticate(username, password):
                    st.success("✅ Login successful!")
                    st.experimental_rerun()
                else:
                    st.error("❌ Invalid username or password")
    
    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate user"""
        auth_data = self._load_auth_data()
        users = auth_data.get('users', {})
        
        if username in users:
            user_data = users[username]
            password_hash = self._hash_password(password)
            
            if password_hash == user_data['password_hash']:
                # Create session
                token = self._generate_session_token()
                timeout_minutes = auth_data.get('settings', {}).get('session_timeout_minutes', 480)
                
                sessions = auth_data.get('sessions', {})
                sessions[token] = {
                    'username': username,
                    'created_at': datetime.now().isoformat(),
                    'expires_at': (datetime.now() + timedelta(minutes=timeout_minutes)).isoformat()
                }
                
                auth_data['sessions'] = sessions
                self._save_auth_data(auth_data)
                
                # Set session state
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.session_token = token
                
                return True
        
        return False
    
    def logout(self):
        """Logout current user"""
        if 'session_token' in st.session_state:
            # Remove session from auth file
            auth_data = self._load_auth_data()
            sessions = auth_data.get('sessions', {})
            
            if st.session_state.session_token in sessions:
                del sessions[st.session_state.session_token]
                auth_data['sessions'] = sessions
                self._save_auth_data(auth_data)
        
        # Clear session state
        st.session_state.authenticated = False
        for key in ['username', 'session_token']:
            if key in st.session_state:
                del st.session_state[key]
    
    def require_auth(self, func: Callable) -> Callable:
        """Decorator to require authentication for a function"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.check_authentication():
                self.login_form()
                st.stop()
            else:
                # Show user info and logout button in sidebar
                with st.sidebar:
                    st.markdown("---")
                    st.markdown(f"👤 **User:** {st.session_state.username}")
                    if st.button("🚪 Logout"):
                        self.logout()
                        st.experimental_rerun()
                    st.markdown("---")
                
                return func(*args, **kwargs)
        return wrapper
    
    def protect_app(self):
        """Protect the entire Streamlit app"""
        if not self.check_authentication():
            # Show login page
            st.set_page_config(
                page_title=f"{self.app_name} - Login",
                page_icon="🔒",
                layout="centered"
            )
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"# {self.app_name}")
                st.markdown("Please login to continue")
                self.login_form()
                
                # Show default credentials warning if using defaults
                auth_data = self._load_auth_data()
                if any(user.get('note') == "DEFAULT PASSWORD - CHANGE IMMEDIATELY" 
                       for user in auth_data.get('users', {}).values()):
                    st.warning("⚠️ Using default credentials (admin/admin123). Please change immediately!")
            
            st.stop()
        else:
            # Show user info in sidebar
            with st.sidebar:
                st.markdown("---")
                st.markdown(f"👤 **User:** {st.session_state.username}")
                if st.button("🚪 Logout"):
                    self.logout()
                    st.experimental_rerun()
                st.markdown("---")


# Utility functions for managing users
def add_user(username: str, password: str, role: str = "viewer"):
    """Add a new user (utility function for administrators)"""
    auth = DashboardAuth()
    auth_data = auth._load_auth_data()
    
    if username in auth_data.get('users', {}):
        return False, "User already exists"
    
    auth_data['users'][username] = {
        'password_hash': auth._hash_password(password),
        'created_at': datetime.now().isoformat(),
        'role': role
    }
    
    auth._save_auth_data(auth_data)
    return True, "User created successfully"


def change_password(username: str, old_password: str, new_password: str):
    """Change user password"""
    auth = DashboardAuth()
    auth_data = auth._load_auth_data()
    
    if username not in auth_data.get('users', {}):
        return False, "User not found"
    
    # Verify old password
    if auth._hash_password(old_password) != auth_data['users'][username]['password_hash']:
        return False, "Invalid old password"
    
    # Update password
    auth_data['users'][username]['password_hash'] = auth._hash_password(new_password)
    auth_data['users'][username]['updated_at'] = datetime.now().isoformat()
    
    # Remove the default password note if present
    if 'note' in auth_data['users'][username]:
        del auth_data['users'][username]['note']
    
    auth._save_auth_data(auth_data)
    return True, "Password changed successfully"


def remove_user(username: str):
    """Remove a user"""
    auth = DashboardAuth()
    auth_data = auth._load_auth_data()
    
    if username not in auth_data.get('users', {}):
        return False, "User not found"
    
    if username == 'admin':
        return False, "Cannot remove admin user"
    
    del auth_data['users'][username]
    
    # Remove any active sessions for this user
    sessions = auth_data.get('sessions', {})
    sessions_to_remove = [token for token, session in sessions.items() 
                         if session['username'] == username]
    for token in sessions_to_remove:
        del sessions[token]
    
    auth_data['sessions'] = sessions
    auth._save_auth_data(auth_data)
    return True, "User removed successfully"