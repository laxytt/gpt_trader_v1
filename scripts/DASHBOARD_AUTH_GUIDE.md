# Dashboard Authentication Guide

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
