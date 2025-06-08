# Dependency Management Guide

This guide explains the dependency management strategy for the GPT Trading System, including version pinning, security updates, and best practices.

## Overview

The project uses a multi-file approach for dependency management:

1. **requirements.txt** - Base requirements with minimum versions
2. **requirements-pinned.txt** - Exact versions for production
3. **requirements-dev-pinned.txt** - Development dependencies
4. **constraints.txt** - Version constraints for indirect dependencies
5. **requirements.in** - Source file for pip-tools (optional)

## Files Description

### requirements.txt
The original requirements file with flexible version specifications using `>=`. This allows for updates while maintaining minimum version requirements.

### requirements-pinned.txt
Production-ready requirements with exact versions (`==`). This ensures reproducible deployments and prevents unexpected behavior from automatic updates.

### requirements-dev-pinned.txt
Development and testing tools with pinned versions:
- Testing: pytest, coverage
- Code quality: black, flake8, mypy
- Security: bandit, safety
- Documentation: sphinx
- Development: jupyter, ipython

### constraints.txt
Additional constraints for indirect dependencies that aren't explicitly listed but are important for security and compatibility.

### requirements.in (pip-tools)
Source file for use with pip-tools, allowing for automatic dependency resolution while maintaining control over versions.

## Installation

### Production Deployment
```bash
# Use pinned versions for reproducible builds
pip install -r requirements-pinned.txt
```

### Development Environment
```bash
# Install both production and dev dependencies
pip install -r requirements-pinned.txt
pip install -r requirements-dev-pinned.txt
```

### With Constraints
```bash
# Apply additional constraints to indirect dependencies
pip install -c constraints.txt -r requirements-pinned.txt
```

### Using pip-tools
```bash
# Install pip-tools
pip install pip-tools

# Compile requirements
pip-compile requirements.in

# Sync environment
pip-sync requirements.txt
```

## Updating Dependencies

### 1. Security Updates Only
```bash
# Check for security vulnerabilities
pip-audit
# or
safety check

# Update specific package
pip install --upgrade package==new.version

# Regenerate pinned file
python scripts/pin_dependencies.py
```

### 2. Minor Updates
```bash
# Update within version constraints
pip install --upgrade -r requirements.txt

# Test thoroughly
pytest

# Regenerate pinned versions
python scripts/pin_dependencies.py --use-installed
```

### 3. Major Updates
1. Update version constraint in requirements.txt
2. Test extensively in development
3. Update pinned versions
4. Document breaking changes

## Version Pinning Strategy

### Semantic Versioning
- **Exact versions (==)**: For production stability
- **Compatible versions (~=)**: For minor updates
- **Minimum versions (>=)**: For flexibility with security patches
- **Range versions (>=X,<Y)**: To avoid major version breaks

### Examples:
```txt
pydantic==2.5.3              # Exact version
pandas>=2.0.0,<3.0.0         # Major version constraint
numpy~=1.24.3                # Equivalent to >=1.24.3,<1.25.0
requests>=2.31.0             # Minimum for security
```

## Security Best Practices

### 1. Regular Security Scans
```bash
# Using pip-audit
pip-audit

# Using safety
safety check

# Using bandit for code
bandit -r core/
```

### 2. Automated Updates
Set up CI/CD to check for security updates:
```yaml
# GitHub Actions example
- name: Check dependencies
  run: |
    pip install pip-audit
    pip-audit
```

### 3. Version Constraints
Always set upper bounds for major versions to prevent breaking changes:
```txt
# Good
pandas>=2.0.0,<3.0.0

# Risky
pandas>=2.0.0
```

## Common Tasks

### Adding a New Dependency
1. Add to requirements.txt with version constraint
2. Install and test
3. Run `python scripts/pin_dependencies.py`
4. Commit both files

### Removing a Dependency
1. Remove from all requirements files
2. Run `pip uninstall package`
3. Test thoroughly
4. Update documentation

### Checking for Updates
```bash
# List outdated packages
pip list --outdated

# Show dependency tree
pip install pipdeptree
pipdeptree
```

### Resolving Conflicts
```bash
# Use pip-tools for automatic resolution
pip-compile --resolver=backtracking requirements.in

# Or manually with pipdeptree
pipdeptree --warn fail
```

## Docker Considerations

### Dockerfile Best Practices
```dockerfile
# Copy requirements first for layer caching
COPY requirements-pinned.txt .
RUN pip install --no-cache-dir -r requirements-pinned.txt

# Verify installations
RUN pip check
```

### Multi-stage Builds
```dockerfile
# Build stage
FROM python:3.11-slim as builder
COPY requirements-pinned.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements-pinned.txt

# Runtime stage
FROM python:3.11-slim
COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*
```

## CI/CD Integration

### GitHub Actions
```yaml
- name: Set up Python
  uses: actions/setup-python@v4
  with:
    python-version: '3.11'
    cache: 'pip'
    cache-dependency-path: '**/requirements-pinned.txt'

- name: Install dependencies
  run: |
    pip install -r requirements-pinned.txt
    pip install -r requirements-dev-pinned.txt
```

### Version Matrix Testing
```yaml
strategy:
  matrix:
    python-version: ['3.9', '3.10', '3.11']
    os: [ubuntu-latest, windows-latest]
```

## Troubleshooting

### "No matching distribution found"
- Check Python version compatibility
- Verify package name spelling
- Try upgrading pip: `pip install --upgrade pip`

### Version conflicts
- Use `pipdeptree` to visualize dependencies
- Consider using pip-tools for resolution
- Check constraints.txt for conflicts

### Platform-specific issues
- Some packages need system libraries
- Use platform markers in requirements:
  ```txt
  pywin32>=306; sys_platform == 'win32'
  ```

### Performance issues
- Use `--no-cache-dir` for Docker builds
- Consider using wheels for faster installs
- Use `pip install --use-deprecated=legacy-resolver` if new resolver is slow

## Maintenance Schedule

### Daily
- Automated security scans in CI/CD

### Weekly
- Review security advisories
- Check for critical updates

### Monthly
- Update dev dependencies
- Review and update constraints

### Quarterly
- Major version updates
- Performance dependency audit
- Clean up unused dependencies

## Emergency Procedures

### Security Vulnerability Found
1. Check severity and affected systems
2. Update immediately if critical:
   ```bash
   pip install --upgrade vulnerable-package==safe.version
   python scripts/pin_dependencies.py
   ```
3. Test core functionality
4. Deploy hotfix
5. Document in CHANGELOG

### Dependency Hijacking
1. Verify package checksums
2. Pin to last known good version
3. Report to PyPI security team
4. Audit all recent changes

By following these practices, you can maintain a secure, stable, and reproducible dependency environment for the GPT Trading System.