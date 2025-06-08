.PHONY: help test build deploy clean install lint format check-env
.PHONY: deps-check deps-update deps-pin deps-security install-prod install-dev-pinned

# Default target
help:
	@echo "GPT Trading System - Available Commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Run all tests"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make lint         - Run code linting"
	@echo "  make format       - Format code with black and isort"
	@echo "  make build        - Build Docker image"
	@echo "  make deploy       - Deploy to production"
	@echo "  make deploy-staging - Deploy to staging"
	@echo "  make clean        - Clean generated files"
	@echo "  make check-env    - Check environment setup"

# Install dependencies
install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

install-prod:
	pip install -r requirements-pinned.txt

install-dev-pinned:
	pip install -r requirements-pinned.txt
	pip install -r requirements-dev-pinned.txt

# Run tests
test:
	python run_tests.py

test-integration:
	python run_tests.py integration

# Code quality
lint:
	flake8 core --count --select=E9,F63,F7,F82 --show-source --statistics
	pylint core --exit-zero
	mypy core --ignore-missing-imports

format:
	black core tests
	isort core tests

# Docker operations
build:
	docker build -t gpt-trader:latest -f deploy/Dockerfile .

build-dashboard:
	docker build -t gpt-trader-dashboard:latest -f deploy/Dockerfile.dashboard .

# Deployment
deploy: check-env test
	python deploy/deployment.py --env production

deploy-staging: check-env
	python deploy/deployment.py --env staging --skip-tests

deploy-dev:
	python deploy/deployment.py --env development --skip-tests --no-backup

# Docker compose operations
up:
	docker-compose -f deploy/docker-compose.yml up -d

down:
	docker-compose -f deploy/docker-compose.yml down

logs:
	docker-compose -f deploy/docker-compose.yml logs -f

# Environment checks
check-env:
	@echo "Checking environment..."
	@test -f .env || (echo "ERROR: .env file not found" && exit 1)
	@grep -q "OPENAI_API_KEY=" .env || (echo "ERROR: OPENAI_API_KEY not set in .env" && exit 1)
	@grep -q "MT5_FILES_DIR=" .env || (echo "ERROR: MT5_FILES_DIR not set in .env" && exit 1)
	@echo "Environment check passed ✓"

# Clean generated files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -rf htmlcov .coverage .pytest_cache
	rm -rf build dist *.egg-info

# Database operations
db-backup:
	python scripts/automation/database_backup.py

db-migrate:
	python -c "from core.infrastructure.database.migrations import run_migrations; run_migrations()"

# Development helpers
shell:
	docker-compose -f deploy/docker-compose.yml exec trading-system /bin/bash

psql:
	docker-compose -f deploy/docker-compose.yml exec database psql -U trader

# Monitoring
monitor:
	@echo "Opening monitoring dashboards..."
	@echo "Grafana: http://localhost:3000"
	@echo "Prometheus: http://localhost:9090"
	@python -m webbrowser http://localhost:3000

# Performance profiling
profile:
	python -m cProfile -o profile.stats trading_loop.py
	python -m pstats profile.stats

# Dependency management
deps-check:
	@echo "Checking for outdated packages..."
	pip list --outdated

deps-update:
	@echo "Updating dependencies..."
	pip install --upgrade -r requirements.txt

deps-pin:
	@echo "Generating pinned requirements..."
	python scripts/pin_dependencies.py --use-installed

deps-security:
	@echo "Running security checks..."
	python scripts/check_dependencies_security.py

deps-security-install:
	@echo "Installing security scanning tools..."
	pip install pip-audit safety

deps-clean:
	@echo "Cleaning pip cache..."
	pip cache purge

# Update all dependencies and check security
update-all: deps-update deps-pin deps-security
	@echo "Dependencies updated and checked"