#!/usr/bin/env python3
"""
Deployment script for GPT Trading System.
Handles environment validation, testing, and deployment.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import argparse
import json


class DeploymentManager:
    """Manages the deployment process."""
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.project_root = Path(__file__).parent.parent
        self.deploy_dir = self.project_root / "deploy"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def validate_environment(self) -> bool:
        """Validate deployment environment."""
        print("🔍 Validating environment...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ required")
            return False
        
        # Check required files
        required_files = [
            ".env",
            "requirements.txt",
            "trading_loop.py",
            "config/settings.py"
        ]
        
        for file in required_files:
            if not (self.project_root / file).exists():
                print(f"❌ Missing required file: {file}")
                return False
        
        # Check environment variables
        env_file = self.project_root / ".env"
        if env_file.exists():
            with open(env_file) as f:
                env_content = f.read()
                
            required_vars = [
                "OPENAI_API_KEY",
                "MT5_FILES_DIR"
            ]
            
            for var in required_vars:
                if f"{var}=" not in env_content:
                    print(f"❌ Missing environment variable: {var}")
                    return False
        
        print("✅ Environment validation passed")
        return True
    
    def run_tests(self) -> bool:
        """Run tests before deployment."""
        print("\n🧪 Running tests...")
        
        test_script = self.project_root / "run_tests.py"
        if not test_script.exists():
            print("⚠️  No test runner found, skipping tests")
            return True
        
        result = subprocess.run(
            [sys.executable, str(test_script)],
            cwd=str(self.project_root)
        )
        
        if result.returncode != 0:
            print("❌ Tests failed")
            return False
        
        print("✅ All tests passed")
        return True
    
    def backup_current(self):
        """Backup current deployment."""
        print("\n💾 Creating backup...")
        
        backup_dir = self.project_root / "backups" / "deployments" / self.timestamp
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup critical files
        files_to_backup = [
            ".env",
            "data/trades.db",
            "models"
        ]
        
        for file in files_to_backup:
            source = self.project_root / file
            if source.exists():
                if source.is_file():
                    shutil.copy2(source, backup_dir)
                else:
                    shutil.copytree(source, backup_dir / file, dirs_exist_ok=True)
        
        print(f"✅ Backup created at: {backup_dir}")
    
    def build_docker_image(self) -> bool:
        """Build Docker image."""
        print("\n🐳 Building Docker image...")
        
        # Generate image tag
        tag = f"gpt-trader:{self.environment}-{self.timestamp}"
        
        # Build image
        result = subprocess.run(
            ["docker", "build", "-t", tag, "-f", "deploy/Dockerfile", "."],
            cwd=str(self.project_root)
        )
        
        if result.returncode != 0:
            print("❌ Docker build failed")
            return False
        
        # Tag as latest
        subprocess.run(
            ["docker", "tag", tag, f"gpt-trader:{self.environment}-latest"]
        )
        
        print(f"✅ Docker image built: {tag}")
        return True
    
    def deploy_docker_compose(self) -> bool:
        """Deploy using docker-compose."""
        print("\n🚀 Deploying with docker-compose...")
        
        compose_file = self.deploy_dir / "docker-compose.yml"
        
        # Stop existing containers
        subprocess.run(
            ["docker-compose", "-f", str(compose_file), "down"],
            cwd=str(self.project_root)
        )
        
        # Start new containers
        result = subprocess.run(
            ["docker-compose", "-f", str(compose_file), "up", "-d"],
            cwd=str(self.project_root)
        )
        
        if result.returncode != 0:
            print("❌ Docker-compose deployment failed")
            return False
        
        print("✅ Services deployed successfully")
        return True
    
    def health_check(self) -> bool:
        """Check if deployed services are healthy."""
        print("\n🏥 Running health checks...")
        
        import time
        time.sleep(10)  # Wait for services to start
        
        # Check main container
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=gpt-trader", "--format", "{{.Status}}"],
            capture_output=True,
            text=True
        )
        
        if "healthy" not in result.stdout.lower() and "up" not in result.stdout.lower():
            print("❌ Trading system container not healthy")
            return False
        
        print("✅ All services healthy")
        return True
    
    def create_deployment_report(self):
        """Create deployment report."""
        report = {
            "timestamp": self.timestamp,
            "environment": self.environment,
            "deployed_by": os.getenv("USER", "unknown"),
            "git_commit": self._get_git_commit(),
            "docker_images": self._get_docker_images(),
            "status": "success"
        }
        
        report_file = self.project_root / "deploy" / "deployment_history.json"
        
        # Load existing history
        if report_file.exists():
            with open(report_file) as f:
                history = json.load(f)
        else:
            history = []
        
        # Add new deployment
        history.append(report)
        
        # Save updated history
        with open(report_file, "w") as f:
            json.dump(history, f, indent=2)
        
        print(f"\n📋 Deployment report saved")
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=str(self.project_root)
            )
            return result.stdout.strip()[:8]
        except:
            return "unknown"
    
    def _get_docker_images(self) -> list:
        """Get list of built docker images."""
        try:
            result = subprocess.run(
                ["docker", "images", "--filter", "reference=gpt-trader*", "--format", "{{.Repository}}:{{.Tag}}"],
                capture_output=True,
                text=True
            )
            return result.stdout.strip().split("\n")
        except:
            return []
    
    def deploy(self) -> bool:
        """Run full deployment process."""
        print(f"🚀 Starting deployment for {self.environment} environment")
        print("=" * 60)
        
        # Validation
        if not self.validate_environment():
            return False
        
        # Testing
        if self.environment == "production":
            if not self.run_tests():
                print("\n⚠️  Deploy anyway? (y/N): ", end="")
                if input().lower() != 'y':
                    return False
        
        # Backup
        if self.environment == "production":
            self.backup_current()
        
        # Build
        if not self.build_docker_image():
            return False
        
        # Deploy
        if not self.deploy_docker_compose():
            return False
        
        # Health check
        if not self.health_check():
            print("⚠️  Health check failed, but services are deployed")
        
        # Report
        self.create_deployment_report()
        
        print("\n" + "=" * 60)
        print("✅ Deployment completed successfully!")
        print(f"📊 Dashboard available at: http://localhost:8050")
        print(f"📈 Grafana available at: http://localhost:3000")
        
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Deploy GPT Trading System")
    parser.add_argument(
        "--env",
        choices=["development", "staging", "production"],
        default="production",
        help="Deployment environment"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backup creation"
    )
    
    args = parser.parse_args()
    
    # Create deployment manager
    manager = DeploymentManager(environment=args.env)
    
    # Override settings based on arguments
    if args.skip_tests:
        manager.run_tests = lambda: True
    if args.no_backup:
        manager.backup_current = lambda: None
    
    # Run deployment
    success = manager.deploy()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()