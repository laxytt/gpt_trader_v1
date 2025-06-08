"""
Real-time health monitoring service for the trading system.
Integrates with the main trading loop for continuous monitoring.
"""

import asyncio
import logging
import psutil
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

from core.infrastructure.mt5.client import MT5Client
from core.infrastructure.database.repositories import TradeRepository, SignalRepository
from core.infrastructure.gpt.client import GPTClient
from core.infrastructure.marketaux.client import MarketAuxClient
from core.utils.circuit_breaker import CircuitBreakerManager
from config.settings import Settings


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthMetric:
    """Individual health metric"""
    name: str
    status: HealthStatus
    value: Any
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """Overall system health"""
    status: HealthStatus
    metrics: Dict[str, HealthMetric]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def is_healthy(self) -> bool:
        return self.status in [HealthStatus.HEALTHY, HealthStatus.WARNING]
    
    @property
    def critical_issues(self) -> List[HealthMetric]:
        return [m for m in self.metrics.values() if m.status == HealthStatus.CRITICAL]
    
    @property
    def warnings(self) -> List[HealthMetric]:
        return [m for m in self.metrics.values() if m.status == HealthStatus.WARNING]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
            'is_healthy': self.is_healthy,
            'metrics': {
                name: {
                    'status': metric.status.value,
                    'value': metric.value,
                    'message': metric.message,
                    'metadata': metric.metadata
                }
                for name, metric in self.metrics.items()
            },
            'critical_issues': [m.name for m in self.critical_issues],
            'warnings': [m.name for m in self.warnings]
        }


class HealthMonitor:
    """Real-time health monitoring for the trading system"""
    
    def __init__(
        self,
        settings: Settings,
        mt5_client: Optional[MT5Client] = None,
        gpt_client: Optional[GPTClient] = None,
        marketaux_client: Optional[MarketAuxClient] = None
    ):
        self.settings = settings
        self.mt5_client = mt5_client
        self.gpt_client = gpt_client
        self.marketaux_client = marketaux_client
        
        # Initialize repositories
        self.trade_repository = TradeRepository(settings.database.db_path)
        self.signal_repository = SignalRepository(settings.database.db_path)
        
        # Monitoring state
        self._last_health_check: Optional[SystemHealth] = None
        self._health_history: List[SystemHealth] = []
        self._max_history = 100
        
        # Alert callbacks
        self._alert_handlers: List[Callable[[SystemHealth], None]] = []
        
        # Performance tracking
        self._performance_metrics = {
            'check_duration_ms': [],
            'api_response_times': {},
            'db_query_times': []
        }
    
    async def check_health(self) -> SystemHealth:
        """Perform comprehensive health check"""
        start_time = time.time()
        metrics = {}
        
        # Run all health checks concurrently
        checks = await asyncio.gather(
            self._check_mt5_connection(),
            self._check_database_health(),
            self._check_api_connectivity(),
            self._check_system_resources(),
            self._check_circuit_breakers(),
            self._check_recent_activity(),
            self._check_error_rates(),
            return_exceptions=True
        )
        
        # Process results
        check_names = [
            'mt5_connection', 'database', 'api_connectivity', 'system_resources',
            'circuit_breakers', 'recent_activity', 'error_rates'
        ]
        
        for name, result in zip(check_names, checks):
            if isinstance(result, Exception):
                metrics[name] = HealthMetric(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    value=None,
                    message=f"Check failed: {str(result)}"
                )
            else:
                metrics[name] = result
        
        # Determine overall status
        statuses = [m.status for m in metrics.values()]
        if any(s == HealthStatus.CRITICAL for s in statuses):
            overall_status = HealthStatus.CRITICAL
        elif any(s == HealthStatus.WARNING for s in statuses):
            overall_status = HealthStatus.WARNING
        elif any(s == HealthStatus.UNKNOWN for s in statuses):
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Create health report
        health = SystemHealth(status=overall_status, metrics=metrics)
        
        # Track performance
        check_duration = (time.time() - start_time) * 1000
        self._performance_metrics['check_duration_ms'].append(check_duration)
        if len(self._performance_metrics['check_duration_ms']) > 100:
            self._performance_metrics['check_duration_ms'].pop(0)
        
        # Update history
        self._last_health_check = health
        self._health_history.append(health)
        if len(self._health_history) > self._max_history:
            self._health_history.pop(0)
        
        # Trigger alerts if needed
        await self._handle_alerts(health)
        
        logger.info(f"Health check completed in {check_duration:.1f}ms - Status: {overall_status.value}")
        
        return health
    
    async def _check_mt5_connection(self) -> HealthMetric:
        """Check MT5 connection health"""
        try:
            if not self.mt5_client:
                return HealthMetric(
                    name="mt5_connection",
                    status=HealthStatus.UNKNOWN,
                    value=None,
                    message="MT5 client not initialized"
                )
            
            # Check if MT5 is initialized
            if not self.mt5_client.is_initialized():
                return HealthMetric(
                    name="mt5_connection",
                    status=HealthStatus.CRITICAL,
                    value=False,
                    message="MT5 not initialized"
                )
            
            # Get account info
            account_info = self.mt5_client.get_account_info()
            if not account_info:
                return HealthMetric(
                    name="mt5_connection",
                    status=HealthStatus.CRITICAL,
                    value=False,
                    message="Cannot retrieve account info"
                )
            
            # Check account metrics
            balance = account_info.get('balance', 0)
            equity = account_info.get('equity', 0)
            margin_level = account_info.get('margin_level', 0)
            
            # Determine status based on account health
            if margin_level > 0 and margin_level < 200:
                status = HealthStatus.WARNING
                message = f"Low margin level: {margin_level:.1f}%"
            elif equity < balance * 0.8:
                status = HealthStatus.WARNING
                message = f"Significant drawdown: {((balance - equity) / balance * 100):.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = "MT5 connection healthy"
            
            return HealthMetric(
                name="mt5_connection",
                status=status,
                value=True,
                message=message,
                metadata={
                    'balance': balance,
                    'equity': equity,
                    'margin_level': margin_level,
                    'login': account_info.get('login')
                }
            )
            
        except Exception as e:
            logger.error(f"MT5 health check error: {e}")
            return HealthMetric(
                name="mt5_connection",
                status=HealthStatus.CRITICAL,
                value=False,
                message=f"MT5 check failed: {str(e)}"
            )
    
    async def _check_database_health(self) -> HealthMetric:
        """Check database health"""
        try:
            start_time = time.time()
            
            # Test database connection
            with self.trade_repository.get_connection() as conn:
                # Check tables
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
                tables = [row[0] for row in cursor.fetchall()]
                
                required_tables = ['trades', 'signals', 'memory_cases', 'gpt_requests', 'ml_predictions']
                missing_tables = [t for t in required_tables if t not in tables]
                
                if missing_tables:
                    return HealthMetric(
                        name="database",
                        status=HealthStatus.CRITICAL,
                        value=False,
                        message=f"Missing tables: {missing_tables}"
                    )
                
                # Check database size
                db_path = self.settings.database.db_path
                db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
                
                # Check query performance
                cursor.execute("SELECT COUNT(*) FROM trades")
                trade_count = cursor.fetchone()[0]
                
                query_time = (time.time() - start_time) * 1000
                self._performance_metrics['db_query_times'].append(query_time)
                if len(self._performance_metrics['db_query_times']) > 100:
                    self._performance_metrics['db_query_times'].pop(0)
                
                # Determine status
                if db_size_mb > 1000:  # 1GB warning
                    status = HealthStatus.WARNING
                    message = f"Database size large: {db_size_mb:.1f}MB"
                elif query_time > 100:  # 100ms warning
                    status = HealthStatus.WARNING
                    message = f"Slow database queries: {query_time:.1f}ms"
                else:
                    status = HealthStatus.HEALTHY
                    message = "Database healthy"
                
                return HealthMetric(
                    name="database",
                    status=status,
                    value=True,
                    message=message,
                    metadata={
                        'size_mb': db_size_mb,
                        'trade_count': trade_count,
                        'query_time_ms': query_time,
                        'tables': tables
                    }
                )
                
        except Exception as e:
            logger.error(f"Database health check error: {e}")
            return HealthMetric(
                name="database",
                status=HealthStatus.CRITICAL,
                value=False,
                message=f"Database check failed: {str(e)}"
            )
    
    async def _check_api_connectivity(self) -> HealthMetric:
        """Check external API connectivity"""
        try:
            issues = []
            metadata = {}
            
            # Check GPT client
            if self.gpt_client:
                try:
                    # Simple test - this should be fast
                    test_response = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.gpt_client.client.completions.create,
                            model="gpt-3.5-turbo-instruct",
                            prompt="test",
                            max_tokens=1
                        ),
                        timeout=5.0
                    )
                    metadata['gpt_status'] = 'connected'
                except asyncio.TimeoutError:
                    issues.append("GPT API timeout")
                    metadata['gpt_status'] = 'timeout'
                except Exception as e:
                    issues.append(f"GPT API error: {str(e)}")
                    metadata['gpt_status'] = 'error'
            
            # Check MarketAux client
            if self.marketaux_client:
                try:
                    # Check rate limit status
                    rate_limit_info = await self.marketaux_client.get_rate_limit_status()
                    metadata['marketaux_status'] = 'connected'
                    metadata['marketaux_remaining'] = rate_limit_info.get('remaining', 0)
                    
                    if rate_limit_info.get('remaining', 0) < 10:
                        issues.append("MarketAux API limit low")
                except Exception as e:
                    issues.append(f"MarketAux API error: {str(e)}")
                    metadata['marketaux_status'] = 'error'
            
            # Determine status
            if len(issues) >= 2:
                status = HealthStatus.CRITICAL
                message = "Multiple API issues"
            elif issues:
                status = HealthStatus.WARNING
                message = issues[0]
            else:
                status = HealthStatus.HEALTHY
                message = "All APIs connected"
            
            return HealthMetric(
                name="api_connectivity",
                status=status,
                value=len(issues) == 0,
                message=message,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"API connectivity check error: {e}")
            return HealthMetric(
                name="api_connectivity",
                status=HealthStatus.UNKNOWN,
                value=None,
                message=f"API check failed: {str(e)}"
            )
    
    async def _check_system_resources(self) -> HealthMetric:
        """Check system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Process-specific metrics
            process = psutil.Process()
            process_memory_mb = process.memory_info().rss / (1024 * 1024)
            
            # Determine status
            issues = []
            if cpu_percent > 80:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            if memory_percent > 85:
                issues.append(f"High memory usage: {memory_percent:.1f}%")
            if disk_percent > 90:
                issues.append(f"Low disk space: {disk_percent:.1f}% used")
            if process_memory_mb > 500:
                issues.append(f"High process memory: {process_memory_mb:.1f}MB")
            
            if len(issues) >= 2:
                status = HealthStatus.CRITICAL
                message = "; ".join(issues)
            elif issues:
                status = HealthStatus.WARNING
                message = issues[0]
            else:
                status = HealthStatus.HEALTHY
                message = "System resources healthy"
            
            return HealthMetric(
                name="system_resources",
                status=status,
                value=True,
                message=message,
                metadata={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'disk_percent': disk_percent,
                    'process_memory_mb': process_memory_mb
                }
            )
            
        except Exception as e:
            logger.error(f"System resources check error: {e}")
            return HealthMetric(
                name="system_resources",
                status=HealthStatus.UNKNOWN,
                value=None,
                message=f"Resource check failed: {str(e)}"
            )
    
    async def _check_circuit_breakers(self) -> HealthMetric:
        """Check circuit breaker states"""
        try:
            manager = CircuitBreakerManager()
            report = manager.get_all_states()
            
            # Count breakers by state
            open_breakers = [name for name, info in report.items() if info['state'] == 'OPEN']
            half_open_breakers = [name for name, info in report.items() if info['state'] == 'HALF_OPEN']
            
            # Determine status
            if open_breakers:
                status = HealthStatus.WARNING
                message = f"Circuit breakers open: {', '.join(open_breakers)}"
            elif half_open_breakers:
                status = HealthStatus.WARNING
                message = f"Circuit breakers recovering: {', '.join(half_open_breakers)}"
            else:
                status = HealthStatus.HEALTHY
                message = "All circuit breakers closed"
            
            return HealthMetric(
                name="circuit_breakers",
                status=status,
                value=len(open_breakers) == 0,
                message=message,
                metadata={
                    'open': open_breakers,
                    'half_open': half_open_breakers,
                    'report': report
                }
            )
            
        except Exception as e:
            logger.error(f"Circuit breaker check error: {e}")
            return HealthMetric(
                name="circuit_breakers",
                status=HealthStatus.UNKNOWN,
                value=None,
                message=f"Circuit breaker check failed: {str(e)}"
            )
    
    async def _check_recent_activity(self) -> HealthMetric:
        """Check recent trading activity"""
        try:
            # Get recent signals
            recent_signals = self.signal_repository.get_recent_signals(hours=24)
            
            # Get recent trades
            recent_trades = self.trade_repository.get_trades_by_date_range(
                start_date=datetime.now(timezone.utc) - timedelta(days=1),
                end_date=datetime.now(timezone.utc)
            )
            
            # Calculate metrics
            signal_count = len(recent_signals)
            trade_count = len(recent_trades)
            
            # Check for activity
            hours_since_last_signal = 24
            if recent_signals:
                last_signal_time = max(s.timestamp for s in recent_signals)
                hours_since_last_signal = (
                    datetime.now(timezone.utc) - last_signal_time
                ).total_seconds() / 3600
            
            # Determine status (only warn during market hours)
            if self._is_market_open():
                if hours_since_last_signal > 12:
                    status = HealthStatus.WARNING
                    message = f"No signals in {hours_since_last_signal:.1f} hours"
                elif signal_count < 5:
                    status = HealthStatus.WARNING
                    message = f"Low signal activity: {signal_count} in 24h"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Active: {signal_count} signals, {trade_count} trades in 24h"
            else:
                status = HealthStatus.HEALTHY
                message = "Market closed"
            
            return HealthMetric(
                name="recent_activity",
                status=status,
                value=True,
                message=message,
                metadata={
                    'signal_count_24h': signal_count,
                    'trade_count_24h': trade_count,
                    'hours_since_last_signal': hours_since_last_signal
                }
            )
            
        except Exception as e:
            logger.error(f"Recent activity check error: {e}")
            return HealthMetric(
                name="recent_activity",
                status=HealthStatus.UNKNOWN,
                value=None,
                message=f"Activity check failed: {str(e)}"
            )
    
    async def _check_error_rates(self) -> HealthMetric:
        """Check system error rates"""
        try:
            # Read recent logs
            log_file = self.settings.logging.file
            error_count = 0
            warning_count = 0
            lines_checked = 0
            
            if os.path.exists(log_file):
                # Read last 1000 lines
                with open(log_file, 'r') as f:
                    lines = f.readlines()[-1000:]
                
                for line in lines:
                    lines_checked += 1
                    if 'ERROR' in line:
                        error_count += 1
                    elif 'WARNING' in line:
                        warning_count += 1
            
            # Calculate error rate
            error_rate = (error_count / lines_checked * 100) if lines_checked > 0 else 0
            warning_rate = (warning_count / lines_checked * 100) if lines_checked > 0 else 0
            
            # Determine status
            if error_rate > 5:
                status = HealthStatus.CRITICAL
                message = f"High error rate: {error_rate:.1f}%"
            elif error_rate > 2 or warning_rate > 10:
                status = HealthStatus.WARNING
                message = f"Elevated errors: {error_rate:.1f}% errors, {warning_rate:.1f}% warnings"
            else:
                status = HealthStatus.HEALTHY
                message = "Error rates normal"
            
            return HealthMetric(
                name="error_rates",
                status=status,
                value=error_rate < 2,
                message=message,
                metadata={
                    'error_count': error_count,
                    'warning_count': warning_count,
                    'lines_checked': lines_checked,
                    'error_rate': error_rate,
                    'warning_rate': warning_rate
                }
            )
            
        except Exception as e:
            logger.error(f"Error rate check error: {e}")
            return HealthMetric(
                name="error_rates",
                status=HealthStatus.UNKNOWN,
                value=None,
                message=f"Error rate check failed: {str(e)}"
            )
    
    def _is_market_open(self) -> bool:
        """Check if forex market is open"""
        now = datetime.now(timezone.utc)
        weekday = now.weekday()
        
        # Market closed on weekends (Friday 21:00 UTC to Sunday 21:00 UTC)
        if weekday == 5:  # Saturday
            return False
        elif weekday == 6:  # Sunday
            return now.hour >= 21
        elif weekday == 4:  # Friday
            return now.hour < 21
        else:
            return True
    
    async def _handle_alerts(self, health: SystemHealth):
        """Handle health alerts"""
        try:
            # Check for critical issues
            if health.status == HealthStatus.CRITICAL:
                for handler in self._alert_handlers:
                    try:
                        handler(health)
                    except Exception as e:
                        logger.error(f"Alert handler error: {e}")
            
            # Log health status changes
            if self._last_health_check and self._last_health_check.status != health.status:
                logger.warning(
                    f"Health status changed: {self._last_health_check.status.value} -> {health.status.value}"
                )
                
        except Exception as e:
            logger.error(f"Alert handling error: {e}")
    
    def add_alert_handler(self, handler: Callable[[SystemHealth], None]):
        """Add an alert handler"""
        self._alert_handlers.append(handler)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary"""
        if not self._last_health_check:
            return {'status': 'unknown', 'message': 'No health check performed yet'}
        
        return self._last_health_check.to_dict()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        avg_check_time = (
            sum(self._performance_metrics['check_duration_ms']) / 
            len(self._performance_metrics['check_duration_ms'])
        ) if self._performance_metrics['check_duration_ms'] else 0
        
        avg_db_time = (
            sum(self._performance_metrics['db_query_times']) / 
            len(self._performance_metrics['db_query_times'])
        ) if self._performance_metrics['db_query_times'] else 0
        
        return {
            'avg_health_check_ms': avg_check_time,
            'avg_db_query_ms': avg_db_time,
            'health_check_count': len(self._health_history)
        }


# Standalone health check function for scripts
async def run_health_check():
    """Run a standalone health check"""
    from config.settings import get_settings
    
    settings = get_settings()
    monitor = HealthMonitor(settings)
    
    health = await monitor.check_health()
    
    print(f"\n{'='*60}")
    print(f"System Health Check - {health.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*60}")
    print(f"Overall Status: {health.status.value.upper()}")
    print(f"{'='*60}\n")
    
    # Print each metric
    for name, metric in health.metrics.items():
        status_symbol = {
            HealthStatus.HEALTHY: "✓",
            HealthStatus.WARNING: "⚠",
            HealthStatus.CRITICAL: "✗",
            HealthStatus.UNKNOWN: "?"
        }[metric.status]
        
        print(f"{status_symbol} {name}: {metric.message}")
        if metric.metadata:
            for key, value in metric.metadata.items():
                print(f"  - {key}: {value}")
        print()
    
    # Print summary
    if health.critical_issues:
        print(f"\nCRITICAL ISSUES ({len(health.critical_issues)}):")
        for issue in health.critical_issues:
            print(f"  - {issue.name}: {issue.message}")
    
    if health.warnings:
        print(f"\nWARNINGS ({len(health.warnings)}):")
        for warning in health.warnings:
            print(f"  - {warning.name}: {warning.message}")
    
    return health


if __name__ == "__main__":
    import sys
    
    # Run health check
    health = asyncio.run(run_health_check())
    
    # Exit with appropriate code
    if health.status == HealthStatus.CRITICAL:
        sys.exit(2)
    elif health.status == HealthStatus.WARNING:
        sys.exit(1)
    else:
        sys.exit(0)