"""
System health monitoring and alerting for GPT Trading System.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
import json
import aiohttp

from config.settings import get_settings
from core.infrastructure.mt5.client import MT5Client
from core.infrastructure.database.repositories import TradeRepository

logger = logging.getLogger(__name__)


class HealthChecker:
    """System health monitoring"""
    
    def __init__(self):
        self.settings = get_settings()
        self.trade_repository = TradeRepository(self.settings.database.db_path)
        self.alert_cooldown = {}  # Prevent alert spam
    
    async def check_all(self) -> Dict[str, Any]:
        """Run all health checks"""
        checks = {
            "mt5_connection": await self._check_mt5_connection(),
            "database": await self._check_database(),
            "recent_activity": await self._check_recent_activity(),
            "error_rate": await self._check_error_rate(),
            "disk_space": await self._check_disk_space(),
            "api_connectivity": await self._check_api_connectivity()
        }
        
        # Overall health status
        healthy = all(check['healthy'] for check in checks.values())
        issues = [f"{name}: {check['message']}" 
                 for name, check in checks.items() 
                 if not check['healthy']]
        
        return {
            'healthy': healthy,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'checks': checks,
            'issues': issues
        }
    
    async def _check_mt5_connection(self) -> Dict[str, Any]:
        """Check MT5 connection status"""
        try:
            # MT5Client needs MT5Settings config
            mt5_client = MT5Client(self.settings.mt5)
            
            if mt5_client.initialize():
                account_info = mt5_client.get_account_info()
                mt5_client.shutdown()
                
                if account_info:
                    return {
                        'healthy': True,
                        'message': f'Connected to account {account_info.get("login", "Unknown")}',
                        'balance': account_info.get('balance', 0),
                        'equity': account_info.get('equity', 0)
                    }
                else:
                    return {
                        'healthy': False,
                        'message': 'Connected but no account info available'
                    }
            else:
                return {
                    'healthy': False,
                    'message': 'Failed to connect to MT5'
                }
                
        except Exception as e:
            return {
                'healthy': False,
                'message': f'MT5 connection error: {str(e)}'
            }
    
    async def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity and integrity"""
        try:
            # Test database connection
            with self.trade_repository.get_connection() as conn:
                # Check tables exist
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
                tables = [row[0] for row in cursor.fetchall()]
                
                required_tables = ['trades', 'signals', 'memory_cases']
                missing_tables = [t for t in required_tables if t not in tables]
                
                if missing_tables:
                    return {
                        'healthy': False,
                        'message': f'Missing tables: {missing_tables}'
                    }
                
                # Check database size
                db_path = self.settings.database.db_path
                db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
                
                return {
                    'healthy': True,
                    'message': f'Database OK, size: {db_size_mb:.1f}MB',
                    'tables': tables,
                    'size_mb': db_size_mb
                }
                
        except Exception as e:
            return {
                'healthy': False,
                'message': f'Database error: {str(e)}'
            }
    
    async def _check_recent_activity(self) -> Dict[str, Any]:
        """Check for recent trading activity"""
        try:
            # Get trades from last 24 hours
            recent_trades = self.trade_repository.get_trades_by_date_range(
                start_date=datetime.now(timezone.utc) - timedelta(days=1),
                end_date=datetime.now(timezone.utc)
            )
            
            # Check if system is generating signals
            last_trade_time = None
            if recent_trades:
                last_trade_time = max(trade.timestamp for trade in recent_trades)
                hours_since_last = (
                    datetime.now(timezone.utc) - last_trade_time
                ).total_seconds() / 3600
                
                # Warning if no trades in last 12 hours during market hours
                if hours_since_last > 12 and self._is_market_open():
                    return {
                        'healthy': False,
                        'message': f'No trades in {hours_since_last:.1f} hours',
                        'last_trade': last_trade_time.isoformat()
                    }
            
            return {
                'healthy': True,
                'message': f'{len(recent_trades)} trades in last 24h',
                'trade_count': len(recent_trades),
                'last_trade': last_trade_time.isoformat() if last_trade_time else None
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'message': f'Activity check error: {str(e)}'
            }
    
    async def _check_error_rate(self) -> Dict[str, Any]:
        """Check system error rate"""
        try:
            # Parse recent log files for errors
            log_path = 'logs/trading_system.log'
            if not os.path.exists(log_path):
                return {
                    'healthy': True,
                    'message': 'No log file found',
                    'error_count': 0
                }
            
            # Count errors in last hour
            one_hour_ago = datetime.now() - timedelta(hours=1)
            error_count = 0
            
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if 'ERROR' in line:
                        # Parse timestamp (assuming format: 2024-01-01 12:00:00)
                        try:
                            timestamp_str = line.split(' | ')[0]
                            timestamp = datetime.strptime(
                                timestamp_str, '%Y-%m-%d %H:%M:%S'
                            )
                            if timestamp > one_hour_ago:
                                error_count += 1
                        except:
                            continue
            
            # Alert if more than 10 errors per hour
            if error_count > 10:
                return {
                    'healthy': False,
                    'message': f'{error_count} errors in last hour',
                    'error_count': error_count
                }
            
            return {
                'healthy': True,
                'message': f'{error_count} errors in last hour',
                'error_count': error_count
            }
            
        except Exception as e:
            return {
                'healthy': True,  # Don't fail health check for log parsing
                'message': f'Log parsing error: {str(e)}',
                'error_count': 0
            }
    
    async def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space"""
        try:
            import shutil
            
            # Get disk usage for data directory
            stat = shutil.disk_usage(self.settings.database.db_path)
            free_gb = stat.free / (1024**3)
            used_percent = (stat.used / stat.total) * 100
            
            # Alert if less than 1GB free or more than 90% used
            if free_gb < 1 or used_percent > 90:
                return {
                    'healthy': False,
                    'message': f'Low disk space: {free_gb:.1f}GB free ({used_percent:.1f}% used)',
                    'free_gb': free_gb,
                    'used_percent': used_percent
                }
            
            return {
                'healthy': True,
                'message': f'{free_gb:.1f}GB free ({used_percent:.1f}% used)',
                'free_gb': free_gb,
                'used_percent': used_percent
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'message': f'Disk check error: {str(e)}'
            }
    
    async def _check_api_connectivity(self) -> Dict[str, Any]:
        """Check connectivity to external APIs"""
        try:
            # Test OpenAI API
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {self.settings.gpt.api_key}',
                    'Content-Type': 'application/json'
                }
                
                # Simple API test
                async with session.get(
                    'https://api.openai.com/v1/models',
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        return {
                            'healthy': True,
                            'message': 'API connectivity OK',
                            'openai_status': 'connected'
                        }
                    else:
                        return {
                            'healthy': False,
                            'message': f'OpenAI API error: {response.status}',
                            'openai_status': 'error'
                        }
                        
        except asyncio.TimeoutError:
            return {
                'healthy': False,
                'message': 'API connection timeout',
                'openai_status': 'timeout'
            }
        except Exception as e:
            return {
                'healthy': False,
                'message': f'API check error: {str(e)}',
                'openai_status': 'error'
            }
    
    def _is_market_open(self) -> bool:
        """Check if forex market is open"""
        now = datetime.now(timezone.utc)
        weekday = now.weekday()
        hour = now.hour
        
        # Market closed on Saturday
        if weekday == 5:
            return False
        
        # Market closed Friday after 22:00 UTC
        if weekday == 4 and hour >= 22:
            return False
        
        # Market closed Sunday before 22:00 UTC
        if weekday == 6 and hour < 22:
            return False
        
        return True
    
    async def send_alert(self, subject: str, message: str):
        """Send alert via configured channels"""
        # Check cooldown to prevent spam
        cooldown_key = f"{subject}:{message[:50]}"
        if cooldown_key in self.alert_cooldown:
            last_sent = self.alert_cooldown[cooldown_key]
            if (datetime.now() - last_sent).total_seconds() < 3600:  # 1 hour cooldown
                return
        
        # Log the alert
        logger.error(f"ALERT: {subject} - {message}")
        
        # Send via Telegram if configured
        if self.settings.telegram.enabled:
            await self._send_telegram_alert(subject, message)
        
        # Update cooldown
        self.alert_cooldown[cooldown_key] = datetime.now()
    
    async def send_alerts(self, issues: List[str]):
        """Send multiple alerts"""
        if issues:
            combined_message = "System Health Issues:\n" + "\n".join(f"• {issue}" for issue in issues)
            await self.send_alert("System Health Alert", combined_message)
    
    async def _send_telegram_alert(self, subject: str, message: str):
        """Send alert via Telegram"""
        try:
            from core.infrastructure.notifications.telegram import TelegramNotifier
            
            notifier = TelegramNotifier(
                token=self.settings.telegram.token,
                chat_id=self.settings.telegram.chat_id
            )
            
            full_message = f"🚨 *{subject}*\n\n{message}\n\n_Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC_"
            await notifier.send_message(full_message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")


async def main():
    """Run health check standalone"""
    import argparse
    
    parser = argparse.ArgumentParser(description='System Health Check')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    args = parser.parse_args()
    
    checker = HealthChecker()
    result = await checker.check_all()
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"System Health: {'✅ HEALTHY' if result['healthy'] else '❌ UNHEALTHY'}")
        print(f"Timestamp: {result['timestamp']}")
        
        for check_name, check_result in result['checks'].items():
            status = '✅' if check_result['healthy'] else '❌'
            print(f"{status} {check_name}: {check_result['message']}")
        
        if result['issues']:
            print("\nIssues found:")
            for issue in result['issues']:
                print(f"  • {issue}")


if __name__ == "__main__":
    asyncio.run(main())