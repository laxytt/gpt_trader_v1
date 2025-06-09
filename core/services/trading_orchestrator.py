"""
Trading orchestrator service that coordinates the entire trading workflow.
Manages the main trading loop, symbol processing, and system coordination.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum

from core.domain.models import MarketData, TradingSignal, Trade, SignalType, TradeStatus
from core.domain.exceptions import (
    TradingSystemError, ErrorContext, ServiceError, ConfigurationError, ValidationError
)
from core.services.council_signal_service import CouncilSignalService
from core.services.trade_service import TradeService  
from core.services.news_service import NewsService
from core.services.memory_service import MemoryService
from core.services.market_service import MarketService
from core.infrastructure.mt5.client import MT5Client
from core.utils.validation import validate_symbol_format
from config.settings import TradingSettings
from config.position_trading_config import PositionTradingConfig
from pathlib import Path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config.settings import Settings
    from core.services.health_monitor import HealthMonitor
    from core.services.position_monitor import PositionMonitor

from core.utils.error_handler import (
    get_error_handler, ErrorSeverity, ErrorCategory,
    handle_errors, error_context
)
from core.utils.structured_logger import get_logger, log_performance


logger = get_logger(__name__)


class TradingState(Enum):
    """Trading system states"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class SymbolProcessor:
    """Processes individual symbols for trading opportunities"""
    
    def __init__(
        self,
        signal_service: CouncilSignalService,
        trade_service: TradeService,
        news_service: NewsService,
        trading_settings: TradingSettings,
        orchestrator: Optional['TradingOrchestrator'] = None
    ):
        self.signal_service = signal_service
        self.trade_service = trade_service
        self.news_service = news_service
        self.trading_settings = trading_settings
        self.orchestrator = orchestrator
    
    async def process_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Process a single symbol for trading opportunities.
        
        Args:
            symbol: Trading symbol to process
            
        Returns:
            Dictionary with processing results
        """
        with ErrorContext("Symbol processing", symbol=symbol) as ctx:
            result = {
                'symbol': symbol,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'action_taken': None,
                'signal': None,
                'trade': None,
                'error': None
            }
            
            try:
                # Validate symbol format
                validate_symbol_format(symbol)
                
                # Check for existing open trade
                existing_trade = await self.trade_service.get_trade_by_symbol(symbol)
                
                if existing_trade:
                    # Manage existing trade
                    ctx.add_detail("existing_trade", existing_trade.id)
                    still_open = await self.trade_service.manage_trade(existing_trade)
                    
                    result['action_taken'] = 'trade_managed'
                    result['trade'] = {
                        'id': existing_trade.id,
                        'status': existing_trade.status.value,
                        'still_open': still_open
                    }
                    
                    logger.info(f"Managed existing trade for {symbol}: {existing_trade.id}")
                    
                else:
                    # Check position trading constraints
                    if self.orchestrator and self.orchestrator._position_trading_mode:
                        can_trade, reason = self.orchestrator._check_position_trading_constraints(symbol)
                        if not can_trade:
                            result['action_taken'] = 'skipped_constraint'
                            result['error'] = reason
                            logger.info(f"Position trading constraint for {symbol}: {reason}")
                            return result
                    
                    # Look for new trading opportunity
                    ctx.add_detail("action", "signal_generation")
                    
                    # Check if symbol is ready for analysis
                    if not await self.signal_service.validate_symbol_readiness(symbol):
                        result['action_taken'] = 'skipped_not_ready'
                        result['error'] = 'Symbol not ready for analysis'
                        return result
                    
                    # Generate signal - use ML if enabled
                    ml_enabled = (self.orchestrator and self.orchestrator.settings and 
                                 hasattr(self.orchestrator.settings, 'ml') and 
                                 self.orchestrator.settings.ml.enabled)
                    
                    if ml_enabled and hasattr(self.signal_service, 'generate_signal_with_ml'):
                        logger.info(f"Using ML-based signal generation for {symbol}")
                        signal = await self.signal_service.generate_signal_with_ml(symbol)
                    else:
                        signal = await self.signal_service.generate_signal(symbol)
                    
                    result['signal'] = {
                        'type': signal.signal.value,
                        'risk_class': signal.risk_class.value,
                        'reason': signal.reason
                    }
                    
                    # Execute signal if actionable
                    if signal.is_actionable:
                        ctx.add_detail("action", "trade_execution")
                        trade = await self.trade_service.execute_signal(signal)
                        
                        if trade:
                            result['action_taken'] = 'trade_executed'
                            result['trade'] = {
                                'id': trade.id,
                                'side': trade.side.value,
                                'entry_price': trade.entry_price
                            }
                            logger.info(f"Executed new trade for {symbol}: {trade.id}")
                        else:
                            result['action_taken'] = 'execution_failed'
                            result['error'] = 'Trade execution failed'
                    else:
                        result['action_taken'] = 'signal_wait'
                        logger.debug(f"WAIT signal for {symbol}: {signal.reason}")
                
                return result
                
            except (ValueError, ValidationError) as e:
                logger.error(f"Symbol validation failed for {symbol}: {e}")
                result['action_taken'] = 'error'
                result['error'] = str(e)
                return result
            except ServiceError as e:
                logger.error(f"Service error processing {symbol}: {e}")
                result['action_taken'] = 'error'
                result['error'] = str(e)
                return result
            except Exception as e:
                logger.error(f"Unexpected error processing {symbol}: {e}")
                result['action_taken'] = 'error'
                result['error'] = str(e)
                return result


class TradingScheduler:
    """Manages trading schedule and timing"""
    
    def __init__(self, trading_config: TradingSettings):
        self.trading_config = trading_config
        self.start_hour = trading_config.start_hour
        self.end_hour = trading_config.end_hour
    
    def is_trading_hours(self, now: Optional[datetime] = None) -> bool:
        """Check if current time is within trading hours and forex market is open"""
        if now is None:
            now = datetime.now(timezone.utc)
        
        # Check if forex market is closed (Friday 22:00 UTC to Sunday 22:00 UTC)
        weekday = now.weekday()  # Monday=0, Sunday=6
        hour = now.hour
        
        # Market closed on Saturday
        if weekday == 5:  # Saturday
            return False
        
        # Market closed Friday after 22:00 UTC
        if weekday == 4 and hour >= 22:  # Friday
            return False
            
        # Market closed Sunday before 22:00 UTC
        if weekday == 6 and hour < 22:  # Sunday
            return False
        
        # Check configured trading hours
        current_hour = now.hour
        
        # Handle overnight trading (e.g., 22:00 to 06:00)
        if self.start_hour > self.end_hour:
            return current_hour >= self.start_hour or current_hour < self.end_hour
        else:
            return self.start_hour <= current_hour < self.end_hour
    
    def time_until_next_session(self, now: Optional[datetime] = None) -> timedelta:
        """Calculate time until next trading session"""
        if now is None:
            now = datetime.now(timezone.utc)
        
        if self.is_trading_hours(now):
            return timedelta(0)
        
        weekday = now.weekday()
        hour = now.hour
        
        # If it's weekend, calculate time until Sunday 22:00 UTC
        if weekday == 5:  # Saturday
            days_until_sunday = 1
            next_session = now.replace(hour=22, minute=0, second=0, microsecond=0)
            next_session += timedelta(days=days_until_sunday)
            return next_session - now
        
        if weekday == 6 and hour < 22:  # Sunday before 22:00
            next_session = now.replace(hour=22, minute=0, second=0, microsecond=0)
            return next_session - now
        
        if weekday == 4 and hour >= 22:  # Friday after 22:00
            # Next session is Sunday 22:00
            days_until_sunday = 2
            next_session = now.replace(hour=22, minute=0, second=0, microsecond=0)
            next_session += timedelta(days=days_until_sunday)
            return next_session - now
        
        # Regular weekday logic
        today_start = now.replace(hour=self.start_hour, minute=0, second=0, microsecond=0)
        
        if now.hour < self.start_hour:
            # Session starts later today
            next_session = today_start
        else:
            # Session starts tomorrow
            next_session = today_start + timedelta(days=1)
        
        return next_session - now
    
    async def wait_for_next_hour_boundary(self):
        """Wait until the next hour boundary for synchronized execution"""
        now = datetime.now(timezone.utc)
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        wait_time = (next_hour - now).total_seconds()
        
        if wait_time > 0:
            logger.info(f"Waiting {wait_time:.1f} seconds until next hour boundary ({next_hour.strftime('%H:%M:%S')})")
            await asyncio.sleep(wait_time)
    
    def seconds_until_trading_hours(self) -> float:
        """Get seconds until trading hours begin"""
        if self.is_trading_hours():
            return 0
        return self.time_until_next_session().total_seconds()
    
    def calculate_cycle_interval(self) -> float:
        """Calculate interval between trading cycles in seconds"""
        # Use configured interval from settings
        return self.trading_config.cycle_interval_minutes * 60.0


class TradingOrchestrator:
    """
    Main orchestrator that coordinates the entire trading system.
    Manages the trading loop, system state, and component coordination.
    """
    
    def __init__(
        self,
        signal_service: CouncilSignalService,
        trade_service: TradeService,
        news_service: NewsService,
        memory_service: MemoryService,
        market_service: MarketService,
        mt5_client: MT5Client,
        trading_config: TradingSettings,
        settings: Optional['Settings'] = None,
        health_monitor: Optional['HealthMonitor'] = None,
        position_monitor: Optional['PositionMonitor'] = None
    ):
        self.signal_service = signal_service
        self.trade_service = trade_service
        self.news_service = news_service
        self.memory_service = memory_service
        self.market_service = market_service
        self.mt5_client = mt5_client
        self.trading_config = trading_config
        self.settings = settings
        self.health_monitor = health_monitor
        self.position_monitor = position_monitor
        
        # Initialize components
        self.symbol_processor = SymbolProcessor(signal_service, trade_service, news_service, trading_config, self)
        self.scheduler = TradingScheduler(trading_config)
        
        # Position trading configuration
        self.position_config = PositionTradingConfig()
        self._position_trading_mode = self._detect_position_trading_mode()
        self._last_trade_times: Dict[str, datetime] = {}  # Track last trade time per symbol
        self._last_position_review: Optional[datetime] = None
        
        # System state
        self.state = TradingState.INITIALIZING
        self.cycle_count = 0
        self.last_cycle_time = None
        self.error_count = 0
        self.max_errors = 10
        self._shutdown_requested = False  # Add this missing attribute
        self._market_closed_logged = False  # Track if we've logged market closed message

        
        # Statistics
        self.stats = {
            'cycles_completed': 0,
            'signals_generated': 0,
            'trades_executed': 0,
            'trades_managed': 0,
            'errors_encountered': 0,
            'start_time': None,
            'position_reviews': 0,
            'positions_adjusted': 0
        }
    
    
    async def run(self):
        """
        Main trading loop that runs continuously.
        """
        logger.info("🚀 Starting GPT Trading System")
        await self._initialize_system()
        
        try:
            self.state = TradingState.RUNNING
            self.stats['start_time'] = datetime.now(timezone.utc).isoformat()
            
            while self.state == TradingState.RUNNING and not self._shutdown_requested:
                # Check for stop file (Windows-friendly graceful shutdown)
                stop_file = Path("STOP_TRADING")
                if stop_file.exists():
                    logger.info("🛑 Stop file detected, initiating graceful shutdown...")
                    self._shutdown_requested = True
                    stop_file.unlink()  # Remove the file
                    break
                    
                try:
                    # Check if we should trade now
                    if not self._should_trade_now():
                        await self._wait_for_trading_hours()
                        continue
                    
                    # Execute trading cycle
                    await self._execute_trading_cycle()
                    
                    # Wait until next cycle (with periodic checks for shutdown)
                    await self._wait_with_shutdown_check()
                    
                except KeyboardInterrupt:
                    logger.info("🛑 Received interrupt signal, stopping...")
                    self.state = TradingState.STOPPING
                    break
                    
                except asyncio.CancelledError:
                    # Propagate cancellation
                    raise
                except (ServiceError, TradingSystemError) as e:
                    await self._handle_cycle_error(e)
                except Exception as e:
                    await self._handle_cycle_error(e)
                    
                    if self.error_count >= self.max_errors:
                        logger.error(f"💥 Maximum error count reached ({self.max_errors}), stopping system")
                        self.state = TradingState.ERROR
                        break
            
            logger.info("🔄 Trading loop ended")
            
        except KeyboardInterrupt:
            logger.info("🛑 Keyboard interrupt received")
        except asyncio.CancelledError:
            logger.info("Trading orchestrator cancelled")
        except (ServiceError, TradingSystemError) as e:
            logger.exception(f"💥 Trading system error: {e}")
            self.state = TradingState.ERROR
        except Exception as e:
            logger.exception(f"💥 Fatal error in trading orchestrator: {e}")
            self.state = TradingState.ERROR
        
        finally:
            await self._shutdown_system()

    
    async def _wait_with_shutdown_check(self):
        """Wait until next hour boundary while checking for shutdown requests"""
        now = datetime.now(timezone.utc)
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        wait_time = (next_hour - now).total_seconds()
        
        if wait_time > 0:
            logger.info(f"⏳ Waiting {wait_time:.1f} seconds until next hour boundary")
            
            # Wait in small increments to check for shutdown
            while wait_time > 0 and self.state == TradingState.RUNNING and not self._shutdown_requested:
                sleep_time = min(10, wait_time)  # Check every 10 seconds
                await asyncio.sleep(sleep_time)
                wait_time -= sleep_time
    
    
    async def _initialize_system(self):
        """Initialize all system components"""
        logger.info("📊 Initializing trading system components...")
        
        try:
            # Initialize MT5 connection
            if not self.mt5_client.initialize():
                raise ConfigurationError("Failed to initialize MT5 connection")
            
            # Validate configuration
            self._validate_configuration()
            
            # Check news data availability
            news_status = self.news_service.get_data_file_status()
            if not news_status['file_exists']:
                logger.warning("⚠️ News data file not found - news filtering disabled")
            elif news_status['file_age_hours'] and news_status['file_age_hours'] > 48:
                logger.warning(f"⚠️ News data is {news_status['file_age_hours']:.1f} hours old")
            
            # Get memory statistics
            memory_stats = self.memory_service.get_memory_stats()
            logger.info(f"🧠 Memory service: {memory_stats}")
            
            logger.info("✅ System initialization complete")
            
        except ConfigurationError:
            # Re-raise configuration errors as-is
            raise
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"❌ Network error during initialization: {e}")
            raise ServiceError(f"Failed to connect to services: {e}") from e
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            raise ServiceError(f"Initialization failed: {e}") from e
    
    def _validate_configuration(self):
        """Validate system configuration"""
        if not self.trading_config.symbols:
            raise ConfigurationError("No trading symbols configured")
        
        if self.trading_config.risk_per_trade_percent <= 0:
            raise ConfigurationError("Risk per trade must be positive")
        
        if self.trading_config.max_open_trades <= 0:
            raise ConfigurationError("Max open trades must be positive")
        
        # Validate symbols
        for symbol in self.trading_config.symbols:
            try:
                validate_symbol_format(symbol)
            except (ValueError, ValidationError) as e:
                raise ConfigurationError(f"Invalid symbol {symbol}: {e}")
    
    def _detect_position_trading_mode(self) -> bool:
        """Detect if we're in position trading mode"""
        # Check if entry timeframe is daily or weekly
        if hasattr(self.trading_config, 'entry_timeframe'):
            return self.trading_config.entry_timeframe in ['D1', 'W1']
        return False
    
    def _should_trade(self, market_data: MarketData) -> bool:
        """Check if current time is good for trading"""
        latest = market_data.latest_candle
        if not latest:
            return False
        
        hour = latest.timestamp.hour
        
        # Skip Asian session for EURUSD/GBPUSD
        if market_data.symbol in ["EURUSD", "GBPUSD"]:
            if 22 <= hour or hour < 6:  # 10 PM - 6 AM UTC
                return False
        
        # Skip low volume hours
        if hour in [23, 0, 1, 2, 3, 4, 5]:
            return False
        
        return True
    
    async def _wait_for_trading_hours(self):
        """Wait until trading hours begin"""
        wait_time = self.scheduler.time_until_next_session()
        hours = int(wait_time.total_seconds() // 3600)
        minutes = int((wait_time.total_seconds() % 3600) // 60)
        
        logger.info(f"⏸️ Outside trading hours. Waiting {hours}h {minutes}m for next session.")
        await asyncio.sleep(min(3600, wait_time.total_seconds()))  # Sleep max 1 hour at a time
    
    @log_performance(threshold=5.0, operation="trading_cycle")
    async def _execute_trading_cycle(self):
        """Execute one complete trading cycle"""
        self.cycle_count += 1
        cycle_start = datetime.now(timezone.utc)
        
        with logger.context(cycle_number=self.cycle_count, cycle_start=cycle_start.isoformat()):
            logger.info(f"🔄 Starting trading cycle #{self.cycle_count} at {cycle_start.strftime('%H:%M:%S')}")
        
        try:
            # Perform health check if available
            if self.health_monitor and self.cycle_count % 5 == 0:  # Check every 5 cycles
                try:
                    health = await self.health_monitor.check_health()
                    if health.status.value == 'critical':
                        logger.error(f"🚨 Critical health issues detected: {[m.name for m in health.critical_issues]}")
                        # Don't trade if system is critical
                        self.stats['cycles_skipped'] = self.stats.get('cycles_skipped', 0) + 1
                        return
                    elif health.warnings:
                        logger.warning(f"⚠️ Health warnings: {[m.name for m in health.warnings]}")
                except (ServiceError, asyncio.TimeoutError) as e:
                    logger.error(f"Health check failed: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error in health check: {e}")
            
            # Perform position review if in position trading mode
            if self._position_trading_mode and self.position_monitor:
                await self._perform_position_review()
            
            # Get active symbols (filter by market conditions if needed)
            active_symbols = await self._get_active_symbols()
            
            # Apply position trading frequency limits
            if self._position_trading_mode:
                active_symbols = self._filter_symbols_by_cooldown(active_symbols)
            
            # Fetch real-time news for all active symbols efficiently (if enhanced news service)
            if hasattr(self.news_service, 'fetch_realtime_market_news'):
                try:
                    logger.info("📰 Fetching real-time market news...")
                    realtime_news = await self.news_service.fetch_realtime_market_news(
                        symbols=active_symbols,
                        hours=24,
                        force_refresh=(self.cycle_count % 10 == 1)  # Force refresh every 10 cycles
                    )
                    total_articles = sum(len(articles) for articles in realtime_news.values())
                    logger.info(f"📰 Retrieved {total_articles} real-time news articles for {len(active_symbols)} symbols")
                except (ServiceError, asyncio.TimeoutError, ConnectionError) as e:
                    logger.warning(f"Failed to fetch real-time news: {e}")
                except Exception as e:
                    logger.warning(f"Unexpected error fetching news: {e}")
            
            # Get market overview for logging
            try:
                market_overview = await self.market_service.get_market_overview(active_symbols)
                session_info = market_overview['session']
                summary = market_overview['summary']
                
                logger.info(f"📊 Market Overview - Session: {session_info['current_session']}")
                logger.info(f"📈 Market Summary - Excellent: {summary['excellent_conditions']}, "
                        f"Good: {summary['good_conditions']}, "
                        f"Avg Score: {summary['average_score']}")
            except (ServiceError, asyncio.TimeoutError) as e:
                logger.warning(f"Failed to get market overview: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error getting market overview: {e}")
        
            if not active_symbols:
                logger.warning("⚠️ No active symbols for trading")
                return
            
            # Process each symbol
            results = await self._process_symbols(active_symbols)
            
            # Update statistics
            self._update_statistics(results)
            
            # Log cycle summary
            self._log_cycle_summary(results, cycle_start)
                
            self.stats['cycles_completed'] += 1
            self.last_cycle_time = cycle_start
            self.error_count = 0  # Reset error count on successful cycle
            
        except asyncio.CancelledError:
            # Propagate cancellation
            raise
        except (ServiceError, TradingSystemError) as e:
            logger.error(f"❌ Trading cycle #{self.cycle_count} failed: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error in trading cycle #{self.cycle_count}: {e}")
            raise
    
    async def _get_active_symbols(self) -> List[str]:
        """Get list of symbols active for trading using market intelligence"""
        try:
            # Get market intelligence for all configured symbols
            market_intelligence = await self.market_service.get_market_intelligence(
                self.trading_config.symbols
            )
            
            # Filter symbols based on market conditions
            active_symbols = []
            for symbol, intel in market_intelligence.items():
                # Only trade symbols with good conditions and decent scores
                if (intel.score >= 60 and 
                    intel.condition.value in ['excellent', 'good'] and
                    intel.data_quality in ['good', 'limited']):
                    active_symbols.append(symbol)
                    logger.debug(f"{symbol}: {intel.condition.value} (score: {intel.score})")
                else:
                    logger.debug(f"{symbol}: {intel.condition.value} (score: {intel.score}) - FILTERED OUT")
            
            if not active_symbols:
                # Fallback: use symbols with minimum acceptable conditions
                fallback_symbols = await self.market_service.filter_tradeable_symbols(
                    self.trading_config.symbols, 
                    min_score=40  # Lower threshold for fallback
                )
                
                if fallback_symbols:
                    logger.info(f"Using fallback symbols with lower threshold: {fallback_symbols}")
                    return fallback_symbols
                else:
                    logger.warning("No symbols meet even minimum trading conditions")
                    return self.trading_config.symbols[:2]  # Emergency fallback to first 2 symbols
            
            logger.info(f"Active symbols based on market conditions: {active_symbols}")
            return active_symbols
            
        except ServiceError as e:
            logger.error(f"Service error getting market intelligence: {e}")
            return self.trading_config.symbols
        except asyncio.TimeoutError:
            logger.error("Timeout getting market intelligence, using all configured symbols")
            return self.trading_config.symbols
        except Exception as e:
            logger.error(f"Unexpected error getting market intelligence: {e}")
            return self.trading_config.symbols
    
    async def _process_symbols(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Process multiple symbols sequentially with production delays"""
        results = []
        
        # Process symbols sequentially to avoid overwhelming the system
        for i, symbol in enumerate(symbols):
            try:
                result = await self.symbol_processor.process_symbol(symbol)
                results.append(result)
                
                # Production delay between symbols (not after last symbol)
                if i < len(symbols) - 1 and self.settings:
                    delay = self.settings.trading.symbol_processing_delay
                    logger.info(f"⏳ Waiting {delay}s before processing next symbol...")
                    await asyncio.sleep(delay)
                
            except asyncio.CancelledError:
                # Propagate cancellation
                raise
            except (ServiceError, TradingSystemError) as e:
                logger.error(f"Service error processing {symbol}: {e}")
                results.append({
                    'symbol': symbol,
                    'action_taken': 'error',
                    'error': str(e)
                })
            except Exception as e:
                logger.error(f"Unexpected error processing {symbol}: {e}")
                results.append({
                    'symbol': symbol,
                    'action_taken': 'error',
                    'error': str(e)
                })
        
        return results
    
    def _update_statistics(self, results: List[Dict[str, Any]]):
        """Update system statistics based on cycle results"""
        for result in results:
            action = result.get('action_taken')
            
            if action == 'trade_executed':
                self.stats['trades_executed'] += 1
                # Update last trade time for position trading
                if self._position_trading_mode:
                    symbol = result.get('symbol')
                    if symbol:
                        self._last_trade_times[symbol] = datetime.now(timezone.utc)
            elif action == 'trade_managed':
                self.stats['trades_managed'] += 1
            elif action in ['signal_wait', 'trade_executed']:
                self.stats['signals_generated'] += 1
            elif action == 'error':
                self.stats['errors_encountered'] += 1
    
    def _log_cycle_summary(self, results: List[Dict[str, Any]], cycle_start: datetime):
        """Log summary of cycle results"""
        cycle_duration = (datetime.now(timezone.utc) - cycle_start).total_seconds()
        
        action_counts = {}
        for result in results:
            action = result.get('action_taken', 'unknown')
            action_counts[action] = action_counts.get(action, 0) + 1
        
        summary_parts = []
        for action, count in action_counts.items():
            if count > 0:
                summary_parts.append(f"{action}: {count}")
        
        summary = ", ".join(summary_parts) if summary_parts else "no actions"
        
        logger.info(f"✅ Cycle #{self.cycle_count} completed in {cycle_duration:.1f}s - {summary}")
    
    async def _handle_cycle_error(self, error: Exception):
        """Handle errors during trading cycles"""
        self.error_count += 1
        
        # Use the error handler for consistent error management
        error_handler = get_error_handler()
        
        # Determine severity based on error count
        if self.error_count >= self.max_errors - 2:
            severity = ErrorSeverity.CRITICAL
        elif self.error_count > self.max_errors // 2:
            severity = ErrorSeverity.HIGH
        else:
            severity = ErrorSeverity.MEDIUM
        
        # Determine category based on error type
        if isinstance(error, ConfigurationError):
            category = ErrorCategory.CONFIGURATION
        elif isinstance(error, ValidationError):
            category = ErrorCategory.VALIDATION
        elif isinstance(error, (ConnectionError, TimeoutError)):
            category = ErrorCategory.NETWORK
        else:
            category = ErrorCategory.TRADING
        
        # Handle the error
        error_handler.handle_error(
            error=error,
            context=f"trading cycle #{self.cycle_count}",
            severity=severity,
            category=category,
            operation="execute_trading_cycle",
            details={
                'cycle_count': self.cycle_count,
                'error_count': self.error_count,
                'max_errors': self.max_errors
            },
            reraise=False  # Don't reraise, we're handling it
        )
        
        # Exponential backoff on errors
        backoff_time = min(300, 30 * (2 ** min(self.error_count - 1, 3)))
        logger.info(f"⏳ Waiting {backoff_time}s before retry...")
        await asyncio.sleep(backoff_time)
    
    def _should_trade_now(self) -> bool:
        """Check if we should trade now based on trading hours"""
        return self.scheduler.is_trading_hours()
    
    async def _wait_for_trading_hours(self):
        """Wait until trading hours begin"""
        wait_time = self.scheduler.seconds_until_trading_hours()
        if wait_time > 0:
            hours = int(wait_time // 3600)
            minutes = int((wait_time % 3600) // 60)
            
            # Get next session start time
            next_session = datetime.now(timezone.utc) + timedelta(seconds=wait_time)
            
            # Show detailed message only once
            if not self._market_closed_logged:
                logger.info(f"⏰ Market is closed. Will open in {hours}h {minutes}m at {next_session.strftime('%Y-%m-%d %H:%M')} UTC")
                logger.info(f"📅 Forex market opens Sunday 22:00 UTC and closes Friday 22:00 UTC")
                self._market_closed_logged = True
            else:
                # Show brief update
                logger.debug(f"Waiting for market to open in {hours}h {minutes}m...")
            
            # Wait longer periods when market is closed to reduce log spam
            if wait_time > 3600:  # More than 1 hour
                # Check every 30 minutes when more than an hour away
                check_interval = 1800  
            elif wait_time > 300:  # More than 5 minutes
                # Check every 5 minutes when less than an hour away
                check_interval = 300
            else:
                # Check every minute when very close to market open
                check_interval = 60
                
            await asyncio.sleep(min(wait_time, check_interval))
        else:
            # Market is open, reset the flag
            self._market_closed_logged = False
    
    async def _wait_with_shutdown_check(self):
        """Wait for next cycle with periodic shutdown checks"""
        wait_time = self.scheduler.calculate_cycle_interval()
        check_interval = 5  # Check for shutdown every 5 seconds
        
        elapsed = 0
        while elapsed < wait_time and not self._shutdown_requested:
            # Check for stop file
            stop_file = Path("STOP_TRADING")
            if stop_file.exists():
                logger.info("🛑 Stop file detected during wait, shutting down...")
                self._shutdown_requested = True
                stop_file.unlink()
                break
                
            await asyncio.sleep(min(check_interval, wait_time - elapsed))
            elapsed += check_interval
    
    async def _shutdown_system(self):
        """Gracefully shutdown the system"""
        logger.info("🔄 Shutting down trading system...")
        
        try:
            self.state = TradingState.STOPPING
            
            # Close any open trades if configured to do so
            open_trades = await self.trade_service.get_open_trades()
            if open_trades:
                logger.info(f"📊 {len(open_trades)} trades remain open")
                # Could add logic to close all trades on shutdown if desired
            
            # Shutdown MT5 connection
            self.mt5_client.shutdown()
            
            # Log final statistics
            self._log_final_statistics()
            
            self.state = TradingState.STOPPED
            logger.info("👋 Trading system shutdown complete")
            
        except asyncio.CancelledError:
            # Expected during shutdown
            pass
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            self.state = TradingState.ERROR
    
    def _log_final_statistics(self):
        """Log final system statistics"""
        if self.stats['start_time']:
            start_time = datetime.fromisoformat(self.stats['start_time'])
            runtime = datetime.now(timezone.utc) - start_time
            
            logger.info("📊 Final Statistics:")
            logger.info(f"   • Runtime: {runtime}")
            logger.info(f"   • Cycles completed: {self.stats['cycles_completed']}")
            logger.info(f"   • Signals generated: {self.stats['signals_generated']}")
            logger.info(f"   • Trades executed: {self.stats['trades_executed']}")
            logger.info(f"   • Trades managed: {self.stats['trades_managed']}")
            logger.info(f"   • Errors encountered: {self.stats['errors_encountered']}")
            if self._position_trading_mode:
                logger.info(f"   • Position reviews: {self.stats['position_reviews']}")
                logger.info(f"   • Positions adjusted: {self.stats['positions_adjusted']}")
    
    # Public methods for external control
    
    async def pause(self):
        """Pause the trading system"""
        if self.state == TradingState.RUNNING:
            self.state = TradingState.PAUSED
            logger.info("⏸️ Trading system paused")
    
    async def resume(self):
        """Resume the trading system"""
        if self.state == TradingState.PAUSED:
            self.state = TradingState.RUNNING
            logger.info("▶️ Trading system resumed")
    
    async def stop(self):
        """Stop the trading system"""
        logger.info("🛑 Stop requested")
        self.state = TradingState.STOPPING
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'state': self.state.value,
            'cycle_count': self.cycle_count,
            'last_cycle_time': self.last_cycle_time.isoformat() if self.last_cycle_time else None,
            'error_count': self.error_count,
            'statistics': self.stats.copy(),
            'trading_hours': {
                'start_hour': self.trading_config.start_hour,
                'end_hour': self.trading_config.end_hour,
                'currently_trading_hours': self.scheduler.is_trading_hours()
            }
        }


    def _check_position_trading_constraints(self, symbol: str) -> Tuple[bool, str]:
        """Check if position trading constraints allow new trade"""
        # Check minimum time between trades
        if symbol in self._last_trade_times:
            hours_since_last = (datetime.now(timezone.utc) - self._last_trade_times[symbol]).total_seconds() / 3600
            min_hours = self.position_config.frequency.MIN_HOURS_BETWEEN_SAME_SYMBOL
            if hours_since_last < min_hours:
                return False, f"Only {hours_since_last:.1f}h since last {symbol} trade (need {min_hours}h)"
        
        # Check global minimum time between any trades
        if self._last_trade_times:
            last_trade_time = max(self._last_trade_times.values())
            hours_since_any = (datetime.now(timezone.utc) - last_trade_time).total_seconds() / 3600
            min_hours_any = self.position_config.frequency.MIN_HOURS_BETWEEN_TRADES
            if hours_since_any < min_hours_any:
                return False, f"Only {hours_since_any:.1f}h since last trade (need {min_hours_any}h)"
        
        # Check max concurrent positions
        open_trades = self.trade_service.get_open_trades() if hasattr(self.trade_service, 'get_open_trades') else []
        if len(open_trades) >= self.position_config.frequency.MAX_CONCURRENT_POSITIONS:
            return False, f"Maximum {self.position_config.frequency.MAX_CONCURRENT_POSITIONS} concurrent positions reached"
        
        # Check session filter
        current_hour = datetime.now(timezone.utc).hour
        if not self.position_config.get_session_filter(symbol, current_hour):
            return False, f"Current hour ({current_hour}) not suitable for {symbol} position trades"
        
        return True, "OK"
    
    def _filter_symbols_by_cooldown(self, symbols: List[str]) -> List[str]:
        """Filter symbols based on position trading cooldown periods"""
        filtered = []
        current_time = datetime.now(timezone.utc)
        
        for symbol in symbols:
            if symbol in self._last_trade_times:
                hours_since = (current_time - self._last_trade_times[symbol]).total_seconds() / 3600
                if hours_since < self.position_config.frequency.MIN_HOURS_BETWEEN_SAME_SYMBOL:
                    logger.debug(f"Filtering out {symbol} - cooldown period ({hours_since:.1f}h < {self.position_config.frequency.MIN_HOURS_BETWEEN_SAME_SYMBOL}h)")
                    continue
            filtered.append(symbol)
        
        return filtered
    
    async def _perform_position_review(self):
        """Perform daily position review for position trading"""
        try:
            # Check if it's time for review (once per day)
            current_time = datetime.now(timezone.utc)
            if self._last_position_review:
                hours_since_review = (current_time - self._last_position_review).total_seconds() / 3600
                if hours_since_review < 24:
                    return
            
            logger.info("📊 Performing daily position review...")
            
            # Monitor all positions
            position_health = await self.position_monitor.monitor_all_positions()
            self.stats['position_reviews'] += 1
            
            # Execute recommended adjustments
            if position_health:
                adjustments = await self.position_monitor.execute_position_adjustments(position_health)
                if adjustments:
                    self.stats['positions_adjusted'] += len(adjustments)
                    logger.info(f"Executed {len(adjustments)} position adjustments")
            
            # Check for time-based exits
            exits_needed = await self.position_monitor.check_time_based_exits()
            if exits_needed:
                logger.warning(f"Time-based exits needed for {len(exits_needed)} positions")
                # In production, would execute these exits
            
            # Generate and log daily review
            review_report = await self.position_monitor.generate_daily_review()
            logger.info(f"Daily position review completed\n{review_report}")
            
            self._last_position_review = current_time
            
        except Exception as e:
            logger.error(f"Error in position review: {e}")


# Export main orchestrator
__all__ = ['TradingOrchestrator', 'TradingState', 'SymbolProcessor', 'TradingScheduler']