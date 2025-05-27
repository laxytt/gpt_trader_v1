"""
Trading orchestrator service that coordinates the entire trading workflow.
Manages the main trading loop, symbol processing, and system coordination.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from enum import Enum

from core.domain.models import TradingSignal, Trade, SignalType, TradeStatus
from core.domain.exceptions import (
    TradingSystemError, ErrorContext, ServiceError, ConfigurationError
)
from core.services.signal_service import SignalService
from core.services.trade_service import TradeService  
from core.services.news_service import NewsService
from core.services.memory_service import MemoryService
from core.services.market_service import MarketService
from core.infrastructure.mt5.client import MT5Client
from core.utils.validation import validate_symbol_format
from config.settings import TradingSettings
from pathlib import Path


logger = logging.getLogger(__name__)


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
        signal_service: SignalService,
        trade_service: TradeService,
        news_service: NewsService
    ):
        self.signal_service = signal_service
        self.trade_service = trade_service
        self.news_service = news_service
    
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
                    # Look for new trading opportunity
                    ctx.add_detail("action", "signal_generation")
                    
                    # Check if symbol is ready for analysis
                    if not await self.signal_service.validate_symbol_readiness(symbol):
                        result['action_taken'] = 'skipped_not_ready'
                        result['error'] = 'Symbol not ready for analysis'
                        return result
                    
                    # Generate signal
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
                
            except Exception as e:
                logger.error(f"Symbol processing failed for {symbol}: {e}")
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
        """Check if current time is within trading hours"""
        if now is None:
            now = datetime.now(timezone.utc)
        
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
        
        # Calculate next session start
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


class TradingOrchestrator:
    """
    Main orchestrator that coordinates the entire trading system.
    Manages the trading loop, system state, and component coordination.
    """
    
    def __init__(
        self,
        signal_service: SignalService,
        trade_service: TradeService,
        news_service: NewsService,
        memory_service: MemoryService,
        market_service: MarketService,
        mt5_client: MT5Client,
        trading_config: TradingSettings
    ):
        self.signal_service = signal_service
        self.trade_service = trade_service
        self.news_service = news_service
        self.memory_service = memory_service
        self.market_service = market_service
        self.mt5_client = mt5_client
        self.trading_config = trading_config
        
        # Initialize components
        self.symbol_processor = SymbolProcessor(signal_service, trade_service, news_service)
        self.scheduler = TradingScheduler(trading_config)
        
        # System state
        self.state = TradingState.INITIALIZING
        self.cycle_count = 0
        self.last_cycle_time = None
        self.error_count = 0
        self.max_errors = 10
        
        # Statistics
        self.stats = {
            'cycles_completed': 0,
            'signals_generated': 0,
            'trades_executed': 0,
            'trades_managed': 0,
            'errors_encountered': 0,
            'start_time': None
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
            
            while self.state == TradingState.RUNNING:
                try:
                    # Check if we should trade now
                    if not self._should_trade_now():
                        await self._wait_for_trading_hours()
                        continue
                    
                    # Execute trading cycle
                    await self._execute_trading_cycle()
                    
                    # Wait until next cycle
                    await self.scheduler.wait_for_next_hour_boundary()
                    
                except KeyboardInterrupt:
                    logger.info("🛑 Received interrupt signal, stopping...")
                    self.state = TradingState.STOPPING
                    break
                    
                except Exception as e:
                    await self._handle_cycle_error(e)
                    
                    if self.error_count >= self.max_errors:
                        logger.error(f"💥 Maximum error count reached ({self.max_errors}), stopping system")
                        self.state = TradingState.ERROR
                        break
            
        except Exception as e:
            logger.exception(f"💥 Fatal error in trading orchestrator: {e}")
            self.state = TradingState.ERROR
        
        finally:
            await self._shutdown_system()
    
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
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            raise
    
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
            except Exception as e:
                raise ConfigurationError(f"Invalid symbol {symbol}: {e}")
    
    def _should_trade_now(self) -> bool:
        """Check if trading should occur now"""
        return self.scheduler.is_trading_hours()
    
    async def _wait_for_trading_hours(self):
        """Wait until trading hours begin"""
        wait_time = self.scheduler.time_until_next_session()
        hours = int(wait_time.total_seconds() // 3600)
        minutes = int((wait_time.total_seconds() % 3600) // 60)
        
        logger.info(f"⏸️ Outside trading hours. Waiting {hours}h {minutes}m for next session.")
        await asyncio.sleep(min(3600, wait_time.total_seconds()))  # Sleep max 1 hour at a time
    
    async def _execute_trading_cycle(self):
        """Execute one complete trading cycle"""
        self.cycle_count += 1
        cycle_start = datetime.now(timezone.utc)
        
        logger.info(f"🔄 Starting trading cycle #{self.cycle_count} at {cycle_start.strftime('%H:%M:%S')}")
        
        try:
            # Get active symbols (filter by market conditions if needed)
            active_symbols = await self._get_active_symbols()
            
            # Get market overview for logging
            try:
                market_overview = await self.market_service.get_market_overview(active_symbols)
                session_info = market_overview['session']
                summary = market_overview['summary']
                
                logger.info(f"📊 Market Overview - Session: {session_info['current_session']}")
                logger.info(f"📈 Market Summary - Excellent: {summary['excellent_conditions']}, "
                        f"Good: {summary['good_conditions']}, "
                        f"Avg Score: {summary['average_score']}")
            except Exception as e:
                logger.warning(f"Failed to get market overview: {e}")
        
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
            
        except Exception as e:
            logger.error(f"❌ Trading cycle #{self.cycle_count} failed: {e}")
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
            
        except Exception as e:
            logger.error(f"Failed to get market intelligence, using all configured symbols: {e}")
            return self.trading_config.symbols
    
    async def _process_symbols(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Process multiple symbols concurrently"""
        results = []
        
        # Process symbols sequentially to avoid overwhelming the system
        for symbol in symbols:
            try:
                result = await self.symbol_processor.process_symbol(symbol)
                results.append(result)
                
                # Small delay between symbols to be respectful to APIs
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Symbol processing failed for {symbol}: {e}")
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
        logger.error(f"❌ Trading cycle error ({self.error_count}/{self.max_errors}): {error}")
        
        # Exponential backoff on errors
        backoff_time = min(300, 30 * (2 ** min(self.error_count - 1, 3)))
        logger.info(f"⏳ Waiting {backoff_time}s before retry...")
        await asyncio.sleep(backoff_time)
    
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


# Export main orchestrator
__all__ = ['TradingOrchestrator', 'TradingState', 'SymbolProcessor', 'TradingScheduler']