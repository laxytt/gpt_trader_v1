"""
Telegram notification service for trading system alerts.
Sends trading notifications, alerts, and system status updates via Telegram bot.
"""

import logging
import asyncio
from typing import Optional
from datetime import datetime, timezone

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from config.settings import TelegramSettings
from core.domain.exceptions import NotificationError, TelegramError


logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Telegram notification service for trading system.
    Sends messages via Telegram Bot API.
    """
    
    def __init__(self, config: TelegramSettings):
        self.config = config
        self.base_url = f"https://api.telegram.org/bot{config.token}"
        self.session = None
        
        if not REQUESTS_AVAILABLE:
            logger.warning("Requests library not available. Telegram notifications disabled.")
            self.enabled = False
        else:
            self.enabled = config.enabled
    
    async def send_message(
        self, 
        text: str, 
        parse_mode: str = "Markdown",
        disable_notification: bool = False
    ) -> bool:
        """
        Send a text message to Telegram.
        
        Args:
            text: Message text to send
            parse_mode: Message formatting ("Markdown" or "HTML")
            disable_notification: Send silently
            
        Returns:
            True if message sent successfully
        """
        if not self.enabled or not REQUESTS_AVAILABLE:
            logger.debug(f"Telegram disabled - would send: {text}")
            return False
        
        try:
            # Add timestamp to message
            timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
            formatted_text = f"🕐 {timestamp}\n{text}"
            
            payload = {
                "chat_id": self.config.chat_id,
                "text": formatted_text,
                "parse_mode": parse_mode,
                "disable_notification": disable_notification
            }
            
            url = f"{self.base_url}/sendMessage"
            
            # Use asyncio to make non-blocking request
            response = await self._make_request(url, payload)
            
            if response and response.get("ok"):
                logger.debug("Telegram message sent successfully")
                return True
            else:
                error_desc = response.get("description", "Unknown error") if response else "No response"
                logger.error(f"Telegram API error: {error_desc}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    async def send_trade_alert(
        self, 
        symbol: str, 
        action: str, 
        details: dict
    ) -> bool:
        """
        Send a trading alert message.
        
        Args:
            symbol: Trading symbol
            action: Action taken (e.g., "TRADE_OPENED", "TRADE_CLOSED")
            details: Additional trade details
            
        Returns:
            True if sent successfully
        """
        try:
            if action == "TRADE_OPENED":
                message = self._format_trade_opened_message(symbol, details)
            elif action == "TRADE_CLOSED":
                message = self._format_trade_closed_message(symbol, details)
            elif action == "SIGNAL_GENERATED":
                message = self._format_signal_message(symbol, details)
            else:
                message = f"📊 *{symbol}*\n{action}\n{details}"
            
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Failed to send trade alert: {e}")
            return False
    
    def _format_trade_opened_message(self, symbol: str, details: dict) -> str:
        """Format trade opened message"""
        side = details.get('side', 'Unknown')
        entry = details.get('entry_price', 'Unknown')
        sl = details.get('stop_loss', 'Unknown')
        tp = details.get('take_profit', 'Unknown')
        risk_class = details.get('risk_class', 'Unknown')
        
        emoji = "🟢" if side == "BUY" else "🔴"
        
        return f"""
{emoji} *TRADE OPENED*
📊 Symbol: `{symbol}`
📈 Side: *{side}*
💰 Entry: `{entry}`
🛑 Stop Loss: `{sl}`
🎯 Take Profit: `{tp}`
⭐ Risk Class: `{risk_class}`
        """.strip()
    
    def _format_trade_closed_message(self, symbol: str, details: dict) -> str:
        """Format trade closed message"""
        result = details.get('result', 'Unknown')
        pnl = details.get('pnl', 'Unknown')
        duration = details.get('duration_minutes', 'Unknown')
        
        if result == 'WIN':
            emoji = "✅"
        elif result == 'LOSS':
            emoji = "❌"
        else:
            emoji = "⚪"
        
        return f"""
{emoji} *TRADE CLOSED*
📊 Symbol: `{symbol}`
📊 Result: *{result}*
💰 P&L: `{pnl}`
⏱️ Duration: `{duration} min`
        """.strip()
    
    def _format_signal_message(self, symbol: str, details: dict) -> str:
        """Format signal generated message"""
        signal = details.get('signal', 'Unknown')
        risk_class = details.get('risk_class', 'Unknown')
        reason = details.get('reason', 'No reason provided')
        
        if signal == 'BUY':
            emoji = "📈"
        elif signal == 'SELL':
            emoji = "📉"
        else:
            emoji = "⏸️"
        
        return f"""
{emoji} *SIGNAL: {signal}*
📊 Symbol: `{symbol}`
⭐ Risk Class: `{risk_class}`
💭 Reason: _{reason}_
        """.strip()
    
    async def send_system_status(self, status: dict) -> bool:
        """
        Send system status update.
        
        Args:
            status: System status dictionary
            
        Returns:
            True if sent successfully
        """
        try:
            state = status.get('state', 'Unknown')
            cycles = status.get('cycle_count', 0)
            trades_executed = status.get('statistics', {}).get('trades_executed', 0)
            trades_managed = status.get('statistics', {}).get('trades_managed', 0)
            
            message = f"""
📊 *SYSTEM STATUS*
🔄 State: `{state}`
📈 Cycles: `{cycles}`
⚡ Trades Executed: `{trades_executed}`
🔧 Trades Managed: `{trades_managed}`
            """.strip()
            
            return await self.send_message(message, disable_notification=True)
            
        except Exception as e:
            logger.error(f"Failed to send system status: {e}")
            return False
    
    async def send_error_alert(self, error_message: str, context: str = "") -> bool:
        """
        Send error alert message.
        
        Args:
            error_message: Error description
            context: Additional context
            
        Returns:
            True if sent successfully
        """
        try:
            message = f"""
🚨 *ERROR ALERT*
❌ Error: `{error_message}`
📍 Context: `{context}`
            """.strip()
            
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Failed to send error alert: {e}")
            return False
    
    async def _make_request(self, url: str, payload: dict) -> Optional[dict]:
        """Make async HTTP request to Telegram API"""
        try:
            # Use asyncio to run requests in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: requests.post(url, data=payload, timeout=10)
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Telegram API HTTP {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Telegram request failed: {e}")
            return None
    
    async def test_connection(self) -> bool:
        """
        Test Telegram bot connection.
        
        Returns:
            True if connection successful
        """
        if not self.enabled:
            return False
        
        try:
            test_message = "🔧 GPT Trading System - Connection Test"
            return await self.send_message(test_message, disable_notification=True)
            
        except Exception as e:
            logger.error(f"Telegram connection test failed: {e}")
            return False
    
    def is_enabled(self) -> bool:
        """Check if Telegram notifications are enabled"""
        return self.enabled and REQUESTS_AVAILABLE


# Export main class
__all__ = ['TelegramNotifier']