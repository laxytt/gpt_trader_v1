import logging
import sys
import time
from datetime import datetime, timedelta, timezone

from config import (
    TRADING_START_HOUR,
    TRADING_END_HOUR,
    SYMBOL_LIST,
    TELEGRAM_TOKEN,
    TELEGRAM_CHAT_ID
)
from core.database import init_db, get_trade_by_symbol
from core.mt5_utils import ensure_mt5_initialized, prefilter_instruments
from core.trade_cycle import trade_cycle
from core.trade_manager import manage_active_trade
from core.news_filter import is_news_restricted_now
from core.telegram_logger import TelegramLogger

# ========== LOGGER SETUP ==========
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def global_except_hook(exctype, value, tb):
    logger.exception("Unhandled exception occurred", exc_info=(exctype, value, tb))

sys.excepthook = global_except_hook

# ========== LOOP CLASS ==========
class TradingLoopManager:
    def __init__(self, symbols: list[str]):
        self.symbols = symbols

    def run(self):
        logger.info("📆 GPT Trader loop started (multi-symbol, synced to H1 chart close)")
        if not ensure_mt5_initialized():
            logger.error("❌ Could not initialize MT5.")
            return

        while True:
            if not self.is_trading_time():
                sleep_seconds = self.seconds_until_trading_start()
                logger.info(f"⏸️ Outside trading hours. Sleeping {int(sleep_seconds)}s.")
                time.sleep(sleep_seconds)
                continue

            logger.info(f"🔄 Starting cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            candidate_symbols = self.get_candidate_symbols()

            for symbol in candidate_symbols:
                try:
                    self.handle_symbol(symbol)
                except Exception as e:
                    logger.exception(f"❌ Error while handling symbol {symbol}: {e}")

            self.wait_until_next_h1_boundary()

    def is_trading_time(self) -> bool:
        now = datetime.now()
        return TRADING_START_HOUR <= now.hour < TRADING_END_HOUR

    def seconds_until_trading_start(self) -> float:
        now = datetime.now()
        start_today = now.replace(hour=TRADING_START_HOUR, minute=0, second=0, microsecond=0)
        if now < start_today:
            return (start_today - now).total_seconds()
        return ((start_today + timedelta(days=1)) - now).total_seconds()

    def wait_until_next_h1_boundary(self):
        now = datetime.now()
        next_boundary = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        delay = (next_boundary - now).total_seconds()
        logger.info(f"⏳ Sleeping {delay:.2f} seconds until {next_boundary}.")
        time.sleep(delay)

    def get_candidate_symbols(self) -> list[str]:
        try:
            symbols = prefilter_instruments(self.symbols)
            logger.info(f"🔎 Candidates after ATR/vol filter: {symbols}")
            return symbols
        except Exception as e:
            logger.exception("Prefiltering failed.")
            return ["EURUSD"]

    def handle_symbol(self, symbol: str):
        if is_news_restricted_now(symbol, now=datetime.now(timezone.utc)):
            logger.info(f"🚫 News block — skipping {symbol}")
            return

        trade = get_trade_by_symbol(symbol)
        if trade and trade.get("status") == "open":
            logger.info(f"🟡 Managing active trade for {symbol} (ticket={trade['ticket']})")
            manage_active_trade(trade["ticket"])
        else:
            logger.info(f"🔍 Looking for new setup on {symbol}")
            trade_cycle(symbol=symbol)

# ========== MAIN ENTRY ==========
if __name__ == "__main__":
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        sys.stdout = TelegramLogger(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)

    init_db()
    loop = TradingLoopManager(SYMBOL_LIST)
    loop.run()
