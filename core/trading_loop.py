import logging
import sys
import time
from datetime import datetime, timedelta, timezone

from core.database import init_db
from core.main_cycle import main_cycle
from core.news_filter import is_news_restricted_now
from core.mt5_utils import prefilter_instruments, ensure_mt5_initialized
from core.trade_status import load_all_open_trades, save_all_open_trades
from core.trade_manager import manage_active_trade
from core.utils import sync_open_trades_from_mt5
from config import (
    TRADING_START_HOUR,
    TRADING_END_HOUR,
    SYMBOL_LIST,
    TELEGRAM_TOKEN,
    TELEGRAM_CHAT_ID
)
from core.telegram_logger import TelegramLogger

# Konfiguracja logowania
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Obsługa globalnych wyjątków
def global_except_hook(exctype, value, tb):
    logger.exception("Unhandled exception occurred", exc_info=(exctype, value, tb))

sys.excepthook = global_except_hook

class TradingLoopManager:
    def __init__(self, symbols: list[str]):
        self.symbols = symbols
        self.open_trades = load_all_open_trades()

    def run(self):
        logger.info("📆 GPT Trader loop started (multi-symbol, synced to H1 chart close)")
        if not ensure_mt5_initialized():
            logger.error("❌ Could not initialize MT5 for instrument scanning.")
            return

        while True:
            if not self.is_trading_time():
                sleep_seconds = self.seconds_until_trading_start()
                next_start = (datetime.now() + timedelta(seconds=sleep_seconds)).strftime('%Y-%m-%d %H:%M:%S')
                logger.info(f"⏸️ Outside trading hours. Sleeping {int(sleep_seconds)} seconds until {next_start}.")
                time.sleep(sleep_seconds)
                continue

            ensure_mt5_initialized()
            logger.info(f"🔄 Starting multi-symbol main cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            candidate_symbols = self.get_candidate_symbols()

            for symbol in candidate_symbols:
                self.handle_symbol(symbol)

            save_all_open_trades(self.open_trades)
            self.wait_until_next_h1_boundary()

    def is_trading_time(self) -> bool:
        now = datetime.now()
        return TRADING_START_HOUR <= now.hour < TRADING_END_HOUR

    def seconds_until_trading_start(self) -> float:
        now = datetime.now()
        start_today = now.replace(hour=TRADING_START_HOUR, minute=0, second=0, microsecond=0)
        if now < start_today:
            delta = start_today - now
        else:
            delta = (start_today + timedelta(days=1)) - now
        return delta.total_seconds()

    def wait_until_next_h1_boundary(self):
        now = datetime.now()
        next_boundary = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        delay = (next_boundary - now).total_seconds()
        logger.info(f"⏳ Sleeping {delay:.2f} seconds until {next_boundary} (next H1 boundary).")
        time.sleep(delay)

    def get_candidate_symbols(self) -> list[str]:
        try:
            symbols = prefilter_instruments(self.symbols)
            logger.info(f"🔎 Candidates passing ATR/volume filter: {symbols}")
            return symbols
        except Exception as e:
            logger.exception(f"Error during prefiltering instruments: {e}")
            return ["EURUSD"]  # fallback symbol

    def handle_symbol(self, symbol: str):
        if is_news_restricted_now(symbol, datetime.now(timezone.utc)):
            logger.info(f"🚫 Skipping {symbol} due to restricted macroeconomic event.")
            return

        trade = self.open_trades.get(symbol)
        if trade:
            logger.info(f"🟡 Managing active trade for {symbol}")
            manage_active_trade(trade)

            if trade.get("status") == "closed":
                self.open_trades.pop(symbol, None)
                logger.info(f"✅ Trade for {symbol} closed and removed from active trades.")
            return

        logger.info(f"🔍 Looking for new trade signal for {symbol}")
        main_cycle(symbol=symbol)

if __name__ == "__main__":
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        log = TelegramLogger(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
        sys.stdout = log

    init_db()  # inicjalizacja bazy danych
    sync_open_trades_from_mt5()
    loop_manager = TradingLoopManager(SYMBOL_LIST)
    loop_manager.run()
