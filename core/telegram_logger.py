import sys
import requests
from datetime import datetime, timezone

class TelegramLogger:
    def __init__(self, token, chat_id, prefix="🖥️ Log"):
        self.token = token
        self.chat_id = chat_id
        self.prefix = prefix
        self._stdout = sys.stdout  # Save original stdout

    def write(self, message):
        # Filter repetitive 404
        if "Telegram HTTP error: 404" in message:
            return
        message = message.strip()
        if message:
            self._stdout.write(message + "\n")
            try:
                timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
                text = f"*{self.prefix} {timestamp}*\n`{message}`"
                self.send_telegram(text)
            except Exception as e:
                self._stdout.write(f"(Telegram send failed: {e})\n")

    def flush(self):
        # Needed for sys.stdout compatibility
        pass

    def send_telegram(self, text):
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }
        try:
            resp = requests.post(url, data=payload, timeout=5)
            if not resp.ok:
                self._stdout.write(f"(Telegram HTTP error: {resp.status_code})\n")
        except Exception as e:
            self._stdout.write(f"(Telegram send error: {e})\n")
