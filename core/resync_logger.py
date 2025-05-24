from datetime import datetime, timezone

def log_resync(event, details, file_path="resync_log.txt"):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} | {event} | {details}\n")
