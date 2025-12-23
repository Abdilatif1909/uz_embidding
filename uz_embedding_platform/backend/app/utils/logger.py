import os
from datetime import datetime

def log_event(msg):
    log_path = os.path.join(
        os.path.dirname(__file__), "../../../logs/system.log"
    )

    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} {msg}\n")
