"""
config.py — Fail-fast environment variable loader.
All required vars must be set at startup or the app crashes immediately.
"""
import os
from functools import lru_cache


class Config:
    # Supabase
    SUPABASE_URL: str
    SUPABASE_SERVICE_ROLE_KEY: str

    # Stream — permanent ipcamlive camera alias (never changes, no session expiry)
    CAMERA_ALIAS: str
    URL_REFRESH_INTERVAL: int  # seconds between proactive URL refreshes (default 240)

    # WebSocket auth (HMAC)
    WS_AUTH_SECRET: str
    ALLOWED_ORIGIN: str

    # AI config
    YOLO_MODEL: str
    YOLO_CONF: float
    COUNT_LINE_RATIO: float  # fallback ratio if no DB line
    DB_SNAPSHOT_INTERVAL_SEC: float

    # Bet logic
    BET_LOCK_SECONDS: int

    # Server
    WS_PORT: int

    def __init__(self):
        required = [
            "SUPABASE_URL",
            "SUPABASE_SERVICE_ROLE_KEY",
            "CAMERA_ALIAS",
            "WS_AUTH_SECRET",
            "ALLOWED_ORIGIN",
        ]
        missing = [k for k in required if not os.getenv(k)]
        if missing:
            raise RuntimeError(
                f"[STARTUP FAILURE] Missing required environment variables: {missing}"
            )

        self.SUPABASE_URL = os.environ["SUPABASE_URL"]
        self.SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
        self.CAMERA_ALIAS = os.environ["CAMERA_ALIAS"]
        self.WS_AUTH_SECRET = os.environ["WS_AUTH_SECRET"]
        self.ALLOWED_ORIGIN = os.environ["ALLOWED_ORIGIN"]

        self.URL_REFRESH_INTERVAL = int(os.getenv("URL_REFRESH_INTERVAL", "240"))
        self.YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
        self.YOLO_CONF = float(os.getenv("YOLO_CONF", "0.45"))
        self.COUNT_LINE_RATIO = float(os.getenv("COUNT_LINE_RATIO", "0.55"))
        self.DB_SNAPSHOT_INTERVAL_SEC = float(os.getenv("DB_SNAPSHOT_INTERVAL_SEC", "0.75"))
        self.BET_LOCK_SECONDS = int(os.getenv("BET_LOCK_SECONDS", "10"))
        self.WS_PORT = int(os.getenv("WS_PORT", "8000"))


@lru_cache(maxsize=1)
def get_config() -> Config:
    return Config()
