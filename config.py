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
    DETECT_INFER_SIZE: int
    DETECT_IOU: float
    DETECT_MAX_DET: int
    TRACK_ACTIVATION_THRESHOLD: float
    TRACK_LOST_BUFFER: int
    TRACK_MATCH_THRESHOLD: float
    TRACK_FRAME_RATE: int
    COUNT_LINE_RATIO: float  # fallback ratio if no DB line
    DB_SNAPSHOT_INTERVAL_SEC: float
    STREAM_GRAB_LATEST: int
    OPENAI_API_KEY: str
    OPENAI_MODEL: str
    TRAINER_WEBHOOK_URL: str
    TRAINER_WEBHOOK_SECRET: str
    TRAINER_DATASET_YAML_URL: str
    TRAINER_EPOCHS: int
    TRAINER_IMGSZ: int
    TRAINER_BATCH: int
    ML_AUTO_RETRAIN_ENABLED: int
    ML_AUTO_RETRAIN_INTERVAL_MIN: int
    ML_AUTO_RETRAIN_HOURS: int
    ML_AUTO_RETRAIN_MIN_ROWS: int
    ML_AUTO_RETRAIN_MIN_SCORE_GAIN: float
    AUTO_CAPTURE_ENABLED: int
    AUTO_CAPTURE_DATASET_ROOT: str
    AUTO_CAPTURE_CLASSES: str
    AUTO_CAPTURE_MIN_CONF: float
    AUTO_CAPTURE_COOLDOWN_SEC: float
    AUTO_CAPTURE_VAL_SPLIT: float
    AUTO_CAPTURE_JPEG_QUALITY: int
    AUTO_CAPTURE_MAX_BOXES_PER_FRAME: int
    AUTO_CAPTURE_UPLOAD_ENABLED: int
    AUTO_CAPTURE_UPLOAD_BUCKET: str
    AUTO_CAPTURE_UPLOAD_PREFIX: str
    AUTO_CAPTURE_DELETE_LOCAL_AFTER_UPLOAD: int
    AUTO_CAPTURE_UPLOAD_TIMEOUT_SEC: float

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
        self.DETECT_INFER_SIZE = int(os.getenv("DETECT_INFER_SIZE", "448"))
        self.DETECT_IOU = float(os.getenv("DETECT_IOU", "0.50"))
        self.DETECT_MAX_DET = int(os.getenv("DETECT_MAX_DET", "80"))
        self.TRACK_ACTIVATION_THRESHOLD = float(os.getenv("TRACK_ACTIVATION_THRESHOLD", "0.2"))
        self.TRACK_LOST_BUFFER = int(os.getenv("TRACK_LOST_BUFFER", "20"))
        self.TRACK_MATCH_THRESHOLD = float(os.getenv("TRACK_MATCH_THRESHOLD", "0.65"))
        self.TRACK_FRAME_RATE = int(os.getenv("TRACK_FRAME_RATE", "25"))
        self.COUNT_LINE_RATIO = float(os.getenv("COUNT_LINE_RATIO", "0.55"))
        self.DB_SNAPSHOT_INTERVAL_SEC = float(os.getenv("DB_SNAPSHOT_INTERVAL_SEC", "0.75"))
        self.STREAM_GRAB_LATEST = int(os.getenv("STREAM_GRAB_LATEST", "1"))
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.TRAINER_WEBHOOK_URL = os.getenv("TRAINER_WEBHOOK_URL", "")
        self.TRAINER_WEBHOOK_SECRET = os.getenv("TRAINER_WEBHOOK_SECRET", "")
        self.TRAINER_DATASET_YAML_URL = os.getenv("TRAINER_DATASET_YAML_URL", "")
        self.TRAINER_EPOCHS = int(os.getenv("TRAINER_EPOCHS", "20"))
        self.TRAINER_IMGSZ = int(os.getenv("TRAINER_IMGSZ", "640"))
        self.TRAINER_BATCH = int(os.getenv("TRAINER_BATCH", "16"))
        self.ML_AUTO_RETRAIN_ENABLED = int(os.getenv("ML_AUTO_RETRAIN_ENABLED", "0"))
        self.ML_AUTO_RETRAIN_INTERVAL_MIN = int(os.getenv("ML_AUTO_RETRAIN_INTERVAL_MIN", "180"))
        self.ML_AUTO_RETRAIN_HOURS = int(os.getenv("ML_AUTO_RETRAIN_HOURS", "24"))
        self.ML_AUTO_RETRAIN_MIN_ROWS = int(os.getenv("ML_AUTO_RETRAIN_MIN_ROWS", "1000"))
        self.ML_AUTO_RETRAIN_MIN_SCORE_GAIN = float(os.getenv("ML_AUTO_RETRAIN_MIN_SCORE_GAIN", "0.015"))
        self.AUTO_CAPTURE_ENABLED = int(os.getenv("AUTO_CAPTURE_ENABLED", "0"))
        self.AUTO_CAPTURE_DATASET_ROOT = os.getenv("AUTO_CAPTURE_DATASET_ROOT", "dataset")
        self.AUTO_CAPTURE_CLASSES = os.getenv("AUTO_CAPTURE_CLASSES", "car")
        self.AUTO_CAPTURE_MIN_CONF = float(os.getenv("AUTO_CAPTURE_MIN_CONF", "0.45"))
        self.AUTO_CAPTURE_COOLDOWN_SEC = float(os.getenv("AUTO_CAPTURE_COOLDOWN_SEC", "5.0"))
        self.AUTO_CAPTURE_VAL_SPLIT = float(os.getenv("AUTO_CAPTURE_VAL_SPLIT", "0.2"))
        self.AUTO_CAPTURE_JPEG_QUALITY = int(os.getenv("AUTO_CAPTURE_JPEG_QUALITY", "90"))
        self.AUTO_CAPTURE_MAX_BOXES_PER_FRAME = int(os.getenv("AUTO_CAPTURE_MAX_BOXES_PER_FRAME", "30"))
        self.AUTO_CAPTURE_UPLOAD_ENABLED = int(os.getenv("AUTO_CAPTURE_UPLOAD_ENABLED", "0"))
        self.AUTO_CAPTURE_UPLOAD_BUCKET = os.getenv("AUTO_CAPTURE_UPLOAD_BUCKET", "ml-datasets")
        self.AUTO_CAPTURE_UPLOAD_PREFIX = os.getenv("AUTO_CAPTURE_UPLOAD_PREFIX", "datasets/live-capture")
        self.AUTO_CAPTURE_DELETE_LOCAL_AFTER_UPLOAD = int(os.getenv("AUTO_CAPTURE_DELETE_LOCAL_AFTER_UPLOAD", "0"))
        self.AUTO_CAPTURE_UPLOAD_TIMEOUT_SEC = float(os.getenv("AUTO_CAPTURE_UPLOAD_TIMEOUT_SEC", "20"))
        self.BET_LOCK_SECONDS = int(os.getenv("BET_LOCK_SECONDS", "10"))
        self.WS_PORT = int(os.getenv("WS_PORT", "8000"))


@lru_cache(maxsize=1)
def get_config() -> Config:
    return Config()
