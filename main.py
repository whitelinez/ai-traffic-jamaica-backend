"""
main.py — FastAPI application entry point.
- Validates env at startup (fail-fast)
- Starts URL refresher first, waits for first live stream URL
- Starts AI background task with live URL
- Mounts all routers and WebSocket endpoints
"""
import asyncio
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from datetime import datetime, timezone

from config import get_config
from middleware.rate_limiter import limiter
from routers import bets, rounds, admin, stream
from websocket.ws_public import router as ws_public_router
from websocket.ws_account import router as ws_account_router
from websocket.ws_manager import manager
from supabase_client import get_supabase, close_supabase
from ai.stream import HLSStream
from ai.detector import VehicleDetector
from ai.tracker import VehicleTracker
from ai.counter import LineCounter, write_snapshot
from ai.url_refresher import url_refresh_loop, get_current_url

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Background task handles ────────────────────────────────────────────────────
_refresh_task: asyncio.Task | None = None
_ai_task: asyncio.Task | None = None
_round_task: asyncio.Task | None = None

# ── Shared state between ai_loop and round_monitor_loop ───────────────────────
_active_round: dict | None = None
_counter_ref: LineCounter | None = None


async def round_monitor_loop() -> None:
    """
    Poll Supabase every 5s for an open bet round.
    - Resets counter when a new round becomes active.
    - Broadcasts 'round' event to public WS on change.
    """
    global _active_round, _counter_ref
    last_round_id: str | None = None

    while True:
        try:
            sb = await get_supabase()
            now_iso = datetime.now(timezone.utc).isoformat()

            # Auto-open: upcoming rounds whose opens_at has passed
            up_resp = await sb.table("bet_rounds") \
                .select("id") \
                .eq("status", "upcoming") \
                .lte("opens_at", now_iso) \
                .execute()
            for row in up_resp.data or []:
                await sb.table("bet_rounds").update({"status": "open"}).eq("id", row["id"]).execute()
                logger.info("Auto-opened round: %s", row["id"])

            # Auto-lock: open rounds whose ends_at has passed
            ended_resp = await sb.table("bet_rounds") \
                .select("id") \
                .eq("status", "open") \
                .lte("ends_at", now_iso) \
                .execute()
            for row in ended_resp.data or []:
                await sb.table("bet_rounds").update({"status": "locked"}).eq("id", row["id"]).execute()
                logger.info("Auto-locked round: %s", row["id"])

            resp = await sb.table("bet_rounds") \
                .select("id, status, market_type, opens_at, closes_at, ends_at") \
                .eq("status", "open") \
                .order("opens_at", ascending=False) \
                .limit(1) \
                .execute()

            round_data = resp.data[0] if resp.data else None
            _active_round = round_data

            if round_data:
                round_id = round_data["id"]
                if round_id != last_round_id:
                    logger.info("New active round: %s — resetting counter", round_id)
                    if _counter_ref:
                        _counter_ref.reset()
                    last_round_id = round_id
                    if manager.public_count > 0:
                        await manager.broadcast_public({"type": "round", "round": round_data})
            else:
                if last_round_id is not None:
                    logger.info("Active round ended — counter idle")
                    last_round_id = None
                    if manager.public_count > 0:
                        await manager.broadcast_public({"type": "round", "round": None})

        except Exception as exc:
            logger.warning("Round monitor error: %s", exc)

        await asyncio.sleep(5)


async def ai_loop(cfg, hls_stream: HLSStream) -> None:
    """
    Continuous AI pipeline:
    HLS frames → YOLO detect → ByteTrack → LineZone → Supabase snapshot + WS broadcast

    Checks for URL updates from the refresh loop on every reconnect.
    """
    try:
        await _ai_loop_inner(cfg, hls_stream)
    except Exception as exc:
        logger.error("AI loop crashed: %s", exc, exc_info=True)
        raise


async def _ai_loop_inner(cfg, hls_stream: HLSStream) -> None:
    global _counter_ref

    logger.info("AI loop inner: initialising detector")
    detector = VehicleDetector(model_path=cfg.YOLO_MODEL, conf_threshold=cfg.YOLO_CONF)
    logger.info("AI loop inner: initialising tracker")
    tracker = VehicleTracker()

    logger.info("AI loop inner: querying camera_id")
    sb = await get_supabase()
    cam_resp = await sb.table("cameras").select("id").eq("ipcam_alias", cfg.CAMERA_ALIAS).limit(1).execute()
    camera_id = cam_resp.data[0]["id"] if cam_resp.data else "default"
    logger.info("AI loop using camera_id: %s", camera_id)

    logger.info("AI loop inner: opening HLS stream")
    frame_buf = None
    counter: LineCounter | None = None

    async for frame in hls_stream.frames():
        # Hot-reload stream URL if the refresher has a newer one
        fresh = get_current_url()
        if fresh and fresh != hls_stream.url:
            logger.info("AI loop: updating stream URL → %s", fresh)
            hls_stream.url = fresh

        if frame_buf is None:
            h, w = frame.shape[:2]
            counter = LineCounter(camera_id, w, h)
            _counter_ref = counter
            frame_buf = True
            logger.info("AI loop started: frame size %dx%d", w, h)

        detections = await asyncio.to_thread(detector.detect, frame)
        # Filter to zone before tracking so only relevant vehicles are tracked
        detections = counter.zone_filter(detections)
        tracked = tracker.update(detections)
        snapshot = await counter.process(frame, tracked)

        # Write snapshot to DB — exclude detection boxes (WS-only, not persisted)
        db_snapshot = {k: v for k, v in snapshot.items() if k != "detections"}
        asyncio.create_task(write_snapshot(db_snapshot))

        if manager.public_count > 0:
            payload: dict = {"type": "count", **snapshot}
            if _active_round:
                payload["round"] = _active_round
            await manager.broadcast_public(payload)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _refresh_task, _ai_task, _round_task

    cfg = get_config()
    logger.info("Config validated — starting WHITELINEZ backend")

    await get_supabase()

    # 1. Start URL refresher — fetches immediately, then every interval
    _refresh_task = asyncio.create_task(
        url_refresh_loop(cfg.CAMERA_ALIAS, cfg.URL_REFRESH_INTERVAL),
        name="url_refresh_loop",
    )
    logger.info("URL refresh task started (alias=%s, interval=%ds)", cfg.CAMERA_ALIAS, cfg.URL_REFRESH_INTERVAL)

    # 2. Wait up to 30s for first URL before starting AI loop
    stream_url = None
    for _ in range(30):
        stream_url = get_current_url()
        if stream_url:
            break
        await asyncio.sleep(1)

    if not stream_url:
        logger.error("No stream URL after 30s — AI loop will not start. Check CAMERA_ALIAS and ipcamlive connectivity.")
    else:
        hls_stream = HLSStream(stream_url)
        _ai_task = asyncio.create_task(ai_loop(cfg, hls_stream), name="ai_loop")
        logger.info("AI loop started with URL: %s", stream_url)

    _round_task = asyncio.create_task(round_monitor_loop(), name="round_monitor")
    logger.info("Round monitor started")

    yield

    # Shutdown
    for task in (_ai_task, _refresh_task, _round_task):
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    await close_supabase()
    logger.info("WHITELINEZ backend shutdown complete")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="WHITELINEZ API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

cfg_once = None
try:
    cfg_once = get_config()
    allowed_origins = [cfg_once.ALLOWED_ORIGIN]
except Exception:
    allowed_origins = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(bets.router)
app.include_router(rounds.router)
app.include_router(admin.router)
app.include_router(stream.router)
app.include_router(ws_public_router)
app.include_router(ws_account_router)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "stream_url": get_current_url(),
        "public_ws_connections": manager.public_count,
        "user_ws_connections": manager.user_count,
        "ai_task_running": _ai_task is not None and not _ai_task.done(),
        "refresh_task_running": _refresh_task is not None and not _refresh_task.done(),
        "round_task_running": _round_task is not None and not _round_task.done(),
        "active_round_id": _active_round["id"] if _active_round else None,
    }


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logger.error("Unhandled error on %s: %s", request.url, exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
