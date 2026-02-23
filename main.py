"""
main.py — FastAPI application entry point.
- Validates env at startup (fail-fast)
- Starts URL refresher first, waits for first live stream URL
- Starts AI background task with live URL
- Starts bet resolver loop (every 2s, resolves expired exact-count bets)
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

from datetime import datetime, timezone, timedelta

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
from ai.dataset_capture import LiveDatasetCapture
from ai.dataset_upload import SupabaseDatasetUploader
from ai.url_refresher import url_refresh_loop, get_current_url
from services.round_service import resolve_round_from_latest_snapshot
from services.round_session_service import session_scheduler_tick, next_session_round_at
from services.analytics_service import write_ml_detection_event
from services.ml_pipeline_service import auto_retrain_cycle
from services.ml_capture_monitor import record_capture_event

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
_resolver_task: asyncio.Task | None = None
_ml_retrain_task: asyncio.Task | None = None

# ── Shared state between ai_loop and round_monitor_loop ───────────────────────
_active_round: dict | None = None
_counter_ref: LineCounter | None = None


async def round_monitor_loop() -> None:
    """
    Poll Supabase every 5s for round lifecycle changes.
    - Resets counter when a new round becomes active.
    - Auto-opens upcoming rounds when opens_at passes.
    - Auto-resolves open rounds when ends_at passes.
    - Broadcasts 'round' event to public WS on change.
    """
    global _active_round, _counter_ref
    last_round_id: str | None = None

    while True:
        try:
            sb = await get_supabase()
            now_iso = datetime.now(timezone.utc).isoformat()

            # Auto-create rounds from active session loops.
            await session_scheduler_tick()

            # Auto-open: upcoming rounds whose opens_at has passed
            up_resp = await sb.table("bet_rounds") \
                .select("id") \
                .eq("status", "upcoming") \
                .lte("opens_at", now_iso) \
                .execute()
            for row in up_resp.data or []:
                await sb.table("bet_rounds").update({"status": "open"}).eq("id", row["id"]).execute()
                logger.info("Auto-opened round: %s", row["id"])

            # Auto-resolve: open rounds whose ends_at has passed
            ended_resp = await sb.table("bet_rounds") \
                .select("id") \
                .eq("status", "open") \
                .lte("ends_at", now_iso) \
                .execute()
            for row in ended_resp.data or []:
                round_id = row["id"]
                try:
                    result = await resolve_round_from_latest_snapshot(round_id)
                    logger.info("Auto-resolved round: %s result=%s", round_id, result)
                except Exception as resolve_exc:
                    logger.warning("Auto-resolve failed for round %s: %s", round_id, resolve_exc)

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


async def bet_resolver_loop() -> None:
    """
    Resolve expired exact-count micro-bets every 2 seconds.
    - Queries pending exact_count bets whose window has expired
    - For each: fetches end snapshot total, computes actual = end_total - baseline
    - Credits winner, updates status, broadcasts via ws_account
    """
    global _counter_ref

    while True:
        try:
            sb = await get_supabase()
            now_iso = datetime.now(timezone.utc).isoformat()

            # Find expired pending exact_count bets
            # window_start + window_duration_sec seconds <= now
            resp = await sb.table("bets") \
                .select("id, user_id, round_id, amount, potential_payout, exact_count, baseline_count, vehicle_class, window_start, window_duration_sec") \
                .eq("bet_type", "exact_count") \
                .eq("status", "pending") \
                .lte("window_start", now_iso) \
                .execute()

            for bet in (resp.data or []):
                try:
                    # Check if window has actually expired
                    ws_str = bet.get("window_start", "")
                    if not ws_str:
                        continue
                    window_start = datetime.fromisoformat(ws_str.replace("Z", "+00:00"))
                    window_dur = bet.get("window_duration_sec", 0) or 0
                    window_end = window_start + timedelta(seconds=window_dur)
                    now = datetime.now(timezone.utc)
                    if now < window_end:
                        continue  # not expired yet

                    bet_id = bet["id"]
                    user_id = bet["user_id"]
                    exact_count = bet.get("exact_count", 0)
                    baseline = bet.get("baseline_count", 0) or 0
                    vehicle_class = bet.get("vehicle_class")

                    # Get current count from in-memory counter if available
                    end_count = 0
                    if _counter_ref is not None:
                        end_count = _counter_ref.get_class_total(vehicle_class)
                    else:
                        # Fall back to latest DB snapshot
                        rnd_resp = await sb.table("bet_rounds") \
                            .select("camera_id") \
                            .eq("id", bet["round_id"]) \
                            .single() \
                            .execute()
                        camera_id = rnd_resp.data.get("camera_id") if rnd_resp.data else None
                        if camera_id:
                            snap_resp = await sb.table("count_snapshots") \
                                .select("total, vehicle_breakdown") \
                                .eq("camera_id", camera_id) \
                                .order("captured_at", desc=True) \
                                .limit(1) \
                                .execute()
                            if snap_resp.data:
                                snap = snap_resp.data[0]
                                if vehicle_class is None:
                                    end_count = snap.get("total", 0) or 0
                                else:
                                    bd = snap.get("vehicle_breakdown") or {}
                                    end_count = bd.get(vehicle_class, 0)

                    actual = max(0, end_count - baseline)
                    won = (actual == exact_count)

                    update_data = {
                        "status": "won" if won else "lost",
                        "actual_count": actual,
                        "resolved_at": now.isoformat(),
                    }
                    await sb.table("bets").update(update_data).eq("id", bet_id).execute()

                    # Credit winner
                    if won:
                        await sb.rpc(
                            "credit_user_balance",
                            {
                                "p_user_id": user_id,
                                "p_amount": int(bet["potential_payout"]),
                            },
                        ).execute()

                    # Broadcast resolution to user
                    await manager.send_to_user(user_id, {
                        "type": "bet_resolved",
                        "bet_id": str(bet_id),
                        "won": won,
                        "payout": bet["potential_payout"] if won else 0,
                        "actual": actual,
                        "exact": exact_count,
                        "vehicle_class": vehicle_class,
                    })

                    logger.info(
                        "Resolved bet %s: actual=%d exact=%d → %s",
                        bet_id, actual, exact_count, "WON" if won else "LOST"
                    )

                except Exception as bet_exc:
                    logger.warning("Error resolving bet %s: %s", bet.get("id"), bet_exc)

            # Early resolve market bets when "over" is already guaranteed.
            rounds_resp = await (
                sb.table("bet_rounds")
                .select("id, camera_id, market_type, params, status")
                .in_("status", ["open", "locked"])
                .in_("market_type", ["over_under", "vehicle_count"])
                .execute()
            )
            for rnd in (rounds_resp.data or []):
                try:
                    round_id = rnd.get("id")
                    params = rnd.get("params") or {}
                    threshold = int(params.get("threshold", 0) or 0)
                    vehicle_class = params.get("vehicle_class") if rnd.get("market_type") == "vehicle_count" else None

                    current_count = 0
                    if _counter_ref is not None:
                        current_count = _counter_ref.get_class_total(vehicle_class)
                    else:
                        camera_id = rnd.get("camera_id")
                        if camera_id:
                            snap_resp = await (
                                sb.table("count_snapshots")
                                .select("total, vehicle_breakdown")
                                .eq("camera_id", camera_id)
                                .order("captured_at", desc=True)
                                .limit(1)
                                .execute()
                            )
                            if snap_resp.data:
                                snap = snap_resp.data[0]
                                if vehicle_class is None:
                                    current_count = int(snap.get("total", 0) or 0)
                                else:
                                    current_count = int((snap.get("vehicle_breakdown") or {}).get(vehicle_class, 0) or 0)

                    bets_resp = await (
                        sb.table("bets")
                        .select("id, user_id, potential_payout, baseline_count, placed_at, markets(outcome_key)")
                        .eq("round_id", round_id)
                        .eq("status", "pending")
                        .or_("bet_type.eq.market,bet_type.is.null")
                        .execute()
                    )
                    for bet in (bets_resp.data or []):
                        market = bet.get("markets") or {}
                        outcome = str(market.get("outcome_key") or "").lower()
                        if outcome not in {"over", "under", "exact"}:
                            continue

                        if bet.get("baseline_count") is not None:
                            baseline = int(bet.get("baseline_count") or 0)
                        else:
                            baseline = 0
                            if rnd.get("camera_id") and bet.get("placed_at"):
                                try:
                                    snap_before_bet = await (
                                        sb.table("count_snapshots")
                                        .select("total, vehicle_breakdown")
                                        .eq("camera_id", rnd["camera_id"])
                                        .lte("captured_at", bet.get("placed_at"))
                                        .order("captured_at", desc=True)
                                        .limit(1)
                                        .execute()
                                    )
                                    if snap_before_bet.data:
                                        snap = snap_before_bet.data[0]
                                        if vehicle_class is None:
                                            baseline = int(snap.get("total", 0) or 0)
                                        else:
                                            baseline = int((snap.get("vehicle_breakdown") or {}).get(vehicle_class, 0) or 0)
                                except Exception:
                                    baseline = 0
                        actual = max(0, current_count - baseline)
                        if actual <= threshold:
                            continue

                        # Once threshold is exceeded, over is won; under/exact cannot win.
                        won = (outcome == "over")
                        await sb.table("bets").update({
                            "status": "won" if won else "lost",
                            "actual_count": actual,
                            "resolved_at": datetime.now(timezone.utc).isoformat(),
                        }).eq("id", bet["id"]).execute()

                        if won:
                            await sb.rpc(
                                "credit_user_balance",
                                {
                                    "p_user_id": bet["user_id"],
                                    "p_amount": int(bet["potential_payout"]),
                                },
                            ).execute()

                        await manager.send_to_user(bet["user_id"], {
                            "type": "bet_resolved",
                            "bet_id": str(bet["id"]),
                            "won": won,
                            "payout": bet["potential_payout"] if won else 0,
                            "actual": actual,
                            "exact": threshold,
                            "vehicle_class": vehicle_class,
                        })
                except Exception as early_exc:
                    logger.warning("Early market resolve error for round %s: %s", rnd.get("id"), early_exc)

        except Exception as exc:
            logger.warning("Bet resolver loop error: %s", exc)

        await asyncio.sleep(2)


async def ai_loop(cfg, hls_stream: HLSStream) -> None:
    """
    Continuous AI pipeline:
    HLS frames → YOLO detect → ByteTrack → LineZone → Supabase snapshot + WS broadcast
    """
    try:
        await _ai_loop_inner(cfg, hls_stream)
    except Exception as exc:
        logger.error("AI loop crashed: %s", exc, exc_info=True)
        raise


async def ml_auto_retrain_loop(cfg) -> None:
    """
    Run periodic ML retrain cycles.
    Real training is delegated to an external webhook GPU trainer.
    """
    if cfg.ML_AUTO_RETRAIN_ENABLED != 1:
        logger.info("ML auto-retrain is disabled")
        return

    while True:
        try:
            result = await auto_retrain_cycle(
                hours=cfg.ML_AUTO_RETRAIN_HOURS,
                min_rows=cfg.ML_AUTO_RETRAIN_MIN_ROWS,
                min_score_gain=cfg.ML_AUTO_RETRAIN_MIN_SCORE_GAIN,
                base_model=cfg.YOLO_MODEL,
                provider="webhook",
                params={
                    "trainer_webhook_url": cfg.TRAINER_WEBHOOK_URL,
                    "trainer_webhook_secret": cfg.TRAINER_WEBHOOK_SECRET,
                    "dataset_yaml_url": cfg.TRAINER_DATASET_YAML_URL,
                    "epochs": cfg.TRAINER_EPOCHS,
                    "imgsz": cfg.TRAINER_IMGSZ,
                    "batch": cfg.TRAINER_BATCH,
                },
            )
            logger.info("ML auto-retrain cycle: %s", result)
        except Exception as exc:
            logger.warning("ML auto-retrain cycle failed: %s", exc)

        await asyncio.sleep(max(60, cfg.ML_AUTO_RETRAIN_INTERVAL_MIN * 60))


async def _ai_loop_inner(cfg, hls_stream: HLSStream) -> None:
    global _counter_ref

    def _is_night_hour() -> bool:
        if int(getattr(cfg, "NIGHT_PROFILE_ENABLED", 0) or 0) != 1:
            return False
        hour = datetime.now().hour
        start = int(getattr(cfg, "NIGHT_PROFILE_START_HOUR", 18) or 18) % 24
        end = int(getattr(cfg, "NIGHT_PROFILE_END_HOUR", 6) or 6) % 24
        if start == end:
            return True
        if start < end:
            return start <= hour < end
        return hour >= start or hour < end

    def _apply_detector_profile(*, night: bool) -> None:
        if night:
            detector.conf = float(getattr(cfg, "NIGHT_YOLO_CONF", cfg.YOLO_CONF))
            detector.infer_size = int(getattr(cfg, "NIGHT_DETECT_INFER_SIZE", cfg.DETECT_INFER_SIZE))
            detector.iou = float(getattr(cfg, "NIGHT_DETECT_IOU", cfg.DETECT_IOU))
            detector.max_det = int(getattr(cfg, "NIGHT_DETECT_MAX_DET", cfg.DETECT_MAX_DET))
        else:
            detector.conf = float(cfg.YOLO_CONF)
            detector.infer_size = int(cfg.DETECT_INFER_SIZE)
            detector.iou = float(cfg.DETECT_IOU)
            detector.max_det = int(cfg.DETECT_MAX_DET)

    logger.info("AI loop inner: initialising detector")
    detector = VehicleDetector(
        model_path=cfg.YOLO_MODEL,
        conf_threshold=cfg.YOLO_CONF,
        infer_size=cfg.DETECT_INFER_SIZE,
        iou_threshold=cfg.DETECT_IOU,
        max_det=cfg.DETECT_MAX_DET,
    )
    profile_is_night: bool | None = None
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
    last_db_write = 0.0
    capture = LiveDatasetCapture(
        enabled=(cfg.AUTO_CAPTURE_ENABLED == 1),
        dataset_root=cfg.AUTO_CAPTURE_DATASET_ROOT,
        classes=[c.strip() for c in cfg.AUTO_CAPTURE_CLASSES.split(",")],
        min_conf=cfg.AUTO_CAPTURE_MIN_CONF,
        cooldown_sec=cfg.AUTO_CAPTURE_COOLDOWN_SEC,
        val_split=cfg.AUTO_CAPTURE_VAL_SPLIT,
        jpeg_quality=cfg.AUTO_CAPTURE_JPEG_QUALITY,
        max_boxes_per_frame=cfg.AUTO_CAPTURE_MAX_BOXES_PER_FRAME,
    )
    uploader = SupabaseDatasetUploader(
        enabled=(cfg.AUTO_CAPTURE_UPLOAD_ENABLED == 1),
        supabase_url=cfg.SUPABASE_URL,
        service_role_key=cfg.SUPABASE_SERVICE_ROLE_KEY,
        bucket=cfg.AUTO_CAPTURE_UPLOAD_BUCKET,
        prefix=cfg.AUTO_CAPTURE_UPLOAD_PREFIX,
        timeout_sec=cfg.AUTO_CAPTURE_UPLOAD_TIMEOUT_SEC,
        delete_local_after_upload=(cfg.AUTO_CAPTURE_DELETE_LOCAL_AFTER_UPLOAD == 1),
    )

    async def _upload_capture_async(capture_payload: dict) -> None:
        upload_result = await uploader.upload_capture(
            image_path=capture_payload["image_path"],
            label_path=capture_payload["label_path"],
            split=capture_payload["split"],
            camera_id=str(camera_id),
        )
        if upload_result.get("ok"):
            record_capture_event(
                "upload_success",
                "Uploaded capture to Supabase storage",
                {
                    "split": capture_payload["split"],
                    "remote_image": upload_result.get("remote_image"),
                    "remote_label": upload_result.get("remote_label"),
                },
            )
        else:
            record_capture_event(
                "upload_failed",
                "Failed to upload capture",
                {
                    "split": capture_payload["split"],
                    "error": upload_result.get("error"),
                    "image_path": capture_payload["image_path"],
                    "label_path": capture_payload["label_path"],
                },
            )

    async for frame in hls_stream.frames():
        now_is_night = _is_night_hour()
        if profile_is_night is None or now_is_night != profile_is_night:
            _apply_detector_profile(night=now_is_night)
            profile_is_night = now_is_night
            logger.info(
                "AI detector profile switched: %s (conf=%.2f, infer=%s, iou=%.2f, max_det=%s)",
                "night" if now_is_night else "day",
                detector.conf,
                detector.infer_size,
                detector.iou,
                detector.max_det,
            )

        # Hot-reload stream URL if the refresher has a newer one
        fresh = get_current_url()
        if fresh and fresh != hls_stream.url:
            logger.info("AI loop: updating stream URL → %s", fresh)
            hls_stream.url = fresh

        if frame_buf is None:
            h, w = frame.shape[:2]
            counter = LineCounter(camera_id, w, h)
            await counter.bootstrap_from_latest_snapshot()
            _counter_ref = counter
            frame_buf = True
            logger.info("AI loop started: frame size %dx%d", w, h)

        detections = await asyncio.to_thread(detector.detect, frame)
        # Keep full-frame detections for tracking/overlay visibility.
        # Count logic still applies detect/count zones inside LineCounter.process().
        tracked = tracker.update(detections)
        snapshot = await counter.process(frame, tracked)
        capture_result = await asyncio.to_thread(
            capture.maybe_capture,
            frame,
            snapshot.get("detections", []),
            str(camera_id),
        )
        if capture_result is not None:
            logger.info(
                "Captured sample split=%s boxes=%s image=%s",
                capture_result["split"],
                capture_result["boxes"],
                capture_result["image_path"],
            )
            record_capture_event(
                "capture_saved",
                "Captured live frame for dataset",
                {
                    "split": capture_result["split"],
                    "boxes": capture_result["boxes"],
                    "image_path": capture_result["image_path"],
                    "label_path": capture_result["label_path"],
                },
            )
            if cfg.AUTO_CAPTURE_UPLOAD_ENABLED == 1:
                asyncio.create_task(_upload_capture_async(capture_result))

        # Write snapshots at a fixed interval to reduce DB pressure without slowing live WS updates.
        loop_now = asyncio.get_running_loop().time()
        if (loop_now - last_db_write) >= cfg.DB_SNAPSHOT_INTERVAL_SEC:
            db_snapshot = {k: v for k, v in snapshot.items() if k not in ("detections", "new_crossings", "per_class_total")}
            asyncio.create_task(write_snapshot(db_snapshot))
            asyncio.create_task(write_ml_detection_event(camera_id, snapshot, cfg.YOLO_MODEL, detector.conf))
            last_db_write = loop_now

        if manager.public_count > 0:
            payload: dict = {"type": "count", **snapshot}
            if _active_round:
                payload["round"] = _active_round
            await manager.broadcast_public(payload)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _refresh_task, _ai_task, _round_task, _resolver_task, _ml_retrain_task

    cfg = get_config()
    logger.info("Config validated — starting WHITELINEZ backend")

    await get_supabase()

    # 1. Start URL refresher
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
        logger.error("No stream URL after 30s — AI loop will not start.")
    else:
        hls_stream = HLSStream(stream_url)
        _ai_task = asyncio.create_task(ai_loop(cfg, hls_stream), name="ai_loop")
        logger.info("AI loop started with URL: %s", stream_url)

    _round_task = asyncio.create_task(round_monitor_loop(), name="round_monitor")
    logger.info("Round monitor started")

    _resolver_task = asyncio.create_task(bet_resolver_loop(), name="bet_resolver")
    logger.info("Bet resolver started")

    if cfg.ML_AUTO_RETRAIN_ENABLED == 1:
        _ml_retrain_task = asyncio.create_task(ml_auto_retrain_loop(cfg), name="ml_auto_retrain")
        logger.info("ML auto-retrain loop started")
    else:
        logger.info("ML auto-retrain loop disabled")

    yield

    # Shutdown
    for task in (_ai_task, _refresh_task, _round_task, _resolver_task, _ml_retrain_task):
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
    next_round_at: str | None = None
    active_round_id: str | None = _active_round["id"] if _active_round else None
    active_round_status: str | None = _active_round.get("status") if _active_round else None
    try:
        sb = await get_supabase()
        now_iso = datetime.now(timezone.utc).isoformat()

        # Prefer DB truth for health checks in case in-memory round cache is stale.
        if not active_round_id:
            open_resp = await (
                sb.table("bet_rounds")
                .select("id, status, opens_at")
                .eq("status", "open")
                .order("opens_at", ascending=False)
                .limit(1)
                .maybeSingle()
                .execute()
            )
            active_row = open_resp.data

            if not active_row:
                locked_resp = await (
                    sb.table("bet_rounds")
                    .select("id, status, closes_at")
                    .eq("status", "locked")
                    .order("closes_at", ascending=False)
                    .limit(1)
                    .maybeSingle()
                    .execute()
                )
                active_row = locked_resp.data

            if active_row:
                active_round_id = active_row.get("id")
                active_round_status = active_row.get("status")

        up_resp = await (
            sb.table("bet_rounds")
            .select("opens_at")
            .eq("status", "upcoming")
            .gte("opens_at", now_iso)
            .order("opens_at", ascending=True)
            .limit(1)
            .maybeSingle()
            .execute()
        )
        next_round_at = (up_resp.data or {}).get("opens_at")
        if not next_round_at:
            next_round_at = await next_session_round_at()
    except Exception:
        next_round_at = None

    return {
        "status": "ok",
        "stream_url": get_current_url(),
        "public_ws_connections": manager.public_count,
        "user_ws_connections": manager.user_count,
        "ai_task_running": _ai_task is not None and not _ai_task.done(),
        "refresh_task_running": _refresh_task is not None and not _refresh_task.done(),
        "round_task_running": _round_task is not None and not _round_task.done(),
        "resolver_task_running": _resolver_task is not None and not _resolver_task.done(),
        "ml_retrain_task_running": _ml_retrain_task is not None and not _ml_retrain_task.done(),
        "active_round_id": active_round_id,
        "active_round_status": active_round_status,
        "next_round_at": next_round_at,
    }


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logger.error("Unhandled error on %s: %s", request.url, exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
