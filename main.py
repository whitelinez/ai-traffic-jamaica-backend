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
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any
import urllib.request
import json

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import cv2
import numpy as np

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
from ai.box_smoother import BoxSmoother
from ai.live_state import set_live_snapshot
from ai.dataset_capture import LiveDatasetCapture
from ai.dataset_upload import SupabaseDatasetUploader
from ai.url_refresher import url_refresh_loop, get_current_url, get_current_alias
from services.round_service import resolve_round_from_latest_snapshot
from services.round_session_service import session_scheduler_tick, next_session_round_at
from services.analytics_service import write_ml_detection_event
from services.ml_pipeline_service import auto_retrain_cycle
from services.ml_capture_monitor import record_capture_event, is_capture_paused
from services.runtime_tuner import (
    RUNTIME_PROFILES,
    TrafficStats,
    select_runtime_profile,
)

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
_watchdog_task: asyncio.Task | None = None

_WATCHDOG_INTERVAL_SEC = 8.0
_WATCHDOG_RESTART_COOLDOWN_SEC = 12.0
_WATCHDOG_STARTUP_WAIT_SEC = 30
_AI_HEARTBEAT_STALE_SEC = 35.0
_MARKET_EARLY_RESOLVE_GRACE_SEC = 6
_watchdog_restart_counts: dict[str, int] = {
    "refresh": 0,
    "ai": 0,
    "round": 0,
    "resolver": 0,
    "ml_retrain": 0,
}
_watchdog_last_restart_iso: dict[str, str | None] = {
    "refresh": None,
    "ai": None,
    "round": None,
    "resolver": None,
    "ml_retrain": None,
}
_watchdog_last_restart_monotonic: dict[str, float] = {
    "refresh": 0.0,
    "ai": 0.0,
    "round": 0.0,
    "resolver": 0.0,
    "ml_retrain": 0.0,
}
_ai_runtime_state: dict[str, Any] = {
    "started_at": None,
    "started_monotonic": 0.0,
    "last_frame_at": None,
    "last_frame_monotonic": 0.0,
    "frames_total": 0,
    "last_db_write_at": None,
    "last_db_write_monotonic": 0.0,
    "last_error": None,
    "fps_window_start_monotonic": 0.0,
    "fps_window_frames": 0,
    "fps_estimate": 0.0,
}
_ai_inference_runtime: dict[str, Any] = {
    "device": "unknown",
    "cuda_available": None,
    "device_name": None,
}

# ── Shared state between ai_loop and round_monitor_loop ───────────────────────
_active_round: dict | None = None
_counter_ref: LineCounter | None = None
_weather_cache: dict[str, Any] = {
    "ts": 0.0,
    "payload": None,
    "last_ok": None,
    "last_error": None,
}


def _map_weather_code_to_label(code: int) -> str:
    rainy_codes = {
        51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99
    }
    if code in rainy_codes:
        return "raining"
    if code in {0, 1}:
        return "sunny"
    if code in {2, 3, 45, 48}:
        return "cloudy"
    return "scanning"


def _fetch_jamaica_weather_cached() -> dict[str, Any]:
    """
    Fetches lightweight live weather for Kingston, Jamaica from Open-Meteo.
    Cached for 10 minutes to avoid per-frame network calls.
    """
    now = time.time()
    cached = _weather_cache.get("payload")
    if cached and (now - float(_weather_cache.get("ts", 0.0))) < 600.0:
        return cached

    try:
        # Kingston, Jamaica
        lat = 17.9970
        lon = -76.7936
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current=is_day,weather_code&timezone=America%2FJamaica"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "whitelinez/1.0"})
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        current = data.get("current") or {}
        is_day = int(current.get("is_day", 0) or 0) == 1
        code = int(current.get("weather_code", -1) or -1)
        payload = {
            "lighting": "day" if is_day else "night",
            "weather": _map_weather_code_to_label(code),
            "confidence": 0.9 if code >= 0 else 0.0,
            "source": "open-meteo",
        }
        _weather_cache["ts"] = now
        _weather_cache["payload"] = payload
        _weather_cache["last_ok"] = True
        _weather_cache["last_error"] = None
        return payload
    except Exception as exc:
        _weather_cache["ts"] = now
        _weather_cache["last_ok"] = False
        _weather_cache["last_error"] = str(exc)
        raise


def _infer_scene_status(frame: np.ndarray) -> dict[str, Any]:
    """
    Lightweight vision-based scene classification.
    Uses per-frame image statistics, not time-of-day scripting.
    """
    try:
        if frame is None or frame.size == 0:
            return {
                "scene_lighting": "unknown",
                "scene_weather": "unknown",
                "scene_confidence": 0.0,
            }

        h, w = frame.shape[:2]
        scale = max(1.0, max(h, w) / 360.0)
        sw = max(96, int(w / scale))
        sh = max(64, int(h / scale))
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1].astype(np.float32)
        v = hsv[:, :, 2].astype(np.float32)

        brightness = float(np.mean(v))
        contrast = float(np.std(gray))
        saturation = float(np.mean(s))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        highlight_ratio = float(np.mean(v >= 235.0))

        edges = cv2.Canny(gray, 60, 150)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=16,
            minLineLength=8,
            maxLineGap=3,
        )
        total_lines = 0
        streak_lines = 0
        if lines is not None:
            for raw in lines:
                ln = raw[0]
                x1, y1, x2, y2 = int(ln[0]), int(ln[1]), int(ln[2]), int(ln[3])
                dx = x2 - x1
                dy = y2 - y1
                length = float(np.hypot(dx, dy))
                if length < 6.0:
                    continue
                total_lines += 1
                angle = abs(float(np.degrees(np.arctan2(dy, dx))))
                # Rain streaks tend to be short/vertical-ish line segments.
                if 65.0 <= angle <= 115.0 and length <= 42.0:
                    streak_lines += 1

        rain_ratio = (float(streak_lines) / float(total_lines)) if total_lines > 0 else 0.0

        low_light = brightness < 70.0
        twilight = 70.0 <= brightness < 95.0
        glare = (highlight_ratio > 0.06 and brightness > 125.0) or (brightness > 150.0 and saturation < 55.0)
        hazy = brightness >= 95.0 and contrast < 38.0 and saturation < 60.0
        rainy = rain_ratio > 0.42 and contrast < 58.0

        if low_light:
            lighting = "night"
        elif twilight:
            lighting = "day"
        elif glare:
            lighting = "day"
        else:
            lighting = "day"

        if rainy:
            weather = "raining"
            conf = min(0.95, 0.55 + rain_ratio * 0.6)
        elif hazy:
            weather = "cloudy"
            conf = 0.70
        elif glare:
            weather = "sunny"
            conf = 0.76
        elif low_light:
            weather = "clear"
            conf = 0.82
        else:
            weather = "sunny"
            conf = 0.8

        return {
            "scene_lighting": lighting,
            "scene_weather": weather,
            "scene_confidence": round(float(max(0.0, min(1.0, conf))), 3),
            "scene_source": "vision",
            "scene_metrics": {
                "brightness": round(brightness, 1),
                "contrast": round(contrast, 1),
                "saturation": round(saturation, 1),
                "sharpness": round(sharpness, 1),
                "rain_ratio": round(rain_ratio, 3),
            },
        }
    except Exception:
        return {
            "scene_lighting": "unknown",
            "scene_weather": "unknown",
            "scene_confidence": 0.0,
            "scene_source": "vision",
        }


def _merge_scene_and_weather(
    vision_status: dict[str, Any],
    weather_status: dict[str, Any] | None,
) -> dict[str, Any]:
    if not weather_status:
        return vision_status

    merged = dict(vision_status or {})
    vision_weather = str((vision_status or {}).get("scene_weather") or "").strip().lower()
    vision_conf = float((vision_status or {}).get("scene_confidence") or 0.0)
    api_lighting = str(weather_status.get("lighting") or "").strip().lower()
    api_weather = str(weather_status.get("weather") or "").strip().lower()
    api_conf = float(weather_status.get("confidence") or 0.0)

    if api_lighting in {"day", "night"}:
        merged["scene_lighting"] = api_lighting

    # Keep strong visual rain detection when present, otherwise prefer API weather.
    if vision_weather == "raining" and vision_conf >= 0.7:
        merged["scene_weather"] = "raining"
        merged["scene_confidence"] = max(vision_conf, api_conf)
        merged["scene_source"] = "vision+weather"
    elif api_weather:
        merged["scene_weather"] = api_weather
        merged["scene_confidence"] = max(vision_conf, api_conf)
        merged["scene_source"] = "weather"
    else:
        merged["scene_source"] = "vision"

    return merged


def _task_running(task: asyncio.Task | None) -> bool:
    return task is not None and not task.done()


def _task_failure(task: asyncio.Task | None) -> str | None:
    if task is None or not task.done():
        return None
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        return "cancelled"
    except Exception as exc:  # pragma: no cover
        return f"exception_lookup_failed: {exc}"
    return str(exc) if exc else "completed"


def _can_restart(task_name: str, now_mono: float) -> bool:
    last = float(_watchdog_last_restart_monotonic.get(task_name, 0.0) or 0.0)
    return (now_mono - last) >= _WATCHDOG_RESTART_COOLDOWN_SEC


def _mark_restart(task_name: str, reason: str) -> None:
    _watchdog_restart_counts[task_name] = int(_watchdog_restart_counts.get(task_name, 0)) + 1
    _watchdog_last_restart_monotonic[task_name] = asyncio.get_running_loop().time()
    _watchdog_last_restart_iso[task_name] = datetime.now(timezone.utc).isoformat()
    logger.warning("Watchdog restarted %s task (reason=%s)", task_name, reason)


def _reset_ai_runtime_state(reason: str | None = None) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    now_mono = asyncio.get_running_loop().time()
    _ai_runtime_state["started_at"] = now_iso
    _ai_runtime_state["started_monotonic"] = now_mono
    _ai_runtime_state["last_frame_at"] = None
    _ai_runtime_state["last_frame_monotonic"] = 0.0
    _ai_runtime_state["frames_total"] = 0
    _ai_runtime_state["last_db_write_at"] = None
    _ai_runtime_state["last_db_write_monotonic"] = 0.0
    _ai_runtime_state["last_error"] = reason
    _ai_runtime_state["fps_window_start_monotonic"] = now_mono
    _ai_runtime_state["fps_window_frames"] = 0
    _ai_runtime_state["fps_estimate"] = 0.0


def _mark_ai_frame_processed() -> None:
    now_mono = asyncio.get_running_loop().time()
    now_iso = datetime.now(timezone.utc).isoformat()
    _ai_runtime_state["last_frame_monotonic"] = now_mono
    _ai_runtime_state["last_frame_at"] = now_iso
    _ai_runtime_state["frames_total"] = int(_ai_runtime_state.get("frames_total", 0) or 0) + 1

    win_start = float(_ai_runtime_state.get("fps_window_start_monotonic", 0.0) or 0.0)
    if win_start <= 0.0:
        _ai_runtime_state["fps_window_start_monotonic"] = now_mono
        _ai_runtime_state["fps_window_frames"] = 1
        return
    win_frames = int(_ai_runtime_state.get("fps_window_frames", 0) or 0) + 1
    _ai_runtime_state["fps_window_frames"] = win_frames
    elapsed = now_mono - win_start
    if elapsed >= 4.0:
        _ai_runtime_state["fps_estimate"] = round(win_frames / max(elapsed, 0.001), 2)
        _ai_runtime_state["fps_window_start_monotonic"] = now_mono
        _ai_runtime_state["fps_window_frames"] = 0


def _mark_ai_db_write() -> None:
    _ai_runtime_state["last_db_write_monotonic"] = asyncio.get_running_loop().time()
    _ai_runtime_state["last_db_write_at"] = datetime.now(timezone.utc).isoformat()


async def _wait_for_stream_url(timeout_sec: int) -> str | None:
    for _ in range(max(1, int(timeout_sec))):
        stream_url = get_current_url()
        if stream_url:
            return stream_url
        await asyncio.sleep(1)
    return None


async def _ensure_ai_task(cfg, timeout_sec: int = 6, reason: str = "watchdog") -> bool:
    global _ai_task
    if _task_running(_ai_task):
        return True
    stream_url = await _wait_for_stream_url(timeout_sec)
    if not stream_url:
        logger.error("AI loop restart skipped (%s): stream URL unavailable", reason)
        _ai_runtime_state["last_error"] = f"stream_unavailable:{reason}"
        return False
    hls_stream = HLSStream(stream_url)
    _reset_ai_runtime_state(reason=f"start:{reason}")
    _ai_task = asyncio.create_task(ai_loop(cfg, hls_stream), name="ai_loop")
    logger.info("AI loop started (%s) with URL: %s", reason, stream_url)
    return True


async def health_watchdog_loop(cfg) -> None:
    global _refresh_task, _round_task, _resolver_task, _ml_retrain_task
    while True:
        loop_now = asyncio.get_running_loop().time()
        try:
            if not _task_running(_refresh_task) and _can_restart("refresh", loop_now):
                reason = _task_failure(_refresh_task) or "not_running"
                _refresh_task = asyncio.create_task(
                    url_refresh_loop(cfg.CAMERA_ALIAS, cfg.URL_REFRESH_INTERVAL),
                    name="url_refresh_loop",
                )
                _mark_restart("refresh", reason)

            ai_running = _task_running(_ai_task)
            ai_restart_reason: str | None = None
            if ai_running:
                last_frame_mono = float(_ai_runtime_state.get("last_frame_monotonic", 0.0) or 0.0)
                started_mono = float(_ai_runtime_state.get("started_monotonic", 0.0) or 0.0)
                since_start = loop_now - started_mono if started_mono > 0 else 0.0
                since_frame = loop_now - last_frame_mono if last_frame_mono > 0 else None
                if (
                    since_start > _AI_HEARTBEAT_STALE_SEC
                    and (since_frame is None or since_frame > _AI_HEARTBEAT_STALE_SEC)
                    and _can_restart("ai", loop_now)
                ):
                    ai_restart_reason = f"stale_heartbeat:{round(since_start, 1)}s"
                    _ai_runtime_state["last_error"] = ai_restart_reason
                    if _ai_task and not _ai_task.done():
                        _ai_task.cancel()
                        try:
                            await _ai_task
                        except asyncio.CancelledError:
                            pass
                    ai_running = False

            if not ai_running and _can_restart("ai", loop_now):
                reason = ai_restart_reason or _task_failure(_ai_task) or "not_running"
                started = await _ensure_ai_task(cfg, timeout_sec=6, reason="watchdog")
                if started:
                    _mark_restart("ai", reason)

            if not _task_running(_round_task) and _can_restart("round", loop_now):
                reason = _task_failure(_round_task) or "not_running"
                _round_task = asyncio.create_task(round_monitor_loop(), name="round_monitor")
                _mark_restart("round", reason)

            if not _task_running(_resolver_task) and _can_restart("resolver", loop_now):
                reason = _task_failure(_resolver_task) or "not_running"
                _resolver_task = asyncio.create_task(bet_resolver_loop(), name="bet_resolver")
                _mark_restart("resolver", reason)

            if cfg.ML_AUTO_RETRAIN_ENABLED == 1:
                if not _task_running(_ml_retrain_task) and _can_restart("ml_retrain", loop_now):
                    reason = _task_failure(_ml_retrain_task) or "not_running"
                    _ml_retrain_task = asyncio.create_task(
                        ml_auto_retrain_loop(cfg),
                        name="ml_auto_retrain",
                    )
                    _mark_restart("ml_retrain", reason)
        except Exception as exc:
            logger.exception("Watchdog loop error: %s", exc)

        await asyncio.sleep(_WATCHDOG_INTERVAL_SEC)


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
                .select("id, camera_id, opens_at, params") \
                .eq("status", "upcoming") \
                .lte("opens_at", now_iso) \
                .execute()
            for row in up_resp.data or []:
                params = row.get("params") or {}
                baseline_total = 0
                baseline_by_class = {}
                baseline_captured_at = None
                camera_id = row.get("camera_id")
                opens_at = row.get("opens_at")
                if camera_id and opens_at:
                    snap = None
                    try:
                        snap_before = await sb.table("count_snapshots") \
                            .select("captured_at, total, vehicle_breakdown") \
                            .eq("camera_id", camera_id) \
                            .lte("captured_at", opens_at) \
                            .order("captured_at", desc=True) \
                            .limit(1) \
                            .execute()
                        if snap_before.data:
                            snap = snap_before.data[0]
                        else:
                            snap_after = await sb.table("count_snapshots") \
                                .select("captured_at, total, vehicle_breakdown") \
                                .eq("camera_id", camera_id) \
                                .gte("captured_at", opens_at) \
                                .order("captured_at", desc=False) \
                                .limit(1) \
                                .execute()
                            if snap_after.data:
                                snap = snap_after.data[0]
                            else:
                                latest_cam = await sb.table("count_snapshots") \
                                    .select("captured_at, total, vehicle_breakdown") \
                                    .eq("camera_id", camera_id) \
                                    .order("captured_at", desc=True) \
                                    .limit(1) \
                                    .execute()
                                if latest_cam.data:
                                    snap = latest_cam.data[0]
                    except Exception:
                        snap = None

                    if snap:
                        baseline_total = int(snap.get("total", 0) or 0)
                        baseline_by_class = snap.get("vehicle_breakdown") or {}
                        baseline_captured_at = snap.get("captured_at")
                    elif _counter_ref:
                        live = _counter_ref.get_snapshot(None)
                        baseline_total = int(live.get("total", 0) or 0)
                        baseline_by_class = live.get("vehicle_breakdown") or {}
                        baseline_captured_at = datetime.now(timezone.utc).isoformat()

                next_params = {
                    **params,
                    "round_baseline_total": baseline_total,
                    "round_baseline_by_class": baseline_by_class,
                    "round_baseline_captured_at": baseline_captured_at,
                }
                await sb.table("bet_rounds").update({
                    "status": "open",
                    "params": next_params,
                }).eq("id", row["id"]).execute()
                logger.info("Auto-opened round: %s", row["id"])

            # Auto-resolve: open/locked rounds whose ends_at has passed.
            # Locked rounds can exist briefly after cutoff; if they linger, resolve them.
            ended_resp = await sb.table("bet_rounds") \
                .select("id") \
                .in_("status", ["open", "locked"]) \
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
                .order("opens_at", desc=True) \
                .limit(1) \
                .execute()

            round_data = resp.data[0] if resp.data else None
            _active_round = round_data

            if round_data:
                round_id = round_data["id"]
                if round_id != last_round_id:
                    logger.info("New active round: %s — setting round baseline", round_id)
                    if _counter_ref:
                        _counter_ref.reset_round()
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

            round_camera_cache: dict[str, str | None] = {}

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
                    round_id = str(bet.get("round_id") or "")

                    camera_id = None
                    if round_id:
                        if round_id not in round_camera_cache:
                            try:
                                rnd_resp = await sb.table("bet_rounds") \
                                    .select("camera_id") \
                                    .eq("id", round_id) \
                                    .single() \
                                    .execute()
                                round_camera_cache[round_id] = (rnd_resp.data or {}).get("camera_id")
                            except Exception:
                                round_camera_cache[round_id] = None
                        camera_id = round_camera_cache.get(round_id)

                    # Re-anchor baseline to the snapshot at/before bet window start when available.
                    if camera_id and ws_str:
                        try:
                            start_snap_resp = await sb.table("count_snapshots") \
                                .select("total, vehicle_breakdown") \
                                .eq("camera_id", camera_id) \
                                .lte("captured_at", window_start.isoformat()) \
                                .order("captured_at", desc=True) \
                                .limit(1) \
                                .execute()
                            if start_snap_resp.data:
                                snap = start_snap_resp.data[0]
                                if vehicle_class is None:
                                    baseline = max(int(baseline or 0), int(snap.get("total", 0) or 0))
                                else:
                                    bd = snap.get("vehicle_breakdown") or {}
                                    baseline = max(int(baseline or 0), int(bd.get(vehicle_class, 0) or 0))
                        except Exception:
                            pass

                    # Resolve end count at/before window_end from DB first to keep window-bounded fairness.
                    end_count: int | None = None
                    if camera_id:
                        try:
                            end_snap_resp = await sb.table("count_snapshots") \
                                .select("total, vehicle_breakdown") \
                                .eq("camera_id", camera_id) \
                                .lte("captured_at", window_end.isoformat()) \
                                .order("captured_at", desc=True) \
                                .limit(1) \
                                .execute()
                            if end_snap_resp.data:
                                snap = end_snap_resp.data[0]
                                if vehicle_class is None:
                                    end_count = int(snap.get("total", 0) or 0)
                                else:
                                    bd = snap.get("vehicle_breakdown") or {}
                                    end_count = int(bd.get(vehicle_class, 0) or 0)
                        except Exception:
                            end_count = None

                    # Fallback to in-memory/live state if no bounded DB snapshot is available yet.
                    if end_count is None:
                        if _counter_ref is not None:
                            end_count = int(_counter_ref.get_class_total(vehicle_class))
                        else:
                            end_count = 0

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
                        "user_id": str(user_id),
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

            # Market bets are resolved only at round end through round_service.
            # This avoids instant win/loss decisions from transient global counts.

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
        _ai_runtime_state["last_error"] = str(exc)
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

    def _bounded_float(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(v)))

    def _bounded_int(v: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, int(v)))

    logger.info("AI loop inner: initialising detector")
    detector = VehicleDetector(
        model_path=cfg.YOLO_MODEL,
        conf_threshold=cfg.YOLO_CONF,
        infer_size=cfg.DETECT_INFER_SIZE,
        iou_threshold=cfg.DETECT_IOU,
        max_det=cfg.DETECT_MAX_DET,
    )
    _ai_inference_runtime.update(detector.runtime_info())
    logger.info("AI inference runtime: %s", _ai_inference_runtime)
    profile_is_night: bool | None = None
    logger.info("AI loop inner: initialising tracker")
    tracker = VehicleTracker()
    box_smoother = BoxSmoother()

    async def _resolve_camera_id_for_alias(alias: str) -> str:
        resp = await (
            sb.table("cameras")
            .select("id")
            .eq("ipcam_alias", alias)
            .limit(1)
            .execute()
        )
        if resp.data:
            return resp.data[0]["id"]
        fallback_resp = await (
            sb.table("cameras")
            .select("id")
            .eq("is_active", True)
            .order("created_at", desc=False)
            .limit(1)
            .execute()
        )
        return fallback_resp.data[0]["id"] if fallback_resp.data else "default"

    logger.info("AI loop inner: querying camera_id")
    sb = await get_supabase()
    camera_alias = get_current_alias() or cfg.CAMERA_ALIAS
    camera_id = await _resolve_camera_id_for_alias(camera_alias)
    logger.info("AI loop using camera alias=%s camera_id=%s", camera_alias, camera_id)

    logger.info("AI loop inner: opening HLS stream")
    frame_buf = None
    counter: LineCounter | None = None
    last_db_write = 0.0
    frame_index = 0
    process_every_n = 1
    runtime_profile_name = ""
    runtime_profile_reason = ""
    last_runtime_eval = 0.0
    last_runtime_switch = 0.0
    runtime_events: deque[tuple[float, int, int, float]] = deque()
    scene_status: dict[str, Any] = {
        "scene_lighting": "unknown",
        "scene_weather": "unknown",
        "scene_confidence": 0.0,
        "scene_source": "vision",
    }
    last_scene_eval = 0.0
    weather_status: dict[str, Any] | None = None
    last_weather_eval = 0.0
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
        frame_index += 1
        _mark_ai_frame_processed()

        latest_alias = get_current_alias() or cfg.CAMERA_ALIAS
        if latest_alias != camera_alias:
            old_alias = camera_alias
            camera_alias = latest_alias
            camera_id = await _resolve_camera_id_for_alias(camera_alias)
            logger.info("AI camera switched alias %s -> %s (camera_id=%s)", old_alias, camera_alias, camera_id)
            counter = None
            _counter_ref = None
            frame_buf = None      # reset so counter re-initialises on next frame
            process_every_n = 1   # reset frame-skip so old camera's perf state doesn't carry over
            box_smoother.reset()  # clear stale EMA state for previous camera
            try:
                runtime_profile_name = ""
                last_runtime_eval = 0.0
                runtime_events.clear()
            except Exception:
                pass

        now_is_night = _is_night_hour()
        if profile_is_night is None or now_is_night != profile_is_night:
            detector.set_night_mode(now_is_night)
            tracker.set_night_mode(now_is_night)
            profile_is_night = now_is_night

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

        loop_now = asyncio.get_running_loop().time()
        runtime_interval_sec = 20.0
        runtime_cooldown_sec = 600.0
        controls: dict[str, object] = {}
        if counter is not None:
            controls = {
                "runtime_profile_mode": counter.get_setting("runtime_profile_mode", "auto"),
                "runtime_manual_profile": counter.get_setting("runtime_manual_profile", ""),
                "runtime_manual_until": counter.get_setting("runtime_manual_until"),
                "runtime_auto_enabled": counter.get_setting("runtime_auto_enabled", 1),
                "runtime_profile_cooldown_sec": counter.get_setting("runtime_profile_cooldown_sec", 600),
                "runtime_autotune_interval_sec": counter.get_setting("runtime_autotune_interval_sec", 20),
                "runtime_stream_grab_latest": counter.get_setting("runtime_stream_grab_latest"),
            }
            try:
                runtime_interval_sec = max(5.0, float(controls.get("runtime_autotune_interval_sec", 20) or 20))
            except Exception:
                runtime_interval_sec = 20.0
            try:
                runtime_cooldown_sec = max(15.0, float(controls.get("runtime_profile_cooldown_sec", 600) or 600))
            except Exception:
                runtime_cooldown_sec = 600.0

        force_eval = (runtime_profile_name == "")
        if (loop_now - last_runtime_eval) >= runtime_interval_sec or force_eval:
            now_utc = datetime.now(timezone.utc)
            horizon_sec = 120.0
            while runtime_events and (loop_now - runtime_events[0][0]) > horizon_sec:
                runtime_events.popleft()
            if runtime_events:
                window_sec = max(1.0, loop_now - runtime_events[0][0])
                det_pm = (sum(ev[1] for ev in runtime_events) / window_sec) * 60.0
                cross_pm = (sum(ev[2] for ev in runtime_events) / window_sec) * 60.0
                conf_vals = [ev[3] for ev in runtime_events if ev[3] > 0]
                avg_conf = (sum(conf_vals) / len(conf_vals)) if conf_vals else 0.0
            else:
                det_pm = 0.0
                cross_pm = 0.0
                avg_conf = 0.0

            stats = TrafficStats(
                detections_per_min=det_pm,
                crossings_per_min=cross_pm,
                avg_confidence=avg_conf,
            )

            mode = str(controls.get("runtime_profile_mode", "auto") or "auto").strip().lower()
            try:
                auto_enabled = 1 if int(controls.get("runtime_auto_enabled", 1) or 0) == 1 else 0
            except Exception:
                auto_enabled = 1
            if mode != "manual" and auto_enabled != 1:
                desired_profile = "night_balanced" if now_is_night else "day_balanced"
                reason = "auto_disabled"
            else:
                desired_profile, reason = select_runtime_profile(
                    now_utc=now_utc,
                    stats=stats,
                    controls=controls,
                    night_start_hour=int(getattr(cfg, "NIGHT_PROFILE_START_HOUR", 18) or 18),
                    night_end_hour=int(getattr(cfg, "NIGHT_PROFILE_END_HOUR", 6) or 6),
                )

            can_switch = (
                runtime_profile_name == ""
                or desired_profile == runtime_profile_name
                or reason == "manual_override"
                or (loop_now - last_runtime_switch) >= runtime_cooldown_sec
            )
            if can_switch and desired_profile in RUNTIME_PROFILES and desired_profile != runtime_profile_name:
                profile = RUNTIME_PROFILES[desired_profile]
                det = profile.get("detector", {})
                trk = profile.get("tracker", {})
                lp = profile.get("loop", {})

                detector.conf = _bounded_float(det.get("conf", detector.conf), 0.05, 0.95)
                detector.infer_size = _bounded_int(det.get("infer_size", detector.infer_size), 320, 1280)
                detector.iou = _bounded_float(det.get("iou", detector.iou), 0.05, 0.95)
                detector.max_det = _bounded_int(det.get("max_det", detector.max_det), 10, 600)
                tracker.apply_runtime_profile(trk)
                process_every_n = _bounded_int(lp.get("process_every_n", process_every_n), 1, 4)

                stream_latest = controls.get("runtime_stream_grab_latest")
                if stream_latest is not None:
                    hls_stream._grab_latest = bool(stream_latest)

                runtime_profile_name = desired_profile
                runtime_profile_reason = reason
                last_runtime_switch = loop_now
                logger.info(
                    "Runtime profile applied: %s (%s) conf=%.2f infer=%s iou=%.2f max_det=%s process_every_n=%s",
                    runtime_profile_name,
                    runtime_profile_reason,
                    detector.conf,
                    detector.infer_size,
                    detector.iou,
                    detector.max_det,
                    process_every_n,
                )
            last_runtime_eval = loop_now

        if process_every_n > 1 and (frame_index % process_every_n) != 0:
            await asyncio.sleep(0)
            continue

        detections = await asyncio.to_thread(detector.detect, frame)
        # Keep full-frame detections for tracking/overlay visibility.
        # Count logic still applies detect/count zones inside LineCounter.process().
        tracked = tracker.update(detections)
        snapshot = await counter.process(frame, tracked)
        det_boxes = snapshot.get("detections") or []
        conf_vals = [float(d.get("conf")) for d in det_boxes if d.get("conf") is not None]
        runtime_events.append((
            loop_now,
            len(det_boxes),
            int(snapshot.get("new_crossings", 0) or 0),
            (sum(conf_vals) / len(conf_vals)) if conf_vals else 0.0,
        ))
        if (loop_now - last_scene_eval) >= 2.0:
            vision_scene = await asyncio.to_thread(_infer_scene_status, frame)
            if (loop_now - last_weather_eval) >= 300.0 or weather_status is None:
                try:
                    weather_status = await asyncio.to_thread(_fetch_jamaica_weather_cached)
                except Exception:
                    weather_status = None
                last_weather_eval = loop_now
            scene_status = _merge_scene_and_weather(vision_scene, weather_status)
            last_scene_eval = loop_now
        capture_result = None
        capture_is_paused = is_capture_paused()
        if capture_is_paused:
            if not getattr(_ai_loop_inner, "_capture_pause_logged", False):
                logger.info("Live capture is paused by admin control")
                setattr(_ai_loop_inner, "_capture_pause_logged", True)
        else:
            if getattr(_ai_loop_inner, "_capture_pause_logged", False):
                logger.info("Live capture resumed by admin control")
                setattr(_ai_loop_inner, "_capture_pause_logged", False)
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

        # Keep live_state current so bet_service can read the exact count at bet placement time.
        set_live_snapshot(snapshot)

        # Write snapshots at a fixed interval to reduce DB pressure without slowing live WS updates.
        loop_now = asyncio.get_running_loop().time()
        if (loop_now - last_db_write) >= cfg.DB_SNAPSHOT_INTERVAL_SEC:
            db_snapshot = {k: v for k, v in snapshot.items() if k not in ("detections", "new_crossings", "per_class_total")}
            asyncio.create_task(write_snapshot(db_snapshot))
            asyncio.create_task(write_ml_detection_event(camera_id, snapshot, cfg.YOLO_MODEL, detector.conf))
            last_db_write = loop_now
            _mark_ai_db_write()

        if manager.public_count > 0:
            payload: dict = {"type": "count", **snapshot}
            payload["runtime_profile"] = runtime_profile_name or None
            payload["runtime_profile_reason"] = runtime_profile_reason or None
            payload["scene_lighting"] = scene_status.get("scene_lighting")
            payload["scene_weather"] = scene_status.get("scene_weather")
            payload["scene_confidence"] = scene_status.get("scene_confidence")
            payload["scene_source"] = scene_status.get("scene_source") or "vision"
            if _active_round:
                payload["round"] = _active_round
            _ws_fps = max(1.0, float(_ai_runtime_state.get("fps_estimate", 15.0) or 15.0))
            payload["detections"] = box_smoother.smooth_detections(
                list(payload.get("detections") or []), fps=_ws_fps
            )
            await manager.broadcast_public(payload)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _refresh_task, _ai_task, _round_task, _resolver_task, _ml_retrain_task, _watchdog_task

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
    stream_url = await _wait_for_stream_url(_WATCHDOG_STARTUP_WAIT_SEC)
    if not stream_url:
        logger.error("No stream URL after 30s — AI loop will not start.")
    else:
        await _ensure_ai_task(cfg, timeout_sec=1, reason="startup")

    _round_task = asyncio.create_task(round_monitor_loop(), name="round_monitor")
    logger.info("Round monitor started")

    _resolver_task = asyncio.create_task(bet_resolver_loop(), name="bet_resolver")
    logger.info("Bet resolver started")

    if cfg.ML_AUTO_RETRAIN_ENABLED == 1:
        _ml_retrain_task = asyncio.create_task(ml_auto_retrain_loop(cfg), name="ml_auto_retrain")
        logger.info("ML auto-retrain loop started")
    else:
        logger.info("ML auto-retrain loop disabled")

    _watchdog_task = asyncio.create_task(health_watchdog_loop(cfg), name="health_watchdog")
    logger.info("Health watchdog started")

    yield

    # Shutdown
    for task in (_watchdog_task, _ai_task, _refresh_task, _round_task, _resolver_task, _ml_retrain_task):
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
    latest_snapshot: dict | None = None
    latest_ml_detection: dict | None = None
    weather_api: dict[str, Any] | None = None
    try:
        sb = await get_supabase()
        now_iso = datetime.now(timezone.utc).isoformat()

        # Prefer DB truth for health checks in case in-memory round cache is stale.
        if not active_round_id:
            open_resp = await (
                sb.table("bet_rounds")
                .select("id, status, opens_at")
                .eq("status", "open")
                .order("opens_at", desc=True)
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
                    .order("closes_at", desc=True)
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
            .order("opens_at", desc=False)
            .limit(1)
            .maybeSingle()
            .execute()
        )
        next_round_at = (up_resp.data or {}).get("opens_at")
        if not next_round_at:
            next_round_at = await next_session_round_at()

        cam_resp = await (
            sb.table("cameras")
            .select("id")
            .eq("ipcam_alias", get_current_alias() or get_config().CAMERA_ALIAS)
            .limit(1)
            .execute()
        )
        if cam_resp.data:
            camera_id = cam_resp.data[0]["id"]
        else:
            fallback_resp = await (
                sb.table("cameras")
                .select("id")
                .eq("is_active", True)
                .order("created_at", desc=False)
                .limit(1)
                .execute()
            )
            camera_id = fallback_resp.data[0]["id"] if fallback_resp.data else None
        if camera_id:
            snap_resp = await (
                sb.table("count_snapshots")
                .select("camera_id,captured_at,count_in,count_out,total,vehicle_breakdown")
                .eq("camera_id", camera_id)
                .order("captured_at", desc=True)
                .limit(1)
                .execute()
            )
            snap_rows = snap_resp.data or []
            if snap_rows:
                latest_snapshot = snap_rows[0]

            ml_resp = await (
                sb.table("ml_detection_events")
                .select("captured_at,avg_confidence,model_name,detections_count,new_crossings")
                .eq("camera_id", camera_id)
                .order("captured_at", desc=True)
                .limit(1)
                .execute()
            )
            ml_rows = ml_resp.data or []
            if ml_rows:
                latest_ml_detection = ml_rows[0]
    except Exception:
        next_round_at = None

    now_ts = time.time()
    weather_ts = float(_weather_cache.get("ts", 0.0) or 0.0)
    weather_age = (now_ts - weather_ts) if weather_ts > 0 else None
    weather_last_ok = _weather_cache.get("last_ok")
    weather_payload = _weather_cache.get("payload")
    weather_status = "not_checked"
    if weather_last_ok is True:
        weather_status = "ok" if (weather_age is not None and weather_age <= 900) else "stale"
    elif weather_last_ok is False:
        weather_status = "error"
    weather_api = {
        "provider": "open-meteo",
        "status": weather_status,
        "last_ok": weather_last_ok,
        "cache_age_sec": int(weather_age) if weather_age is not None else None,
        "last_error": _weather_cache.get("last_error"),
        "latest": weather_payload,
    }

    now_mono = asyncio.get_running_loop().time()
    ai_last_frame_mono = float(_ai_runtime_state.get("last_frame_monotonic", 0.0) or 0.0)
    ai_last_db_mono = float(_ai_runtime_state.get("last_db_write_monotonic", 0.0) or 0.0)
    ai_last_frame_age = (now_mono - ai_last_frame_mono) if ai_last_frame_mono > 0 else None
    ai_last_db_age = (now_mono - ai_last_db_mono) if ai_last_db_mono > 0 else None
    ai_heartbeat_stale = bool(
        ai_last_frame_age is not None and ai_last_frame_age > _AI_HEARTBEAT_STALE_SEC
    )

    return {
        "status": "ok",
        "stream_configured": bool(get_current_url()),
        "public_ws_connections": manager.public_count,
        "user_ws_connections": manager.user_count,
        "user_ws_sockets": manager.user_socket_count,
        "total_ws_connections": manager.public_count + manager.user_socket_count,
        "public_ws_total_visits": manager.public_connection_events,
        "user_ws_total_visits": manager.user_connection_events,
        "ai_task_running": _ai_task is not None and not _ai_task.done(),
        "refresh_task_running": _refresh_task is not None and not _refresh_task.done(),
        "round_task_running": _round_task is not None and not _round_task.done(),
        "resolver_task_running": _resolver_task is not None and not _resolver_task.done(),
        "ml_retrain_task_running": _ml_retrain_task is not None and not _ml_retrain_task.done(),
        "watchdog_task_running": _watchdog_task is not None and not _watchdog_task.done(),
        "watchdog_restart_counts": _watchdog_restart_counts,
        "watchdog_last_restart_at": _watchdog_last_restart_iso,
        "ai_last_frame_at": _ai_runtime_state.get("last_frame_at"),
        "ai_last_frame_age_sec": round(ai_last_frame_age, 2) if ai_last_frame_age is not None else None,
        "ai_last_db_write_at": _ai_runtime_state.get("last_db_write_at"),
        "ai_last_db_write_age_sec": round(ai_last_db_age, 2) if ai_last_db_age is not None else None,
        "ai_frames_total": int(_ai_runtime_state.get("frames_total", 0) or 0),
        "ai_fps_estimate": float(_ai_runtime_state.get("fps_estimate", 0.0) or 0.0),
        "ai_heartbeat_stale": ai_heartbeat_stale,
        "ai_last_error": _ai_runtime_state.get("last_error"),
        "ai_inference": _ai_inference_runtime,
        "active_round_id": active_round_id,
        "active_round_status": active_round_status,
        "next_round_at": next_round_at,
        "latest_snapshot": latest_snapshot,
        "latest_ml_detection": latest_ml_detection,
        "weather_api": weather_api,
    }


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logger.error("Unhandled error on %s: %s", request.url, exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
