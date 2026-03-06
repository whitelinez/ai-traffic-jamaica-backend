"""
ai/counter.py - LineZone crossing counter + Supabase snapshot writer.
Polls the cameras table for admin-defined count and detect zones every 30s.
"""
import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import supervision as sv

from ai.detector import CLASS_NAMES
from config import get_config
from supabase_client import get_supabase

logger = logging.getLogger(__name__)

LINE_REFRESH_INTERVAL = 30  # seconds
TRACK_TTL_SEC = 12.0
DEFAULT_COUNT_SETTINGS = {
    "min_track_frames": 2,  # reduced from 3: low-FPS stream means fewer frames per crossing
    "min_box_area_ratio": 0.0015,
    "min_confidence": 0.22,
    "allowed_classes": ["car", "truck", "bus", "motorcycle"],
    "class_min_confidence": {
        "car": 0.20,
        "truck": 0.28,
        "bus": 0.30,
        "motorcycle": 0.22,
    },
    # If detector cannot classify a moving vehicle, count it as car.
    "count_unknown_as_car": True,
    # Track-level temporal smoothing + hysteresis for stable counting.
    "track_conf_smoothing_alpha": 0.30,
    "track_conf_enter": 0.30,
    "track_conf_exit": 0.18,
    # Polygon count confirmation: require N consecutive inside frames.
    "count_zone_confirm_frames": 1,
    # Congestion assist: in dense traffic, ease gating so short-lived tracks
    # are still countable instead of being dropped before confirmation.
    "burst_mode_enabled": True,
    "burst_density_min_detections": 18,
    "burst_min_track_frames": 1,
    "burst_zone_confirm_frames": 1,
}


class LineCounter:
    """
    Manages count-zone and detect-zone logic.
    - Detect zone: pre-validation (candidate)
    - Count zone: confirmation (count increment)
    """

    def __init__(self, camera_id: str, frame_width: int, frame_height: int):
        self.camera_id = camera_id
        self.frame_width = frame_width
        self.frame_height = frame_height

        self._zone = None
        self._zone_type = "line"  # "line" | "polygon"
        self._detect_zone = None
        self._detect_zone_data = None
        self._last_refresh = 0.0
        self._count_settings = dict(DEFAULT_COUNT_SETTINGS)
        self._raw_count_settings: dict[str, Any] = {}

        self._counts: dict[str, dict[str, int]] = {}
        self._per_class_total: dict[str, int] = {}

        self._prevalidated_track_ids: set[int] = set()
        self._confirmed_track_ids: set[int] = set()
        self._track_last_seen: dict[int, float] = {}
        self._track_seen_frames: dict[int, int] = {}
        self._track_in_count_zone: set[int] = set()
        self._track_in_zone_frames: dict[int, int] = {}
        self._track_conf_ema: dict[int, float] = {}
        self._track_conf_stable: dict[int, bool] = {}
        self._tracker_history: dict[int, dict] = {}  # track_id → {cls, last_conf}
        self._scene_status: dict[str, Any] = {}  # set externally by ai_loop after scene eval
        self._exclusion_zones: list["sv.PolygonZone"] = []  # polygons from scene_map exclusion/parking/sidewalk
        self._road_zones: list["sv.PolygonZone"] = []       # polygons from scene_map road features (positive filter)
        self._analytics = AnalyticsZoneProcessor(camera_id, frame_width, frame_height)

        self._confirmed_in = 0
        self._confirmed_out = 0
        self._confirmed_total = 0
        self._round_baseline_in = 0
        self._round_baseline_out = 0
        self._round_baseline_total = 0
        self._round_baseline_per_class: dict[str, int] = {}

    @staticmethod
    def _normalize_polygon(points: list[tuple[int, int]]) -> np.ndarray:
        """
        Return a stable, clockwise polygon for zone checks.
        Admin clicks can arrive in arbitrary order; sorting around centroid avoids
        self-intersections that break PolygonZone.trigger().
        """
        if len(points) < 3:
            return np.empty((0, 2), dtype=np.int32)
        uniq = list(dict.fromkeys((int(x), int(y)) for x, y in points))
        if len(uniq) < 3:
            return np.empty((0, 2), dtype=np.int32)
        cx = sum(p[0] for p in uniq) / len(uniq)
        cy = sum(p[1] for p in uniq) / len(uniq)
        ordered = sorted(uniq, key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
        return np.array(ordered, dtype=np.int32)

    @staticmethod
    def _safe_tracker_id(raw_tid: Any) -> int | None:
        """Coerce tracker id and drop invalid/untracked values."""
        if raw_tid is None:
            return None
        try:
            tid = int(raw_tid)
        except Exception:
            return None
        if tid < 0:
            return None
        return tid

    @staticmethod
    def _to_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return default

    def _count_class_name(self, cls_id: int, count_unknown_as_car: bool) -> str:
        cls_name = CLASS_NAMES.get(int(cls_id), "unknown").lower()
        if cls_name == "unknown" and count_unknown_as_car:
            return "car"
        return cls_name

    async def bootstrap_from_latest_snapshot(self) -> None:
        """
        Restore in-memory totals from the latest DB snapshot so redeploys/restarts
        do not reset the visible running count.
        """
        try:
            sb = await get_supabase()
            resp = await (
                sb.table("count_snapshots")
                .select("total, count_in, count_out, vehicle_breakdown")
                .eq("camera_id", self.camera_id)
                .gt("total", 0)
                .order("captured_at", desc=True)
                .limit(1)
                .execute()
            )
            rows = resp.data or []
            if not rows:
                # Fallback to latest persisted snapshot across cameras.
                # This keeps visible totals stable across deploys if alias/camera_id
                # changed before camera metadata catches up.
                fallback = await (
                    sb.table("count_snapshots")
                    .select("total, count_in, count_out, vehicle_breakdown")
                    .gt("total", 0)
                    .order("captured_at", desc=True)
                    .limit(1)
                    .execute()
                )
                rows = fallback.data or []
            if not rows:
                return

            latest = rows[0] or {}
            breakdown = latest.get("vehicle_breakdown") or {}
            if not isinstance(breakdown, dict):
                breakdown = {}

            rebuilt_per_class: dict[str, int] = {}
            for cls_name, raw_val in breakdown.items():
                try:
                    rebuilt_per_class[str(cls_name)] = int(raw_val or 0)
                except Exception:
                    rebuilt_per_class[str(cls_name)] = 0

            self._per_class_total = rebuilt_per_class
            self._confirmed_total = int(
                latest.get("total", 0) or sum(rebuilt_per_class.values())
            )
            self._confirmed_in = int(latest.get("count_in", 0) or 0)
            self._confirmed_out = int(latest.get("count_out", 0) or 0)
            self._round_baseline_in = self._confirmed_in
            self._round_baseline_out = self._confirmed_out
            self._round_baseline_total = self._confirmed_total
            self._round_baseline_per_class = dict(self._per_class_total)

            # Preserve per-class totals while rebuilding display buckets.
            self._counts = {
                cls_name: {"in": total, "out": 0}
                for cls_name, total in rebuilt_per_class.items()
            }

            logger.info(
                "Counter bootstrapped from snapshot: total=%s classes=%s",
                self._confirmed_total,
                len(rebuilt_per_class),
            )
        except Exception as exc:
            logger.warning("Counter bootstrap failed: %s", exc)

    async def _refresh_line(self) -> None:
        """Fetch count_line, detect_zone, and count_settings from Supabase cameras table."""
        sb = await get_supabase()
        resp = await (
            sb.table("cameras")
            .select("count_line, detect_zone, count_settings, scene_map")
            .eq("id", self.camera_id)
            .maybe_single()
            .execute()
        )
        if resp.data is None:
            logger.warning(
                "_refresh_line: camera_id=%s not found in DB — skipping refresh",
                self.camera_id,
            )
            return
        data = resp.data
        line_data = data.get("count_line")
        detect_data = data.get("detect_zone")
        count_settings = data.get("count_settings") or {}
        if not isinstance(count_settings, dict):
            count_settings = {}
        self._raw_count_settings = dict(count_settings)

        merged = dict(DEFAULT_COUNT_SETTINGS)
        merged.update(count_settings)
        merged["min_track_frames"] = min(
            8,       # raised from 3 — admin can now honour values up to 8 frames
            max(1, int(merged.get("min_track_frames", DEFAULT_COUNT_SETTINGS["min_track_frames"]) or DEFAULT_COUNT_SETTINGS["min_track_frames"]))
        )
        merged["min_box_area_ratio"] = min(
            0.01,    # raised from 0.0015 — allows filtering out small noise at admin's discretion
            max(0.0, min(1.0, float(merged.get("min_box_area_ratio", 0.0) or 0.0)))
        )
        merged["min_confidence"] = min(
            0.45,    # raised from 0.22 — admin can now set a stricter global floor
            max(0.0, min(1.0, float(merged.get("min_confidence", 0.0) or 0.0)))
        )
        allowed_classes = merged.get("allowed_classes", [])
        merged["allowed_classes"] = (
            [str(x).strip().lower() for x in allowed_classes if str(x).strip()]
            if isinstance(allowed_classes, list)
            else []
        )
        raw_class_conf = merged.get("class_min_confidence", {})
        class_conf: dict[str, float] = {}
        if isinstance(raw_class_conf, dict):
            for k, v in raw_class_conf.items():
                try:
                    class_conf[str(k).strip().lower()] = max(0.0, min(1.0, float(v)))
                except Exception:
                    continue
        class_conf_caps = {
            "car": 0.50,        # raised from 0.20
            "truck": 0.50,      # raised from 0.28
            "bus": 0.50,        # raised from 0.30
            "motorcycle": 0.50, # raised from 0.22
        }
        for cls_name, cap in class_conf_caps.items():
            if cls_name in class_conf:
                class_conf[cls_name] = min(class_conf[cls_name], cap)
            else:
                class_conf[cls_name] = cap
        merged["class_min_confidence"] = class_conf
        merged["track_conf_smoothing_alpha"] = max(
            0.05, min(1.0, float(merged.get("track_conf_smoothing_alpha", 0.35) or 0.35))
        )
        merged["track_conf_enter"] = max(
            0.0, min(1.0, float(merged.get("track_conf_enter", 0.42) or 0.42))
        )
        merged["track_conf_exit"] = max(
            0.0, min(1.0, float(merged.get("track_conf_exit", 0.30) or 0.30))
        )
        if merged["track_conf_exit"] > merged["track_conf_enter"]:
            merged["track_conf_exit"] = merged["track_conf_enter"]
        merged["count_zone_confirm_frames"] = max(
            1, int(merged.get("count_zone_confirm_frames", 2) or 2)
        )
        merged["burst_mode_enabled"] = self._to_bool(
            merged.get("burst_mode_enabled", True),
            True,
        )
        merged["burst_density_min_detections"] = max(
            6, min(120, int(merged.get("burst_density_min_detections", 18) or 18))
        )
        merged["burst_min_track_frames"] = max(
            1, min(3, int(merged.get("burst_min_track_frames", 1) or 1))
        )
        merged["burst_zone_confirm_frames"] = max(
            1, min(3, int(merged.get("burst_zone_confirm_frames", 1) or 1))
        )
        merged["count_unknown_as_car"] = self._to_bool(
            merged.get("count_unknown_as_car", True),
            True,
        )
        self._count_settings = merged

        w, h = self.frame_width, self.frame_height

        if line_data and "x3" in line_data:
            polygon = self._normalize_polygon([
                (int(line_data["x1"] * w), int(line_data["y1"] * h)),
                (int(line_data["x2"] * w), int(line_data["y2"] * h)),
                (int(line_data["x3"] * w), int(line_data["y3"] * h)),
                (int(line_data["x4"] * w), int(line_data["y4"] * h)),
            ])
            if len(polygon) >= 3:
                self._zone = sv.PolygonZone(polygon=polygon)
                self._zone_type = "polygon"
            else:
                self._zone = None
        elif line_data:
            x1, y1 = int(line_data["x1"] * w), int(line_data["y1"] * h)
            x2, y2 = int(line_data["x2"] * w), int(line_data["y2"] * h)
            self._zone = sv.LineZone(start=sv.Point(x1, y1), end=sv.Point(x2, y2))
            self._zone_type = "line"
        else:
            y = int(get_config().COUNT_LINE_RATIO * h)
            self._zone = sv.LineZone(start=sv.Point(0, y), end=sv.Point(w, y))
            self._zone_type = "line"

        self._detect_zone_data = detect_data
        if detect_data and isinstance(detect_data.get("points"), list):
            pts = detect_data.get("points") or []
            pts = [p for p in pts if isinstance(p, dict) and "x" in p and "y" in p]
            if len(pts) >= 3:
                dz_polygon = self._normalize_polygon(
                    [(int(float(p["x"]) * w), int(float(p["y"]) * h)) for p in pts]
                )
                self._detect_zone = sv.PolygonZone(polygon=dz_polygon) if len(dz_polygon) >= 3 else None
            else:
                self._detect_zone = None
        elif detect_data and "x3" in detect_data:
            dz_polygon = self._normalize_polygon([
                (int(detect_data["x1"] * w), int(detect_data["y1"] * h)),
                (int(detect_data["x2"] * w), int(detect_data["y2"] * h)),
                (int(detect_data["x3"] * w), int(detect_data["y3"] * h)),
                (int(detect_data["x4"] * w), int(detect_data["y4"] * h)),
            ])
            self._detect_zone = sv.PolygonZone(polygon=dz_polygon) if len(dz_polygon) >= 3 else None
        elif detect_data and "x1" in detect_data:
            dz_polygon = np.array([
                [int(detect_data["x1"] * w), int(detect_data["y1"] * h)],
                [int(detect_data["x2"] * w), int(detect_data["y1"] * h)],
                [int(detect_data["x2"] * w), int(detect_data["y2"] * h)],
                [int(detect_data["x1"] * w), int(detect_data["y2"] * h)],
            ], dtype=np.int32)
            self._detect_zone = sv.PolygonZone(polygon=dz_polygon)
        else:
            self._detect_zone = None

        # ── Exclusion zones from scene_map ───────────────────────────────────────
        # Admin-drawn polygons of type "exclusion", "parking", or "sidewalk" are
        # treated as exclusion zones. Detections whose center falls inside any of
        # these are silently dropped before counting — reduces false positives from
        # static objects, parked vehicles, and pedestrian areas.
        _EXCLUSION_TYPES = {"exclusion", "parking", "sidewalk"}
        scene_map = data.get("scene_map") or {}
        features = scene_map.get("features") if isinstance(scene_map, dict) else []
        exclusion_zones: list[sv.PolygonZone] = []
        if isinstance(features, list):
            for feat in features:
                if not isinstance(feat, dict) or feat.get("type") not in _EXCLUSION_TYPES:
                    continue
                pts = feat.get("points") or []
                if not isinstance(pts, list) or len(pts) < 3:
                    continue
                pixel_pts = [
                    (int(float(p["x"]) * w), int(float(p["y"]) * h))
                    for p in pts
                    if isinstance(p, dict) and "x" in p and "y" in p
                ]
                if len(pixel_pts) < 3:
                    continue
                poly_arr = self._normalize_polygon(pixel_pts)
                if len(poly_arr) >= 3:
                    exclusion_zones.append(sv.PolygonZone(polygon=poly_arr))
        self._exclusion_zones = exclusion_zones

        # ── Road zones (positive filter) ─────────────────────────────────────────
        # If any road polygons are defined, detections whose centroid does NOT land
        # on any road surface are suppressed. Falls back to no filter if no roads mapped.
        road_zones: list[sv.PolygonZone] = []
        if isinstance(features, list):
            for feat in features:
                if not isinstance(feat, dict) or feat.get("type") != "road":
                    continue
                pts = feat.get("points") or []
                if not isinstance(pts, list) or len(pts) < 3:
                    continue
                pixel_pts = [
                    (int(float(p["x"]) * w), int(float(p["y"]) * h))
                    for p in pts
                    if isinstance(p, dict) and "x" in p and "y" in p
                ]
                if len(pixel_pts) < 3:
                    continue
                poly_arr = self._normalize_polygon(pixel_pts)
                if len(poly_arr) >= 3:
                    road_zones.append(sv.PolygonZone(polygon=poly_arr))
        self._road_zones = road_zones

        if exclusion_zones or road_zones:
            logger.debug(
                "Counter: loaded %d exclusion zone(s), %d road zone(s) for camera %s",
                len(exclusion_zones), len(road_zones), self.camera_id,
            )

        self._last_refresh = time.monotonic()

    def _touch_track(self, tid: int, now_mono: float) -> None:
        self._track_last_seen[tid] = now_mono

    def _cleanup_stale_tracks(self, now_mono: float) -> None:
        stale = [tid for tid, ts in self._track_last_seen.items() if (now_mono - ts) > TRACK_TTL_SEC]
        if not stale:
            return
        stale_set = set(stale)
        for tid in stale:
            self._track_last_seen.pop(tid, None)
            self._track_seen_frames.pop(tid, None)
            self._track_in_zone_frames.pop(tid, None)
            self._track_conf_ema.pop(tid, None)
            self._track_conf_stable.pop(tid, None)
            self._tracker_history.pop(tid, None)
        self._prevalidated_track_ids.difference_update(stale_set)
        self._track_in_count_zone.difference_update(stale_set)

    def _confirm_crossing(self, cls_name: str, in_flag: bool, out_flag: bool, tid: int | None) -> int:
        if tid is not None and tid in self._confirmed_track_ids:
            return 0

        bucket = self._counts.setdefault(cls_name, {"in": 0, "out": 0})
        if in_flag:
            bucket["in"] += 1
            self._confirmed_in += 1
        if out_flag:
            bucket["out"] += 1
            self._confirmed_out += 1

        self._per_class_total[cls_name] = self._per_class_total.get(cls_name, 0) + 1
        self._confirmed_total += 1

        if tid is not None:
            self._confirmed_track_ids.add(tid)

        return 1

    async def process(self, frame: np.ndarray, detections: sv.Detections) -> dict[str, Any]:
        now_mono = time.monotonic()
        if self._zone is None or (now_mono - self._last_refresh) > LINE_REFRESH_INTERVAL:
            try:
                await self._refresh_line()
            except Exception as exc:
                logger.warning("_refresh_line failed (camera_id=%s): %s", self.camera_id, exc)
                if self._zone is None:
                    # No zone yet and DB unreachable — return empty snapshot so AI loop stays alive.
                    _bd = {cls: v["in"] + v["out"] for cls, v in self._counts.items()}
                    return {
                        "camera_id": self.camera_id,
                        "captured_at": datetime.now(timezone.utc).isoformat(),
                        "count_in": self._confirmed_in,
                        "count_out": self._confirmed_out,
                        "total": self._confirmed_total,
                        "vehicle_breakdown": _bd,
                        "round_count_in": max(0, self._confirmed_in - self._round_baseline_in),
                        "round_count_out": max(0, self._confirmed_out - self._round_baseline_out),
                        "round_total": max(0, self._confirmed_total - self._round_baseline_total),
                        "round_vehicle_breakdown": {
                            cls: max(0, int(v["in"] + v["out"]) - int(self._round_baseline_per_class.get(cls, 0)))
                            for cls, v in self._counts.items()
                        },
                        "detections": [],
                        "new_crossings": 0,
                        "per_class_total": dict(self._per_class_total),
                        "pre_count_total": 0,
                        "confirmed_crossings_total": self._confirmed_total,
                        "burst_mode_active": False,
                    }
                # Zone exists from prior refresh — continue processing with stale config.

        new_crossings = 0
        tracker_ids = getattr(detections, "tracker_id", None)
        has_tracker_ids = tracker_ids is not None and len(tracker_ids) == len(detections)
        detect_inside_mask: list[bool] = [False] * len(detections)

        eligible_mask = [True] * len(detections)
        allowed_classes = set(self._count_settings.get("allowed_classes", []))
        class_min_conf = self._count_settings.get("class_min_confidence", {})
        count_unknown_as_car = self._to_bool(self._count_settings.get("count_unknown_as_car", True), True)
        min_conf = float(self._count_settings.get("min_confidence", 0.0) or 0.0)
        min_area_ratio = float(self._count_settings.get("min_box_area_ratio", 0.0) or 0.0)
        min_track_frames = int(
            self._count_settings.get("min_track_frames", DEFAULT_COUNT_SETTINGS["min_track_frames"])
            or DEFAULT_COUNT_SETTINGS["min_track_frames"]
        )
        conf_alpha = float(self._count_settings.get("track_conf_smoothing_alpha", 0.35) or 0.35)
        conf_enter = float(self._count_settings.get("track_conf_enter", 0.42) or 0.42)
        conf_exit = float(self._count_settings.get("track_conf_exit", 0.30) or 0.30)
        zone_confirm_frames = int(self._count_settings.get("count_zone_confirm_frames", 2) or 2)
        burst_mode_enabled = self._to_bool(self._count_settings.get("burst_mode_enabled", True), True)
        burst_density_min_detections = int(self._count_settings.get("burst_density_min_detections", 18) or 18)
        burst_min_track_frames = int(self._count_settings.get("burst_min_track_frames", 1) or 1)
        burst_zone_confirm_frames = int(self._count_settings.get("burst_zone_confirm_frames", 1) or 1)

        burst_mode_active = burst_mode_enabled and len(detections) >= burst_density_min_detections
        effective_min_track_frames = (
            min(min_track_frames, burst_min_track_frames) if burst_mode_active else min_track_frames
        )
        effective_zone_confirm_frames = (
            min(zone_confirm_frames, burst_zone_confirm_frames) if burst_mode_active else zone_confirm_frames
        )
        frame_area = float(max(1, self.frame_width * self.frame_height))

        for i in range(len(detections)):
            if i >= len(detections.class_id):
                eligible_mask[i] = False
                continue
            cls_name = self._count_class_name(int(detections.class_id[i]), count_unknown_as_car)

            if allowed_classes and cls_name not in allowed_classes:
                eligible_mask[i] = False
                continue

            if detections.confidence is not None and i < len(detections.confidence):
                conf = float(detections.confidence[i])
                if conf < min_conf:
                    eligible_mask[i] = False
                    continue
                cls_floor = class_min_conf.get(cls_name)
                if cls_floor is not None and conf < float(cls_floor):
                    eligible_mask[i] = False
                    continue

            if detections.xyxy is not None and i < len(detections.xyxy):
                x1, y1, x2, y2 = detections.xyxy[i]
                area_ratio = max(0.0, (float(x2) - float(x1)) * (float(y2) - float(y1)) / frame_area)
                if area_ratio < min_area_ratio:
                    eligible_mask[i] = False
                    continue

            # Track-level confidence smoothing + hysteresis:
            # avoids one-frame confidence dips causing flicker/miscounts.
            if has_tracker_ids and detections.confidence is not None and i < len(detections.confidence):
                tid = self._safe_tracker_id(tracker_ids[i])
                if tid is not None:
                    conf_now = float(detections.confidence[i])
                    prev_ema = self._track_conf_ema.get(tid, conf_now)
                    ema = (conf_alpha * conf_now) + ((1.0 - conf_alpha) * prev_ema)
                    self._track_conf_ema[tid] = ema
                    was_stable = self._track_conf_stable.get(tid, False)
                    if was_stable:
                        is_stable = ema >= conf_exit
                    else:
                        is_stable = ema >= conf_enter
                    self._track_conf_stable[tid] = is_stable
                    if not is_stable:
                        eligible_mask[i] = False
                        continue

        # ── Exclusion zone filter ────────────────────────────────────────────────
        # Batch-evaluate all exclusion zones. Any detection whose centroid lands
        # inside an exclusion polygon is suppressed before counting/tracking logic.
        if self._exclusion_zones and len(detections) > 0:
            for excl_zone in self._exclusion_zones:
                try:
                    excl_mask = excl_zone.trigger(detections=detections)
                    for i, is_excluded in enumerate(excl_mask):
                        if is_excluded:
                            eligible_mask[i] = False
                except Exception:
                    pass

        # ── Road zone filter (positive) ──────────────────────────────────────────
        # If road polygons are mapped, only detections on a road surface are counted.
        # A detection is kept if its centroid falls inside ANY of the road polygons.
        if self._road_zones and len(detections) > 0:
            try:
                on_road = [False] * len(detections)
                for road_zone in self._road_zones:
                    road_mask = road_zone.trigger(detections=detections)
                    for i, is_on_road in enumerate(road_mask):
                        if is_on_road:
                            on_road[i] = True
                for i, on in enumerate(on_road):
                    if not on:
                        eligible_mask[i] = False
            except Exception:
                pass  # if road filter errors, fall back to no filter

        if has_tracker_ids:
            seen_this_frame: set[int] = set()
            for raw_tid in tracker_ids:
                tid = self._safe_tracker_id(raw_tid)
                if tid is None:
                    continue
                self._touch_track(tid, now_mono)
                if tid not in seen_this_frame:
                    self._track_seen_frames[tid] = self._track_seen_frames.get(tid, 0) + 1
                    seen_this_frame.add(tid)
            self._cleanup_stale_tracks(now_mono)
            # Update tracker history with latest class + confidence per track
            for i in range(len(detections)):
                tid = self._safe_tracker_id(tracker_ids[i])
                if tid is None:
                    continue
                cls_h = self._count_class_name(int(detections.class_id[i]), count_unknown_as_car) if (
                    detections.class_id is not None and i < len(detections.class_id)
                ) else "car"
                conf_h = float(detections.confidence[i]) if (
                    detections.confidence is not None and i < len(detections.confidence)
                ) else 0.0
                h = self._tracker_history.setdefault(tid, {"cls": cls_h, "last_conf": conf_h})
                h["cls"] = cls_h
                h["last_conf"] = conf_h

        if self._detect_zone is not None and len(detections) > 0:
            detect_mask = self._detect_zone.trigger(detections=detections)
            detect_inside_mask = [bool(x) for x in detect_mask]
            if has_tracker_ids:
                for i, inside in enumerate(detect_mask):
                    if not inside or not eligible_mask[i]:
                        continue
                    tid = self._safe_tracker_id(tracker_ids[i])
                    if tid is None:
                        continue
                    self._prevalidated_track_ids.add(tid)
        if self._detect_zone is None:
            detect_inside_mask = [True] * len(detections)

        _crossing_details: list[dict] = []
        confirmed_before = frozenset(self._confirmed_track_ids)

        if self._zone_type == "polygon":
            inside_mask = self._zone.trigger(detections=detections) if len(detections) > 0 else []
            inside_now: set[int] = set()

            for i, is_inside in enumerate(inside_mask):
                if not is_inside or i >= len(detections.class_id) or not eligible_mask[i]:
                    continue

                cls_name = self._count_class_name(int(detections.class_id[i]), count_unknown_as_car)

                tid = None
                if has_tracker_ids:
                    tid = self._safe_tracker_id(tracker_ids[i])

                if tid is None:
                    continue

                inside_now.add(tid)
                self._track_in_zone_frames[tid] = self._track_in_zone_frames.get(tid, 0) + 1

                if self._detect_zone is not None and tid not in self._prevalidated_track_ids:
                    continue
                if self._track_seen_frames.get(tid, 0) < effective_min_track_frames:
                    continue
                if self._track_in_zone_frames.get(tid, 0) < effective_zone_confirm_frames:
                    continue

                # Use only _confirmed_track_ids to prevent double-counting.
                # Checking _track_in_count_zone here would conflict with
                # zone_confirm_frames > 1: on frame 2+ in-zone, the vehicle IS
                # in _track_in_count_zone so the "not in" check would suppress
                # the count even after the required N frames have elapsed.
                if tid not in self._confirmed_track_ids:
                    added = self._confirm_crossing(cls_name, True, False, tid)
                    new_crossings += added
                    if added:
                        h = self._tracker_history.get(tid, {})
                        _crossing_details.append({
                            "track_id": tid,
                            "vehicle_class": h.get("cls", cls_name),
                            "confidence": round(h.get("last_conf", 0.0), 4),
                            "direction": "in",
                            "dwell_frames": self._track_seen_frames.get(tid, 0),
                        })

            self._track_in_count_zone = inside_now
            # Reset inside streak for tracks no longer inside.
            for tid in list(self._track_in_zone_frames.keys()):
                if tid not in inside_now:
                    self._track_in_zone_frames[tid] = 0
            total_in = self._confirmed_in
            total_out = self._confirmed_out
            total = self._confirmed_total
            breakdown = {cls: v["in"] + v["out"] for cls, v in self._counts.items()}

        else:
            crossed_in, crossed_out = self._zone.trigger(detections=detections)

            for i, (in_flag, out_flag) in enumerate(zip(crossed_in, crossed_out)):
                if i >= len(detections.class_id):
                    continue
                if not eligible_mask[i]:
                    continue
                if not in_flag and not out_flag:
                    continue

                tid = None
                if has_tracker_ids:
                    tid = self._safe_tracker_id(tracker_ids[i])

                if tid is None:
                    continue

                if self._detect_zone is not None and tid not in self._prevalidated_track_ids:
                    continue
                if self._track_seen_frames.get(tid, 0) < effective_min_track_frames:
                    continue

                cls_name = self._count_class_name(int(detections.class_id[i]), count_unknown_as_car)
                added = self._confirm_crossing(cls_name, bool(in_flag), bool(out_flag), tid)
                new_crossings += added
                if added:
                    h = self._tracker_history.get(tid, {})
                    _crossing_details.append({
                        "track_id": tid,
                        "vehicle_class": h.get("cls", cls_name),
                        "confidence": round(h.get("last_conf", 0.0), 4),
                        "direction": "in" if in_flag else "out",
                        "dwell_frames": self._track_seen_frames.get(tid, 0),
                    })

            total_in = self._confirmed_in
            total_out = self._confirmed_out
            total = self._confirmed_total
            breakdown = {cls: v["in"] + v["out"] for cls, v in self._counts.items()}

        valid_cls_ids = set(CLASS_NAMES.keys())
        boxes = []
        if len(detections) > 0 and detections.xyxy is not None:
            for i in range(min(len(detections.xyxy), 60)):
                x1, y1, x2, y2 = detections.xyxy[i]
                cls_id = int(detections.class_id[i]) if (
                    detections.class_id is not None and i < len(detections.class_id)
                ) else -1

                if cls_id not in valid_cls_ids:
                    continue

                conf = None
                if detections.confidence is not None and i < len(detections.confidence):
                    conf = round(float(detections.confidence[i]), 4)
                boxes.append({
                    "x1": round(float(x1) / self.frame_width, 4),
                    "y1": round(float(y1) / self.frame_height, 4),
                    "x2": round(float(x2) / self.frame_width, 4),
                    "y2": round(float(y2) / self.frame_height, 4),
                    "cls": CLASS_NAMES[cls_id],
                    "conf": conf,
                    "in_detect_zone": bool(detect_inside_mask[i]) if i < len(detect_inside_mask) else True,
                })

        snapshot = {
            "camera_id": self.camera_id,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "count_in": total_in,
            "count_out": total_out,
            "total": total,
            "vehicle_breakdown": breakdown,
            "round_count_in": max(0, total_in - self._round_baseline_in),
            "round_count_out": max(0, total_out - self._round_baseline_out),
            "round_total": max(0, total - self._round_baseline_total),
            "round_vehicle_breakdown": {
                cls: max(0, int(val) - int(self._round_baseline_per_class.get(cls, 0)))
                for cls, val in breakdown.items()
            },
            "detections": boxes,
            "new_crossings": new_crossings,
            "per_class_total": dict(self._per_class_total),
            "pre_count_total": max(0, len(self._prevalidated_track_ids - self._confirmed_track_ids)),
            "confirmed_crossings_total": self._confirmed_total,
            "burst_mode_active": burst_mode_active,
        }

        if _crossing_details:
            asyncio.create_task(self._write_vehicle_crossings(_crossing_details, snapshot))

        asyncio.create_task(
            self._analytics.process(detections, self._tracker_history, snapshot, now_mono)
        )

        return snapshot

    def set_scene_status(self, scene: dict[str, Any]) -> None:
        """Called by ai_loop after scene inference to keep scene fields fresh for crossing writes."""
        self._scene_status = scene or {}

    async def _write_vehicle_crossings(self, crossings: list[dict], snapshot: dict[str, Any]) -> None:
        """Bulk-insert confirmed vehicle crossings into vehicle_crossings table."""
        try:
            sb = await get_supabase()
            captured_at = snapshot.get("captured_at") or datetime.now(timezone.utc).isoformat()
            lighting = self._scene_status.get("scene_lighting")
            weather = self._scene_status.get("scene_weather")
            rows = [
                {
                    "camera_id": self.camera_id,
                    "captured_at": captured_at,
                    "track_id": c["track_id"],
                    "vehicle_class": c["vehicle_class"],
                    "confidence": c["confidence"],
                    "direction": c["direction"],
                    "dwell_frames": c["dwell_frames"],
                    "scene_lighting": lighting,
                    "scene_weather": weather,
                    "zone_source": "game",   # game count line — NOT intersection analytics
                    "zone_name": None,
                }
                for c in crossings
            ]
            await sb.table("vehicle_crossings").insert(rows).execute()
        except Exception as exc:
            logger.debug("vehicle_crossings write failed: %s", exc)

    def zone_filter(self, detections: sv.Detections) -> sv.Detections:
        """
        Filter detections before tracking:
        - If detect_zone is set: keep all detections for track continuity
        - Else if count_zone is polygon: keep only detections inside count zone
        - Else (line zone): keep all detections
        """
        if len(detections) == 0:
            return detections

        if self._detect_zone is None and self._zone is not None and self._zone_type == "polygon":
            mask = self._zone.trigger(detections=detections)
            return detections[mask]

        return detections

    def get_class_total(self, vehicle_class: str | None) -> int:
        if vehicle_class is None:
            return sum(self._per_class_total.values())
        return self._per_class_total.get(vehicle_class, 0)

    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Return a raw per-camera count_settings value.
        Useful for runtime controls that are not part of core counter filters.
        """
        return self._raw_count_settings.get(key, default)

    def reset(self) -> None:
        self._counts.clear()
        self._per_class_total.clear()
        self._prevalidated_track_ids.clear()
        self._confirmed_track_ids.clear()
        self._track_last_seen.clear()
        self._track_seen_frames.clear()
        self._track_in_count_zone.clear()
        self._track_in_zone_frames.clear()
        self._track_conf_ema.clear()
        self._track_conf_stable.clear()
        self._tracker_history.clear()
        self._confirmed_in = 0
        self._confirmed_out = 0
        self._confirmed_total = 0
        self._round_baseline_in = 0
        self._round_baseline_out = 0
        self._round_baseline_total = 0
        self._round_baseline_per_class = {}

        if self._zone and self._zone_type == "line":
            self._zone.in_count = 0
            self._zone.out_count = 0

    def reset_round(self) -> None:
        """
        Start a new round baseline without resetting lifetime counters.
        """
        self._round_baseline_in = self._confirmed_in
        self._round_baseline_out = self._confirmed_out
        self._round_baseline_total = self._confirmed_total
        self._round_baseline_per_class = dict(self._per_class_total)


_snapshot_write_count = 0
_SNAPSHOT_PURGE_EVERY = 1000
_SNAPSHOT_KEEP_HOURS = 24


async def write_snapshot(snapshot: dict[str, Any]) -> None:
    """Write a count snapshot to Supabase (non-blocking, fire-and-forget).

    Skips writes where total == 0 to prevent poisoning the bootstrap query
    on the next redeploy (bootstrap reads the latest row by captured_at).
    Auto-purges rows older than 24h every 1000 writes to keep table small.
    """
    global _snapshot_write_count
    if not snapshot.get("total"):
        return
    try:
        sb = await get_supabase()
        await sb.table("count_snapshots").insert(snapshot).execute()
        _snapshot_write_count += 1
        if _snapshot_write_count % _SNAPSHOT_PURGE_EVERY == 0:
            from datetime import datetime, timezone, timedelta
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=_SNAPSHOT_KEEP_HOURS)).isoformat()
            await sb.table("count_snapshots").delete().lt("captured_at", cutoff).execute()
            logger.info("count_snapshots purged rows older than %sh", _SNAPSHOT_KEEP_HOURS)
    except Exception as exc:
        logger.warning("Snapshot write failed: %s", exc)


class AnalyticsZoneProcessor:
    """
    Processes named analytics zones (queue, entry, exit, speed_a, speed_b, roi).
    Zones loaded from camera_zones table, refreshed every ZONE_REFRESH_INTERVAL seconds.

    Per frame it:
    - Counts occupancy of queue/roi zones → periodic traffic_snapshots writes
    - Tracks entry→exit zone transitions per track_id → turning_movements writes
    - Measures speed between speed_a and speed_b zones → updates vehicle_crossings.speed_kmh
    - Writes daily aggregates to traffic_daily every DAILY_WRITE_INTERVAL seconds
    """

    ZONE_REFRESH_INTERVAL = 60    # seconds between zone config reloads
    SNAPSHOT_INTERVAL     = 30    # seconds between traffic_snapshot writes
    DAILY_WRITE_INTERVAL  = 300   # seconds between traffic_daily upserts
    TRACK_TTL_SEC         = 15.0  # evict stale per-track state after this many seconds

    def __init__(self, camera_id: str, frame_width: int, frame_height: int) -> None:
        self.camera_id    = camera_id
        self.frame_width  = frame_width
        self.frame_height = frame_height

        self._zone_polys: dict[str, "sv.PolygonZone"] = {}  # name → PolygonZone
        self._zone_types: dict[str, str] = {}               # name → zone_type
        self._zone_meta:  dict[str, dict] = {}              # name → metadata

        self._last_refresh       = 0.0
        self._last_snapshot_write = 0.0
        self._last_daily_write   = 0.0

        # Per-track state
        # entry tracking: track_id → {entry_zone, entry_time_ms, cls, conf}
        self._track_entry: dict[int, dict] = {}
        # speed_a crossing: track_id → {time_ms, zone_name}
        self._track_speed_a: dict[int, dict] = {}
        # TTL tracking
        self._track_last_seen: dict[int, float] = {}

        # Queue depth accumulator between snapshot writes
        self._queue_acc: list[int] = []

    # ── Zone loading ──────────────────────────────────────────────────────────

    async def refresh_zones(self) -> None:
        """Reload zone definitions from camera_zones table."""
        try:
            sb = await get_supabase()
            resp = await (
                sb.table("camera_zones")
                .select("id,name,zone_type,points,metadata")
                .eq("camera_id", self.camera_id)
                .eq("active", True)
                .execute()
            )
            zones = resp.data or []
            w, h = self.frame_width, self.frame_height
            new_polys: dict[str, sv.PolygonZone] = {}
            new_types: dict[str, str] = {}
            new_meta:  dict[str, dict] = {}

            for z in zones:
                name      = z.get("name", "")
                zone_type = z.get("zone_type", "roi")
                pts       = z.get("points") or []
                meta      = z.get("metadata") or {}
                if not isinstance(pts, list) or len(pts) < 3:
                    continue
                pixel_pts = []
                for p in pts:
                    if isinstance(p, dict) and "x" in p and "y" in p:
                        pixel_pts.append((int(float(p["x"]) * w), int(float(p["y"]) * h)))
                if len(pixel_pts) < 3:
                    continue
                # Reuse LineCounter's polygon normaliser for consistent winding
                poly_arr = LineCounter._normalize_polygon(pixel_pts)
                if len(poly_arr) >= 3:
                    new_polys[name] = sv.PolygonZone(polygon=poly_arr)
                    new_types[name] = zone_type
                    new_meta[name]  = meta

            self._zone_polys = new_polys
            self._zone_types = new_types
            self._zone_meta  = new_meta
            self._last_refresh = time.monotonic()
            logger.debug(
                "AnalyticsZoneProcessor: loaded %d zones for camera %s",
                len(new_polys), self.camera_id,
            )
        except Exception as exc:
            logger.warning("AnalyticsZoneProcessor.refresh_zones failed: %s", exc)

    # ── Per-frame processing ──────────────────────────────────────────────────

    async def process(
        self,
        detections: "sv.Detections",
        tracker_history: dict,
        snapshot: dict,
        now_mono: float,
    ) -> None:
        """
        Called after LineCounter.process().
        Must not raise — all errors are caught internally.
        """
        try:
            await self._process_inner(detections, tracker_history, snapshot, now_mono)
        except Exception as exc:
            logger.debug("AnalyticsZoneProcessor.process error: %s", exc)

    async def _process_inner(
        self,
        detections: "sv.Detections",
        tracker_history: dict,
        snapshot: dict,
        now_mono: float,
    ) -> None:
        # Refresh zone definitions periodically
        if (now_mono - self._last_refresh) > self.ZONE_REFRESH_INTERVAL:
            await self.refresh_zones()

        if not self._zone_polys:
            self._queue_acc.append(0)
            await self._maybe_write_snapshot(0, 0, snapshot, now_mono)
            return

        tracker_ids = getattr(detections, "tracker_id", None)
        has_trackers = (
            tracker_ids is not None and len(tracker_ids) == len(detections) and len(detections) > 0
        )

        # Evaluate each zone → occupancy count + per-track membership
        zone_occupancy: dict[str, int] = {}
        track_zones: dict[int, set[str]] = {}  # track_id → set of zone names it's currently inside

        for name, poly in self._zone_polys.items():
            if len(detections) == 0:
                zone_occupancy[name] = 0
                continue
            try:
                inside_mask = poly.trigger(detections=detections)
            except Exception:
                continue
            zone_occupancy[name] = int(np.sum(inside_mask))
            if has_trackers:
                for i, is_inside in enumerate(inside_mask):
                    if not is_inside:
                        continue
                    try:
                        tid = int(tracker_ids[i])
                    except Exception:
                        continue
                    if tid < 0:
                        continue
                    track_zones.setdefault(tid, set()).add(name)
                    self._track_last_seen[tid] = now_mono

        # Process turning movements and speed traps
        new_turnings: list[dict] = []
        new_entry_crossings: list[dict] = []   # intersection analytics — entry zone hits
        for tid, zones_in in track_zones.items():
            h   = tracker_history.get(tid, {})
            cls = h.get("cls", "car")
            conf = h.get("last_conf", 0.0)
            for zone_name in zones_in:
                ztype = self._zone_types.get(zone_name, "roi")

                if ztype == "entry" and tid not in self._track_entry:
                    # Vehicle just arrived at an approach leg — record as analytics crossing
                    self._track_entry[tid] = {
                        "entry_zone":    zone_name,
                        "entry_time_ms": int(now_mono * 1000),
                        "cls":           cls,
                        "conf":          conf,
                    }
                    new_entry_crossings.append({
                        "camera_id":     self.camera_id,
                        "captured_at":   snapshot.get("captured_at"),
                        "track_id":      tid,
                        "vehicle_class": cls,
                        "confidence":    round(conf, 4),
                        "direction":     "entry",
                        "dwell_frames":  0,
                        "zone_source":   "entry",   # intersection analytics — NOT game line
                        "zone_name":     zone_name,
                    })

                elif ztype == "exit":
                    entry_info = self._track_entry.get(tid)
                    if entry_info:
                        dwell_ms = int(now_mono * 1000) - entry_info["entry_time_ms"]
                        new_turnings.append({
                            "camera_id":     self.camera_id,
                            "captured_at":   snapshot.get("captured_at"),
                            "track_id":      tid,
                            "vehicle_class": entry_info["cls"],
                            "entry_zone":    entry_info["entry_zone"],
                            "exit_zone":     zone_name,
                            "dwell_ms":      max(0, dwell_ms),
                            "confidence":    round(entry_info["conf"], 4),
                        })
                        del self._track_entry[tid]

                elif ztype == "speed_a" and tid not in self._track_speed_a:
                    self._track_speed_a[tid] = {
                        "time_ms":  int(now_mono * 1000),
                        "zone":     zone_name,
                    }

                elif ztype == "speed_b":
                    sa = self._track_speed_a.get(tid)
                    if sa:
                        delta_ms = int(now_mono * 1000) - sa["time_ms"]
                        if delta_ms > 50:  # ignore sub-50ms (tracker jitter)
                            dist_m = float(
                                self._zone_meta.get(sa["zone"], {}).get("distance_m", 0) or 0
                            )
                            if dist_m > 0:
                                speed_kmh = round((dist_m / (delta_ms / 1000.0)) * 3.6, 1)
                                asyncio.create_task(
                                    self._update_crossing_speed(tid, speed_kmh)
                                )
                        del self._track_speed_a[tid]

        if new_turnings:
            asyncio.create_task(self._write_turnings(new_turnings))
        if new_entry_crossings:
            asyncio.create_task(self._write_entry_crossings(new_entry_crossings))

        # Queue depth = total vehicles in all queue-type zones
        total_queue = sum(
            v for k, v in zone_occupancy.items()
            if self._zone_types.get(k) == "queue"
        )
        roi_visible = sum(
            v for k, v in zone_occupancy.items()
            if self._zone_types.get(k) == "roi"
        )
        self._queue_acc.append(total_queue)

        # Evict stale track state
        stale = [
            tid for tid, ts in self._track_last_seen.items()
            if (now_mono - ts) > self.TRACK_TTL_SEC
        ]
        for tid in stale:
            self._track_last_seen.pop(tid, None)
            self._track_entry.pop(tid, None)
            self._track_speed_a.pop(tid, None)

        await self._maybe_write_snapshot(total_queue, roi_visible, snapshot, now_mono)
        await self._maybe_write_daily(snapshot, now_mono)

    # ── Periodic writers ──────────────────────────────────────────────────────

    async def _maybe_write_snapshot(
        self, queue_depth: int, roi_visible: int, snapshot: dict, now_mono: float
    ) -> None:
        if (now_mono - self._last_snapshot_write) < self.SNAPSHOT_INTERVAL:
            return
        self._last_snapshot_write = now_mono
        avg_queue = (
            int(sum(self._queue_acc) / len(self._queue_acc))
            if self._queue_acc else queue_depth
        )
        self._queue_acc.clear()
        total_visible = len(self._track_last_seen)
        asyncio.create_task(
            self._write_snapshot_row(avg_queue, roi_visible, total_visible, snapshot)
        )

    async def _maybe_write_daily(self, snapshot: dict, now_mono: float) -> None:
        if (now_mono - self._last_daily_write) < self.DAILY_WRITE_INTERVAL:
            return
        self._last_daily_write = now_mono
        asyncio.create_task(self._upsert_daily(snapshot))

    # ── DB writers ────────────────────────────────────────────────────────────

    async def _write_snapshot_row(
        self, queue_depth: int, roi_visible: int, total_visible: int, snapshot: dict
    ) -> None:
        try:
            sb = await get_supabase()
            await sb.table("traffic_snapshots").insert({
                "camera_id":    self.camera_id,
                "captured_at":  snapshot.get("captured_at"),
                "queue_depth":  queue_depth,
                "roi_visible":  roi_visible,
                "total_visible": total_visible,
            }).execute()
        except Exception as exc:
            logger.debug("traffic_snapshots write failed: %s", exc)

    async def _write_turnings(self, rows: list[dict]) -> None:
        try:
            sb = await get_supabase()
            await sb.table("turning_movements").insert(rows).execute()
        except Exception as exc:
            logger.debug("turning_movements write failed: %s", exc)

    async def _write_entry_crossings(self, rows: list[dict]) -> None:
        """Write entry-zone crossings to vehicle_crossings as zone_source='entry'.
        These represent true intersection throughput — one row per unique vehicle
        entering the intersection through any named entry zone.
        """
        try:
            sb = await get_supabase()
            await sb.table("vehicle_crossings").insert(rows).execute()
        except Exception as exc:
            logger.debug("entry_crossings write failed: %s", exc)

    async def _update_crossing_speed(self, track_id: int, speed_kmh: float) -> None:
        try:
            sb = await get_supabase()
            await (
                sb.table("vehicle_crossings")
                .update({"speed_kmh": speed_kmh})
                .eq("track_id", track_id)
                .eq("camera_id", self.camera_id)
                .is_("speed_kmh", "null")   # only update if not already set
                .execute()
            )
        except Exception as exc:
            logger.debug("speed update failed track_id=%s: %s", track_id, exc)

    async def _upsert_daily(self, snapshot: dict) -> None:
        """Aggregate today's data from vehicle_crossings and traffic_snapshots into traffic_daily."""
        try:
            sb = await get_supabase()
            today = datetime.now(timezone.utc).date().isoformat()
            day_start = today + "T00:00:00+00:00"
            day_end   = today + "T23:59:59+00:00"

            # Crossing totals for today
            xresp = await (
                sb.table("vehicle_crossings")
                .select("vehicle_class,direction")
                .eq("camera_id", self.camera_id)
                .gte("captured_at", day_start)
                .lte("captured_at", day_end)
                .execute()
            )
            rows = xresp.data or []
            total = len(rows)
            car_c = truck_c = bus_c = moto_c = in_c = out_c = 0
            for r in rows:
                cls = (r.get("vehicle_class") or "car").lower()
                if cls == "car":        car_c  += 1
                elif cls == "truck":    truck_c += 1
                elif cls == "bus":      bus_c   += 1
                elif cls == "motorcycle": moto_c += 1
                if r.get("direction") == "in":  in_c  += 1
                if r.get("direction") == "out": out_c += 1

            # Queue stats for today
            sresp = await (
                sb.table("traffic_snapshots")
                .select("queue_depth,captured_at")
                .eq("camera_id", self.camera_id)
                .gte("captured_at", day_start)
                .lte("captured_at", day_end)
                .execute()
            )
            snaps = sresp.data or []
            avg_queue = peak_queue = 0
            if snaps:
                depths = [s.get("queue_depth", 0) for s in snaps]
                avg_queue  = round(sum(depths) / len(depths), 2)
                peak_queue = max(depths)

            # Speed average
            vresp = await (
                sb.table("vehicle_crossings")
                .select("speed_kmh")
                .eq("camera_id", self.camera_id)
                .gte("captured_at", day_start)
                .lte("captured_at", day_end)
                .not_.is_("speed_kmh", "null")
                .execute()
            )
            speeds = [r["speed_kmh"] for r in (vresp.data or []) if r.get("speed_kmh")]
            avg_speed = round(sum(speeds) / len(speeds), 2) if speeds else None

            await (
                sb.table("traffic_daily")
                .upsert({
                    "camera_id":        self.camera_id,
                    "date":             today,
                    "total_crossings":  total,
                    "car_count":        car_c,
                    "truck_count":      truck_c,
                    "bus_count":        bus_c,
                    "motorcycle_count": moto_c,
                    "count_in":         in_c,
                    "count_out":        out_c,
                    "avg_queue_depth":  avg_queue,
                    "peak_queue_depth": peak_queue,
                    "avg_speed_kmh":    avg_speed,
                }, on_conflict="camera_id,date")
                .execute()
            )
        except Exception as exc:
            logger.debug("traffic_daily upsert failed: %s", exc)
