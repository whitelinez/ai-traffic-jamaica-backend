"""
ai/counter.py — Vehicle counter. Supports LineZone (2-point) and PolygonZone (4-point).
Polls cameras table every 30 s for zone config. Exclusion zones suppress detections.
No detect-zone prevalidation, no EMA hysteresis, no burst mode.
"""
from __future__ import annotations

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
TRACK_TTL_SEC         = 10.0

# ── defaults (all overridable via cameras.count_settings) ─────────────────────
DEFAULTS: dict[str, Any] = {
    "min_confidence":       0.22,
    "min_box_area_ratio":   0.001,
    "min_track_frames":     2,
    "allowed_classes":      ["car", "truck", "bus", "motorcycle"],
    "class_min_confidence": {"car": 0.22, "truck": 0.25, "bus": 0.25, "motorcycle": 0.22},
    "count_unknown_as_car": True,
}


class LineCounter:
    """
    Counts vehicles crossing a LineZone or dwelling in a PolygonZone.

    Public API used by main.py:
        process(frame, detections) → snapshot dict
        set_scene_status(scene)
        get_setting(key, default)
        bootstrap_from_latest_snapshot()
        reset_round()
        _confirmed_total  (int, read by main.py on startup)
    """

    def __init__(self, camera_id: str, frame_width: int, frame_height: int) -> None:
        self.camera_id    = camera_id
        self.frame_width  = frame_width
        self.frame_height = frame_height

        self._zone:       sv.LineZone | sv.PolygonZone | None = None
        self._zone_type:  str = "line"   # "line" | "polygon"
        self._excl_zones: list[sv.PolygonZone] = []
        self._last_refresh = 0.0

        # counts
        self._confirmed_in:    int = 0
        self._confirmed_out:   int = 0
        self._confirmed_total: int = 0
        self._counts: dict[str, dict[str, int]] = {}   # cls → {in, out}

        # round baseline
        self._round_in:    int = 0
        self._round_out:   int = 0
        self._round_total: int = 0
        self._round_cls:   dict[str, int] = {}

        # track state
        self._confirmed_ids:    set[int] = set()   # already counted
        self._track_frames:     dict[int, int] = {}  # tid → frames seen
        self._track_last_seen:  dict[int, float] = {}

        # polygon zone: inside-last-frame memory
        self._inside_ids: set[int] = set()

        # settings + scene
        self._settings:     dict[str, Any] = dict(DEFAULTS)
        self._scene_status: dict[str, Any] = {}

    # ── zone loading ──────────────────────────────────────────────────────────

    async def _refresh(self) -> None:
        sb = get_supabase()
        resp = (
            await asyncio.to_thread(
                lambda: sb.table("cameras")
                .select("count_line, detect_zone, count_settings, scene_map")
                .eq("id", self.camera_id)
                .maybe_single()
                .execute()
            )
        )
        if resp.data is None:
            logger.warning("Counter: camera %s not found", self.camera_id)
            return

        data     = resp.data
        w, h     = self.frame_width, self.frame_height
        cfg_raw  = data.get("count_settings") or {}
        if not isinstance(cfg_raw, dict):
            cfg_raw = {}

        # merge settings
        merged = dict(DEFAULTS)
        merged.update({k: v for k, v in cfg_raw.items() if v is not None})
        merged["min_track_frames"]   = max(1, min(4, int(merged.get("min_track_frames", 2) or 2)))
        merged["min_confidence"]     = max(0.10, min(0.60, float(merged.get("min_confidence", 0.22) or 0.22)))
        merged["min_box_area_ratio"] = max(0.0, min(0.05, float(merged.get("min_box_area_ratio", 0.001) or 0.001)))
        cls_conf_raw = merged.get("class_min_confidence", {})
        if not isinstance(cls_conf_raw, dict):
            cls_conf_raw = {}
        merged["class_min_confidence"] = {
            k: max(0.10, min(0.60, float(v)))
            for k, v in cls_conf_raw.items()
        }
        allowed = merged.get("allowed_classes", [])
        merged["allowed_classes"] = [str(x).strip().lower() for x in allowed] if isinstance(allowed, list) else []
        self._settings = merged

        # ── count zone ────────────────────────────────────────────────────────
        line = data.get("count_line")
        if line and "x3" in line:
            # 4-point polygon — use admin point order directly
            poly = np.array([
                [int(line["x1"] * w), int(line["y1"] * h)],
                [int(line["x2"] * w), int(line["y2"] * h)],
                [int(line["x3"] * w), int(line["y3"] * h)],
                [int(line["x4"] * w), int(line["y4"] * h)],
            ], dtype=np.int32)
            self._zone      = sv.PolygonZone(polygon=poly)
            self._zone_type = "polygon"
        elif line and "x1" in line and "x2" in line:
            # 2-point line
            x1 = int(line["x1"] * w); y1 = int(line["y1"] * h)
            x2 = int(line["x2"] * w); y2 = int(line["y2"] * h)
            self._zone      = sv.LineZone(start=sv.Point(x1, y1), end=sv.Point(x2, y2))
            self._zone_type = "line"
        else:
            # fallback horizontal line at 55%
            y = int(0.55 * h)
            self._zone      = sv.LineZone(start=sv.Point(0, y), end=sv.Point(w, y))
            self._zone_type = "line"

        # ── exclusion zones from scene_map ────────────────────────────────────
        excl: list[sv.PolygonZone] = []
        scene_map = data.get("scene_map") or {}
        features  = scene_map.get("features") if isinstance(scene_map, dict) else []
        EXCL_TYPES = {"exclusion", "parking", "sidewalk", "crossing"}
        if isinstance(features, list):
            for feat in features:
                if not isinstance(feat, dict):
                    continue
                if feat.get("type") not in EXCL_TYPES:
                    continue
                pts = feat.get("points") or []
                pixel_pts = [
                    (int(float(p["x"]) * w), int(float(p["y"]) * h))
                    for p in pts
                    if isinstance(p, dict) and "x" in p and "y" in p
                ]
                if len(pixel_pts) >= 3:
                    excl.append(sv.PolygonZone(polygon=np.array(pixel_pts, dtype=np.int32)))
        self._excl_zones = excl

        self._last_refresh = time.monotonic()
        logger.debug(
            "Counter refreshed: zone=%s excl=%d camera=%s",
            self._zone_type, len(excl), self.camera_id,
        )

    # ── track bookkeeping ─────────────────────────────────────────────────────

    def _touch(self, tid: int) -> None:
        now = time.monotonic()
        self._track_frames[tid]    = self._track_frames.get(tid, 0) + 1
        self._track_last_seen[tid] = now

    def _cleanup(self) -> None:
        now    = time.monotonic()
        stale  = [t for t, ts in self._track_last_seen.items() if now - ts > TRACK_TTL_SEC]
        for t in stale:
            self._track_frames.pop(t, None)
            self._track_last_seen.pop(t, None)
            self._inside_ids.discard(t)
            # do NOT remove from _confirmed_ids — prevents double-count on re-entry

    def _add_count(self, cls_name: str, direction_in: bool) -> None:
        if direction_in:
            self._confirmed_in    += 1
            self._confirmed_total += 1
        else:
            self._confirmed_out   += 1
            self._confirmed_total += 1
        bucket = self._counts.setdefault(cls_name, {"in": 0, "out": 0})
        bucket["in" if direction_in else "out"] += 1

    # ── eligibility filter ────────────────────────────────────────────────────

    def _eligible_mask(self, detections: sv.Detections) -> list[bool]:
        n           = len(detections)
        mask        = [True] * n
        s           = self._settings
        min_conf    = float(s.get("min_confidence", 0.22))
        min_area    = float(s.get("min_box_area_ratio", 0.001))
        cls_floor   = s.get("class_min_confidence", {})
        allowed     = s.get("allowed_classes", [])
        unk_as_car  = bool(s.get("count_unknown_as_car", True))
        frame_area  = float(max(1, self.frame_width * self.frame_height))

        for i in range(n):
            cls_id   = int(detections.class_id[i]) if detections.class_id is not None else -1
            cls_name = CLASS_NAMES.get(cls_id, "unknown")
            if cls_name == "unknown":
                if unk_as_car:
                    cls_name = "car"
                else:
                    mask[i] = False
                    continue

            if allowed and cls_name not in allowed:
                mask[i] = False
                continue

            if detections.confidence is not None and i < len(detections.confidence):
                conf = float(detections.confidence[i])
                if conf < min_conf:
                    mask[i] = False
                    continue
                floor = cls_floor.get(cls_name)
                if floor is not None and conf < float(floor):
                    mask[i] = False
                    continue

            if detections.xyxy is not None and i < len(detections.xyxy):
                x1, y1, x2, y2 = detections.xyxy[i]
                area = (float(x2) - float(x1)) * (float(y2) - float(y1)) / frame_area
                if area < min_area:
                    mask[i] = False
                    continue

        # exclusion zones
        if self._excl_zones and n > 0:
            for excl in self._excl_zones:
                try:
                    em = excl.trigger(detections=detections)
                    for i, hit in enumerate(em):
                        if hit:
                            mask[i] = False
                except Exception:
                    pass

        return mask

    # ── main process ──────────────────────────────────────────────────────────

    async def process(self, frame: np.ndarray, detections: sv.Detections) -> dict[str, Any]:
        now_mono = time.monotonic()
        if self._zone is None or (now_mono - self._last_refresh) > LINE_REFRESH_INTERVAL:
            try:
                await self._refresh()
            except Exception as exc:
                logger.warning("Counter._refresh failed: %s", exc)

        if self._zone is None:
            return self._empty_snapshot()

        has_ids = (
            detections.tracker_id is not None
            and len(detections.tracker_id) == len(detections)
        )
        tracker_ids: list[int] = []
        if has_ids:
            tracker_ids = [int(t) for t in detections.tracker_id]

        # eligibility
        eligible = self._eligible_mask(detections)

        # touch tracks
        if has_ids:
            for i, tid in enumerate(tracker_ids):
                self._touch(tid)
            self._cleanup()

        # build filtered subset for zone trigger
        eligible_indices = [i for i, ok in enumerate(eligible) if ok]
        new_crossings    = 0
        min_tf           = int(self._settings.get("min_track_frames", 2))

        s           = self._settings
        unk_as_car  = bool(s.get("count_unknown_as_car", True))

        if self._zone_type == "line":
            # LineZone: use sv.LineZone.trigger which tracks crossed_in/crossed_out
            if len(detections) > 0:
                try:
                    crossed_in, crossed_out = self._zone.trigger(detections=detections)
                except Exception:
                    crossed_in = crossed_out = np.array([], dtype=bool)

                for i in eligible_indices:
                    if i >= len(crossed_in):
                        continue
                    tid = tracker_ids[i] if has_ids and i < len(tracker_ids) else None

                    # require min_track_frames before counting
                    if tid is not None and self._track_frames.get(tid, 0) < min_tf:
                        continue
                    if tid is not None and tid in self._confirmed_ids:
                        continue

                    cls_id   = int(detections.class_id[i]) if detections.class_id is not None else -1
                    cls_name = CLASS_NAMES.get(cls_id, "unknown")
                    if cls_name == "unknown" and unk_as_car:
                        cls_name = "car"

                    if crossed_in[i]:
                        self._add_count(cls_name, True)
                        new_crossings += 1
                        if tid is not None:
                            self._confirmed_ids.add(tid)
                    elif crossed_out[i]:
                        self._add_count(cls_name, False)
                        new_crossings += 1
                        if tid is not None:
                            self._confirmed_ids.add(tid)

        else:  # polygon
            if len(detections) > 0:
                try:
                    inside_mask = self._zone.trigger(detections=detections)
                except Exception:
                    inside_mask = np.zeros(len(detections), dtype=bool)

                inside_now: set[int] = set()

                for i in eligible_indices:
                    if i >= len(inside_mask) or not inside_mask[i]:
                        continue
                    tid = tracker_ids[i] if has_ids and i < len(tracker_ids) else None
                    if tid is None:
                        continue
                    if tid in self._confirmed_ids:
                        continue
                    if self._track_frames.get(tid, 0) < min_tf:
                        continue

                    inside_now.add(tid)

                # count tracks that just entered (were outside last frame, inside now)
                newly_entered = inside_now - self._inside_ids
                for tid in newly_entered:
                    # find class for this tid
                    cls_name = "car"
                    if has_ids:
                        for i, t in enumerate(tracker_ids):
                            if t == tid and detections.class_id is not None and i < len(detections.class_id):
                                c = CLASS_NAMES.get(int(detections.class_id[i]), "unknown")
                                cls_name = c if c != "unknown" else ("car" if unk_as_car else "unknown")
                                break
                    if cls_name == "unknown":
                        continue
                    self._add_count(cls_name, True)
                    self._confirmed_ids.add(tid)
                    new_crossings += 1

                self._inside_ids = inside_now

        # build snapshot
        breakdown = {cls: v["in"] + v["out"] for cls, v in self._counts.items()}

        # detection boxes for WS broadcast
        boxes: list[dict] = []
        if len(detections) > 0 and detections.xyxy is not None:
            for i in range(min(len(detections.xyxy), 60)):
                cls_id = int(detections.class_id[i]) if detections.class_id is not None and i < len(detections.class_id) else -1
                if cls_id not in CLASS_NAMES:
                    continue
                conf = round(float(detections.confidence[i]), 4) if detections.confidence is not None and i < len(detections.confidence) else None
                x1, y1, x2, y2 = detections.xyxy[i]
                boxes.append({
                    "x1": round(float(x1) / self.frame_width, 4),
                    "y1": round(float(y1) / self.frame_height, 4),
                    "x2": round(float(x2) / self.frame_width, 4),
                    "y2": round(float(y2) / self.frame_height, 4),
                    "cls": CLASS_NAMES[cls_id],
                    "conf": conf,
                    "in_detect_zone": True,
                })

        return {
            "camera_id":              self.camera_id,
            "captured_at":            datetime.now(timezone.utc).isoformat(),
            "count_in":               self._confirmed_in,
            "count_out":              self._confirmed_out,
            "total":                  self._confirmed_total,
            "vehicle_breakdown":      breakdown,
            "round_count_in":         max(0, self._confirmed_in  - self._round_in),
            "round_count_out":        max(0, self._confirmed_out - self._round_out),
            "round_total":            max(0, self._confirmed_total - self._round_total),
            "round_vehicle_breakdown": {
                cls: max(0, int(v) - int(self._round_cls.get(cls, 0)))
                for cls, v in breakdown.items()
            },
            "detections":             boxes,
            "new_crossings":          new_crossings,
            "per_class_total":        {cls: v["in"] + v["out"] for cls, v in self._counts.items()},
            "pre_count_total":        0,
            "confirmed_crossings_total": self._confirmed_total,
            "burst_mode_active":      False,
        }

    def _empty_snapshot(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "count_in": self._confirmed_in, "count_out": self._confirmed_out,
            "total": self._confirmed_total, "vehicle_breakdown": {},
            "round_count_in": 0, "round_count_out": 0, "round_total": 0,
            "round_vehicle_breakdown": {}, "detections": [], "new_crossings": 0,
            "per_class_total": {}, "pre_count_total": 0,
            "confirmed_crossings_total": self._confirmed_total,
            "burst_mode_active": False,
        }

    # ── helpers used by main.py ───────────────────────────────────────────────

    def set_scene_status(self, scene: dict[str, Any]) -> None:
        self._scene_status = scene or {}

    def get_setting(self, key: str, default: Any = None) -> Any:
        return self._settings.get(key, default)

    def reset_round(self) -> None:
        self._round_in    = self._confirmed_in
        self._round_out   = self._confirmed_out
        self._round_total = self._confirmed_total
        self._round_cls   = {cls: v["in"] + v["out"] for cls, v in self._counts.items()}

    async def bootstrap_from_latest_snapshot(self) -> None:
        """Restore confirmed_total from the latest DB snapshot on startup."""
        try:
            sb = get_supabase()
            resp = await asyncio.to_thread(
                lambda: sb.table("count_snapshots")
                .select("total, vehicle_breakdown")
                .eq("camera_id", self.camera_id)
                .order("captured_at", desc=True)
                .limit(1)
                .execute()
            )
            rows = resp.data or []
            if rows:
                snap = rows[0]
                total = int(snap.get("total") or 0)
                self._confirmed_total = total
                self._confirmed_in    = total
                bd = snap.get("vehicle_breakdown") or {}
                if isinstance(bd, dict):
                    for cls, cnt in bd.items():
                        self._counts[cls] = {"in": int(cnt), "out": 0}
                logger.info("Counter bootstrapped: total=%d camera=%s", total, self.camera_id)
        except Exception as exc:
            logger.warning("Counter bootstrap failed: %s", exc)


# ── Analytics zone processor stub (keeps analytics_service import happy) ──────

class AnalyticsZoneProcessor:
    """Stub — zone analytics handled by analytics_service separately."""
    def __init__(self, *a, **kw): pass
    def process(self, *a, **kw) -> list: return []
