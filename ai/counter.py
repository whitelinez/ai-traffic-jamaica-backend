"""
ai/counter.py - LineZone crossing counter + Supabase snapshot writer.
Polls the cameras table for admin-defined count and detect zones every 30s.
"""
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

        self._counts: dict[str, dict[str, int]] = {}
        self._per_class_total: dict[str, int] = {}

        self._prevalidated_track_ids: set[int] = set()
        self._confirmed_track_ids: set[int] = set()
        self._track_last_seen: dict[int, float] = {}
        self._track_in_count_zone: set[int] = set()

        self._confirmed_in = 0
        self._confirmed_out = 0
        self._confirmed_total = 0

    async def _refresh_line(self) -> None:
        """Fetch count_line and detect_zone from Supabase cameras table."""
        sb = await get_supabase()
        resp = await sb.table("cameras").select("count_line, detect_zone").eq("id", self.camera_id).single().execute()
        data = resp.data or {}
        line_data = data.get("count_line")
        detect_data = data.get("detect_zone")

        w, h = self.frame_width, self.frame_height

        if line_data and "x3" in line_data:
            polygon = np.array([
                [int(line_data["x1"] * w), int(line_data["y1"] * h)],
                [int(line_data["x2"] * w), int(line_data["y2"] * h)],
                [int(line_data["x3"] * w), int(line_data["y3"] * h)],
                [int(line_data["x4"] * w), int(line_data["y4"] * h)],
            ], dtype=np.int32)
            self._zone = sv.PolygonZone(polygon=polygon)
            self._zone_type = "polygon"
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
        if detect_data and "x3" in detect_data:
            dz_polygon = np.array([
                [int(detect_data["x1"] * w), int(detect_data["y1"] * h)],
                [int(detect_data["x2"] * w), int(detect_data["y2"] * h)],
                [int(detect_data["x3"] * w), int(detect_data["y3"] * h)],
                [int(detect_data["x4"] * w), int(detect_data["y4"] * h)],
            ], dtype=np.int32)
            self._detect_zone = sv.PolygonZone(polygon=dz_polygon)
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
            await self._refresh_line()

        new_crossings = 0
        tracker_ids = getattr(detections, "tracker_id", None)
        has_tracker_ids = tracker_ids is not None and len(tracker_ids) == len(detections)

        if has_tracker_ids:
            for raw_tid in tracker_ids:
                if raw_tid is None:
                    continue
                self._touch_track(int(raw_tid), now_mono)
            self._cleanup_stale_tracks(now_mono)

        if self._detect_zone is not None and len(detections) > 0 and has_tracker_ids:
            detect_mask = self._detect_zone.trigger(detections=detections)
            for i, inside in enumerate(detect_mask):
                if not inside:
                    continue
                raw_tid = tracker_ids[i]
                if raw_tid is None:
                    continue
                self._prevalidated_track_ids.add(int(raw_tid))

        if self._zone_type == "polygon":
            inside_mask = self._zone.trigger(detections=detections) if len(detections) > 0 else []
            inside_now: set[int] = set()

            for i, is_inside in enumerate(inside_mask):
                if not is_inside or i >= len(detections.class_id):
                    continue

                cls_name = CLASS_NAMES.get(int(detections.class_id[i]), "unknown")

                tid = None
                if has_tracker_ids:
                    raw_tid = tracker_ids[i]
                    if raw_tid is not None:
                        tid = int(raw_tid)

                if tid is None:
                    continue

                inside_now.add(tid)

                if self._detect_zone is not None and tid not in self._prevalidated_track_ids:
                    continue

                if tid not in self._track_in_count_zone and tid not in self._confirmed_track_ids:
                    new_crossings += self._confirm_crossing(cls_name, True, False, tid)

            self._track_in_count_zone = inside_now
            total_in = self._confirmed_in
            total_out = self._confirmed_out
            total = self._confirmed_total
            breakdown = {cls: v["in"] + v["out"] for cls, v in self._counts.items()}

        else:
            crossed_in, crossed_out = self._zone.trigger(detections=detections)

            for i, (in_flag, out_flag) in enumerate(zip(crossed_in, crossed_out)):
                if i >= len(detections.class_id):
                    continue
                if not in_flag and not out_flag:
                    continue

                tid = None
                if has_tracker_ids:
                    raw_tid = tracker_ids[i]
                    if raw_tid is not None:
                        tid = int(raw_tid)

                if tid is None:
                    continue

                if self._detect_zone is not None and tid not in self._prevalidated_track_ids:
                    continue

                cls_name = CLASS_NAMES.get(int(detections.class_id[i]), "unknown")
                new_crossings += self._confirm_crossing(cls_name, bool(in_flag), bool(out_flag), tid)

            total_in = self._confirmed_in
            total_out = self._confirmed_out
            total = self._confirmed_total
            breakdown = {cls: v["in"] + v["out"] for cls, v in self._counts.items()}

        valid_cls_ids = set(CLASS_NAMES.keys())
        boxes = []
        if len(detections) > 0 and detections.xyxy is not None:
            detect_mask = self._detect_zone.trigger(detections=detections) if self._detect_zone is not None else None
            for i in range(min(len(detections.xyxy), 60)):
                x1, y1, x2, y2 = detections.xyxy[i]
                cls_id = int(detections.class_id[i]) if (
                    detections.class_id is not None and i < len(detections.class_id)
                ) else -1

                if cls_id not in valid_cls_ids:
                    continue
                if detect_mask is not None and i < len(detect_mask) and not detect_mask[i]:
                    continue

                boxes.append({
                    "x1": round(float(x1) / self.frame_width, 4),
                    "y1": round(float(y1) / self.frame_height, 4),
                    "x2": round(float(x2) / self.frame_width, 4),
                    "y2": round(float(y2) / self.frame_height, 4),
                    "cls": CLASS_NAMES[cls_id],
                })

        snapshot = {
            "camera_id": self.camera_id,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "count_in": total_in,
            "count_out": total_out,
            "total": total,
            "vehicle_breakdown": breakdown,
            "detections": boxes,
            "new_crossings": new_crossings,
            "per_class_total": dict(self._per_class_total),
            "pre_count_total": max(0, len(self._prevalidated_track_ids - self._confirmed_track_ids)),
            "confirmed_crossings_total": self._confirmed_total,
        }
        return snapshot

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

    def reset(self) -> None:
        self._counts.clear()
        self._per_class_total.clear()
        self._prevalidated_track_ids.clear()
        self._confirmed_track_ids.clear()
        self._track_last_seen.clear()
        self._track_in_count_zone.clear()
        self._confirmed_in = 0
        self._confirmed_out = 0
        self._confirmed_total = 0

        if self._zone and self._zone_type == "line":
            self._zone.in_count = 0
            self._zone.out_count = 0


async def write_snapshot(snapshot: dict[str, Any]) -> None:
    """Write a count snapshot to Supabase (non-blocking, fire-and-forget)."""
    try:
        sb = await get_supabase()
        await sb.table("count_snapshots").insert(snapshot).execute()
    except Exception as exc:
        logger.warning("Snapshot write failed: %s", exc)
