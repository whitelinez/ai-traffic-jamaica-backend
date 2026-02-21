"""
ai/counter.py — LineZone crossing counter + Supabase snapshot writer.
Polls the cameras table for the admin-defined count line every 30s.
Also loads detect_zone (separate bounding-box zone) every 30s.
Writes count_snapshots to Supabase every frame.
"""
import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import supervision as sv

from ai.detector import CLASS_NAMES
from supabase_client import get_supabase
from config import get_config

logger = logging.getLogger(__name__)

LINE_REFRESH_INTERVAL = 30  # seconds


class LineCounter:
    """
    Manages a sv.LineZone that counts vehicles crossing a user-defined line.
    Also manages a separate detect_zone (bounding-box filter for detections).
    Hot-reloads both zones from Supabase every LINE_REFRESH_INTERVAL seconds.
    """

    def __init__(self, camera_id: str, frame_width: int, frame_height: int):
        self.camera_id = camera_id
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._zone = None
        self._zone_type = "line"  # "line" | "polygon"
        self._detect_zone = None   # sv.PolygonZone for bounding-box filtering
        self._detect_zone_data = None  # raw dict from DB
        self._last_refresh = 0.0
        self._counts: dict[str, dict] = {}  # {class_name: {in, out}}
        # Cumulative per-class total for exact-count bet resolution
        self._per_class_total: dict[str, int] = {}
        # Double-validation state:
        # 1) track first seen in detect zone
        # 2) confirm when it crosses count zone
        self._prevalidated_track_ids: set[int] = set()
        self._confirmed_track_ids: set[int] = set()
        self._confirmed_total: int = 0

    async def _refresh_line(self) -> None:
        """Fetch count_line and detect_zone from Supabase cameras table."""
        cfg = get_config()
        sb = await get_supabase()
        resp = await sb.table("cameras").select("count_line, detect_zone").eq("id", self.camera_id).single().execute()
        data = resp.data or {}
        line_data = data.get("count_line")
        detect_data = data.get("detect_zone")

        w, h = self.frame_width, self.frame_height

        # Build count zone
        if line_data and "x3" in line_data:
            polygon = np.array([
                [int(line_data["x1"] * w), int(line_data["y1"] * h)],
                [int(line_data["x2"] * w), int(line_data["y2"] * h)],
                [int(line_data["x3"] * w), int(line_data["y3"] * h)],
                [int(line_data["x4"] * w), int(line_data["y4"] * h)],
            ], dtype=np.int32)
            self._zone = sv.PolygonZone(polygon=polygon)
            self._zone_type = "polygon"
            logger.debug("PolygonZone set with 4 points")
        elif line_data:
            x1, y1 = int(line_data["x1"] * w), int(line_data["y1"] * h)
            x2, y2 = int(line_data["x2"] * w), int(line_data["y2"] * h)
            self._zone = sv.LineZone(start=sv.Point(x1, y1), end=sv.Point(x2, y2))
            self._zone_type = "line"
            logger.debug("LineZone set: (%d,%d)→(%d,%d)", x1, y1, x2, y2)
        else:
            ratio = cfg.COUNT_LINE_RATIO
            y = int(ratio * h)
            self._zone = sv.LineZone(start=sv.Point(0, y), end=sv.Point(w, y))
            self._zone_type = "line"
            logger.debug("No DB zone — using fallback line at ratio %.2f", ratio)

        # Build detect zone (separate from count zone)
        self._detect_zone_data = detect_data
        if detect_data and "x3" in detect_data:
            dz_polygon = np.array([
                [int(detect_data["x1"] * w), int(detect_data["y1"] * h)],
                [int(detect_data["x2"] * w), int(detect_data["y2"] * h)],
                [int(detect_data["x3"] * w), int(detect_data["y3"] * h)],
                [int(detect_data["x4"] * w), int(detect_data["y4"] * h)],
            ], dtype=np.int32)
            self._detect_zone = sv.PolygonZone(polygon=dz_polygon)
            logger.debug("DetectZone polygon set")
        elif detect_data and "x1" in detect_data:
            # Simple 2-point bounding box stored as (x1,y1,x2,y2) rect
            dz_polygon = np.array([
                [int(detect_data["x1"] * w), int(detect_data["y1"] * h)],
                [int(detect_data["x2"] * w), int(detect_data["y1"] * h)],
                [int(detect_data["x2"] * w), int(detect_data["y2"] * h)],
                [int(detect_data["x1"] * w), int(detect_data["y2"] * h)],
            ], dtype=np.int32)
            self._detect_zone = sv.PolygonZone(polygon=dz_polygon)
            logger.debug("DetectZone rect set from 2-point data")
        else:
            self._detect_zone = None
            logger.debug("No detect zone configured")

        self._last_refresh = time.monotonic()

    async def process(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> dict[str, Any]:
        """
        Update LineZone with tracked detections.
        Returns snapshot dict suitable for DB insert + WS broadcast.
        """
        now = time.monotonic()
        if self._zone is None or (now - self._last_refresh) > LINE_REFRESH_INTERVAL:
            await self._refresh_line()

        new_crossings = 0
        tracker_ids = getattr(detections, "tracker_id", None)
        has_tracker_ids = tracker_ids is not None and len(tracker_ids) == len(detections)

        # Pre-validation: if a detect zone exists, mark tracked vehicles that have been
        # observed inside detect zone at least once.
        if self._detect_zone is not None and len(detections) > 0 and has_tracker_ids:
            detect_mask_for_pre = self._detect_zone.trigger(detections=detections)
            for i, inside in enumerate(detect_mask_for_pre):
                if not inside:
                    continue
                tid = tracker_ids[i]
                if tid is None:
                    continue
                self._prevalidated_track_ids.add(int(tid))

        if self._zone_type == "polygon":
            inside_mask = self._zone.trigger(detections=detections)
            total_in = int(inside_mask.sum())
            total_out = 0
            total = total_in
            breakdown: dict[str, int] = {}
            for i, is_inside in enumerate(inside_mask):
                if is_inside and i < len(detections.class_id):
                    cls_name = CLASS_NAMES.get(int(detections.class_id[i]), "unknown")
                    breakdown[cls_name] = breakdown.get(cls_name, 0) + 1
            prev_total = getattr(self, "_prev_total", 0)
            if total > prev_total:
                new_crossings = total - prev_total
            self._prev_total = total
            # Update per-class totals
            for cls, cnt in breakdown.items():
                self._per_class_total[cls] = cnt
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

                # Double validation:
                # - if detect zone exists, vehicle must have been seen there first
                if self._detect_zone is not None and tid is not None and tid not in self._prevalidated_track_ids:
                    continue
                # Count each tracked vehicle crossing only once
                if tid is not None and tid in self._confirmed_track_ids:
                    continue

                cls_name = CLASS_NAMES.get(int(detections.class_id[i]), "unknown")
                bucket = self._counts.setdefault(cls_name, {"in": 0, "out": 0})
                if in_flag:
                    bucket["in"] += 1
                if out_flag:
                    bucket["out"] += 1
                new_crossings += 1
                self._per_class_total[cls_name] = self._per_class_total.get(cls_name, 0) + 1
                self._confirmed_total += 1
                if tid is not None:
                    self._confirmed_track_ids.add(tid)
            total_in = self._zone.in_count
            total_out = self._zone.out_count
            total = total_in + total_out
            breakdown = {cls: v["in"] + v["out"] for cls, v in self._counts.items()}

        # Build bounding boxes — filter to detect_zone if configured, else count_zone
        valid_cls_ids = set(CLASS_NAMES.keys())
        boxes = []
        if len(detections) > 0 and detections.xyxy is not None:
            # Determine which detections to show boxes for
            if self._detect_zone is not None:
                # Show boxes only for vehicles inside detect_zone
                detect_mask = self._detect_zone.trigger(detections=detections)
            else:
                # No separate detect zone — show all valid detections
                detect_mask = None

            for i in range(min(len(detections.xyxy), 60)):
                x1, y1, x2, y2 = detections.xyxy[i]
                cls_id = int(detections.class_id[i]) if (
                    detections.class_id is not None and i < len(detections.class_id)
                ) else -1
                if cls_id not in valid_cls_ids:
                    continue
                if detect_mask is not None and i < len(detect_mask) and not detect_mask[i]:
                    continue  # outside detect zone
                boxes.append({
                    "x1": round(float(x1) / self.frame_width,  4),
                    "y1": round(float(y1) / self.frame_height, 4),
                    "x2": round(float(x2) / self.frame_width,  4),
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
            "pre_count_total": max(0, len(self._prevalidated_track_ids) - len(self._confirmed_track_ids)),
            "confirmed_crossings_total": self._confirmed_total,
        }
        return snapshot

    def zone_filter(self, detections: sv.Detections) -> sv.Detections:
        """
        Filter detections before tracking:
        - If detect_zone is set, filter to detect_zone
        - Else if count_zone is polygon, filter to count_zone
        - Else (line zone) return all detections unchanged
        """
        if len(detections) == 0:
            return detections

        # Do not filter by detect_zone here; we need full track continuity so that
        # detect-zone prevalidation can later be confirmed at count zone crossing.
        # Keep polygon count-zone fallback when no detect zone exists.
        if self._zone is not None and self._zone_type == "polygon":
            mask = self._zone.trigger(detections=detections)
            return detections[mask]

        return detections

    def get_class_total(self, vehicle_class: str | None) -> int:
        """
        Return cumulative total for a vehicle class (or all classes).
        Used by bet resolver to compute actual count delta.
        """
        if vehicle_class is None:
            return sum(self._per_class_total.values())
        return self._per_class_total.get(vehicle_class, 0)

    def reset(self) -> None:
        """Reset counters (called at round start)."""
        self._counts.clear()
        self._per_class_total.clear()
        self._prevalidated_track_ids.clear()
        self._confirmed_track_ids.clear()
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
