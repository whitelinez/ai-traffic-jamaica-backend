"""
ai/counter.py — LineZone crossing counter + Supabase snapshot writer.
Polls the cameras table for the admin-defined count line every 30s.
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
    Hot-reloads the line from Supabase every LINE_REFRESH_INTERVAL seconds.
    """

    def __init__(self, camera_id: str, frame_width: int, frame_height: int):
        self.camera_id = camera_id
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._zone = None
        self._zone_type = "line"  # "line" | "polygon"
        self._last_refresh = 0.0
        self._counts: dict[str, dict] = {}  # {class_name: {in, out}}

    async def _refresh_line(self) -> None:
        """Fetch count_line from Supabase cameras table and build zone/line."""
        cfg = get_config()
        sb = await get_supabase()
        resp = await sb.table("cameras").select("count_line").eq("id", self.camera_id).single().execute()
        line_data = resp.data.get("count_line") if resp.data else None

        w, h = self.frame_width, self.frame_height

        if line_data and "x3" in line_data:
            # 4-point polygon zone
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
            # Legacy 2-point line
            x1, y1 = int(line_data["x1"] * w), int(line_data["y1"] * h)
            x2, y2 = int(line_data["x2"] * w), int(line_data["y2"] * h)
            self._zone = sv.LineZone(start=sv.Point(x1, y1), end=sv.Point(x2, y2))
            self._zone_type = "line"
            logger.debug("LineZone set: (%d,%d)→(%d,%d)", x1, y1, x2, y2)
        else:
            # Fallback: horizontal line at COUNT_LINE_RATIO
            ratio = cfg.COUNT_LINE_RATIO
            y = int(ratio * h)
            self._zone = sv.LineZone(start=sv.Point(0, y), end=sv.Point(w, y))
            self._zone_type = "line"
            logger.debug("No DB zone — using fallback line at ratio %.2f", ratio)

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

        if self._zone_type == "polygon":
            # PolygonZone: count vehicles currently inside the zone
            inside_mask = self._zone.trigger(detections=detections)
            total_in = int(inside_mask.sum())
            total_out = 0
            total = total_in
            breakdown: dict[str, int] = {}
            for i, is_inside in enumerate(inside_mask):
                if is_inside and i < len(detections.class_id):
                    cls_name = CLASS_NAMES.get(int(detections.class_id[i]), "unknown")
                    breakdown[cls_name] = breakdown.get(cls_name, 0) + 1
        else:
            # LineZone: cumulative crossing count
            crossed_in, crossed_out = self._zone.trigger(detections=detections)
            for i, (in_flag, out_flag) in enumerate(zip(crossed_in, crossed_out)):
                if i >= len(detections.class_id):
                    continue
                cls_name = CLASS_NAMES.get(int(detections.class_id[i]), "unknown")
                bucket = self._counts.setdefault(cls_name, {"in": 0, "out": 0})
                if in_flag:
                    bucket["in"] += 1
                if out_flag:
                    bucket["out"] += 1
            total_in = self._zone.in_count
            total_out = self._zone.out_count
            total = total_in + total_out
            breakdown = {cls: v["in"] + v["out"] for cls, v in self._counts.items()}

        # Build relative-coord bounding boxes for frontend visualization
        boxes = []
        if len(detections) > 0 and detections.xyxy is not None:
            for i in range(min(len(detections.xyxy), 60)):  # cap at 60 boxes
                x1, y1, x2, y2 = detections.xyxy[i]
                cls_id = int(detections.class_id[i]) if (
                    detections.class_id is not None and i < len(detections.class_id)
                ) else -1
                boxes.append({
                    "x1": round(float(x1) / self.frame_width,  4),
                    "y1": round(float(y1) / self.frame_height, 4),
                    "x2": round(float(x2) / self.frame_width,  4),
                    "y2": round(float(y2) / self.frame_height, 4),
                    "cls": CLASS_NAMES.get(cls_id, "vehicle"),
                })

        snapshot = {
            "camera_id": self.camera_id,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "count_in": total_in,
            "count_out": total_out,
            "total": total,
            "vehicle_breakdown": breakdown,
            "detections": boxes,
        }
        return snapshot

    def zone_filter(self, detections: sv.Detections) -> sv.Detections:
        """
        Filter detections to only those inside the polygon zone before tracking.
        For line zones returns all detections unchanged (LineZone handles crossing).
        Returns all detections if zone not yet initialized.
        """
        if self._zone is None or self._zone_type != "polygon" or len(detections) == 0:
            return detections
        mask = self._zone.trigger(detections=detections)
        return detections[mask]

    def reset(self) -> None:
        """Reset counters (called at round start)."""
        self._counts.clear()
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
