"""
ai/turning_tracker.py — Tracks vehicle turning movements between entry/exit zones.

Loads entry and exit zone definitions from camera_zones table.
For each frame's tracked detections, detects when a vehicle:
  1. Enters an entry zone  → records (entry_zone, entry_time, class, conf)
  2. Later enters an exit zone → writes a turning_movements row

Hit test uses center-of-zone + adaptive radius, which works for
the thin triangular zones drawn by the admin zone editor.

Results from process() are lists of dicts ready for batch insert
into turning_movements via write_turning_movements().
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import supervision as sv

from ai.detector import CLASS_NAMES
from supabase_client import get_supabase

logger = logging.getLogger(__name__)

ZONE_REFRESH_SEC = 60       # reload camera_zones every 60 s
TRANSIT_TTL_SEC  = 45       # discard unresolved entry after 45 s
MIN_ZONE_RADIUS  = 45       # minimum hit-circle radius (pixels)
ZONE_RADIUS_PAD  = 1.8      # expand zone radius by this factor


def _zone_hit_circle(
    points: list[dict], frame_w: int, frame_h: int
) -> tuple[float, float, float]:
    """
    Convert relative zone points to a (cx, cy, radius) hit-circle in pixels.
    Works reliably for thin/degenerate triangular zones drawn as line segments.
    radius = max(MIN_ZONE_RADIUS, ZONE_RADIUS_PAD * max_dist_from_centroid).
    """
    px = [p["x"] * frame_w for p in points]
    py = [p["y"] * frame_h for p in points]
    cx = sum(px) / len(px)
    cy = sum(py) / len(py)
    max_d = max(((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 for x, y in zip(px, py))
    radius = max(float(MIN_ZONE_RADIUS), max_d * ZONE_RADIUS_PAD)
    return cx, cy, radius


class TurningMovementTracker:
    """
    Per-camera turning movement tracker. Loaded once per AI loop session.

    Usage (in main.py):
        tt = TurningMovementTracker(camera_id, frame_w, frame_h)
        ...
        movements = await tt.process(detections, tracker_ids, class_ids, confidences)
        if movements:
            asyncio.create_task(write_turning_movements(movements))
    """

    def __init__(self, camera_id: str, frame_width: int, frame_height: int) -> None:
        self.camera_id    = camera_id
        self.frame_width  = frame_width
        self.frame_height = frame_height

        # (name, cx, cy, radius)
        self._entry_zones: list[tuple[str, float, float, float]] = []
        self._exit_zones:  list[tuple[str, float, float, float]] = []

        # tid → (entry_zone_name, entry_mono, vehicle_class, confidence)
        self._in_entry: dict[int, tuple[str, float, str, float | None]] = {}

        self._last_refresh = 0.0

    # ── zone loading ──────────────────────────────────────────────────────────

    async def _refresh(self) -> None:
        try:
            sb = await get_supabase()
            resp = await (
                sb.table("camera_zones")
                .select("name,zone_type,points")
                .eq("camera_id", self.camera_id)
                .eq("active", True)
                .execute()
            )
        except Exception as exc:
            logger.warning("TurningTracker: zone refresh failed: %s", exc)
            return

        rows = resp.data or []
        entry, exit_ = [], []
        for z in rows:
            pts = z.get("points") or []
            if len(pts) < 2:
                continue
            try:
                hit = _zone_hit_circle(pts, self.frame_width, self.frame_height)
            except Exception:
                continue
            rec = (z["name"], *hit)
            if z["zone_type"] == "entry":
                entry.append(rec)
            elif z["zone_type"] in {"exit"}:
                exit_.append(rec)

        self._entry_zones = entry
        self._exit_zones  = exit_
        self._last_refresh = time.monotonic()
        logger.info(
            "TurningTracker refreshed: %d entry zones, %d exit zones, camera=%s",
            len(entry), len(exit_), self.camera_id,
        )

    # ── frame processing ──────────────────────────────────────────────────────

    async def process(
        self,
        detections: sv.Detections,
        tracker_ids: list[int],
        class_ids: list[int],
        confidences: list[float],
    ) -> list[dict[str, Any]]:
        """
        Check detections against entry/exit zones.
        Returns list of completed turning_movements rows (may be empty).
        """
        now = time.monotonic()

        # Refresh zones periodically
        if not self._entry_zones or (now - self._last_refresh) > ZONE_REFRESH_SEC:
            await self._refresh()

        if not self._entry_zones or not self._exit_zones:
            return []

        if len(detections) == 0 or detections.xyxy is None:
            return []

        # Expire stale in-entry records
        self._in_entry = {
            tid: v for tid, v in self._in_entry.items()
            if now - v[1] < TRANSIT_TTL_SEC
        }

        completed: list[dict[str, Any]] = []

        for i, tid in enumerate(tracker_ids):
            if i >= len(detections.xyxy):
                continue
            x1, y1, x2, y2 = detections.xyxy[i]
            cx = float((x1 + x2) / 2)
            cy = float((y1 + y2) / 2)

            cls_id = int(class_ids[i]) if i < len(class_ids) else -1
            cls    = CLASS_NAMES.get(cls_id, "car")
            conf   = round(float(confidences[i]), 4) if i < len(confidences) else None

            # ── exit check first ─────────────────────────────────────────────
            if tid in self._in_entry:
                entry_name, entry_ts, entry_cls, entry_conf = self._in_entry[tid]
                for (ez_name, ez_cx, ez_cy, ez_r) in self._exit_zones:
                    if (cx - ez_cx) ** 2 + (cy - ez_cy) ** 2 <= ez_r ** 2:
                        dwell_ms = max(0, int((now - entry_ts) * 1000))
                        completed.append({
                            "camera_id":    self.camera_id,
                            "captured_at":  datetime.now(timezone.utc).isoformat(),
                            "track_id":     int(tid),
                            "vehicle_class": entry_cls,
                            "entry_zone":   entry_name,
                            "exit_zone":    ez_name,
                            "dwell_ms":     dwell_ms,
                            "confidence":   entry_conf,
                        })
                        del self._in_entry[tid]
                        break
                continue   # already in transit — skip entry check

            # ── entry check ──────────────────────────────────────────────────
            for (ez_name, ez_cx, ez_cy, ez_r) in self._entry_zones:
                if (cx - ez_cx) ** 2 + (cy - ez_cy) ** 2 <= ez_r ** 2:
                    self._in_entry[tid] = (ez_name, now, cls, conf)
                    break

        return completed


# ── DB writer ─────────────────────────────────────────────────────────────────

async def write_turning_movements(movements: list[dict]) -> None:
    """Batch-insert completed turning movements into Supabase."""
    if not movements:
        return
    try:
        sb = await get_supabase()
        await sb.table("turning_movements").insert(movements).execute()
        logger.debug("TurningTracker: wrote %d movement(s)", len(movements))
    except Exception as exc:
        logger.warning("write_turning_movements failed (%d rows): %s", len(movements), exc)
