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

import cv2
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

        self._zone:        sv.LineZone | sv.PolygonZone | None = None
        self._zone_type:   str = "line"   # "line" | "polygon"
        self._zone_coords: tuple[int, int, int, int] = (0, 0, 0, 0)  # (x1,y1,x2,y2) pixels
        self._excl_polys:  list[np.ndarray] = []  # pixel-coord polygons for CENTER exclusion
        self._detect_poly: np.ndarray | None = None  # inclusion zone — None = whole frame
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
        # pending crossings: vehicles that crossed the line but hadn't reached
        # min_track_frames yet. Keyed by tid, value = direction_in bool.
        # Counted as soon as the track accumulates enough frames.
        self._pending_crossings: dict[int, bool] = {}

        # polygon zone: inside-last-frame memory
        self._inside_ids: set[int] = set()

        # diagnostic counter for throttled logging
        self._process_calls: int = 0

        # settings + scene
        self._settings:     dict[str, Any] = dict(DEFAULTS)
        self._scene_status: dict[str, Any] = {}

    # ── zone loading ──────────────────────────────────────────────────────────

    async def _refresh(self) -> None:
        sb = await get_supabase()
        resp = (
            await sb.table("cameras")
            .select("count_line, detect_zone, count_settings, scene_map")
            .eq("id", self.camera_id)
            .maybe_single()
            .execute()
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
            # 4-point polygon — derive midline LineZone for reliable crossing detection.
            # supervision's PolygonZone uses BOTTOM_CENTER anchor which misses most
            # vehicles on road cameras (bbox bottom falls below the zone band).
            # The midline (midpoint(p1,p4) → midpoint(p2,p3)) is perpendicular to
            # traffic flow and correctly fires once per crossing.
            x1, y1 = int(line["x1"] * w), int(line["y1"] * h)
            x2, y2 = int(line["x2"] * w), int(line["y2"] * h)
            x3, y3 = int(line["x3"] * w), int(line["y3"] * h)
            x4, y4 = int(line["x4"] * w), int(line["y4"] * h)
            mx1, my1 = (x1 + x4) // 2, (y1 + y4) // 2
            mx2, my2 = (x2 + x3) // 2, (y2 + y3) // 2
            self._zone        = sv.LineZone(start=sv.Point(mx1, my1), end=sv.Point(mx2, my2))
            self._zone_type   = "line"
            self._zone_coords = (mx1, my1, mx2, my2)
            logger.info(
                "Counter zone: polygon→midline LineZone (%d,%d)→(%d,%d) camera=%s",
                mx1, my1, mx2, my2, self.camera_id,
            )
        elif line and "x1" in line and "x2" in line:
            # 2-point line
            x1 = int(line["x1"] * w); y1 = int(line["y1"] * h)
            x2 = int(line["x2"] * w); y2 = int(line["y2"] * h)
            self._zone        = sv.LineZone(start=sv.Point(x1, y1), end=sv.Point(x2, y2))
            self._zone_type   = "line"
            self._zone_coords = (x1, y1, x2, y2)
        else:
            # fallback horizontal line at 55%
            y = int(0.55 * h)
            self._zone        = sv.LineZone(start=sv.Point(0, y), end=sv.Point(w, y))
            self._zone_type   = "line"
            self._zone_coords = (0, y, w, y)

        # ── exclusion zones from scene_map ────────────────────────────────────
        # Use CENTER-point-in-polygon check (cv2.pointPolygonTest) instead of
        # supervision's PolygonZone which uses BOTTOM_CENTER anchor.  This prevents
        # large sidewalk/exclusion zones from excluding vehicles that are near-but-not-
        # inside the zone, and correctly ignores zones that overlap the count band.
        # "crossing" is intentionally excluded: pedestrian crossings should not suppress
        # vehicle detections (vehicles are counted as they cross the intersection).
        excl_polys: list[np.ndarray] = []
        scene_map = data.get("scene_map") or {}
        features  = scene_map.get("features") if isinstance(scene_map, dict) else []
        EXCL_TYPES = {"exclusion", "parking", "sidewalk"}
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
                    excl_polys.append(np.array(pixel_pts, dtype=np.int32))
        self._excl_polys = excl_polys

        # ── detect zone (inclusion filter) ────────────────────────────────────
        # If detect_zone is configured, only detections whose CENTER falls inside
        # this polygon are counted.  This is the primary way to restrict counting
        # to the road area and exclude sky/buildings/parked-car false-positives.
        self._detect_poly = None
        dz = data.get("detect_zone")
        if isinstance(dz, dict):
            dz_pts_raw = dz.get("points") or []
            if not dz_pts_raw:
                # legacy 4-key format {x1,y1,x2,y2,x3,y3,x4,y4}
                if "x1" in dz and "y1" in dz:
                    dz_pts_raw = [
                        {"x": dz["x1"], "y": dz["y1"]},
                        {"x": dz["x2"], "y": dz["y2"]},
                        {"x": dz["x3"], "y": dz["y3"]},
                        {"x": dz["x4"], "y": dz["y4"]},
                    ]
            dz_pixels = [
                (int(float(p["x"]) * w), int(float(p["y"]) * h))
                for p in dz_pts_raw
                if isinstance(p, dict) and "x" in p and "y" in p
            ]
            if len(dz_pixels) >= 3:
                self._detect_poly = np.array(dz_pixels, dtype=np.int32)

        self._last_refresh = time.monotonic()
        logger.info(
            "Counter refreshed: zone=%s excl=%d detect_zone=%s camera=%s",
            self._zone_type, len(excl_polys),
            "yes" if self._detect_poly is not None else "none",
            self.camera_id,
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
            self._pending_crossings.pop(t, None)  # discard pending if track died
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

        # detect zone — inclusion filter: center must be INSIDE detect_zone polygon
        if self._detect_poly is not None and n > 0 and detections.xyxy is not None:
            for i in range(n):
                if not mask[i]:
                    continue
                if i >= len(detections.xyxy):
                    continue
                x1d, y1d, x2d, y2d = detections.xyxy[i]
                cx = float((x1d + x2d) / 2)
                cy = float((y1d + y2d) / 2)
                try:
                    if cv2.pointPolygonTest(self._detect_poly, (cx, cy), False) < 0:
                        mask[i] = False
                except Exception:
                    pass

        # exclusion zones — center-point-in-polygon (avoids BOTTOM_CENTER anchor issue)
        if self._excl_polys and n > 0 and detections.xyxy is not None:
            for i in range(n):
                if not mask[i]:
                    continue
                if i >= len(detections.xyxy):
                    continue
                x1e, y1e, x2e, y2e = detections.xyxy[i]
                cx = float((x1e + x2e) / 2)
                cy = float((y1e + y2e) / 2)
                for poly in self._excl_polys:
                    try:
                        if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                            mask[i] = False
                            break
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

        self._process_calls += 1
        do_diag = (self._process_calls % 50 == 1)  # log every 50 calls (~12 min at 1/15s)

        if self._zone_type == "line":
            # LineZone: use sv.LineZone.trigger which tracks crossed_in/crossed_out
            if len(detections) > 0:
                try:
                    crossed_in, crossed_out = self._zone.trigger(detections=detections)
                except Exception as e:
                    logger.warning("LineZone.trigger failed: %s", e)
                    crossed_in = crossed_out = np.array([], dtype=bool)

                # ── diagnostic log ──────────────────────────────────────────
                if do_diag and detections.xyxy is not None:
                    lx1, ly1, lx2, ly2 = self._zone_coords
                    centers = []
                    for i in range(min(len(detections.xyxy), 8)):
                        bx1, by1, bx2, by2 = detections.xyxy[i]
                        cx = (float(bx1) + float(bx2)) / 2
                        cy = (float(by1) + float(by2)) / 2
                        tf = self._track_frames.get(tracker_ids[i] if has_ids and i < len(tracker_ids) else -1, 0)
                        ci = bool(crossed_in[i]) if i < len(crossed_in) else False
                        co = bool(crossed_out[i]) if i < len(crossed_out) else False
                        elig = i in eligible_indices
                        centers.append(f"({cx/self.frame_width:.2f},{cy/self.frame_height:.2f})tf={tf}e={elig}ci={ci}co={co}")
                    logger.info(
                        "DIAG frame=%dx%d line=(%d,%d)->(%d,%d) rel=(%.2f,%.2f)->(%.2f,%.2f) "
                        "n=%d eligible=%d pending=%d det=%s",
                        self.frame_width, self.frame_height,
                        int(lx1), int(ly1), int(lx2), int(ly2),
                        lx1/self.frame_width, ly1/self.frame_height,
                        lx2/self.frame_width, ly2/self.frame_height,
                        len(detections), len(eligible_indices),
                        len(self._pending_crossings),
                        " ".join(centers),
                    )
                # ────────────────────────────────────────────────────────────

                for i in eligible_indices:
                    if i >= len(crossed_in):
                        continue
                    tid = tracker_ids[i] if has_ids and i < len(tracker_ids) else None
                    if tid is not None and tid in self._confirmed_ids:
                        continue

                    cls_id   = int(detections.class_id[i]) if detections.class_id is not None else -1
                    cls_name = CLASS_NAMES.get(cls_id, "unknown")
                    if cls_name == "unknown" and unk_as_car:
                        cls_name = "car"

                    direction_in: bool | None = None
                    if crossed_in[i]:
                        direction_in = True
                    elif crossed_out[i]:
                        direction_in = False

                    if direction_in is not None:
                        frames_seen = self._track_frames.get(tid, 0) if tid is not None else min_tf
                        if frames_seen >= min_tf:
                            # Immediately count — track is mature enough.
                            self._add_count(cls_name, direction_in)
                            new_crossings += 1
                            if tid is not None:
                                self._confirmed_ids.add(tid)
                                self._pending_crossings.pop(tid, None)
                        elif tid is not None and tid not in self._pending_crossings:
                            # Track crossed but not mature yet — queue it.
                            self._pending_crossings[tid] = direction_in
                            logger.info(
                                "Crossing queued: tid=%d frames=%d/%d cls=%s dir=%s",
                                tid, frames_seen, min_tf, cls_name,
                                "in" if direction_in else "out",
                            )

                # ── flush pending crossings for now-mature tracks ────────────
                if self._pending_crossings and has_ids:
                    for tid, direction_in in list(self._pending_crossings.items()):
                        if tid in self._confirmed_ids:
                            del self._pending_crossings[tid]
                            continue
                        if self._track_frames.get(tid, 0) >= min_tf:
                            # Find class for this tid
                            cls_name = "car"
                            for i, t in enumerate(tracker_ids):
                                if t == tid and detections.class_id is not None and i < len(detections.class_id):
                                    c = CLASS_NAMES.get(int(detections.class_id[i]), "unknown")
                                    cls_name = c if c != "unknown" else ("car" if unk_as_car else "unknown")
                                    break
                            self._add_count(cls_name, direction_in)
                            new_crossings += 1
                            self._confirmed_ids.add(tid)
                            del self._pending_crossings[tid]
                            logger.info(
                                "Pending crossing flushed: tid=%d cls=%s dir=%s total=%d",
                                tid, cls_name, "in" if direction_in else "out",
                                self._confirmed_total,
                            )

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
            sb = await get_supabase()
            resp = await (
                sb.table("count_snapshots")
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


async def write_snapshot(snapshot: dict) -> None:
    """Write a count snapshot row to Supabase. Called by main.py at DB_SNAPSHOT_INTERVAL_SEC."""
    try:
        sb = await get_supabase()
        row = {
            "camera_id":         snapshot.get("camera_id"),
            "captured_at":       snapshot.get("captured_at"),
            "total":             snapshot.get("total", 0),
            "count_in":          snapshot.get("count_in", 0),
            "count_out":         snapshot.get("count_out", 0),
            "vehicle_breakdown": snapshot.get("vehicle_breakdown", {}),
            "round_total":       snapshot.get("round_total", 0),
            "round_count_in":    snapshot.get("round_count_in", 0),
            "round_count_out":   snapshot.get("round_count_out", 0),
        }
        await sb.table("count_snapshots").insert(row).execute()
    except Exception as exc:
        logger.warning("write_snapshot failed: %s", exc)


# ── Analytics zone processor stub (keeps analytics_service import happy) ──────

class AnalyticsZoneProcessor:
    """Stub — zone analytics handled by analytics_service separately."""
    def __init__(self, *a, **kw): pass
    def process(self, *a, **kw) -> list: return []
