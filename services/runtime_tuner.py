"""
services/runtime_tuner.py - Adaptive runtime profile selection for live AI loop.

Profiles tune detector/tracker/loop behavior based on:
- day vs night window
- traffic flow (detections and crossings rate)
- observed detection confidence

Manual override can be carried in cameras.count_settings:
- runtime_profile_mode: "auto" | "manual"
- runtime_manual_profile: profile name
- runtime_manual_until: ISO timestamp (optional)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


RUNTIME_PROFILES: dict[str, dict[str, Any]] = {
    # Day profile for normal traffic.
    "day_balanced": {
        "detector": {"conf": 0.34, "infer_size": 512, "iou": 0.50, "max_det": 100},
        "tracker": {"lost_buffer": 18, "fallback_ttl_sec": 1.00, "fallback_dist_ratio": 0.065},
        "loop": {"process_every_n": 1},
    },
    # Day profile when many vehicles are present.
    "day_heavy": {
        "detector": {"conf": 0.30, "infer_size": 448, "iou": 0.48, "max_det": 120},
        "tracker": {"lost_buffer": 14, "fallback_ttl_sec": 0.75, "fallback_dist_ratio": 0.055},
        "loop": {"process_every_n": 1},
    },
    # Day profile for harsh light/reflections where false positives rise.
    "day_glare": {
        "detector": {"conf": 0.38, "infer_size": 512, "iou": 0.52, "max_det": 90},
        "tracker": {"lost_buffer": 16, "fallback_ttl_sec": 0.90, "fallback_dist_ratio": 0.060},
        "loop": {"process_every_n": 1},
    },
    "night_balanced": {
        "detector": {"conf": 0.28, "infer_size": 640, "iou": 0.45, "max_det": 120},
        "tracker": {"lost_buffer": 20, "fallback_ttl_sec": 1.20, "fallback_dist_ratio": 0.075},
        "loop": {"process_every_n": 1},
    },
    "night_heavy": {
        "detector": {"conf": 0.26, "infer_size": 576, "iou": 0.44, "max_det": 140},
        "tracker": {"lost_buffer": 18, "fallback_ttl_sec": 1.00, "fallback_dist_ratio": 0.070},
        "loop": {"process_every_n": 1},
    },
}


def _as_utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


@dataclass
class TrafficStats:
    detections_per_min: float
    crossings_per_min: float
    avg_confidence: float


def is_night_hour(hour: int, start_hour: int, end_hour: int) -> bool:
    start = int(start_hour) % 24
    end = int(end_hour) % 24
    h = int(hour) % 24
    if start == end:
        return True
    if start < end:
        return start <= h < end
    return h >= start or h < end


def select_runtime_profile(
    *,
    now_utc: datetime,
    stats: TrafficStats,
    controls: dict[str, Any],
    night_start_hour: int,
    night_end_hour: int,
) -> tuple[str, str]:
    """
    Return (profile_name, reason).
    Manual override takes precedence over auto mode.
    """
    mode = str(controls.get("runtime_profile_mode", "auto") or "auto").strip().lower()
    manual_name = str(controls.get("runtime_manual_profile", "") or "").strip()
    manual_until = _as_utc(controls.get("runtime_manual_until"))

    if mode == "manual":
        if manual_name in RUNTIME_PROFILES:
            if manual_until is None or now_utc <= manual_until:
                return manual_name, "manual_override"
        # Expired/invalid manual profile: fall through to auto.

    night = is_night_hour(
        hour=now_utc.hour,
        start_hour=night_start_hour,
        end_hour=night_end_hour,
    )
    det_pm = float(max(0.0, stats.detections_per_min))
    cross_pm = float(max(0.0, stats.crossings_per_min))
    avg_conf = float(max(0.0, min(1.0, stats.avg_confidence)))

    if night:
        if det_pm >= 45 or cross_pm >= 14:
            return "night_heavy", "night_heavy_traffic"
        return "night_balanced", "night_default"

    # Daytime decisions
    if avg_conf < 0.42 and det_pm < 20 and cross_pm < 5:
        return "day_glare", "day_glare_low_conf"
    if det_pm >= 55 or cross_pm >= 16:
        return "day_heavy", "day_heavy_traffic"
    return "day_balanced", "day_default"
