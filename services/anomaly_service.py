"""
services/anomaly_service.py — Rolling statistical anomaly detection on vehicle counts.

Uses Welford's online algorithm to maintain a running mean and variance over
the last _WINDOW count readings. Fires an alert when the current count
deviates by more than _Z_THRESH standard deviations from the rolling mean.

Two integration modes:
  1. Inline (AI loop): call detector.feed(count, camera_id) every frame.
     Zero latency, fires WS broadcast immediately.
  2. Polled (background loop): anomaly_monitor_loop() polls count_snapshots
     every _POLL_INTERVAL seconds. Use when AI loop is unavailable or for
     secondary cameras.

Broadcasts {"type": "count_anomaly", ...} via manager.broadcast_public().
"""
import asyncio
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_WINDOW        = 60      # rolling window size (count readings)
_MIN_SAMPLES   = 15      # minimum samples before anomaly detection activates
_Z_THRESH      = 3.0     # z-score threshold (3σ ≈ 0.27% false-positive rate)
_ALERT_COOLDOWN_SEC = 90  # minimum seconds between anomaly alerts per camera
_POLL_INTERVAL = 30       # seconds between snapshot polls (background loop)
_POLL_LOOKBACK = 120      # seconds of count_snapshots to load per poll cycle


class CountAnomalyDetector:
    """
    Welford online mean/variance over a sliding window of vehicle counts.
    One instance per camera (or share a single instance for the active AI cam).
    """

    def __init__(self, camera_id: str = "") -> None:
        self.camera_id = camera_id
        self._window: deque[float] = deque(maxlen=_WINDOW)
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0           # Welford accumulator
        self._last_alert_at: float = 0.0

    def feed(self, count: float) -> dict[str, Any] | None:
        """
        Feed a new count reading. Returns an alert dict if anomalous, else None.
        Thread-safe for single-producer use (AI loop is single-threaded).
        """
        import time

        old_val = None
        if len(self._window) == _WINDOW:
            old_val = self._window[0]   # value about to be evicted

        self._window.append(float(count))
        self._n = min(self._n + 1, _WINDOW)

        # Welford update (approximate sliding — recompute on eviction)
        if old_val is not None:
            self._recompute()
        else:
            delta = count - self._mean
            self._mean += delta / self._n
            delta2 = count - self._mean
            self._m2 += delta * delta2

        if self._n < _MIN_SAMPLES:
            return None

        variance = self._m2 / self._n
        std = variance ** 0.5
        if std < 1e-6:
            return None

        z = abs(count - self._mean) / std
        if z < _Z_THRESH:
            return None

        now = time.monotonic()
        if (now - self._last_alert_at) < _ALERT_COOLDOWN_SEC:
            return None
        self._last_alert_at = now

        direction = "spike" if count > self._mean else "drop"
        logger.warning(
            "AnomalyDetector camera=%s count=%.0f mean=%.1f std=%.1f z=%.2f [%s]",
            self.camera_id, count, self._mean, std, z, direction,
        )
        return {
            "camera_id": self.camera_id,
            "count": count,
            "rolling_mean": round(self._mean, 1),
            "rolling_std": round(std, 1),
            "z_score": round(z, 2),
            "direction": direction,
            "detected_at": datetime.now(timezone.utc).isoformat(),
        }

    def _recompute(self) -> None:
        """Full recompute of mean/M2 from current window (on eviction)."""
        vals = list(self._window)
        n = len(vals)
        if n == 0:
            self._mean = 0.0
            self._m2 = 0.0
            self._n = 0
            return
        mean = sum(vals) / n
        m2 = sum((v - mean) ** 2 for v in vals)
        self._mean = mean
        self._m2 = m2
        self._n = n

    def reset(self) -> None:
        self._window.clear()
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._last_alert_at = 0.0


# ── Background poll loop (for secondary cameras not in AI loop) ───────────────

async def anomaly_monitor_loop() -> None:
    """
    Polls count_snapshots for all active cameras every _POLL_INTERVAL seconds
    and broadcasts anomaly alerts via the public WS channel.

    The AI camera is also covered here as a cross-check, but the primary
    inline feed() call in the AI loop is more responsive.
    """
    from supabase_client import get_supabase
    from websocket.ws_manager import manager

    await asyncio.sleep(60)   # let things warm up

    detectors: dict[str, CountAnomalyDetector] = {}

    while True:
        try:
            sb = await get_supabase()
            since = (
                datetime.now(timezone.utc)
                .isoformat()
                .replace(
                    datetime.now(timezone.utc).strftime("%H:%M:%S.%f"),
                    (datetime.now(timezone.utc).replace(
                        second=max(0, datetime.now(timezone.utc).second - _POLL_LOOKBACK)
                    )).strftime("%H:%M:%S.%f"),
                )
            )
            # Simpler: compute since as ISO string from current time minus lookback
            from datetime import timedelta
            since_dt = datetime.now(timezone.utc) - timedelta(seconds=_POLL_LOOKBACK)
            since_iso = since_dt.isoformat()

            resp = await (
                sb.table("count_snapshots")
                .select("camera_id, total, captured_at")
                .gte("captured_at", since_iso)
                .order("captured_at", desc=False)
                .limit(5000)
                .execute()
            )
            rows = resp.data or []

            for row in rows:
                cam_id = row.get("camera_id")
                total = row.get("total")
                if not cam_id or total is None:
                    continue
                if cam_id not in detectors:
                    detectors[cam_id] = CountAnomalyDetector(camera_id=cam_id)
                alert = detectors[cam_id].feed(float(total))
                if alert and manager.public_count > 0:
                    await manager.broadcast_public({"type": "count_anomaly", **alert})

        except Exception as exc:
            logger.warning("AnomalyMonitor loop error: %s", exc)

        await asyncio.sleep(_POLL_INTERVAL)
