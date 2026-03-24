"""
ai/ptz_guard.py — PTZ camera motion detector.

Uses sparse Lucas-Kanade optical flow on a grid of points to distinguish
global camera movement (pan/tilt/zoom) from local vehicle motion.

When the camera is moving:
  - Crossings are suppressed (prevents phantom counts)
  - Tracker is reset after motion stops (stale tracks from old position discarded)

Tunable via env vars:
  PTZ_DETECTION_ENABLED    1/0 (default 1)
  PTZ_MOTION_THRESHOLD     mean pixel displacement per frame to trigger (default 2.0)
  PTZ_ANGLE_STD_MAX        max angle std-dev for global motion classification (default 0.9)
  PTZ_FRAMES_TO_TRIGGER    consecutive moving frames before flagging (default 2)
  PTZ_FRAMES_TO_CLEAR      consecutive still frames before clearing (default 6)
"""
from __future__ import annotations

import os
import logging
import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

logger = logging.getLogger(__name__)

_FLOW_W = 160
_FLOW_H = 90
_GRID_X = 8
_GRID_Y = 6


class PTZGuard:
    def __init__(self) -> None:
        self._enabled: bool = int(os.getenv("PTZ_DETECTION_ENABLED", "1")) == 1
        self._threshold: float = float(os.getenv("PTZ_MOTION_THRESHOLD", "2.0"))
        self._angle_std_max: float = float(os.getenv("PTZ_ANGLE_STD_MAX", "0.9"))
        self._frames_to_trigger: int = max(1, int(os.getenv("PTZ_FRAMES_TO_TRIGGER", "2")))
        self._frames_to_clear: int = max(1, int(os.getenv("PTZ_FRAMES_TO_CLEAR", "6")))

        self._prev_gray: np.ndarray | None = None
        self._is_moving: bool = False
        self._was_moving: bool = False   # True on the one frame transition moving→still
        self._consec_moving: int = 0
        self._consec_still: int = 0

        if not _CV2_AVAILABLE:
            logger.warning("PTZGuard: cv2 not available — disabled")
            self._enabled = False
        if not self._enabled:
            logger.info("PTZGuard: disabled")
        else:
            logger.info(
                "PTZGuard: enabled (threshold=%.1f angle_std_max=%.2f "
                "trigger=%d clear=%d)",
                self._threshold, self._angle_std_max,
                self._frames_to_trigger, self._frames_to_clear,
            )

    # ------------------------------------------------------------------
    def update(self, frame: np.ndarray) -> tuple[bool, bool]:
        """
        Process one frame.

        Returns:
            (is_moving, just_stopped)
            is_moving   — True while camera is panning/tilting/zooming
            just_stopped — True on the single frame the camera transitions to still
                           (caller should reset tracker on this signal)
        """
        if not self._enabled:
            return False, False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (_FLOW_W, _FLOW_H), interpolation=cv2.INTER_AREA)

        if self._prev_gray is None:
            self._prev_gray = small
            return False, False

        # Build grid of feature points
        xs = np.linspace(10, _FLOW_W - 10, _GRID_X)
        ys = np.linspace(10, _FLOW_H - 10, _GRID_Y)
        pts = np.array(
            [[x, y] for y in ys for x in xs], dtype=np.float32
        ).reshape(-1, 1, 2)

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, small, pts, None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        self._prev_gray = small

        good = (status.ravel() == 1)
        min_pts = max(8, (_GRID_X * _GRID_Y) // 4)
        if good.sum() < min_pts:
            # Not enough tracked points — can't reliably classify
            return self._is_moving, False

        flow = (next_pts[good] - pts[good]).reshape(-1, 2)
        magnitude = float(np.linalg.norm(flow, axis=1).mean())

        # Global motion check: camera pan → all points move in a similar direction
        # Local/vehicle motion → high angular variance
        angles = np.arctan2(flow[:, 1], flow[:, 0])
        # unwrap to avoid discontinuities at ±π
        angle_std = float(np.std(np.unwrap(angles)))

        is_global = magnitude > self._threshold and angle_std < self._angle_std_max

        prev_is_moving = self._is_moving

        if is_global:
            self._consec_moving += 1
            self._consec_still = 0
        else:
            self._consec_still += 1
            self._consec_moving = 0

        if self._consec_moving >= self._frames_to_trigger:
            self._is_moving = True
        elif self._consec_still >= self._frames_to_clear:
            self._is_moving = False

        just_stopped = prev_is_moving and not self._is_moving

        if self._is_moving and not prev_is_moving:
            logger.info(
                "PTZGuard: camera moving (mag=%.2f angle_std=%.2f)",
                magnitude, angle_std,
            )
        elif just_stopped:
            logger.info("PTZGuard: camera stopped — tracker will reset")

        return self._is_moving, just_stopped

    def reset(self) -> None:
        """Reset state — call when stream restarts."""
        self._prev_gray = None
        self._is_moving = False
        self._consec_moving = 0
        self._consec_still = 0
