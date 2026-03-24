"""
ai/ptz_guard.py — PTZ camera motion detector.

Uses sparse Lucas-Kanade optical flow + RANSAC homography estimation to
robustly distinguish global camera movement (pan/tilt/zoom) from local
vehicle motion.

Why RANSAC homography over angle-std:
  A pure angle-std check breaks when all vehicles move in the same direction
  (one-way street, convoy). RANSAC separates inlier background motion (camera)
  from outlier vehicle motion geometrically — giving a clean signal even in
  heavy traffic.

  - High inlier ratio + significant homography translation → camera panning
  - Low inlier ratio (many outliers) + near-identity H → vehicles only
  - Near-identity H regardless of inliers → scene is static / very slow pan

When camera is moving:
  - Crossings suppressed (prevents phantom counts from zone mismatch)
  - Pending crossings cleared (pre-pan queue discarded)

When camera stops:
  - Settle delay applied before crossings re-enable (lets tracker rebuild
    clean IDs in the new scene before accepting any crossings)
  - Tracker + box_smoother reset (caller responsibility on just_stopped=True)

Tunable via env vars:
  PTZ_DETECTION_ENABLED    1/0 (default 1)
  PTZ_MOTION_THRESHOLD     min mean inlier displacement (pixels) to trigger (default 1.8)
  PTZ_INLIER_RATIO_MIN     min RANSAC inlier ratio to confirm global motion (default 0.55)
  PTZ_FRAMES_TO_TRIGGER    consecutive moving frames before flagging (default 2)
  PTZ_FRAMES_TO_CLEAR      consecutive still frames before clearing moving flag (default 6)
  PTZ_SETTLE_FRAMES        extra frames after stop before crossings re-enable (default 12)
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
_GRID_X = 10
_GRID_Y = 7


class PTZGuard:
    def __init__(self) -> None:
        self._enabled: bool = int(os.getenv("PTZ_DETECTION_ENABLED", "1")) == 1
        self._threshold: float = float(os.getenv("PTZ_MOTION_THRESHOLD", "1.8"))
        self._inlier_ratio_min: float = float(os.getenv("PTZ_INLIER_RATIO_MIN", "0.55"))
        self._frames_to_trigger: int = max(1, int(os.getenv("PTZ_FRAMES_TO_TRIGGER", "2")))
        self._frames_to_clear: int = max(1, int(os.getenv("PTZ_FRAMES_TO_CLEAR", "6")))
        self._settle_frames: int = max(0, int(os.getenv("PTZ_SETTLE_FRAMES", "12")))

        self._prev_gray: np.ndarray | None = None
        self._is_moving: bool = False
        self._settle_countdown: int = 0   # frames remaining in post-stop settle window
        self._consec_moving: int = 0
        self._consec_still: int = 0

        # Last computed homography — exposed for potential zone adaptation callers
        self.last_homography: np.ndarray | None = None
        self.last_inlier_ratio: float = 0.0
        self.last_displacement: float = 0.0

        if not _CV2_AVAILABLE:
            logger.warning("PTZGuard: cv2 not available — disabled")
            self._enabled = False
        if not self._enabled:
            logger.info("PTZGuard: disabled")
        else:
            logger.info(
                "PTZGuard: enabled (threshold=%.1f inlier_min=%.2f "
                "trigger=%d clear=%d settle=%d)",
                self._threshold, self._inlier_ratio_min,
                self._frames_to_trigger, self._frames_to_clear,
                self._settle_frames,
            )

    @property
    def is_suppressed(self) -> bool:
        """True when crossings should be suppressed — either moving or settling."""
        return self._is_moving or self._settle_countdown > 0

    def update(self, frame: np.ndarray) -> tuple[bool, bool]:
        """
        Process one frame.

        Returns:
            (is_suppressed, just_stopped)

            is_suppressed — True while camera is moving OR in post-stop settle window.
                            Caller should pass suppress_crossings=True to counter.
            just_stopped  — True on the single frame the camera transitions to still
                            (before settle window starts). Caller should reset tracker.
        """
        if not self._enabled:
            return False, False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (_FLOW_W, _FLOW_H), interpolation=cv2.INTER_AREA)

        if self._prev_gray is None:
            self._prev_gray = small
            return False, False

        # ── Sparse optical flow ───────────────────────────────────────────────
        xs = np.linspace(8, _FLOW_W - 8, _GRID_X)
        ys = np.linspace(8, _FLOW_H - 8, _GRID_Y)
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
        n_good = int(good.sum())
        min_pts = max(10, (_GRID_X * _GRID_Y) // 4)

        if n_good < min_pts:
            # Insufficient tracking points — inherit last state, tick settle
            self._tick_settle()
            return self.is_suppressed, False

        src = pts[good]
        dst = next_pts[good]

        # ── RANSAC homography ─────────────────────────────────────────────────
        # Separates inlier background motion (camera) from outlier vehicles.
        # Requires ≥4 point correspondences.
        is_global = False
        self.last_homography = None
        self.last_inlier_ratio = 0.0
        self.last_displacement = 0.0

        if n_good >= 4:
            try:
                H, mask = cv2.findHomography(
                    src, dst,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=2.5,
                    maxIters=500,
                )
                if H is not None and mask is not None:
                    inlier_ratio = float(mask.sum()) / float(n_good)
                    self.last_inlier_ratio = inlier_ratio

                    # Mean displacement of inlier points under H
                    inlier_src = src[mask.ravel() == 1]
                    inlier_dst = dst[mask.ravel() == 1]
                    displacement = float(
                        np.linalg.norm(inlier_dst - inlier_src, axis=-1).mean()
                    ) if len(inlier_src) > 0 else 0.0
                    self.last_displacement = displacement

                    # Translation component of H (normalised to 160x90 space)
                    tx = float(H[0, 2])
                    ty = float(H[1, 2])
                    h_translation = float(np.hypot(tx, ty))

                    # Camera is moving when:
                    #   - Most points are inliers (coherent background motion)
                    #   - The inlier displacement is significant
                    #   - H has a non-trivial translation (rules out zoom-only)
                    is_global = (
                        inlier_ratio >= self._inlier_ratio_min
                        and displacement >= self._threshold
                    )

                    if is_global:
                        self.last_homography = H

            except cv2.error:
                # findHomography can fail on degenerate point configs
                pass

        # Fallback for very low point counts: use raw mean flow magnitude
        if not is_global and n_good >= min_pts:
            raw_flow = (dst - src).reshape(-1, 2)
            mean_mag = float(np.linalg.norm(raw_flow, axis=1).mean())
            angles = np.arctan2(raw_flow[:, 1], raw_flow[:, 0])
            angle_std = float(np.std(np.unwrap(angles)))
            is_global = mean_mag >= self._threshold * 1.5 and angle_std < 0.8

        # ── Hysteresis ────────────────────────────────────────────────────────
        prev_is_moving = self._is_moving

        if is_global:
            self._consec_moving += 1
            self._consec_still = 0
        else:
            self._consec_still += 1
            self._consec_moving = 0

        if self._consec_moving >= self._frames_to_trigger:
            self._is_moving = True
            self._settle_countdown = 0   # reset settle; we're still moving

        elif self._consec_still >= self._frames_to_clear and self._is_moving:
            self._is_moving = False

        # ── Settle window ─────────────────────────────────────────────────────
        just_stopped = prev_is_moving and not self._is_moving
        if just_stopped and self._settle_frames > 0:
            self._settle_countdown = self._settle_frames
        else:
            self._tick_settle()

        # ── Logging ───────────────────────────────────────────────────────────
        if self._is_moving and not prev_is_moving:
            logger.info(
                "PTZGuard: camera moving — inlier_ratio=%.2f displacement=%.2fpx",
                self.last_inlier_ratio, self.last_displacement,
            )
        elif just_stopped:
            logger.info(
                "PTZGuard: camera stopped — settle=%d frames, tracker will reset",
                self._settle_frames,
            )
        elif self._settle_countdown > 0 and (self._settle_countdown % 4 == 0):
            logger.debug("PTZGuard: settling (%d frames left)", self._settle_countdown)

        return self.is_suppressed, just_stopped

    def _tick_settle(self) -> None:
        if self._settle_countdown > 0:
            self._settle_countdown -= 1

    def reset(self) -> None:
        """Reset all state — call when stream restarts or camera switches."""
        self._prev_gray = None
        self._is_moving = False
        self._settle_countdown = 0
        self._consec_moving = 0
        self._consec_still = 0
        self.last_homography = None
        self.last_inlier_ratio = 0.0
        self.last_displacement = 0.0
