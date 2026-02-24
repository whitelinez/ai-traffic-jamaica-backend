"""
ai/stream.py - HLS stream reader with exponential backoff reconnect.
Yields raw BGR frames from the live feed.
"""
import asyncio
import logging
import time
from typing import AsyncGenerator

import cv2
import numpy as np

from config import get_config

logger = logging.getLogger(__name__)

MAX_BACKOFF = 60.0
BASE_BACKOFF = 2.0


class HLSStream:
    """OpenCV-based HLS reader with auto-reconnect."""

    def __init__(self, url: str):
        self.url = url
        self._cap: cv2.VideoCapture | None = None
        self._backoff = BASE_BACKOFF
        self._grab_latest = get_config().STREAM_GRAB_LATEST == 1
        self._source_type = self._infer_source_type(url)
        self._reported_fps = 0.0
        self._measured_decode_fps = 0.0
        self._frame_interval_ms = 0.0
        self._resolution = {"width": 0, "height": 0}
        self._codec = "unknown"
        self._bitrate_kbps: int | None = None
        self._keyframe_interval = "unknown"
        self._timestamp_mode = "wallclock"
        self._decode_latency_ms = 0.0
        self._end_to_end_latency_estimate_ms = 0.0
        self._opened_wallclock = 0.0
        self._first_pts_ms: float | None = None
        self._last_frame_wallclock = 0.0
        self._last_stats_log_mono = 0.0

    def _open(self) -> bool:
        if self._cap:
            self._cap.release()
        logger.info("Opening HLS stream: %s", self.url)
        self._cap = cv2.VideoCapture(self.url)
        # FFMPEG backend, minimal buffering
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok = self._cap.isOpened()
        if ok:
            self._backoff = BASE_BACKOFF
            self._opened_wallclock = time.time()
            self._first_pts_ms = None
            self._last_frame_wallclock = 0.0
            self._source_type = self._infer_source_type(self.url)
            self._read_static_props()
            logger.info("Stream opened successfully")
            logger.info("Stream diagnostics (open): %s", self.diagnostics_report())
        else:
            logger.warning("Failed to open stream")
        return ok

    @staticmethod
    def _infer_source_type(url: str) -> str:
        raw = str(url or "").lower()
        if raw.startswith("rtsp://"):
            return "rtsp"
        if raw.startswith("webrtc://") or "webrtc" in raw:
            return "webrtc"
        if raw.endswith(".m3u8") or ".m3u8?" in raw:
            return "hls"
        if raw.endswith(".mp4") or ".mp4?" in raw:
            return "mp4"
        return "unknown"

    @staticmethod
    def _decode_fourcc(value: float) -> str:
        try:
            num = int(value)
            if num <= 0:
                return "unknown"
            return "".join([chr((num >> 8 * i) & 0xFF) for i in range(4)]).strip() or "unknown"
        except Exception:
            return "unknown"

    def _read_static_props(self) -> None:
        if not self._cap:
            return
        try:
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            self._resolution = {"width": max(0, w), "height": max(0, h)}
        except Exception:
            pass
        try:
            fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if np.isfinite(fps) and fps > 0:
                self._reported_fps = fps
                self._frame_interval_ms = 1000.0 / max(fps, 0.001)
        except Exception:
            pass
        try:
            self._codec = self._decode_fourcc(self._cap.get(cv2.CAP_PROP_FOURCC))
        except Exception:
            pass
        try:
            # CAP_PROP_BITRATE is backend-dependent; keep optional.
            bitrate = float(self._cap.get(cv2.CAP_PROP_BITRATE) or 0.0)
            if np.isfinite(bitrate) and bitrate > 0:
                self._bitrate_kbps = int(round(bitrate))
        except Exception:
            self._bitrate_kbps = None

    def _update_runtime_stats(self) -> None:
        if not self._cap:
            return
        now = time.time()
        if self._last_frame_wallclock > 0:
            dt = max(0.0001, now - self._last_frame_wallclock)
            inst_decode_fps = 1.0 / dt
            if self._measured_decode_fps <= 0:
                self._measured_decode_fps = inst_decode_fps
            else:
                # Smooth to avoid noisy one-frame swings.
                self._measured_decode_fps = (self._measured_decode_fps * 0.85) + (inst_decode_fps * 0.15)
        self._last_frame_wallclock = now

        pts_ms = 0.0
        try:
            pts_ms = float(self._cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        except Exception:
            pts_ms = 0.0

        has_pts = np.isfinite(pts_ms) and pts_ms > 0.0
        if has_pts and self._first_pts_ms is None:
            self._first_pts_ms = pts_ms

        if has_pts and self._first_pts_ms is not None:
            self._timestamp_mode = "pts"
            source_elapsed_sec = max(0.0, (pts_ms - self._first_pts_ms) / 1000.0)
            wall_elapsed_sec = max(0.0, now - self._opened_wallclock)
            inst_latency_ms = max(0.0, (wall_elapsed_sec - source_elapsed_sec) * 1000.0)
            self._decode_latency_ms = (self._decode_latency_ms * 0.85) + (inst_latency_ms * 0.15)
        else:
            self._timestamp_mode = "wallclock"

        if self._measured_decode_fps > 0:
            self._frame_interval_ms = 1000.0 / max(self._measured_decode_fps, 0.001)
        elif self._reported_fps > 0:
            self._frame_interval_ms = 1000.0 / max(self._reported_fps, 0.001)

        # Without player-side timing we expose a backend-side estimate.
        self._end_to_end_latency_estimate_ms = max(self._decode_latency_ms, self._frame_interval_ms * 1.5)

    def get_frame_interval_sec(self, default_fps: float = 15.0) -> float:
        if self._frame_interval_ms > 0:
            return max(1.0 / 60.0, self._frame_interval_ms / 1000.0)
        fps = self._reported_fps if self._reported_fps > 0 else default_fps
        return max(1.0 / 60.0, 1.0 / max(1.0, fps))

    def diagnostics_report(self) -> dict:
        return {
            "source_type": self._source_type,
            "reported_fps": round(float(self._reported_fps or 0.0), 3),
            "measured_fps": round(float(self._measured_decode_fps or 0.0), 3),
            "frame_interval_ms": round(float(self._frame_interval_ms or 0.0), 2),
            "resolution": dict(self._resolution),
            "codec": self._codec,
            "bitrate_kbps": self._bitrate_kbps,
            "keyframe_interval": self._keyframe_interval,
            "timestamp_mode": self._timestamp_mode,
            "decode_latency_ms": round(float(self._decode_latency_ms or 0.0), 2),
            "end_to_end_latency_estimate_ms": round(float(self._end_to_end_latency_estimate_ms or 0.0), 2),
        }

    async def frames(self) -> AsyncGenerator[np.ndarray, None]:
        """
        Async generator that yields BGR frames.
        Reconnects with exponential backoff on failure.
        """
        while True:
            if not self._open():
                await asyncio.sleep(self._backoff)
                self._backoff = min(self._backoff * 2, MAX_BACKOFF)
                continue

            while True:
                if self._grab_latest:
                    # Drop one queued frame when possible to reduce stale-latency feel.
                    await asyncio.to_thread(self._cap.grab)
                ret, frame = await asyncio.to_thread(self._cap.read)
                if not ret:
                    logger.warning("Frame read failed - reconnecting in %.1fs", self._backoff)
                    break
                self._update_runtime_stats()
                now_mono = asyncio.get_running_loop().time()
                if (now_mono - self._last_stats_log_mono) >= 15.0:
                    self._last_stats_log_mono = now_mono
                    logger.info("Stream diagnostics (runtime): %s", self.diagnostics_report())
                yield frame
                # Yield control so asyncio can process other tasks
                await asyncio.sleep(0)

            await asyncio.sleep(self._backoff)
            self._backoff = min(self._backoff * 2, MAX_BACKOFF)

    def release(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None
