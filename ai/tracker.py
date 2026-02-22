"""
ai/tracker.py - Supervision ByteTrack wrapper.
Assigns persistent track IDs to detections across frames.
"""
import supervision as sv

from config import get_config


class VehicleTracker:
    def __init__(self):
        cfg = get_config()
        self.tracker = sv.ByteTrack(
            track_activation_threshold=cfg.TRACK_ACTIVATION_THRESHOLD,
            lost_track_buffer=cfg.TRACK_LOST_BUFFER,   # keep IDs stable through brief occlusions/frame drops
            minimum_matching_threshold=cfg.TRACK_MATCH_THRESHOLD,
            frame_rate=cfg.TRACK_FRAME_RATE,
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Update tracker and return detections with track IDs."""
        return self.tracker.update_with_detections(detections)
