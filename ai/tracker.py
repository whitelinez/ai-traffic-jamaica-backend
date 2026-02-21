"""
ai/tracker.py — Supervision ByteTrack wrapper.
Assigns persistent track IDs to detections across frames.
"""
import supervision as sv


class VehicleTracker:
    def __init__(self):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=5,   # 5 frames @ 25fps = 0.2s — minimise ghost boxes
            minimum_matching_threshold=0.8,
            frame_rate=25,
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Update tracker and return detections with track IDs."""
        return self.tracker.update_with_detections(detections)
