"""
ai/tracker.py - Supervision ByteTrack wrapper.
Assigns persistent track IDs to detections across frames.
"""
import supervision as sv


class VehicleTracker:
    def __init__(self):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.2,
            lost_track_buffer=20,   # keep IDs stable through brief occlusions/frame drops
            minimum_matching_threshold=0.65,
            frame_rate=25,
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Update tracker and return detections with track IDs."""
        return self.tracker.update_with_detections(detections)
