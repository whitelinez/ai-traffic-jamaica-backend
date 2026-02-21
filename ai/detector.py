"""
ai/detector.py — YOLOv8n vehicle detector using Ultralytics + Supervision.
COCO classes used: 2=car, 3=motorcycle, 5=bus, 7=truck
"""
import logging

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# COCO class IDs for vehicles
VEHICLE_CLASSES = [2, 3, 5, 7]
CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

_INFER_SIZE = 640  # YOLO input square size


class VehicleDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.50):
        logger.info("Loading YOLO model: %s (conf=%.2f)", model_path, conf_threshold)
        self.model = YOLO(model_path)
        self.conf = conf_threshold

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Run inference on a BGR frame. Must be called from a thread.

        OpenCV 4.11.0 in this build rejects Python-created numpy arrays in cv2
        operations (PyArray_Check ABI mismatch). Only arrays allocated by cv2's
        own C code pass the check. Strategy:
          1. Resize the frame ourselves using cv2.resize — frame is cv2-allocated
             (from VideoCapture), and cv2.resize returns a cv2-allocated result.
          2. Pass ONLY the resized frame to ultralytics. ultralytics' LetterBox
             sees it already fits the width (640), so cv2.resize is skipped.
             It then calls cv2.copyMakeBorder on our cv2-allocated array → works.
          3. Scale detected boxes back to original frame dimensions.
        """
        h, w = frame.shape[:2]

        # Scale to fit _INFER_SIZE on the longer side, maintaining aspect ratio.
        # cv2.resize on a cv2-allocated frame returns a cv2-allocated result.
        scale = min(_INFER_SIZE / h, _INFER_SIZE / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        results = self.model.predict(
            source=resized,
            conf=self.conf,
            classes=VEHICLE_CLASSES,
            verbose=False,
            imgsz=_INFER_SIZE,
        )[0]

        detections = sv.Detections.from_ultralytics(results)

        # ultralytics maps boxes back to source (resized) space.
        # Scale from resized to original frame coordinates.
        if len(detections) > 0:
            detections.xyxy /= scale

        return detections

    @staticmethod
    def class_name(class_id: int) -> str:
        return CLASS_NAMES.get(class_id, "unknown")
