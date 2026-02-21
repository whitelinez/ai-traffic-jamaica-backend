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

        OpenCV 4.11 in this build cannot call cv2.resize on Python-created numpy
        arrays (PyArray_Check fails), only on arrays it created internally.
        Work-around: letterbox the frame ourselves here (frame came from
        VideoCapture so it passes cv2's check), then pass a pre-sized 640x640
        padded image to ultralytics so its LetterBox finds no resize needed and
        never calls cv2.resize internally.  Detection boxes are scaled back to
        original frame coordinates before returning.
        """
        h, w = frame.shape[:2]

        # Letterbox: scale to fit _INFER_SIZE, preserve aspect ratio
        scale = min(_INFER_SIZE / h, _INFER_SIZE / w)
        new_w, new_h = int(w * scale), int(h * scale)

        # cv2.resize works here — frame was allocated by VideoCapture (C side)
        resized = cv2.resize(frame, (new_w, new_h))
        padded = np.zeros((_INFER_SIZE, _INFER_SIZE, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        results = self.model.predict(
            source=padded,
            conf=self.conf,
            classes=VEHICLE_CLASSES,
            verbose=False,
            imgsz=_INFER_SIZE,
        )[0]

        detections = sv.Detections.from_ultralytics(results)

        # Remap boxes from 640x640 padded space back to original frame space
        if len(detections) > 0:
            detections.xyxy[:, [0, 2]] /= scale  # x
            detections.xyxy[:, [1, 3]] /= scale  # y

        return detections

    @staticmethod
    def class_name(class_id: int) -> str:
        return CLASS_NAMES.get(class_id, "unknown")
