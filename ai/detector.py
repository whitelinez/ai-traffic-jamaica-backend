"""
ai/detector.py — YOLOv8n vehicle detector using Ultralytics + Supervision.
COCO classes used: 2=car, 3=motorcycle, 5=bus, 7=truck
"""
import logging

import cv2
import numpy as np
import supervision as sv
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# COCO class IDs for vehicles
VEHICLE_CLASSES = [2, 3, 5, 7]
CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class VehicleDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.50):
        logger.info("Loading YOLO model: %s (conf=%.2f)", model_path, conf_threshold)
        self.model = YOLO(model_path)
        self.conf = conf_threshold

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Run inference on a BGR frame.
        Converts to PIL Image (RGB) before passing to ultralytics — PIL is an
        explicitly supported type in ultralytics 8.3.x and avoids internal
        LetterBox preprocessing issues with raw cv2 numpy arrays.
        """
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        results = self.model.predict(
            source=pil_image,
            conf=self.conf,
            classes=VEHICLE_CLASSES,
            verbose=False,
        )[0]

        return sv.Detections.from_ultralytics(results)

    @staticmethod
    def class_name(class_id: int) -> str:
        return CLASS_NAMES.get(class_id, "unknown")
