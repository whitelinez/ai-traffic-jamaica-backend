"""
ai/detector.py — YOLOv8n vehicle detector using Ultralytics + Supervision.
COCO classes used: 2=car, 3=motorcycle, 5=bus, 7=truck
"""
import logging
import os

import numpy as np
import supervision as sv
import torch
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# COCO class IDs for vehicles
VEHICLE_CLASSES = [2, 3, 5, 7]
CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

_INFER_SIZE = int(os.getenv("DETECT_INFER_SIZE", "448"))


class VehicleDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.50):
        logger.info("Loading YOLO model: %s (conf=%.2f)", model_path, conf_threshold)
        self.model = YOLO(model_path)
        self.conf = conf_threshold

    def detect(self, frame) -> sv.Detections:
        """
        Run inference on a BGR frame (from cv2.VideoCapture). Thread-safe.

        Root cause: OpenCV 4.11 in this env has two incompatible numpy ABIs.
          - cv2-internal arrays (VideoCapture, cv2.resize output) pass cv2's
            PyArray_Check but fail Python isinstance(x, np.ndarray).
          - System numpy arrays (np.zeros etc.) pass isinstance but fail
            cv2's PyArray_Check, crashing cv2.resize / cv2.copyMakeBorder.

        Fix:
          1. np.array(frame) uses __array_interface__ to copy the cv2 array
             into system numpy — works across ABI versions.
          2. PIL letterboxes the frame with zero cv2 calls.
          3. A float torch.Tensor is passed as source.
          4. ultralytics detects torch.Tensor → LoadTensor path → skips
             pre_transform / LetterBox / all cv2 entirely.
          5. Predictions are in 640×640 padded space; we un-letterbox manually.
        """
        h, w = frame.shape[:2]
        scale = min(_INFER_SIZE / h, _INFER_SIZE / w)
        new_w, new_h = int(w * scale), int(h * scale)
        pad_top = (_INFER_SIZE - new_h) // 2
        pad_left = (_INFER_SIZE - new_w) // 2

        # Bridge cv2 array → system numpy via __array_interface__
        frame_np = np.array(frame, dtype=np.uint8)                   # BGR, system numpy
        frame_rgb = np.ascontiguousarray(frame_np[:, :, ::-1])       # BGR → RGB

        # Letterbox entirely in PIL — no cv2 calls
        pil = Image.fromarray(frame_rgb)
        pil = pil.resize((new_w, new_h), Image.BILINEAR)
        padded = Image.new("RGB", (_INFER_SIZE, _INFER_SIZE), (114, 114, 114))
        padded.paste(pil, (pad_left, pad_top))

        # PIL → tensor via raw bytes — avoids torch.from_numpy() which requires
        # identical numpy ABIs between torch's C extension and the system numpy.
        # tobytes() returns a plain Python bytes object; frombuffer() on a
        # bytearray uses Python's buffer protocol (no numpy involved at all).
        raw = padded.tobytes()  # RGB, length = _INFER_SIZE * _INFER_SIZE * 3
        tensor = (
            torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            .reshape(_INFER_SIZE, _INFER_SIZE, 3)
            .to(dtype=torch.float32)
            .div(255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

        # Tensor source → ultralytics uses LoadTensor, skips LetterBox/cv2 entirely
        results = self.model.predict(
            source=tensor,
            conf=self.conf,
            classes=VEHICLE_CLASSES,
            verbose=False,
        )[0]

        # from_ultralytics() passes torch-internal numpy arrays which fail
        # supervision's isinstance(xyxy, np.ndarray) check (ABI mismatch).
        # Bridge each array through np.array() to re-allocate in system numpy.
        boxes = results.boxes
        if boxes is not None and len(boxes):
            xyxy      = np.array(boxes.xyxy.cpu().numpy(),      dtype=np.float32)
            confidence = np.array(boxes.conf.cpu().numpy(),     dtype=np.float32)
            class_id  = np.array(boxes.cls.cpu().numpy(),       dtype=np.int32)
        else:
            xyxy       = np.empty((0, 4), dtype=np.float32)
            confidence = np.empty((0,),   dtype=np.float32)
            class_id   = np.empty((0,),   dtype=np.int32)

        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

        # Un-letterbox: predictions are in 640×640 padded tensor space
        if len(detections) > 0:
            detections.xyxy[:, [0, 2]] -= pad_left
            detections.xyxy[:, [1, 3]] -= pad_top
            detections.xyxy /= scale
            detections.xyxy[:, [0, 2]] = detections.xyxy[:, [0, 2]].clip(0, w)
            detections.xyxy[:, [1, 3]] = detections.xyxy[:, [1, 3]].clip(0, h)

        return detections

    @staticmethod
    def class_name(class_id: int) -> str:
        return CLASS_NAMES.get(class_id, "unknown")
