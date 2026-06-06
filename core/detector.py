"""
YOLOv8 Detector Wrapper
========================
Clean interface untuk deteksi — handles model loading, inference, dan result parsing.
"""
import numpy as np
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class Detection:
    """Single detection result."""
    bbox: np.ndarray      # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str


@dataclass
class FrameDetections:
    """All detections in a single frame."""
    boxes: np.ndarray          # (N, 4) [x1,y1,x2,y2]
    confidences: np.ndarray    # (N,)
    class_ids: np.ndarray      # (N,)
    track_ids: Optional[np.ndarray] = None  # (N,) filled by tracker


class VehicleDetector:
    """YOLOv8 detector tuned for Buah Batu traffic."""

    def __init__(self, model_path: str, conf: float = 0.25,
                 iou: float = 0.6, imgsz: int = 880, device: str = "0"):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.class_names = self.model.names  # {0: 'Bis mini', ...}

    def detect(self, frame: np.ndarray) -> FrameDetections:
        """Run detection on a single frame."""
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        boxes_obj = results[0].boxes
        if boxes_obj is None or len(boxes_obj) == 0:
            return FrameDetections(
                boxes=np.empty((0, 4)),
                confidences=np.empty(0),
                class_ids=np.empty(0, dtype=int),
            )

        return FrameDetections(
            boxes=boxes_obj.xyxy.cpu().numpy(),
            confidences=boxes_obj.conf.cpu().numpy(),
            class_ids=boxes_obj.cls.cpu().numpy().astype(int),
        )

    def detect_and_track(self, frame: np.ndarray,
                          tracker_config: str = None,
                          persist: bool = True) -> FrameDetections:
        """Run detection + tracking (ByteTrack) on a single frame."""
        track_args = dict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            persist=persist,
            verbose=False,
        )
        if tracker_config:
            track_args["tracker"] = tracker_config

        results = self.model.track(**track_args)
        boxes_obj = results[0].boxes

        if boxes_obj is None or len(boxes_obj) == 0:
            return FrameDetections(
                boxes=np.empty((0, 4)),
                confidences=np.empty(0),
                class_ids=np.empty(0, dtype=int),
                track_ids=np.empty(0, dtype=int),
            )

        # Track IDs may be None if tracking fails on some frames
        track_ids = None
        if boxes_obj.id is not None:
            track_ids = boxes_obj.id.cpu().numpy().astype(int)

        return FrameDetections(
            boxes=boxes_obj.xyxy.cpu().numpy(),
            confidences=boxes_obj.conf.cpu().numpy(),
            class_ids=boxes_obj.cls.cpu().numpy().astype(int),
            track_ids=track_ids,
        )

    def get_class_name(self, class_id: int) -> str:
        return self.class_names.get(class_id, f"unknown_{class_id}")
