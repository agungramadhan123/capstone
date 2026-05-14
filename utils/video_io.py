"""
Video I/O dengan Memory Management
====================================
Handles video reading/writing dengan proper resource cleanup.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Generator, Tuple


class VideoReader:
    """Frame-by-frame video reader dengan metadata."""

    def __init__(self, source: str):
        self.source = str(source)
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {self.source}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_number = 0

    def __iter__(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frame_number += 1
            yield self.frame_number, frame

    def __len__(self):
        return self.total_frames

    def __del__(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

    @property
    def info(self) -> str:
        duration = self.total_frames / self.fps if self.fps > 0 else 0
        return (f"Video: {self.source}\n"
                f"  Resolution: {self.width}x{self.height}\n"
                f"  FPS: {self.fps:.1f}\n"
                f"  Frames: {self.total_frames}\n"
                f"  Duration: {duration:.1f}s ({duration/60:.1f}m)")


class VideoWriter:
    """Video writer dengan auto-codec selection."""

    def __init__(self, output_path: str, fps: float,
                 width: int, height: int):
        self.output_path = str(output_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # MP4V codec — compatible everywhere
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            self.output_path, fourcc, fps, (width, height)
        )
        if not self.writer.isOpened():
            raise IOError(f"Cannot create video writer: {self.output_path}")

        self.frame_count = 0

    def write(self, frame: np.ndarray):
        self.writer.write(frame)
        self.frame_count += 1

    def release(self):
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.release()

    def __del__(self):
        self.release()
