"""
Visualization Utilities
========================
Drawing bounding boxes, track IDs, virtual line, dan counting overlay.
"""
import cv2
import numpy as np
from typing import Dict, Tuple, Optional

# Color palette — distinct colors for each class
CLASS_COLORS = {
    0: (255, 165, 0),    # Bis mini — orange
    1: (0, 255, 0),      # Manusia — green
    2: (0, 0, 255),      # Mobil — red
    3: (0, 255, 255),    # Motor — yellow
    4: (128, 0, 128),    # Truk berat — purple
    5: (255, 0, 128),    # Truk menengah — pink
    6: (0, 165, 255),    # Truk ringan — orange-light
    7: (255, 100, 0),    # bus — blue-orange
}


def draw_detections(
    frame: np.ndarray,
    boxes: np.ndarray,
    class_ids: np.ndarray,
    confidences: np.ndarray,
    track_ids: Optional[np.ndarray],
    class_names: Dict[int, str],
) -> np.ndarray:
    """Draw bounding boxes dengan class label dan track ID."""
    annotated = frame.copy()

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        cls_id = int(class_ids[i])
        conf = float(confidences[i])
        color = CLASS_COLORS.get(cls_id, (200, 200, 200))

        # Box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label
        label = class_names.get(cls_id, f"cls_{cls_id}")
        if track_ids is not None and i < len(track_ids):
            label = f"ID:{int(track_ids[i])} {label} {conf:.2f}"
        else:
            label = f"{label} {conf:.2f}"

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return annotated


def draw_virtual_line(
    frame: np.ndarray,
    line_start: Tuple[int, int],
    line_end: Tuple[int, int],
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 3,
) -> np.ndarray:
    """Draw garis virtual counting."""
    annotated = frame.copy()
    cv2.line(annotated, line_start, line_end, color, thickness)

    # Dash effect
    mid_x = (line_start[0] + line_end[0]) // 2
    mid_y = (line_start[1] + line_end[1]) // 2
    cv2.putText(annotated, "COUNTING LINE", (mid_x - 80, mid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return annotated


def draw_counting_overlay(
    frame: np.ndarray,
    total_count: int,
    class_counts: Dict[str, int],
    fps_display: float = 0.0,
) -> np.ndarray:
    """Draw counting summary overlay (top-left corner)."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Semi-transparent background
    overlay = annotated.copy()
    panel_h = 40 + len(class_counts) * 25 + 30
    cv2.rectangle(overlay, (5, 5), (250, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

    # Total count
    y = 30
    cv2.putText(annotated, f"TOTAL: {total_count}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Per-class
    y += 10
    for cls_name, count in sorted(class_counts.items()):
        y += 25
        cv2.putText(annotated, f"  {cls_name}: {count}", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # FPS
    if fps_display > 0:
        y += 25
        cv2.putText(annotated, f"  FPS: {fps_display:.1f}", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return annotated


def draw_crossing_flash(
    frame: np.ndarray,
    centroid: Tuple[int, int],
    class_name: str,
) -> np.ndarray:
    """Flash effect saat kendaraan melewati garis."""
    annotated = frame.copy()
    cx, cy = int(centroid[0]), int(centroid[1])

    # Green circle flash
    cv2.circle(annotated, (cx, cy), 25, (0, 255, 0), 3)
    cv2.putText(annotated, f"COUNTED: {class_name}", (cx - 60, cy - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return annotated
