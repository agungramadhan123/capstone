"""
Virtual Line Crossing Counter
===============================
Menghitung kendaraan yang melewati garis virtual.
Menggunakan cross-product untuk deteksi crossing + anti double-count.
"""
import numpy as np
from typing import Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class CrossingEvent:
    """Record of a single line crossing."""
    track_id: int
    class_id: int
    class_name: str
    frame_number: int
    timestamp: float          # seconds
    centroid: Tuple[float, float]
    direction: str            # "down" or "up"


class VirtualLineCounter:
    """
    Menghitung kendaraan yang melewati garis virtual.

    Cara kerja:
    1. Definisikan garis dengan 2 titik
    2. Untuk setiap track, simpan posisi centroid sebelumnya
    3. Cek apakah centroid berpindah sisi terhadap garis
    4. Jika ya → crossing event → count (hanya sekali per track ID)
    """

    def __init__(
        self,
        line_start: Tuple[int, int],
        line_end: Tuple[int, int],
        direction: str = "down",    # "down", "up", "both"
        min_track_length: int = 5,  # Min frames sebelum dihitung
    ):
        self.line_start = np.array(line_start, dtype=float)
        self.line_end = np.array(line_end, dtype=float)
        self.direction = direction
        self.min_track_length = min_track_length

        # State
        self._prev_centroids: Dict[int, np.ndarray] = {}
        self._track_frame_counts: Dict[int, int] = defaultdict(int)
        self._counted_ids: Set[int] = set()
        self._class_counts: Dict[str, int] = defaultdict(int)
        self._total_count: int = 0
        self._events: list = []

    def _side_of_line(self, point: np.ndarray) -> float:
        """
        Tentukan sisi titik terhadap garis.
        Positif = satu sisi, Negatif = sisi lain.
        Cross product: (B-A) × (P-A)
        """
        d = ((point[0] - self.line_start[0]) *
             (self.line_end[1] - self.line_start[1]) -
             (point[1] - self.line_start[1]) *
             (self.line_end[0] - self.line_start[0]))
        return d

    def _get_centroid(self, bbox: np.ndarray) -> np.ndarray:
        """Hitung centroid dari bounding box [x1, y1, x2, y2]."""
        return np.array([
            (bbox[0] + bbox[2]) / 2,
            (bbox[1] + bbox[3]) / 2,
        ])

    def update(
        self,
        track_id: int,
        bbox: np.ndarray,
        class_id: int,
        class_name: str,
        frame_number: int,
        fps: float = 30.0,
    ) -> Optional[CrossingEvent]:
        """
        Update tracker dan cek crossing. Return CrossingEvent jika crossing terjadi.
        """
        centroid = self._get_centroid(bbox)
        self._track_frame_counts[track_id] += 1

        # Skip jika track terlalu pendek (noise filter)
        if self._track_frame_counts[track_id] < self.min_track_length:
            self._prev_centroids[track_id] = centroid
            return None

        # Skip jika sudah pernah dihitung
        if track_id in self._counted_ids:
            self._prev_centroids[track_id] = centroid
            return None

        # Cek crossing
        if track_id in self._prev_centroids:
            prev_side = self._side_of_line(self._prev_centroids[track_id])
            curr_side = self._side_of_line(centroid)

            # Crossing terjadi jika sign berubah
            if prev_side * curr_side < 0:
                # Cek arah
                is_valid_direction = False
                if self.direction == "both":
                    is_valid_direction = True
                elif self.direction == "down" and curr_side > 0:
                    is_valid_direction = True
                elif self.direction == "up" and curr_side < 0:
                    is_valid_direction = True

                if is_valid_direction:
                    # Count!
                    self._counted_ids.add(track_id)
                    self._class_counts[class_name] += 1
                    self._total_count += 1

                    event = CrossingEvent(
                        track_id=track_id,
                        class_id=class_id,
                        class_name=class_name,
                        frame_number=frame_number,
                        timestamp=frame_number / fps,
                        centroid=tuple(centroid),
                        direction="down" if curr_side > 0 else "up",
                    )
                    self._events.append(event)
                    self._prev_centroids[track_id] = centroid
                    return event

        self._prev_centroids[track_id] = centroid
        return None

    @property
    def total_count(self) -> int:
        return self._total_count

    @property
    def class_counts(self) -> Dict[str, int]:
        return dict(self._class_counts)

    @property
    def events(self) -> list:
        return self._events

    def get_summary(self) -> str:
        """Human-readable counting summary."""
        lines = [f"Total: {self._total_count}"]
        for cls_name, count in sorted(self._class_counts.items()):
            lines.append(f"  {cls_name}: {count}")
        return "\n".join(lines)

    def cleanup_lost_tracks(self, active_track_ids: set):
        """Bersihkan data track yang sudah tidak aktif (memory management)."""
        lost = set(self._prev_centroids.keys()) - active_track_ids
        for tid in lost:
            if tid not in self._counted_ids:
                # Belum dihitung — biarkan dulu (mungkin kembali)
                pass
            # Hapus dari prev centroids jika track benar-benar hilang
            if self._track_frame_counts.get(tid, 0) > self.min_track_length + 50:
                self._prev_centroids.pop(tid, None)
                self._track_frame_counts.pop(tid, None)
