"""
Visual HUD - Premium Overlay

Head-Up Display premium untuk visualisasi monitoring.
Menampilkan:
- Papan akumulasi total kendaraan per kelas
- Papan arah pergerakan (Origin-Destination)
- Status sumber video
- FPS counter
"""

import numpy as np
import cv2

from .config import CLASS_NAMES, COLOR_HUD_BG, COLOR_HUD_ACCENT


class VisualHUD:
    """
    Head-Up Display premium untuk visualisasi monitoring.
    Menampilkan:
    - Papan akumulasi total kendaraan per kelas (kiri atas)
    - Papan arah pergerakan / OD matrix (kiri bawah)
    - Status sumber video + FPS (kanan atas)
    """

    # Warna per kelas (BGR)
    CLASS_COLORS = {
        "Bis":   (255, 165, 0),    # Orange
        "Mobil": (0, 200, 100),    # Green
        "Motor": (255, 100, 100),  # Light blue
        "Truk":  (0, 100, 255),    # Red-orange
    }

    # Warna per arah untuk panel direction (BGR)
    DIRECTION_COLORS = {
        "Utara":   (0, 255, 0),
        "Selatan": (0, 0, 255),
        "Timur":   (255, 0, 0),
        "Barat":   (0, 255, 255),
    }

    def __init__(self):
        self._fps_history = []
        self._max_fps_samples = 30

    def draw(self, frame: np.ndarray, counts: dict, source_label: str,
             frame_id: int, fps: float,
             direction_counts: dict = None) -> np.ndarray:
        """
        Gambar seluruh HUD overlay pada frame.

        Args:
            frame: frame BGR
            counts: total kendaraan per kelas {"Mobil": 10, ...}
            source_label: label sumber video
            frame_id: nomor frame
            fps: FPS saat ini
            direction_counts: total per arah {"Selatan→Utara": 15, ...}
        """
        result = frame.copy()

        # Update FPS history
        self._fps_history.append(fps)
        if len(self._fps_history) > self._max_fps_samples:
            self._fps_history.pop(0)
        avg_fps = sum(self._fps_history) / len(self._fps_history)

        # Panel kiri atas: Tabel kendaraan
        result = self._draw_vehicle_panel(result, counts)

        # Panel kanan atas: Info status
        result = self._draw_status_panel(result, source_label, frame_id, avg_fps)

        # Panel kiri bawah: Arah pergerakan (jika tersedia)
        if direction_counts:
            result = self._draw_direction_panel(result, direction_counts)

        return result

    def _draw_vehicle_panel(self, frame: np.ndarray, counts: dict) -> np.ndarray:
        """Gambar panel akumulasi kendaraan di pojok kiri atas."""
        h, w = frame.shape[:2]

        # Ukuran panel
        panel_w = 220
        panel_h = 30 + len(CLASS_NAMES) * 35 + 10
        margin = 10

        # Background semi-transparan
        overlay = frame.copy()
        x1, y1 = margin, margin
        x2, y2 = margin + panel_w, margin + panel_h

        # Rounded rectangle effect (drawn as rectangle + gradient)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_HUD_BG, -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

        # Border tipis
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_HUD_ACCENT, 1)

        # Header
        cv2.putText(frame, "VEHICLE COUNT",
                    (x1 + 12, y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    COLOR_HUD_ACCENT, 1, cv2.LINE_AA)

        # Garis separator
        cv2.line(frame, (x1 + 8, y1 + 30), (x2 - 8, y1 + 30),
                 COLOR_HUD_ACCENT, 1)

        # Kelas kendaraan
        y_offset = y1 + 55
        for cls_id, cls_name in CLASS_NAMES.items():
            count = counts.get(cls_name, 0)
            color = self.CLASS_COLORS.get(cls_name, (200, 200, 200))

            # Bullet/dot berwarna
            cv2.circle(frame, (x1 + 20, y_offset - 5), 6, color, -1)

            # Nama kelas
            cv2.putText(frame, cls_name,
                        (x1 + 35, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                        (220, 220, 220), 1, cv2.LINE_AA)

            # Jumlah (rata kanan)
            count_text = str(count)
            (tw, _), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.putText(frame, count_text,
                        (x2 - tw - 15, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)

            y_offset += 35

        return frame

    def _draw_direction_panel(self, frame: np.ndarray,
                              direction_counts: dict) -> np.ndarray:
        """Gambar panel arah pergerakan di pojok kiri bawah."""
        h, w = frame.shape[:2]

        if not direction_counts:
            return frame

        # Hitung ukuran panel berdasarkan jumlah arah
        num_items = len(direction_counts)
        panel_w = 260
        panel_h = 30 + num_items * 28 + 10
        margin = 10

        x1 = margin
        y1 = h - panel_h - margin
        x2 = x1 + panel_w
        y2 = h - margin

        # Background semi-transparan
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_HUD_BG, -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

        # Border
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_HUD_ACCENT, 1)

        # Header
        cv2.putText(frame, "ARAH PERGERAKAN",
                    (x1 + 12, y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    COLOR_HUD_ACCENT, 1, cv2.LINE_AA)

        # Separator
        cv2.line(frame, (x1 + 8, y1 + 30), (x2 - 8, y1 + 30),
                 COLOR_HUD_ACCENT, 1)

        # Arah pergerakan (diurutkan berdasarkan jumlah, terbanyak di atas)
        sorted_dirs = sorted(direction_counts.items(), key=lambda x: x[1],
                             reverse=True)

        y_offset = y1 + 50
        for direction, count in sorted_dirs:
            # Tentukan warna bullet berdasarkan zona tujuan/asal
            dot_color = (200, 200, 200)  # Default abu-abu
            for zone_name, zone_color in self.DIRECTION_COLORS.items():
                if zone_name in direction:
                    dot_color = zone_color
                    break

            # Bullet berwarna
            cv2.circle(frame, (x1 + 16, y_offset - 4), 4, dot_color, -1)

            # Nama arah (potong jika terlalu panjang)
            dir_text = direction
            if len(dir_text) > 20:
                dir_text = dir_text[:18] + ".."
            cv2.putText(frame, dir_text,
                        (x1 + 28, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (210, 210, 210), 1, cv2.LINE_AA)

            # Jumlah (rata kanan)
            count_text = str(count)
            (tw, _), _ = cv2.getTextSize(count_text,
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
            cv2.putText(frame, count_text,
                        (x2 - tw - 12, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 2, cv2.LINE_AA)

            y_offset += 28

        return frame

    def _draw_status_panel(self, frame: np.ndarray, source_label: str,
                           frame_id: int, fps: float) -> np.ndarray:
        """Gambar panel status di pojok kanan atas."""
        h, w = frame.shape[:2]

        panel_w = 230
        panel_h = 75
        margin = 10

        overlay = frame.copy()
        x1 = w - panel_w - margin
        y1 = margin
        x2 = w - margin
        y2 = margin + panel_h

        cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_HUD_BG, -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_HUD_ACCENT, 1)

        # Source
        src_color = (0, 255, 100) if "RTSP" in source_label else (100, 200, 255)
        cv2.putText(frame, source_label,
                    (x1 + 10, y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    src_color, 1, cv2.LINE_AA)

        # Frame counter
        cv2.putText(frame, f"Frame: {frame_id}",
                    (x1 + 10, y1 + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (180, 180, 180), 1, cv2.LINE_AA)

        # FPS
        fps_color = (0, 255, 0) if fps >= 25 else (0, 200, 255) if fps >= 15 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (x1 + 10, y1 + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    fps_color, 1, cv2.LINE_AA)

        return frame
