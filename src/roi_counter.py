"""
Single Polygon Counter - Penghitung Kendaraan Berbasis Satu Trapesium Dinamis

Sistem ini menggunakan satu area Trapesium (Trapezoid) yang otomatis di-generate
di tengah layar untuk menyesuaikan perspektif kamera (Vanishing Point).
Kendaraan dihitung berdasarkan arah perlintasannya (misal: Atas -> Bawah).
"""

import cv2
import numpy as np
from collections import defaultdict

from .config import MIN_DISPLACEMENT_PX, logger

class ROICounter:
    """
    Penghitung kendaraan berbasis satu trapesium dinamis di tengah layar.
    Melacak arah masuk dan keluar kendaraan dari area tersebut.
    """

    STATIONARY_THRESHOLD_FRAMES = 90  # ~3 detik di 30fps

    def __init__(self, class_names: dict, frame_size: tuple):
        self.class_names = class_names
        self.frame_w, self.frame_h = frame_size
        
        # 1. Tentukan ukuran dasar (Garis Poligon Membentang Ujung ke Ujung)
        # Diposisikan 80% dari atas (sehingga dekat dengan frame bawah)
        center_y = int(self.frame_h * 0.65)
        
        # Poligon dibuat dengan jarak (margin) dari tepi kiri dan kanan agar tidak menabrak UI
        margin_x = int(self.frame_w * 0.15)  # Margin 15% di kiri dan kanan
        setengah_tinggi = 30 
        
        p1 = [margin_x, center_y - setengah_tinggi]                             # Kiri Atas
        p2 = [self.frame_w - margin_x, center_y - setengah_tinggi]              # Kanan Atas
        p3 = [self.frame_w - margin_x, center_y + setengah_tinggi]              # Kanan Bawah
        p4 = [margin_x, center_y + setengah_tinggi]                             # Kiri Bawah

        self.polygon = np.array([p1, p2, p3, p4], dtype=np.int32)
        
        self.polygon_contour = self.polygon.reshape((-1, 1, 2)).astype(np.float32)

        # ── State Tracking ──
        self._vehicle_history = {}
        self.trip_counts = defaultdict(lambda: defaultdict(int))
        self.single_zone_counts = defaultdict(lambda: defaultdict(int))
        self._counted_ids = set()
        self._flash_frames_remaining = 0

    def _get_anchor(self, xyxy) -> tuple:
        """Titik tumpu bawah-tengah (roda kendaraan)."""
        x1, y1, x2, y2 = xyxy
        return (float(x1 + x2) / 2.0, float(y2))

    def _is_inside(self, anchor: tuple) -> bool:
        """Cek apakah titik tumpu berada di dalam trapesium."""
        return cv2.pointPolygonTest(self.polygon_contour, anchor, False) >= 0

    def _get_relative_side(self, anchor: tuple) -> str:
        """Tentukan posisi relatif anchor terhadap kotak trapesium."""
        cx, cy = anchor
        center_x = self.frame_w // 2
        center_y = int(self.frame_h * 0.8)
        
        dx = cx - center_x
        dy = cy - center_y
        
        if abs(dx) > abs(dy):
            return "Kanan" if dx > 0 else "Kiri"
        else:
            return "Bawah" if dy > 0 else "Atas"

    def _calc_displacement(self, p1: tuple, p2: tuple) -> float:
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return (dx * dx + dy * dy) ** 0.5

    def update(self, detections, frame_id: int) -> dict:
        new_crossings = {}
        if detections.tracker_id is None:
            return new_crossings

        for i in range(len(detections)):
            tracker_id = int(detections.tracker_id[i])
            if tracker_id in self._counted_ids:
                continue

            bbox = detections.xyxy[i]
            anchor = self._get_anchor(bbox)
            class_id = int(detections.class_id[i])
            confidence = float(detections.confidence[i])
            class_name = self.class_names.get(class_id, f"class_{class_id}")

            is_inside = self._is_inside(anchor)

            if tracker_id not in self._vehicle_history:
                # Kendaraan baru terdeteksi
                self._vehicle_history[tracker_id] = {
                    "entry_side": self._get_relative_side(anchor) if is_inside else None,
                    "is_inside": is_inside,
                    "last_anchor": anchor,
                    "class_name": class_name,
                    "confidence": confidence,
                    "stationary_frames": 0,
                }
            else:
                history = self._vehicle_history[tracker_id]
                prev_anchor = history["last_anchor"]

                # Cek pergerakan (anti-mangkir)
                displacement = self._calc_displacement(anchor, prev_anchor)
                if displacement < MIN_DISPLACEMENT_PX:
                    history["stationary_frames"] += 1
                else:
                    history["stationary_frames"] = 0

                history["last_anchor"] = anchor
                history["confidence"] = max(history["confidence"], confidence)

                if history["stationary_frames"] >= self.STATIONARY_THRESHOLD_FRAMES:
                    continue

                was_inside = history["is_inside"]
                history["is_inside"] = is_inside

                # Transisi Masuk: dari luar ke dalam
                if not was_inside and is_inside:
                    history["entry_side"] = self._get_relative_side(prev_anchor)

                # Transisi Keluar: dari dalam ke luar
                elif was_inside and not is_inside:
                    if history["entry_side"] is not None:
                        exit_side = self._get_relative_side(anchor)
                        
                        # Trip selesai (misal: Atas -> Bawah)
                        direction = f"{history['entry_side']} -> {exit_side}"
                        
                        self.trip_counts[direction][class_name] += 1
                        self._counted_ids.add(tracker_id)
                        self._flash_frames_remaining = 3

                        new_crossings[tracker_id] = {
                            "class_name": class_name,
                            "confidence": history["confidence"],
                            "direction": direction,
                            "origin_zone": history["entry_side"],
                            "destination_zone": exit_side,
                        }

        return new_crossings

    def cleanup_positions(self, active_ids: set) -> dict:
        single_zone_events = {}
        stale_ids = set(self._vehicle_history.keys()) - active_ids

        for tid in stale_ids:
            if tid in self._counted_ids:
                del self._vehicle_history[tid]
                continue

            history = self._vehicle_history[tid]
            if history["entry_side"] is not None and history["is_inside"]:
                # Kendaraan terhenti/hilang di dalam trapesium tanpa pernah keluar
                entry = history["entry_side"]
                class_name = history["class_name"]
                label = f"Masuk dari {entry} (Berhenti)"
                
                self.single_zone_counts[label][class_name] += 1
                self._counted_ids.add(tid)

                single_zone_events[tid] = {
                    "class_name": class_name,
                    "confidence": history["confidence"],
                    "direction": label,
                    "origin_zone": entry,
                    "destination_zone": "Area Deteksi",
                }

            del self._vehicle_history[tid]

        # Bersihkan tracker id dari memori counted_ids
        stale_counted = self._counted_ids - active_ids - set(self._vehicle_history.keys())
        if stale_counted:
            self._counted_ids -= stale_counted

        return single_zone_events

    @property
    def should_flash(self) -> bool:
        if self._flash_frames_remaining > 0:
            self._flash_frames_remaining -= 1
            return True
        return False

    @property
    def total_counts(self) -> dict:
        totals = defaultdict(int)
        for direction_counts in self.trip_counts.values():
            for cls_name, cnt in direction_counts.items():
                totals[cls_name] += cnt
        for direction_counts in self.single_zone_counts.values():
            for cls_name, cnt in direction_counts.items():
                totals[cls_name] += cnt
        return dict(totals)

    @property
    def direction_counts(self) -> dict:
        result = {}
        for direction, class_counts in self.trip_counts.items():
            result[direction] = sum(class_counts.values())
        for direction, class_counts in self.single_zone_counts.items():
            result[direction] = sum(class_counts.values())
        return result

    def draw_zones(self, frame: np.ndarray, alpha: float = 0.25) -> np.ndarray:
        """Menggambar area deteksi di tengah layar beserta namanya."""
        overlay = frame.copy()
        
        color = (0, 255, 0)
        cv2.fillPoly(overlay, [self.polygon], color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        is_flashing = self.should_flash
        border_color = (255, 255, 255) if is_flashing else color
        border_thickness = 3 if is_flashing else 2
        
        cv2.polylines(frame, [self.polygon], True, border_color, border_thickness, cv2.LINE_AA)

        # (Label dinonaktifkan atas permintaan)

        return frame
