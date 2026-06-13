"""
Traffic Logger - Anti-Memory Leak CSV Logging

Logger efisien yang menulis event kendaraan langsung ke CSV.
- Menggunakan csv.writer dengan mode append ('a')
- TIDAK menggunakan Pandas DataFrame dalam loop
- Membersihkan ID lama secara berkala
- Mendukung kolom origin_zone dan destination_zone untuk ROI counting
"""

import os
import csv
from datetime import datetime

from .config import logger


class TrafficLogger:
    """
    Logger efisien yang menulis event kendaraan langsung ke CSV.
    - Menggunakan csv.writer dengan mode append ('a')
    - TIDAK menggunakan Pandas DataFrame dalam loop
    - Membersihkan ID lama secara berkala
    """

    CSV_COLUMNS = [
        "timestamp", "frame_id", "vehicle_id",
        "class_name", "confidence",
        "origin_zone", "destination_zone", "direction"
    ]

    def __init__(self, csv_path: str, flush_interval: int = 100):
        self.csv_path = csv_path
        self.flush_interval = flush_interval

        # Set untuk tracking ID yang sudah dicatat (mencegah duplikasi)
        self._logged_ids = set()
        # Counter untuk flush berkala
        self._write_count = 0

        # Buka file CSV
        self._file = None
        self._writer = None
        self._open_csv()

    def _open_csv(self):
        """Buka atau buat file CSV dengan header."""
        file_exists = os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0

        self._file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)

        if not file_exists:
            self._writer.writerow(self.CSV_COLUMNS)
            self._file.flush()
            logger.info(f"CSV dibuat: {self.csv_path}")
        else:
            logger.info(f"CSV append mode: {self.csv_path}")

    def log_crossing(self, frame_id: int, vehicle_id: int,
                     class_name: str, confidence: float, direction: str,
                     origin_zone: str = "", destination_zone: str = ""):
        """
        Catat event kendaraan melintas zona ke CSV.
        Mencegah duplikasi berdasarkan vehicle_id.

        Args:
            frame_id: nomor frame
            vehicle_id: tracker ID kendaraan
            class_name: nama kelas (Mobil, Motor, dll)
            confidence: confidence score
            direction: arah pergerakan (e.g. "Selatan→Utara")
            origin_zone: zona asal (e.g. "Selatan")
            destination_zone: zona tujuan (e.g. "Utara" atau "Tidak diketahui")
        """
        if vehicle_id in self._logged_ids:
            return  # Sudah pernah dicatat

        self._logged_ids.add(vehicle_id)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [timestamp, frame_id, vehicle_id, class_name,
               f"{confidence:.3f}", origin_zone, destination_zone, direction]

        self._writer.writerow(row)
        self._write_count += 1

        # Flush berkala untuk mencegah data loss
        if self._write_count % self.flush_interval == 0:
            self._file.flush()

    def cleanup_stale_ids(self, active_track_ids: set):
        """
        Bersihkan ID lama dari memori yang sudah tidak dilacak ByteTrack.
        Panggil secara berkala untuk mencegah memory leak pada set.
        """
        stale_ids = self._logged_ids - active_track_ids
        if stale_ids:
            self._logged_ids -= stale_ids
            logger.debug(f"Dibersihkan {len(stale_ids)} ID stale dari memori.")

    @property
    def total_logged(self) -> int:
        return len(self._logged_ids)

    def close(self):
        """Tutup file CSV dengan aman."""
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None
            logger.info(f"CSV ditutup. Total {self.total_logged} event tercatat.")
