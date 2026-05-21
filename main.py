"""
=============================================================
 FASE 2 — CORE MONITORING APPLICATION (PRODUKSI & DEMO)
 
 Vehicle Tracking, Counting & Analytics — Jalan Buah Batu
 Arsitektur: "Antigraviti" (Fail-Safe)
 
 Fitur:
   1. Fail-Safe RTSP → MP4 fallback (anti-crash)
   2. YOLOv8 + ByteTrack dengan occlusion handling
   3. Directional virtual line counting (sv.LineZone)
   4. Anti-memory-leak CSV logging (TrafficLogger)
   5. Premium Visual HUD + trace + dynamic line color
 
 Cara menjalankan:
   python main.py
   python main.py --source video_buahbatu.mp4
   python main.py --source rtsp://user:pass@ip:port/stream
   python main.py --model runs/detect/cctv_bubat/finetune_v1-9/weights/best.pt
   python main.py --show          # tampilkan window OpenCV
   python main.py --save-video    # simpan output ke file
=============================================================
"""

import os
import sys
import cv2
import csv
import time
import numpy as np
import argparse
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from multiprocessing import freeze_support

import supervision as sv
from ultralytics import YOLO

# ── KONFIGURASI DEFAULT ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_MODEL = str(
    PROJECT_ROOT / "runs" / "detect" / "cctv_bubat" / "finetune_v1-9" / "weights" / "best.pt"
)
DEFAULT_RTSP = "rtsp://username:password@ip:port/stream"
DEFAULT_VIDEO = str(PROJECT_ROOT / "video_buahbatu.mp4")
DEFAULT_CSV = str(PROJECT_ROOT / "traffic_logs_buahbatu.csv")
TRACKER_CONFIG = str(PROJECT_ROOT / "custom_bytetrack.yaml")

# Kelas kendaraan (harus sesuai dengan data.yaml yang sudah bersih)
CLASS_NAMES = {0: "Bis", 1: "Mobil", 2: "Motor", 3: "Truk"}
CLASS_EMOJIS = {0: "🚌", 1: "🚗", 2: "🏍", 3: "🚛"}

# Virtual Line — Koordinat default (sesuaikan dengan video Anda)
# Garis horizontal di y=300 pada frame 640x480
LINE_START = sv.Point(x=0, y=300)
LINE_END = sv.Point(x=640, y=300)

# Displacement minimal (piksel) untuk mencegah double-count saat macet
MIN_DISPLACEMENT_PX = 5

# Warna HUD
COLOR_HUD_BG = (30, 30, 30)          # Dark background
COLOR_HUD_TEXT = (255, 255, 255)      # White text
COLOR_HUD_ACCENT = (0, 200, 150)     # Teal accent
COLOR_LINE_DEFAULT = sv.Color(0, 255, 0)   # Hijau
COLOR_LINE_TRIGGER = sv.Color(0, 0, 255)   # Merah

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ══════════════════════════════════════════════════════════════
#  1. VIDEO SOURCE MANAGER — Fail-Safe RTSP → MP4 Fallback
# ══════════════════════════════════════════════════════════════
class VideoSourceManager:
    """
    Mengelola input video dengan mekanisme fail-safe:
    - Coba RTSP stream terlebih dahulu
    - Jika gagal/terputus → otomatis fallback ke video lokal
    - Reconnect berkala ke RTSP setiap N detik
    """

    def __init__(self, rtsp_url: str, fallback_path: str, reconnect_interval: int = 30):
        self.rtsp_url = rtsp_url
        self.fallback_path = fallback_path
        self.reconnect_interval = reconnect_interval

        self._cap = None
        self._using_rtsp = False
        self._last_reconnect_attempt = 0
        self._frame_count = 0
        self._source_label = "NONE"

    @property
    def is_rtsp(self) -> bool:
        return self._using_rtsp

    @property
    def source_label(self) -> str:
        return self._source_label

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def open(self) -> bool:
        """Buka sumber video. Coba RTSP dulu, lalu fallback."""
        # Coba RTSP
        if self._try_open_rtsp():
            return True

        # Fallback ke video lokal
        return self._try_open_fallback()

    def _try_open_rtsp(self) -> bool:
        """Coba membuka RTSP stream."""
        try:
            logger.info(f"Mencoba koneksi RTSP: {self.rtsp_url[:50]}...")
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

            # Set timeout pendek untuk RTSP
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self._release_current()
                    self._cap = cap
                    self._using_rtsp = True
                    self._source_label = "RTSP LIVE"
                    logger.info("✅ RTSP stream berhasil terhubung!")
                    return True
                else:
                    cap.release()
                    logger.warning("⚠️ RTSP terbuka tapi tidak bisa membaca frame.")
            else:
                logger.warning("⚠️ RTSP gagal dibuka.")

        except Exception as e:
            logger.warning(f"⚠️ Exception saat koneksi RTSP: {e}")

        self._last_reconnect_attempt = time.time()
        return False

    def _try_open_fallback(self) -> bool:
        """Buka video lokal sebagai fallback."""
        if not os.path.exists(self.fallback_path):
            logger.error(f"❌ Video fallback tidak ditemukan: {self.fallback_path}")
            return False

        try:
            cap = cv2.VideoCapture(self.fallback_path)
            if cap.isOpened():
                self._release_current()
                self._cap = cap
                self._using_rtsp = False
                self._source_label = f"LOCAL ({Path(self.fallback_path).name})"
                logger.info(f"✅ Fallback ke video lokal: {self.fallback_path}")
                return True
        except Exception as e:
            logger.error(f"❌ Gagal membuka video fallback: {e}")

        return False

    def read(self):
        """
        Baca frame berikutnya. Jika gagal, coba reconnect atau fallback.
        Returns: (success: bool, frame: np.ndarray | None)
        """
        if self._cap is None:
            return False, None

        try:
            ret, frame = self._cap.read()

            if ret and frame is not None:
                self._frame_count += 1

                # Periodically coba reconnect ke RTSP jika sedang di fallback
                if not self._using_rtsp:
                    elapsed = time.time() - self._last_reconnect_attempt
                    if elapsed >= self.reconnect_interval:
                        self._try_rtsp_reconnect()

                return True, frame

            else:
                # Frame gagal dibaca
                if self._using_rtsp:
                    # RTSP terputus mid-stream → switch ke fallback
                    logger.warning("⚠️ RTSP stream terputus! Switching ke fallback...")
                    self._last_reconnect_attempt = time.time()
                    if self._try_open_fallback():
                        return self.read()  # Baca frame pertama dari fallback
                    return False, None
                else:
                    # Video lokal habis
                    logger.info("📹 Video lokal selesai (end of file).")
                    return False, None

        except Exception as e:
            logger.error(f"❌ Exception saat membaca frame: {e}")
            if self._using_rtsp:
                logger.warning("Switching ke fallback...")
                if self._try_open_fallback():
                    return self.read()
            return False, None

    def _try_rtsp_reconnect(self):
        """Coba reconnect ke RTSP stream di background."""
        self._last_reconnect_attempt = time.time()
        logger.info("🔄 Mencoba reconnect ke RTSP...")
        if self._try_open_rtsp():
            logger.info("✅ Reconnect RTSP berhasil! Beralih ke live stream.")

    def get_fps(self) -> float:
        """Ambil FPS dari sumber video."""
        if self._cap:
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            return fps if fps > 0 else 30.0
        return 30.0

    def get_frame_size(self) -> tuple:
        """Ambil ukuran frame (width, height)."""
        if self._cap:
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        return (640, 480)

    def _release_current(self):
        """Release VideoCapture yang sedang aktif."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def release(self):
        """Release semua resource."""
        self._release_current()
        logger.info("📹 Video source ditutup.")


# ══════════════════════════════════════════════════════════════
#  2. TRAFFIC LOGGER — Anti-Memory Leak CSV Logging
# ══════════════════════════════════════════════════════════════
class TrafficLogger:
    """
    Logger efisien yang menulis event kendaraan langsung ke CSV.
    - Menggunakan csv.writer dengan mode append ('a')
    - TIDAK menggunakan Pandas DataFrame dalam loop
    - Membersihkan ID lama secara berkala
    """

    CSV_COLUMNS = [
        "timestamp", "frame_id", "vehicle_id",
        "class_name", "confidence", "direction"
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
            logger.info(f"📄 CSV dibuat: {self.csv_path}")
        else:
            logger.info(f"📄 CSV append mode: {self.csv_path}")

    def log_crossing(self, frame_id: int, vehicle_id: int,
                     class_name: str, confidence: float, direction: str):
        """
        Catat event kendaraan melintas garis ke CSV.
        Mencegah duplikasi berdasarkan vehicle_id.
        """
        if vehicle_id in self._logged_ids:
            return  # Sudah pernah dicatat

        self._logged_ids.add(vehicle_id)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [timestamp, frame_id, vehicle_id, class_name,
               f"{confidence:.3f}", direction]

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
            logger.debug(f"🧹 Dibersihkan {len(stale_ids)} ID stale dari memori.")

    @property
    def total_logged(self) -> int:
        return len(self._logged_ids)

    def close(self):
        """Tutup file CSV dengan aman."""
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None
            logger.info(f"📄 CSV ditutup. Total {self.total_logged} event tercatat.")


# ══════════════════════════════════════════════════════════════
#  3. DIRECTIONAL COUNTER — Virtual Line Crossing
# ══════════════════════════════════════════════════════════════
class DirectionalCounter:
    """
    Penghitung kendaraan dengan arah menggunakan sv.LineZone.
    Melacak displacement untuk mencegah double-count saat macet.
    """

    def __init__(self, line_start: sv.Point, line_end: sv.Point,
                 class_names: dict, min_displacement: int = MIN_DISPLACEMENT_PX):
        self.class_names = class_names
        self.min_displacement = min_displacement

        # sv.LineZone untuk counting
        self.line_zone = sv.LineZone(
            start=line_start,
            end=line_end,
        )

        # Akumulasi total per kelas
        self.counts_in = defaultdict(int)   # Masuk (Selatan → Utara)
        self.counts_out = defaultdict(int)  # Keluar (Utara → Selatan)

        # Tracking posisi terakhir per ID untuk displacement check
        self._last_positions = {}

        # Flash state untuk efek visual garis
        self._flash_frames_remaining = 0

    def update(self, detections: sv.Detections, frame_id: int) -> dict:
        """
        Update counter dengan deteksi terbaru.
        Returns: dict of newly crossed vehicles
        """
        new_crossings = {}

        # Cek displacement sebelum counting
        if detections.tracker_id is not None:
            valid_mask = self._check_displacement(detections)
            # Filter deteksi yang bergerak cukup signifikan
            filtered = detections[valid_mask]
        else:
            filtered = detections

        # Update LineZone crossing
        crossed_in, crossed_out = self.line_zone.trigger(detections=filtered)

        # Proses crossing masuk (Selatan → Utara)
        if crossed_in.any():
            for i, crossed in enumerate(crossed_in):
                if crossed and filtered.tracker_id is not None:
                    tracker_id = int(filtered.tracker_id[i])
                    class_id = int(filtered.class_id[i])
                    confidence = float(filtered.confidence[i])
                    class_name = self.class_names.get(class_id, f"class_{class_id}")

                    self.counts_in[class_name] += 1
                    new_crossings[tracker_id] = {
                        "class_name": class_name,
                        "confidence": confidence,
                        "direction": "Selatan→Utara",
                    }

            self._flash_frames_remaining = 2  # Trigger flash effect

        # Proses crossing keluar (Utara → Selatan)
        if crossed_out.any():
            for i, crossed in enumerate(crossed_out):
                if crossed and filtered.tracker_id is not None:
                    tracker_id = int(filtered.tracker_id[i])
                    class_id = int(filtered.class_id[i])
                    confidence = float(filtered.confidence[i])
                    class_name = self.class_names.get(class_id, f"class_{class_id}")

                    self.counts_out[class_name] += 1
                    new_crossings[tracker_id] = {
                        "class_name": class_name,
                        "confidence": confidence,
                        "direction": "Utara→Selatan",
                    }

            self._flash_frames_remaining = 2  # Trigger flash effect

        return new_crossings

    def _check_displacement(self, detections: sv.Detections) -> np.ndarray:
        """
        Cek apakah setiap objek sudah bergerak cukup jauh.
        Returns: boolean mask (True = cukup bergerak)
        """
        mask = np.ones(len(detections), dtype=bool)

        if detections.tracker_id is None:
            return mask

        for i, tracker_id in enumerate(detections.tracker_id):
            tid = int(tracker_id)
            # Hitung pusat bounding box
            x1, y1, x2, y2 = detections.xyxy[i]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            if tid in self._last_positions:
                last_cx, last_cy = self._last_positions[tid]
                displacement = np.sqrt((cx - last_cx) ** 2 + (cy - last_cy) ** 2)

                if displacement < self.min_displacement:
                    mask[i] = False  # Tidak cukup bergerak → jangan count

            # Update posisi terakhir
            self._last_positions[tid] = (cx, cy)

        return mask

    def cleanup_positions(self, active_ids: set):
        """Bersihkan posisi ID yang sudah tidak aktif."""
        stale = set(self._last_positions.keys()) - active_ids
        for sid in stale:
            del self._last_positions[sid]

    @property
    def should_flash(self) -> bool:
        """Apakah garis harus berkedip merah."""
        if self._flash_frames_remaining > 0:
            self._flash_frames_remaining -= 1
            return True
        return False

    @property
    def total_counts(self) -> dict:
        """Total akumulasi per kelas (in + out)."""
        totals = defaultdict(int)
        for cls, cnt in self.counts_in.items():
            totals[cls] += cnt
        for cls, cnt in self.counts_out.items():
            totals[cls] += cnt
        return dict(totals)


# ══════════════════════════════════════════════════════════════
#  4. VISUAL HUD — Premium Overlay
# ══════════════════════════════════════════════════════════════
class VisualHUD:
    """
    Head-Up Display premium untuk visualisasi monitoring.
    Menampilkan:
    - Papan akumulasi total kendaraan per kelas
    - Status sumber video
    - FPS counter
    """

    # Warna per kelas (BGR)
    CLASS_COLORS = {
        "Bis":   (255, 165, 0),    # Orange
        "Mobil": (0, 200, 100),    # Green
        "Motor": (255, 100, 100),  # Light blue
        "Truk":  (0, 100, 255),    # Red-orange
    }

    def __init__(self):
        self._fps_history = []
        self._max_fps_samples = 30

    def draw(self, frame: np.ndarray, counts: dict, source_label: str,
             frame_id: int, fps: float) -> np.ndarray:
        """Gambar seluruh HUD overlay pada frame."""
        result = frame.copy()

        # Update FPS history
        self._fps_history.append(fps)
        if len(self._fps_history) > self._max_fps_samples:
            self._fps_history.pop(0)
        avg_fps = sum(self._fps_history) / len(self._fps_history)

        # ── Panel kiri atas: Tabel kendaraan ──
        result = self._draw_vehicle_panel(result, counts)

        # ── Panel kanan atas: Info status ──
        result = self._draw_status_panel(result, source_label, frame_id, avg_fps)

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


# ══════════════════════════════════════════════════════════════
#  5. MAIN APPLICATION — Pipeline Utama
# ══════════════════════════════════════════════════════════════
class TrafficMonitorApp:
    """
    Aplikasi utama monitoring lalu lintas.
    Mengorkestrasi seluruh komponen: video source, model, tracker,
    counter, logger, dan visualisasi.
    """

    def __init__(self, args):
        self.args = args
        self._setup_components()

    def _setup_components(self):
        """Inisialisasi seluruh komponen."""
        args = self.args

        # ── 1. Video Source ──
        rtsp_url = args.source if args.source.startswith("rtsp://") else DEFAULT_RTSP
        fallback = args.source if not args.source.startswith("rtsp://") else DEFAULT_VIDEO

        self.video_source = VideoSourceManager(
            rtsp_url=rtsp_url,
            fallback_path=fallback,
            reconnect_interval=30,
        )

        # ── 2. YOLO Model ──
        logger.info(f"Loading model: {args.model}")
        self.model = YOLO(args.model)

        # ── 3. Traffic Logger ──
        self.traffic_logger = TrafficLogger(
            csv_path=args.csv_output,
            flush_interval=100,
        )

        # ── 4. Directional Counter ──
        self.counter = DirectionalCounter(
            line_start=LINE_START,
            line_end=LINE_END,
            class_names=CLASS_NAMES,
            min_displacement=MIN_DISPLACEMENT_PX,
        )

        # ── 5. Visual HUD ──
        self.hud = VisualHUD()

        # ── 6. Supervision Annotators ──
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
        )
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.4,
            text_thickness=1,
            text_padding=4,
        )
        self.trace_annotator = sv.TraceAnnotator(
            thickness=2,
            trace_length=50,
        )
        self.line_annotator = sv.LineZoneAnnotator(
            thickness=2,
            text_scale=0.5,
            text_thickness=1,
        )

        # ── 7. Output video writer ──
        self.video_writer = None

        # ── 8. Cleanup interval ──
        self._cleanup_interval = 300  # Setiap 300 frame
        self._frame_times = []

    def run(self):
        """Jalankan pipeline monitoring utama."""
        logger.info("🚀 Memulai Traffic Monitor — Jalan Buah Batu")

        # Buka video source
        if not self.video_source.open():
            logger.error("❌ Tidak bisa membuka sumber video! Pastikan RTSP atau file MP4 tersedia.")
            return

        fps = self.video_source.get_fps()
        frame_w, frame_h = self.video_source.get_frame_size()
        logger.info(f"📹 Video: {frame_w}x{frame_h} @ {fps:.1f} FPS")

        # Setup video writer jika diminta
        if self.args.save_video:
            output_path = str(PROJECT_ROOT / "output_monitoring.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
            logger.info(f"💾 Output video: {output_path}")

        try:
            self._processing_loop(fps)
        except KeyboardInterrupt:
            logger.info("\n⏹️ Dihentikan oleh pengguna (Ctrl+C)")
        finally:
            self._cleanup()

    def _processing_loop(self, source_fps: float):
        """Loop pemrosesan frame utama."""
        frame_id = 0

        while True:
            loop_start = time.time()

            # ── Baca frame ──
            success, frame = self.video_source.read()
            if not success:
                break

            frame_id += 1

            # ── Deteksi + Tracking ──
            results = self.model.track(
                source=frame,
                persist=True,
                tracker=TRACKER_CONFIG,
                conf=0.25,
                iou=0.5,
                verbose=False,
            )

            # Konversi hasil ke sv.Detections
            detections = sv.Detections.from_ultralytics(results[0])

            # ── Update counter (line crossing) ──
            new_crossings = self.counter.update(detections, frame_id)

            # ── Log crossing events ke CSV ──
            for vehicle_id, info in new_crossings.items():
                self.traffic_logger.log_crossing(
                    frame_id=frame_id,
                    vehicle_id=vehicle_id,
                    class_name=info["class_name"],
                    confidence=info["confidence"],
                    direction=info["direction"],
                )

            # ── Cleanup berkala (anti-memory leak) ──
            if frame_id % self._cleanup_interval == 0:
                active_ids = set()
                if detections.tracker_id is not None:
                    active_ids = set(int(tid) for tid in detections.tracker_id)
                self.traffic_logger.cleanup_stale_ids(active_ids)
                self.counter.cleanup_positions(active_ids)

            # ── Visualisasi ──
            annotated = self._annotate_frame(frame, detections, frame_id, loop_start)

            # ── Output ──
            if self.args.show:
                cv2.imshow("Traffic Monitor - Buah Batu", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 'q' atau ESC
                    logger.info("⏹️ Keluar (tekan 'q')")
                    break

            if self.video_writer:
                self.video_writer.write(annotated)

            # ── Logging periodik ──
            if frame_id % 500 == 0:
                counts = self.counter.total_counts
                total = sum(counts.values())
                logger.info(
                    f"Frame {frame_id} | Total: {total} | "
                    f"Bis:{counts.get('Bis',0)} Mobil:{counts.get('Mobil',0)} "
                    f"Motor:{counts.get('Motor',0)} Truk:{counts.get('Truk',0)}"
                )

    def _annotate_frame(self, frame: np.ndarray, detections: sv.Detections,
                        frame_id: int, loop_start: float) -> np.ndarray:
        """Buat frame teranotasi dengan semua visual overlay."""
        annotated = frame.copy()

        # ── 1. Trace annotator (ekor pergerakan) ──
        annotated = self.trace_annotator.annotate(
            scene=annotated,
            detections=detections,
        )

        # ── 2. Bounding box ──
        annotated = self.box_annotator.annotate(
            scene=annotated,
            detections=detections,
        )

        # ── 3. Label ──
        labels = []
        if detections.tracker_id is not None:
            for i in range(len(detections)):
                cls_id = int(detections.class_id[i])
                conf = float(detections.confidence[i])
                tid = int(detections.tracker_id[i])
                cls_name = CLASS_NAMES.get(cls_id, "?")
                labels.append(f"#{tid} {cls_name} {conf:.2f}")
        else:
            for i in range(len(detections)):
                cls_id = int(detections.class_id[i])
                conf = float(detections.confidence[i])
                cls_name = CLASS_NAMES.get(cls_id, "?")
                labels.append(f"{cls_name} {conf:.2f}")

        annotated = self.label_annotator.annotate(
            scene=annotated,
            detections=detections,
            labels=labels,
        )

        # ── 4. Virtual line (dengan efek warna dinamis) ──
        # Efek flash: merah saat ada crossing, hijau default
        if self.counter.should_flash:
            # Ubah warna line zone ke merah sementara
            self.line_annotator = sv.LineZoneAnnotator(
                thickness=3,
                text_scale=0.5,
                text_thickness=1,
                color=COLOR_LINE_TRIGGER,
            )
        else:
            self.line_annotator = sv.LineZoneAnnotator(
                thickness=2,
                text_scale=0.5,
                text_thickness=1,
                color=COLOR_LINE_DEFAULT,
            )

        annotated = self.line_annotator.annotate(
            frame=annotated,
            line_counter=self.counter.line_zone,
        )

        # ── 5. HUD overlay ──
        elapsed = time.time() - loop_start
        fps = 1.0 / elapsed if elapsed > 0 else 0
        counts = self.counter.total_counts

        annotated = self.hud.draw(
            frame=annotated,
            counts=counts,
            source_label=self.video_source.source_label,
            frame_id=frame_id,
            fps=fps,
        )

        return annotated

    def _cleanup(self):
        """Bersihkan semua resource."""
        self.video_source.release()
        self.traffic_logger.close()

        if self.video_writer:
            self.video_writer.release()

        cv2.destroyAllWindows()

        # Print ringkasan akhir
        counts = self.counter.total_counts
        total = sum(counts.values())
        print(f"\n{'='*55}")
        print(f"  📊 RINGKASAN MONITORING")
        print(f"{'='*55}")
        print(f"  Total kendaraan melintas: {total}")
        for cls_name, cnt in sorted(counts.items()):
            emoji = CLASS_EMOJIS.get(
                next((k for k, v in CLASS_NAMES.items() if v == cls_name), -1), ""
            )
            print(f"    {emoji} {cls_name:8s}: {cnt}")
        print(f"  Frame diproses: {self.video_source.frame_count}")
        print(f"  CSV output: {self.args.csv_output}")
        print(f"{'='*55}\n")


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="Traffic Monitor — Jalan Buah Batu, Bandung",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  python main.py "Dishub Kota Bandung.mp4" --show
  python main.py --source "video dengan spasi.mp4" --show
  python main.py rtsp://user:pass@ip:port/stream --show
  python main.py video.mp4 --save-video
        """,
    )
    # Positional arg: file yang di-drag-drop atau diketik langsung
    # nargs='*' agar nama file dengan spasi (tanpa tanda kutip) tetap terbaca
    parser.add_argument(
        "source_positional", nargs="*", default=None,
        help="Sumber video (drag-and-drop atau ketik path langsung)"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help=f"Sumber video: path ke .mp4 atau URL RTSP (default: {DEFAULT_VIDEO})"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Path ke model YOLO (.pt)"
    )
    parser.add_argument(
        "--csv-output", type=str, default=DEFAULT_CSV,
        help=f"Path output CSV log (default: {DEFAULT_CSV})"
    )
    parser.add_argument(
        "--show", action="store_true", default=True,
        help="Tampilkan jendela OpenCV (default: True)"
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Jangan tampilkan jendela OpenCV"
    )
    parser.add_argument(
        "--save-video", action="store_true",
        help="Simpan video output ke file"
    )

    args = parser.parse_args()

    # Resolve sumber video: prioritas --source > positional > default
    if args.source:
        pass  # Gunakan --source langsung
    elif args.source_positional:
        # Gabungkan token positional (handle nama file dengan spasi tanpa kutip)
        args.source = " ".join(args.source_positional)
    else:
        args.source = DEFAULT_VIDEO

    # Bersihkan path: hapus kutip ganda jika ada (dari drag-drop Windows)
    args.source = args.source.strip().strip('"').strip("'")

    if args.no_show:
        args.show = False

    return args


def main():
    args = parse_args()
    app = TrafficMonitorApp(args)
    app.run()


if __name__ == "__main__":
    freeze_support()
    main()
