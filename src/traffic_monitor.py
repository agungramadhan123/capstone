"""
Traffic Monitor App - Pipeline Utama (Orchestrator)

Aplikasi utama monitoring lalu lintas.
Mengorkestrasi seluruh komponen: video source, model, tracker,
ROI counter, logger, dan visualisasi.
"""

import time

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from .config import (
    PROJECT_ROOT, DEFAULT_VIDEO, DEFAULT_ROI_CONFIG, TRACKER_CONFIG,
    CLASS_NAMES, MIN_DISPLACEMENT_PX,
    logger,
)
from .video_source import VideoSourceManager
from .traffic_logger import TrafficLogger
from .roi_counter import ROICounter
from .visual_hud import VisualHUD


class TrafficMonitorApp:
    """
    Aplikasi utama monitoring lalu lintas.
    Mengorkestrasi seluruh komponen: video source, model, tracker,
    ROI counter, logger, dan visualisasi.
    """

    def __init__(self, args, frame_callback=None):
        self.args = args
        self.frame_callback = frame_callback
        self.is_running = True
        self._setup_components()

    def stop(self):
        """Hentikan pemrosesan loop."""
        self.is_running = False

    def _setup_components(self):
        """Inisialisasi seluruh komponen."""
        args = self.args

        # 1. Video Source
        is_net = args.source.startswith(("rtsp://", "http://", "https://"))
        rtsp_url = args.source if is_net else ""
        fallback = DEFAULT_VIDEO if is_net else args.source

        self.video_source = VideoSourceManager(
            rtsp_url=rtsp_url,
            fallback_path=fallback,
            reconnect_interval=30,
        )

        # 2. YOLO Model
        logger.info(f"Loading model: {args.model}")
        self.model = YOLO(args.model)

        # Sinkronisasi CLASS_NAMES secara dinamis berdasarkan nama kelas model
        # untuk mengantisipasi model 5-kelas (kotor)
        global CLASS_NAMES
        CLASS_NAMES.clear()
        for cid, cname in self.model.names.items():
            if cname not in ["Labelling-data-lalu-lintas", "Labelling-data-laku-lintas"]:
                CLASS_NAMES[cid] = cname
        logger.info(f"Dynamic CLASS_NAMES initialized: {CLASS_NAMES}")

        # 3. Traffic Logger
        self.traffic_logger = TrafficLogger(
            csv_path=args.csv_output,
            flush_interval=100,
        )

        # 4. ROI Counter akan diinisialisasi setelah resolusi video didapatkan

        # 5. Visual HUD
        self.hud = VisualHUD()

        # 6. Supervision Annotators
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

        # 7. Output video writer
        self.video_writer = None

        # 8. Cleanup interval
        self._cleanup_interval = 300  # Setiap 300 frame
        self._frame_times = []

    def run(self):
        """Jalankan pipeline monitoring utama."""
        logger.info("Memulai Traffic Monitor - Jalan Buah Batu")

        # Buka video source
        if not self.video_source.open():
            logger.error("Tidak bisa membuka sumber video! Pastikan RTSP atau file MP4 tersedia.")
            return

        fps = self.video_source.get_fps()
        orig_w, orig_h = self.video_source.get_frame_size()

        # Ide Standardisasi Resolusi: Setel resolusi standar sesuai dengan resolusi asli CCTV
        self.std_w, self.std_h = orig_w, orig_h
        frame_w, frame_h = self.std_w, self.std_h

        logger.info(
            f"Video asli {orig_w}x{orig_h} akan ditampilkan sesuai resolusi CCTV "
            f"yaitu {frame_w}x{frame_h} @ {fps:.1f} FPS"
        )
        # ── Setup ROI Counter ─────────────────────────────────────────────
        logger.info("Menginisialisasi Dynamic Single Polygon Counter")

        self.counter = ROICounter(
            class_names=CLASS_NAMES,
            frame_size=(frame_w, frame_h),
        )
        logger.info("ROI Counter aktif dengan 1 zona dinamis di tengah layar")

        # Setup video writer jika diminta
        if self.args.save_video:
            output_path = str(PROJECT_ROOT / "output_monitoring.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                output_path, fourcc, fps, (frame_w, frame_h)
            )
            logger.info(f"Output video: {output_path}")

        # Buat window resizable (tanpa paksa fullscreen agar sesuai resolusi asli)
        if self.args.show:
            cv2.namedWindow("Traffic Monitor - Buah Batu", cv2.WINDOW_NORMAL)

        try:
            self._processing_loop(fps)
        except KeyboardInterrupt:
            logger.info("\nDihentikan oleh pengguna (Ctrl+C)")
        finally:
            self._cleanup()

    def _processing_loop(self, source_fps: float):
        """Loop pemrosesan frame utama dengan Adaptive Sync."""
        frame_id = 0

        # 1. Tentukan target sinkronisasi FPS
        fps_target = source_fps
        if fps_target <= 0 or fps_target > 60:
            fps_target = 12.0

        time_per_frame = 1.0 / fps_target
        logger.info(
            f"Adaptive Sync aktif: Target {fps_target} FPS "
            f"(Waktu per frame: {time_per_frame:.3f}s)"
        )

        while self.is_running:
            loop_start = time.time()

            # Baca frame
            success, frame = self.video_source.read()
            if not success:
                # Meniru backoff HLS dari hasil observasi check_stream_fps.py
                if self.video_source.is_rtsp:
                    logger.warning(
                        "Stream terputus/buffering. Memberikan jeda 2 detik..."
                    )
                    time.sleep(2.0)
                    continue
                else:
                    break

            # Resize frame ke resolusi standar
            frame = cv2.resize(frame, (self.std_w, self.std_h))

            frame_id += 1

            # Deteksi + Tracking
            results = self.model.track(
                source=frame,
                persist=True,
                tracker=TRACKER_CONFIG,
                conf=0.1,
                iou=0.5,
                verbose=False,
            )

            # Konversi hasil ke sv.Detections
            detections = sv.Detections.from_ultralytics(results[0])

            # ── Update ROI counter ────────────────────────────────────
            new_crossings = self.counter.update(detections, frame_id)

            # Log trip events ke CSV
            for vehicle_id, info in new_crossings.items():
                self.traffic_logger.log_crossing(
                    frame_id=frame_id,
                    vehicle_id=vehicle_id,
                    class_name=info["class_name"],
                    confidence=info["confidence"],
                    direction=info["direction"],
                    origin_zone=info["origin_zone"],
                    destination_zone=info["destination_zone"],
                )

            # ── Cleanup berkala (anti-memory leak) ────────────────────
            if frame_id % self._cleanup_interval == 0:
                active_ids = set()
                if detections.tracker_id is not None:
                    active_ids = set(int(tid) for tid in detections.tracker_id)
                self.traffic_logger.cleanup_stale_ids(active_ids)

                # Cleanup juga mengembalikan single-zone events
                single_zone_events = self.counter.cleanup_positions(active_ids)

                # Log single-zone events (kendaraan masuk 1 zona lalu hilang)
                for vehicle_id, info in single_zone_events.items():
                    self.traffic_logger.log_crossing(
                        frame_id=frame_id,
                        vehicle_id=vehicle_id,
                        class_name=info["class_name"],
                        confidence=info["confidence"],
                        direction=info["direction"],
                        origin_zone=info["origin_zone"],
                        destination_zone=info["destination_zone"],
                    )

            # ── Visualisasi ───────────────────────────────────────────
            annotated = self._annotate_frame(
                frame, detections, frame_id, loop_start
            )

            # Output
            if self.args.show:
                cv2.imshow("Traffic Monitor - Buah Batu", annotated)

            if self.video_writer:
                self.video_writer.write(annotated)
                
            if hasattr(self, 'frame_callback') and self.frame_callback is not None:
                self.frame_callback(annotated)

            # --- ADAPTIVE SYNC MECHANISM ---
            elapsed_time = time.time() - loop_start
            sleep_delay = time_per_frame - elapsed_time

            if sleep_delay > 0:
                time.sleep(sleep_delay)

            # Jeda minimal OpenCV untuk menyegarkan UI
            if self.args.show:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    logger.info("Keluar (tekan 'q')")
                    break

            # Logging periodik
            if frame_id % 500 == 0:
                counts = self.counter.total_counts
                total = sum(counts.values())
                dir_counts = self.counter.direction_counts
                logger.info(
                    f"Frame {frame_id} | Total: {total} | "
                    f"Bis:{counts.get('Bis', 0)} Mobil:{counts.get('Mobil', 0)} "
                    f"Motor:{counts.get('Motor', 0)} Truk:{counts.get('Truk', 0)}"
                )
                if dir_counts:
                    dirs_str = " | ".join(
                        f"{d}:{c}" for d, c in
                        sorted(dir_counts.items(), key=lambda x: x[1],
                               reverse=True)[:4]
                    )
                    logger.info(f"  Arah: {dirs_str}")

    def _annotate_frame(self, frame: np.ndarray, detections: sv.Detections,
                        frame_id: int, loop_start: float) -> np.ndarray:
        """Buat frame teranotasi dengan semua visual overlay."""
        annotated = frame.copy()

        # 1. Gambar zona polygon ROI (semi-transparan)
        annotated = self.counter.draw_zones(annotated, alpha=0.20)

        # 2. Trace annotator (ekor pergerakan)
        if detections.tracker_id is not None:
            annotated = self.trace_annotator.annotate(
                scene=annotated,
                detections=detections,
            )

        # 3. Bounding box
        annotated = self.box_annotator.annotate(
            scene=annotated,
            detections=detections,
        )

        # 4. Label
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

        # 5. HUD overlay (termasuk panel arah pergerakan)
        elapsed = time.time() - loop_start
        fps = 1.0 / elapsed if elapsed > 0 else 0
        counts = self.counter.total_counts
        dir_counts = self.counter.direction_counts

        annotated = self.hud.draw(
            frame=annotated,
            counts=counts,
            source_label=self.video_source.source_label,
            frame_id=frame_id,
            fps=fps,
            direction_counts=dir_counts,
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
        dir_counts = self.counter.direction_counts

        print("\n" + "=" * 50)
        print("RINGKASAN MONITORING")
        print("=" * 50)
        print(f"Total kendaraan terdeteksi: {total}")

        print("\nPer kelas kendaraan:")
        for cls_name, cnt in sorted(counts.items()):
            print(f"    {cls_name:8s}: {cnt}")

        if dir_counts:
            print("\nPer arah pergerakan:")
            for direction, cnt in sorted(dir_counts.items(),
                                         key=lambda x: x[1], reverse=True):
                print(f"    {direction:20s}: {cnt}")

        print(f"\nFrame diproses: {self.video_source.frame_count}")
        print(f"CSV output: {self.args.csv_output}")
        print("=" * 50 + "\n")
