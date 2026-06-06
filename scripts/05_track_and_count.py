"""
TAHAP 3 — Tracking & Counting dengan ByteTrack
================================================
Tujuan: Deteksi + tracking + counting kendaraan via virtual line.
Jalankan: python scripts/05_track_and_count.py --source video.mp4
"""
import sys
import os
import argparse
import time
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import cfg
from core.detector import VehicleDetector
from core.counter import VirtualLineCounter
from utils.video_io import VideoReader, VideoWriter
from utils.visualizer import (
    draw_detections, draw_virtual_line,
    draw_counting_overlay, draw_crossing_flash
)
from utils.logger import setup_logger

logger = setup_logger("tracking", log_dir="outputs/logs")


def parse_args():
    parser = argparse.ArgumentParser(description="Traffic Tracking & Counting")
    parser.add_argument("--source", required=True, help="Path to video file")
    parser.add_argument("--model", default=None, help="Path to model weights")
    parser.add_argument("--output", default=None, help="Output video path")
    parser.add_argument("--show", action="store_true", help="Show live window")
    parser.add_argument("--line-y", type=float, default=0.6,
                        help="Virtual line Y position (0-1 normalized)")
    parser.add_argument("--direction", default="down",
                        choices=["up", "down", "both"])
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def find_model(model_arg: str = None) -> str:
    """Auto-find best model dari training runs."""
    if model_arg:
        return model_arg

    from pathlib import Path
    runs_dir = cfg.paths.runs_dir
    candidates = sorted(runs_dir.glob("buahbatu_*/weights/best.pt"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        logger.info(f"Auto-detected model: {candidates[0]}")
        return str(candidates[0])

    # Fallback to pre-trained
    logger.warning("No fine-tuned model found. Using yolov8m.pt (COCO)")
    return "yolov8m.pt"


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("TRACKING & COUNTING — Buah Batu Traffic")
    logger.info("=" * 60)

    # ─── Init components ───
    model_path = find_model(args.model)
    detector = VehicleDetector(
        model_path=model_path,
        conf=args.conf,
        iou=cfg.model.iou_threshold,
        imgsz=cfg.model.img_size,
    )

    video = VideoReader(args.source)
    logger.info(f"\n{video.info}")

    # Virtual line position
    line_y = int(args.line_y * video.height)
    line_start = (0, line_y)
    line_end = (video.width, line_y)

    counter = VirtualLineCounter(
        line_start=line_start,
        line_end=line_end,
        direction=args.direction,
        min_track_length=cfg.count.min_track_length,
    )

    # Output video
    output_path = args.output or str(
        cfg.paths.output_dir / f"tracked_{os.path.basename(args.source)}"
    )
    writer = VideoWriter(output_path, video.fps, video.width, video.height)

    tracker_config = str(cfg.paths.tracker_config)

    logger.info(f"\n📌 Virtual Line: y={line_y} ({args.direction})")
    logger.info(f"📌 Tracker config: {tracker_config}")
    logger.info(f"📌 Output: {output_path}")
    logger.info("\n🚀 Processing started...")

    # ─── Main loop ───
    fps_counter = 0
    fps_timer = time.time()
    display_fps = 0.0
    crossing_flash_frames = {}  # track_id -> frames remaining

    for frame_num, frame in video:
        # 1. Detect + Track
        detections = detector.detect_and_track(
            frame, tracker_config=tracker_config, persist=True
        )

        # 2. Update counter for each tracked object
        if detections.track_ids is not None:
            for i in range(len(detections.boxes)):
                track_id = int(detections.track_ids[i])
                event = counter.update(
                    track_id=track_id,
                    bbox=detections.boxes[i],
                    class_id=int(detections.class_ids[i]),
                    class_name=detector.get_class_name(
                        int(detections.class_ids[i])
                    ),
                    frame_number=frame_num,
                    fps=video.fps,
                )

                if event:
                    logger.info(
                        f"  🚗 #{event.track_id} {event.class_name} "
                        f"crossed at frame {frame_num} "
                        f"(t={event.timestamp:.1f}s) | "
                        f"Total: {counter.total_count}"
                    )
                    crossing_flash_frames[track_id] = 10  # Flash 10 frames

        # 3. Visualize
        annotated = draw_virtual_line(frame, line_start, line_end)
        annotated = draw_detections(
            annotated,
            detections.boxes,
            detections.class_ids,
            detections.confidences,
            detections.track_ids,
            detector.class_names,
        )

        # Crossing flash effects
        for tid in list(crossing_flash_frames.keys()):
            crossing_flash_frames[tid] -= 1
            if crossing_flash_frames[tid] <= 0:
                del crossing_flash_frames[tid]

        annotated = draw_counting_overlay(
            annotated,
            counter.total_count,
            counter.class_counts,
            fps_display=display_fps,
        )

        # 4. Write output
        writer.write(annotated)

        # 5. Optional live display
        if args.show:
            cv2.imshow("Traffic Counting", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User quit (Q pressed)")
                break

        # FPS calculation
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            display_fps = fps_counter / (time.time() - fps_timer)
            fps_counter = 0
            fps_timer = time.time()

        # Progress log every 500 frames
        if frame_num % 500 == 0:
            progress = frame_num / video.total_frames * 100
            logger.info(
                f"  Progress: {progress:.1f}% | "
                f"Frame {frame_num}/{video.total_frames} | "
                f"Count: {counter.total_count} | "
                f"FPS: {display_fps:.1f}"
            )

    # ─── Cleanup ───
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    # ─── Summary ───
    logger.info("\n" + "=" * 60)
    logger.info("📊 COUNTING RESULTS")
    logger.info("=" * 60)
    logger.info(f"\n{counter.get_summary()}")
    logger.info(f"\nOutput video: {output_path}")

    # Export events CSV
    _export_csv(counter, output_path)

    logger.info("=" * 60)
    logger.info("✅ Tracking & counting complete!")


def _export_csv(counter: VirtualLineCounter, video_path: str):
    """Export crossing events ke CSV."""
    import pandas as pd
    from pathlib import Path

    events = counter.events
    if not events:
        logger.warning("No crossing events to export")
        return

    df = pd.DataFrame([{
        "track_id": e.track_id,
        "class_name": e.class_name,
        "frame": e.frame_number,
        "timestamp_sec": round(e.timestamp, 2),
        "centroid_x": round(e.centroid[0], 1),
        "centroid_y": round(e.centroid[1], 1),
        "direction": e.direction,
    } for e in events])

    csv_path = Path(video_path).with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"CSV exported: {csv_path}")

    # Summary CSV
    summary_path = Path(video_path).with_name(
        Path(video_path).stem + "_summary.csv"
    )
    summary_df = pd.DataFrame([
        {"class": k, "count": v}
        for k, v in sorted(counter.class_counts.items())
    ])
    summary_df.loc[len(summary_df)] = {"class": "TOTAL", "count": counter.total_count}
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    """
    ╔══════════════════════════════════════════════════════════╗
    ║  USAGE:                                                 ║
    ║  python scripts/05_track_and_count.py \                 ║
    ║      --source video.mp4 \                               ║
    ║      --line-y 0.6 --direction down --show               ║
    ║                                                         ║
    ║  DEBUG TIPS:                                            ║
    ║  1. --show untuk live preview                           ║
    ║  2. --line-y mengatur posisi garis (0=atas, 1=bawah)    ║
    ║  3. --conf 0.35 untuk kurangi false positive            ║
    ║  4. Jika track ID tidak stabil → naikkan track_buffer   ║
    ║  5. Double counting → naikkan min_track_length          ║
    ╚══════════════════════════════════════════════════════════╝
    """
    main()
