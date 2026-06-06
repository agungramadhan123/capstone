"""
TAHAP 4 — Full Pipeline: Video → Detection → Tracking → Counting → CSV
========================================================================
Batch processing support & memory-efficient pipeline.
Jalankan: python scripts/06_full_pipeline.py --source video.mp4
          python scripts/06_full_pipeline.py --source folder_videos/
"""
import sys
import os
import argparse
import time
import gc
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import cfg
from core.detector import VehicleDetector
from core.counter import VirtualLineCounter
from utils.video_io import VideoReader, VideoWriter
from utils.visualizer import (
    draw_detections, draw_virtual_line, draw_counting_overlay
)
from utils.logger import setup_logger

logger = setup_logger("pipeline", log_dir="outputs/logs")


def parse_args():
    parser = argparse.ArgumentParser(description="Full Traffic Pipeline")
    parser.add_argument("--source", required=True,
                        help="Video file or folder of videos")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--line-y", type=float, default=0.6)
    parser.add_argument("--direction", default="down")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save-video", action="store_true", default=True)
    parser.add_argument("--no-video", dest="save_video", action="store_false")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Frames to skip (1=process all, 2=skip 1)")
    return parser.parse_args()


def find_model(model_arg=None) -> str:
    if model_arg:
        return model_arg
    candidates = sorted(cfg.paths.runs_dir.glob("buahbatu_*/weights/best.pt"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return str(candidates[0])
    return "yolov8m.pt"


def process_single_video(
    video_path: str,
    detector: VehicleDetector,
    output_dir: Path,
    line_y_ratio: float,
    direction: str,
    save_video: bool,
    frame_skip: int,
) -> dict:
    """Process satu video — return counting results."""
    video_name = Path(video_path).stem
    logger.info(f"\n{'─'*50}")
    logger.info(f"📹 Processing: {video_name}")

    video = VideoReader(video_path)
    logger.info(video.info)

    # Virtual line
    line_y = int(line_y_ratio * video.height)
    counter = VirtualLineCounter(
        line_start=(0, line_y),
        line_end=(video.width, line_y),
        direction=direction,
        min_track_length=cfg.count.min_track_length,
    )

    # Output
    writer = None
    if save_video:
        out_path = str(output_dir / f"{video_name}_counted.mp4")
        writer = VideoWriter(out_path, video.fps, video.width, video.height)

    tracker_config = str(cfg.paths.tracker_config)
    start_time = time.time()

    for frame_num, frame in video:
        # Frame skip for speed
        if frame_skip > 1 and frame_num % frame_skip != 0:
            continue

        # Detect + Track
        detections = detector.detect_and_track(
            frame, tracker_config=tracker_config, persist=True
        )

        # Count
        if detections.track_ids is not None:
            active_ids = set()
            for i in range(len(detections.boxes)):
                tid = int(detections.track_ids[i])
                active_ids.add(tid)
                counter.update(
                    track_id=tid,
                    bbox=detections.boxes[i],
                    class_id=int(detections.class_ids[i]),
                    class_name=detector.get_class_name(int(detections.class_ids[i])),
                    frame_number=frame_num,
                    fps=video.fps,
                )
            # Memory cleanup for long videos
            if frame_num % 1000 == 0:
                counter.cleanup_lost_tracks(active_ids)

        # Write annotated frame
        if writer:
            annotated = draw_virtual_line(frame, (0, line_y), (video.width, line_y))
            annotated = draw_detections(
                annotated, detections.boxes, detections.class_ids,
                detections.confidences, detections.track_ids,
                detector.class_names,
            )
            annotated = draw_counting_overlay(
                annotated, counter.total_count, counter.class_counts
            )
            writer.write(annotated)

        # Progress
        if frame_num % 1000 == 0:
            elapsed = time.time() - start_time
            pct = frame_num / video.total_frames * 100
            proc_fps = frame_num / elapsed if elapsed > 0 else 0
            logger.info(
                f"  [{pct:.0f}%] frame {frame_num}/{video.total_frames} "
                f"| count={counter.total_count} | {proc_fps:.1f} fps"
            )

    # Cleanup
    if writer:
        writer.release()
    elapsed = time.time() - start_time

    # Export CSV
    _export_results(counter, output_dir, video_name)

    # Summary
    result = {
        "video": video_name,
        "total_frames": video.total_frames,
        "processed_time_sec": round(elapsed, 1),
        "total_count": counter.total_count,
        "class_counts": counter.class_counts,
    }

    logger.info(f"\n  ✅ {video_name}: {counter.total_count} vehicles "
                f"in {elapsed:.1f}s")
    logger.info(f"  {counter.get_summary()}")

    # Force garbage collection for long videos
    del video
    gc.collect()

    return result


def _export_results(counter: VirtualLineCounter, output_dir: Path, name: str):
    """Export counting results to CSV."""
    import pandas as pd

    # Events log
    events = counter.events
    if events:
        df = pd.DataFrame([{
            "track_id": e.track_id,
            "class_name": e.class_name,
            "frame": e.frame_number,
            "timestamp_sec": round(e.timestamp, 2),
            "direction": e.direction,
        } for e in events])
        csv_path = output_dir / f"{name}_events.csv"
        df.to_csv(csv_path, index=False)

    # Summary
    summary = [{"class": k, "count": v}
               for k, v in sorted(counter.class_counts.items())]
    summary.append({"class": "TOTAL", "count": counter.total_count})
    pd.DataFrame(summary).to_csv(
        output_dir / f"{name}_summary.csv", index=False
    )


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("FULL PIPELINE — Buah Batu Traffic Counting")
    logger.info("=" * 60)

    # Setup
    model_path = find_model(args.model)
    detector = VehicleDetector(
        model_path=model_path, conf=args.conf,
        iou=cfg.model.iou_threshold, imgsz=cfg.model.img_size,
    )

    output_dir = Path(args.output_dir) if args.output_dir else cfg.paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find videos
    source = Path(args.source)
    if source.is_dir():
        video_paths = sorted(
            p for p in source.glob("*")
            if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
        )
        logger.info(f"Found {len(video_paths)} videos in {source}")
    else:
        video_paths = [source]

    # Process each video
    all_results = []
    for vp in video_paths:
        result = process_single_video(
            str(vp), detector, output_dir,
            args.line_y, args.direction,
            args.save_video, args.batch_size,
        )
        all_results.append(result)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 PIPELINE SUMMARY")
    logger.info("=" * 60)
    for r in all_results:
        logger.info(f"  {r['video']}: {r['total_count']} vehicles "
                     f"({r['processed_time_sec']}s)")
    total = sum(r['total_count'] for r in all_results)
    logger.info(f"\n  Grand total: {total} vehicles")
    logger.info(f"  Output dir: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
