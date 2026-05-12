"""
TAHAP 2B — Evaluasi Model Fine-Tuned
=====================================
Tujuan: Analisis mendalam performa model — per-class, per-size,
        dan identifikasi false positive motor berdekatan.
Jalankan: python scripts/04_evaluate_model.py
"""
import sys
import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import cfg
from utils.logger import setup_logger

logger = setup_logger("evaluate", log_dir="outputs/logs")


def find_best_model() -> Path:
    """Cari model best.pt terbaru dari hasil training."""
    runs_dir = cfg.paths.runs_dir
    candidates = sorted(runs_dir.glob("buahbatu_*/weights/best.pt"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No best.pt found in {runs_dir}. Run training first!"
        )
    logger.info(f"Using model: {candidates[0]}")
    return candidates[0]


def run_validation(model_path: Path):
    """Run official YOLOv8 validation — mAP, precision, recall."""
    from ultralytics import YOLO

    logger.info("📊 Running validation on test set...")
    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(cfg.paths.data_yaml),
        split="test",
        imgsz=cfg.model.img_size,
        conf=cfg.model.conf_threshold,
        iou=cfg.model.iou_threshold,
        plots=True,
        save_json=True,
    )

    logger.info(f"\n{'='*50}")
    logger.info(f"  mAP@50:      {metrics.box.map50:.4f}")
    logger.info(f"  mAP@50-95:   {metrics.box.map:.4f}")
    logger.info(f"  Precision:   {metrics.box.mp:.4f}")
    logger.info(f"  Recall:      {metrics.box.mr:.4f}")

    logger.info(f"\n  Per-class mAP@50:")
    for i, name in enumerate(cfg.class_names):
        if i < len(metrics.box.maps):
            emoji = "✅" if metrics.box.maps[i] > 0.5 else "⚠️"
            logger.info(f"    {emoji} {name:<18}: {metrics.box.maps[i]:.4f}")

    return metrics


def analyze_false_positives(model_path: Path, num_samples: int = 30):
    """Analisis false positive, fokus motor berdekatan."""
    from ultralytics import YOLO

    logger.info("\n🔍 Analisis False Positive (Motor Berdekatan)")
    logger.info("-" * 50)

    model = YOLO(str(model_path))
    test_img_dir = cfg.paths.project_root / "test" / "images"
    test_lbl_dir = cfg.paths.project_root / "test" / "labels"
    output_dir = cfg.paths.output_dir / "fp_analysis"
    output_dir.mkdir(exist_ok=True)

    images = sorted(test_img_dir.glob("*.*"))[:num_samples]

    fp_stats = defaultdict(int)
    close_pair_issues = 0

    for img_path in images:
        # Predict
        results = model.predict(str(img_path), conf=0.25, iou=0.6, verbose=False)
        pred_boxes = results[0].boxes

        # Load ground truth
        lbl_path = test_lbl_dir / f"{img_path.stem}.txt"
        gt_boxes = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    gt_boxes.append((cls_id, cx, cy, w, h))

        # Analyze close motor detections
        if pred_boxes is not None and len(pred_boxes) > 0:
            motor_boxes = []
            for box in pred_boxes:
                if int(box.cls.item()) == 3:  # Motor
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    motor_boxes.append((x1, y1, x2, y2))

            # Check pairwise IoU of predicted motors
            for i in range(len(motor_boxes)):
                for j in range(i + 1, len(motor_boxes)):
                    iou = _compute_iou(motor_boxes[i], motor_boxes[j])
                    if 0.3 < iou < 0.6:
                        close_pair_issues += 1

            # Visualize
            if len(motor_boxes) > 5:  # Dense motor scene
                img = cv2.imread(str(img_path))
                for x1, y1, x2, y2 in motor_boxes:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (0, 255, 255), 2)
                cv2.imwrite(str(output_dir / f"dense_{img_path.name}"), img)

    logger.info(f"  Motor pairs dengan IoU 0.3-0.6: {close_pair_issues}")
    logger.info(f"  → Ini adalah area rawan false positive/duplicate")

    if close_pair_issues > 10:
        logger.warning("  ⚠️  Banyak motor berdekatan! Rekomendasi:")
        logger.warning("     - Naikkan NMS IoU threshold (0.6 → 0.7)")
        logger.warning("     - Atau naikkan conf threshold (0.25 → 0.35)")
    else:
        logger.info("  ✅ Deteksi motor berdekatan cukup baik")


def analyze_by_object_size(model_path: Path):
    """Breakdown performa berdasarkan ukuran objek."""
    from ultralytics import YOLO

    logger.info("\n📏 Analisis per Ukuran Objek")
    logger.info("-" * 50)

    model = YOLO(str(model_path))
    test_lbl_dir = cfg.paths.project_root / "test" / "labels"
    labels = sorted(test_lbl_dir.glob("*.txt"))

    size_counts = {"small": 0, "medium": 0, "large": 0}

    for lbl_path in labels:
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                w, h = float(parts[3]), float(parts[4])
                area = w * h

                if area < 0.01:       # < 1% of image
                    size_counts["small"] += 1
                elif area < 0.05:     # 1-5% of image
                    size_counts["medium"] += 1
                else:
                    size_counts["large"] += 1

    total = sum(size_counts.values())
    for size, count in size_counts.items():
        pct = count / total * 100 if total > 0 else 0
        logger.info(f"  {size:<8}: {count:>5} ({pct:.1f}%)")

    logger.info("\n  💡 Tips berdasarkan distribusi ukuran:")
    if size_counts["small"] / total > 0.4:
        logger.info("  → Banyak objek kecil: Gunakan imgsz=880+, box_loss=7.5+")
    if size_counts["large"] / total < 0.1:
        logger.info("  → Sedikit objek besar: Model sudah optimal untuk close-range")


def _compute_iou(box1, box2):
    """Hitung IoU antara dua box [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def main():
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION — Buah Batu Fine-Tuned")
    logger.info("=" * 60)

    model_path = find_best_model()

    logger.info("\n📌 1/3 Validation Metrics")
    run_validation(model_path)

    logger.info("\n📌 2/3 False Positive Analysis")
    analyze_false_positives(model_path)

    logger.info("\n📌 3/3 Object Size Analysis")
    analyze_by_object_size(model_path)

    logger.info("\n" + "=" * 60)
    logger.info("✅ Evaluation complete! Check outputs/fp_analysis/")
    logger.info("Next: python scripts/05_track_and_count.py --source <video>")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
