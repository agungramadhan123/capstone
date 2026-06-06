"""
TAHAP 5B — Evaluasi Deteksi: mAP Analysis
============================================
Deep analysis mAP, confusion matrix, dan error patterns.
Jalankan: python evaluation/detect_evaluator.py
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import cfg
from utils.logger import setup_logger

logger = setup_logger("eval_detect", log_dir="outputs/logs")


def find_best_model() -> Path:
    candidates = sorted(cfg.paths.runs_dir.glob("buahbatu_*/weights/best.pt"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No trained model found. Run training first!")
    return candidates[0]


def run_full_evaluation():
    """Run comprehensive mAP evaluation."""
    from ultralytics import YOLO

    model_path = find_best_model()
    logger.info(f"Model: {model_path}")

    model = YOLO(str(model_path))

    # ─── Validation Set ───
    logger.info("\n📊 Validation Set Evaluation")
    val_metrics = model.val(
        data=str(cfg.paths.data_yaml),
        split="val",
        imgsz=cfg.model.img_size,
        conf=0.001,       # Low conf for full P-R curve
        iou=0.5,
        plots=True,
        save_json=True,
    )

    # ─── Test Set ───
    logger.info("\n📊 Test Set Evaluation")
    test_metrics = model.val(
        data=str(cfg.paths.data_yaml),
        split="test",
        imgsz=cfg.model.img_size,
        conf=0.001,
        iou=0.5,
        plots=True,
        save_json=True,
    )

    # ─── Comparison ───
    logger.info("\n" + "=" * 65)
    logger.info("DETECTION EVALUATION RESULTS")
    logger.info("=" * 65)

    logger.info(f"\n{'Metric':<20} {'Validation':>12} {'Test':>12}")
    logger.info("-" * 45)
    logger.info(f"{'mAP@50':<20} {val_metrics.box.map50:>12.4f} {test_metrics.box.map50:>12.4f}")
    logger.info(f"{'mAP@50-95':<20} {val_metrics.box.map:>12.4f} {test_metrics.box.map:>12.4f}")
    logger.info(f"{'Precision':<20} {val_metrics.box.mp:>12.4f} {test_metrics.box.mp:>12.4f}")
    logger.info(f"{'Recall':<20} {val_metrics.box.mr:>12.4f} {test_metrics.box.mr:>12.4f}")

    # Per-class breakdown (test set)
    logger.info(f"\n{'Class':<18} {'mAP@50':>8} {'mAP@50-95':>10}")
    logger.info("-" * 38)
    for i, name in enumerate(cfg.class_names):
        if i < len(test_metrics.box.maps):
            map50 = test_metrics.box.ap50[i] if hasattr(test_metrics.box, 'ap50') else 0
            map95 = test_metrics.box.maps[i]
            emoji = "✅" if map95 > 0.4 else "⚠️" if map95 > 0.2 else "❌"
            logger.info(f"  {emoji} {name:<16} {map50:>7.4f} {map95:>10.4f}")

    # ─── Error Analysis ───
    logger.info("\n💡 ERROR ANALYSIS & RECOMMENDATIONS:")

    if test_metrics.box.map50 < 0.5:
        logger.warning("  ⚠️ mAP@50 < 0.5 — Model needs improvement!")
        logger.info("  Recommendations:")
        logger.info("    1. Increase epochs (150 → 250)")
        logger.info("    2. Try larger model (yolov8l.pt)")
        logger.info("    3. Add more training data")

    if test_metrics.box.mp > 0.8 and test_metrics.box.mr < 0.5:
        logger.info("  → High Precision, Low Recall: Model terlalu conservative")
        logger.info("    Fix: Turunkan conf_threshold (0.25 → 0.15)")

    if test_metrics.box.mr > 0.8 and test_metrics.box.mp < 0.5:
        logger.info("  → High Recall, Low Precision: Terlalu banyak false positive")
        logger.info("    Fix: Naikkan conf_threshold (0.25 → 0.40)")

    # Overfitting check
    val_map = val_metrics.box.map50
    test_map = test_metrics.box.map50
    gap = val_map - test_map
    if gap > 0.1:
        logger.warning(f"  ⚠️ Overfitting detected (val-test gap: {gap:.3f})")
        logger.info("    Fix: Increase augmentation, reduce epochs, or add regularization")
    else:
        logger.info(f"  ✅ No overfitting (val-test gap: {gap:.3f})")

    logger.info(f"\n  📁 Plots saved in: {model_path.parent.parent}")
    logger.info("     → confusion_matrix.png, PR_curve.png, F1_curve.png")

    return test_metrics


def main():
    logger.info("=" * 60)
    logger.info("DETECTION EVALUATION — mAP & Error Analysis")
    logger.info("=" * 60)

    run_full_evaluation()

    logger.info("\n" + "=" * 60)
    logger.info("✅ Evaluation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
