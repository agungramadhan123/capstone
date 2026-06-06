"""
TAHAP 2A — Fine-Tuning YOLOv8 pada Dataset Buah Batu
=====================================================
Tujuan: Train model khusus traffic Buah Batu dengan optimasi
        untuk small objects (motor) dan dense scenes.
Jalankan: python scripts/03_train_buahbatu.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import cfg
from utils.logger import setup_logger

logger = setup_logger("train", log_dir="outputs/logs")


def train():
    """Fine-tune YOLOv8m pada dataset Buah Batu."""
    from ultralytics import YOLO

    tc = cfg.train
    mc = cfg.model

    logger.info("=" * 60)
    logger.info("FINE-TUNING YOLOv8 — Dataset Buah Batu")
    logger.info("=" * 60)

    # Load pre-trained model
    logger.info(f"Loading pre-trained: {mc.weights}")
    model = YOLO(mc.weights)

    # ─── Training Parameters ───
    train_args = dict(
        # Dataset
        data=str(cfg.paths.data_yaml),
        imgsz=mc.img_size,          # 880 — match dataset resolution

        # Training schedule
        epochs=tc.epochs,           # 150
        batch=tc.batch_size,        # 8 (safe for 8GB VRAM)
        patience=tc.patience,       # 30 early stopping

        # Optimizer
        optimizer=tc.optimizer,     # AdamW
        lr0=tc.lr0,                 # 0.001
        lrf=tc.lrf,                 # 0.01 (final LR = lr0 * lrf)
        weight_decay=tc.weight_decay,
        warmup_epochs=tc.warmup_epochs,

        # Augmentation — key untuk kamera miring & small objects
        degrees=tc.degrees,         # 10° rotasi
        translate=tc.translate,     # 15% translate
        scale=tc.scale,             # ±50% scale variation
        mosaic=tc.mosaic,           # Mosaic augmentation (ON)
        mixup=tc.mixup,             # 10% mixup
        copy_paste=tc.copy_paste,   # 10% copy-paste (small object boost)
        fliplr=0.5,                 # Horizontal flip
        flipud=0.0,                 # No vertical flip (tidak natural)

        # Loss weights
        cls=tc.cls_loss,            # 1.0 classification
        box=tc.box_loss,            # 7.5 box (HIGHER = better small object)
        dfl=tc.dfl_loss,            # 1.5 distribution focal loss

        # NMS
        iou=mc.iou_threshold,      # 0.6

        # Output
        project=str(cfg.paths.runs_dir),
        name="buahbatu_v1",
        exist_ok=True,

        # System
        device=mc.device,
        workers=4,
        verbose=True,
        plots=True,                 # Generate training plots
    )

    # ─── Log config ───
    logger.info("\n📋 Training Configuration:")
    for k, v in train_args.items():
        logger.info(f"  {k}: {v}")

    # ─── Train ───
    logger.info("\n🚀 Starting training...")
    results = model.train(**train_args)

    # ─── Results ───
    logger.info("\n" + "=" * 60)
    logger.info("📊 TRAINING RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Best model: {results.save_dir}/weights/best.pt")
    logger.info(f"  Last model: {results.save_dir}/weights/last.pt")

    # Validate on test set
    logger.info("\n📌 Running validation on test set...")
    best_model = YOLO(f"{results.save_dir}/weights/best.pt")
    metrics = best_model.val(
        data=str(cfg.paths.data_yaml),
        split="test",
        imgsz=mc.img_size,
        conf=mc.conf_threshold,
        iou=mc.iou_threshold,
    )

    logger.info(f"\n  mAP@50:    {metrics.box.map50:.4f}")
    logger.info(f"  mAP@50-95: {metrics.box.map:.4f}")

    # Per-class mAP
    logger.info("\n  Per-class mAP@50:")
    for i, name in enumerate(cfg.class_names):
        if i < len(metrics.box.maps):
            logger.info(f"    {name:<18}: {metrics.box.maps[i]:.4f}")

    logger.info("\n" + "=" * 60)
    logger.info("✅ Training complete!")
    logger.info("Next: python scripts/04_evaluate_model.py")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    """
    ╔══════════════════════════════════════════════════════════╗
    ║  DEBUG TIPS:                                            ║
    ║  1. OOM Error → Kurangi batch_size (8→4→2)              ║
    ║  2. Loss NaN  → Kurangi lr0 (0.001→0.0005)              ║
    ║  3. Slow conv → Pastikan workers=4, device="0"          ║
    ║  4. Monitor   → tensorboard --logdir runs/buahbatu_v1   ║
    ║  5. Resume    → YOLO("runs/.../last.pt").train(resume=1)║
    ╚══════════════════════════════════════════════════════════╝
    """
    train()
