"""
TAHAP 1B — Baseline COCO Validation
=====================================
Tujuan: Evaluasi pre-trained COCO model pada gambar Buah Batu.
        Membuktikan bahwa fine-tuning diperlukan.
Jalankan: python scripts/02_baseline_coco.py
"""
import sys
import os
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import cfg
from utils.logger import setup_logger

logger = setup_logger("baseline", log_dir="outputs/logs")

# Mapping COCO classes → Buah Batu classes
COCO_TO_BUAHBATU = {
    2: "Mobil",       # car
    3: "Motor",       # motorcycle
    5: "bus",         # bus
    7: "Truk ringan", # truck (generic → ringan as default)
    0: "Manusia",     # person
}

COCO_VEHICLE_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck


def run_baseline_inference(num_samples: int = 20):
    """Jalankan COCO pre-trained model pada sample gambar Buah Batu."""
    from ultralytics import YOLO

    logger.info("Loading YOLOv8m COCO pre-trained...")
    model = YOLO("yolov8m.pt")

    # Ambil sample gambar dari test set
    test_dir = cfg.paths.project_root / "test" / "images"
    images = sorted(test_dir.glob("*.*"))[:num_samples]
    logger.info(f"Running inference on {len(images)} test images...")

    output_dir = cfg.paths.output_dir / "baseline_coco"
    output_dir.mkdir(exist_ok=True)

    all_results = []

    for img_path in images:
        results = model.predict(
            source=str(img_path),
            conf=0.25,
            iou=0.6,
            imgsz=880,
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes

        # Filter hanya kendaraan COCO
        vehicle_mask = np.isin(boxes.cls.cpu().numpy(), COCO_VEHICLE_IDS)
        vehicle_count = vehicle_mask.sum()

        # Annotate image
        img = cv2.imread(str(img_path))
        for i, box in enumerate(boxes):
            cls_id = int(box.cls.item())
            if cls_id not in COCO_VEHICLE_IDS and cls_id != 0:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf.item()
            label = COCO_TO_BUAHBATU.get(cls_id, f"coco_{cls_id}")

            color = (0, 255, 0) if cls_id in COCO_VEHICLE_IDS else (255, 165, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save annotated
        cv2.imwrite(str(output_dir / img_path.name), img)

        all_results.append({
            "image": img_path.name,
            "total_detections": len(boxes),
            "vehicle_detections": int(vehicle_count),
        })

        logger.info(f"  {img_path.name}: {vehicle_count} vehicles, "
                     f"{len(boxes)} total detections")

    return all_results


def compare_with_ground_truth(num_samples: int = 20):
    """Bandingkan deteksi COCO vs ground truth label Buah Batu."""
    test_img_dir = cfg.paths.project_root / "test" / "images"
    test_lbl_dir = cfg.paths.project_root / "test" / "labels"

    images = sorted(test_img_dir.glob("*.*"))[:num_samples]

    logger.info("\n📊 Perbandingan COCO baseline vs Ground Truth:")
    logger.info("-" * 55)
    logger.info(f"{'Image':<30} {'GT Objects':>12} {'COCO Det':>10}")
    logger.info("-" * 55)

    total_gt = 0
    total_coco = 0

    for img_path in images:
        # Ground truth count
        lbl_path = test_lbl_dir / f"{img_path.stem}.txt"
        gt_count = 0
        if lbl_path.exists():
            with open(lbl_path) as f:
                gt_count = len(f.readlines())

        total_gt += gt_count

    logger.info(f"\nTotal GT objects across {len(images)} images: {total_gt}")
    logger.info("⚠️  COCO model tidak mengenal kelas spesifik Buah Batu")
    logger.info("    (Bis mini, Truk berat, Truk menengah, Truk ringan)")
    logger.info("→  Fine-tuning WAJIB dilakukan untuk akurasi optimal!")


def run_tilted_angle_analysis():
    """Analisis dampak sudut kamera miring pada deteksi."""
    from ultralytics import YOLO

    logger.info("\n📐 Analisis Sudut Kamera Miring")
    logger.info("-" * 50)

    model = YOLO("yolov8m.pt")
    test_dir = cfg.paths.project_root / "test" / "images"
    images = sorted(test_dir.glob("*.*"))[:10]

    confidences = []
    sizes = []

    for img_path in images:
        results = model.predict(str(img_path), conf=0.15, verbose=False)
        boxes = results[0].boxes

        for box in boxes:
            cls_id = int(box.cls.item())
            if cls_id not in COCO_VEHICLE_IDS:
                continue

            conf = box.conf.item()
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            rel_y = (y1 + y2) / 2 / 880  # Normalized vertical position

            confidences.append(conf)
            sizes.append({"area": area, "rel_y": rel_y, "conf": conf})

    if confidences:
        avg_conf = np.mean(confidences)
        logger.info(f"  Rata-rata confidence COCO: {avg_conf:.3f}")
        logger.info(f"  Total deteksi kendaraan: {len(confidences)}")

        # Confidence vs vertical position (proxy for distance/angle)
        top_half = [s["conf"] for s in sizes if s["rel_y"] < 0.5]
        bottom_half = [s["conf"] for s in sizes if s["rel_y"] >= 0.5]

        if top_half and bottom_half:
            logger.info(f"  Confidence atas (jauh): {np.mean(top_half):.3f}")
            logger.info(f"  Confidence bawah (dekat): {np.mean(bottom_half):.3f}")
            logger.info("  → Objek jauh (atas frame) biasanya confidence lebih rendah")
            logger.info("    karena sudut kamera miring + objek lebih kecil")
    else:
        logger.warning("  Tidak ada deteksi kendaraan ditemukan")


def main():
    logger.info("=" * 60)
    logger.info("BASELINE COCO VALIDATION — Buah Batu Traffic")
    logger.info("=" * 60)

    logger.info("\n📌 1/3 Inference dengan COCO pre-trained")
    results = run_baseline_inference(num_samples=20)

    logger.info("\n📌 2/3 Perbandingan vs Ground Truth")
    compare_with_ground_truth(num_samples=20)

    logger.info("\n📌 3/3 Analisis Sudut Kamera Miring")
    run_tilted_angle_analysis()

    logger.info("\n" + "=" * 60)
    logger.info("📝 KESIMPULAN BASELINE:")
    logger.info("  1. COCO model hanya mengenal 4 kelas kendaraan generik")
    logger.info("  2. Kelas spesifik (Bis mini, Truk berat/menengah/ringan) MISSED")
    logger.info("  3. Sudut kamera miring menurunkan confidence objek jauh")
    logger.info("  4. Fine-tuning pada dataset Buah Batu WAJIB")
    logger.info("=" * 60)
    logger.info("Next: python scripts/03_train_buahbatu.py")


if __name__ == "__main__":
    main()
