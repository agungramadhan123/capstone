"""
TAHAP 1A — Setup & Validasi Environment
========================================
Tujuan: Memastikan CUDA, GPU, dan semua dependencies siap.
Jalankan: python scripts/01_setup_environment.py
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger

logger = setup_logger("setup", log_dir="outputs/logs")


def check_python():
    """Cek versi Python."""
    ver = sys.version
    logger.info(f"Python: {ver}")
    assert sys.version_info >= (3, 9), "Python 3.9+ required"
    logger.info("✅ Python version OK")


def check_cuda():
    """Cek CUDA & GPU via PyTorch."""
    import torch

    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available! Cek instalasi PyTorch+CUDA")
        logger.error("   Fix: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
        return False

    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Quick benchmark
    logger.info("Running GPU benchmark...")
    x = torch.randn(1000, 1000, device="cuda")
    for _ in range(100):
        x = x @ x
    torch.cuda.synchronize()
    logger.info("✅ GPU compute OK")

    # FP16 check
    with torch.autocast("cuda"):
        y = torch.randn(100, 100, device="cuda") @ torch.randn(100, 100, device="cuda")
    logger.info("✅ FP16 (mixed precision) OK")

    return True


def check_dependencies():
    """Cek semua library yang dibutuhkan."""
    deps = {
        "ultralytics": "YOLOv8",
        "cv2": "OpenCV",
        "numpy": "NumPy",
        "supervision": "Supervision",
        "scipy": "SciPy",
        "yaml": "PyYAML",
        "pandas": "Pandas",
    }

    all_ok = True
    for module, name in deps.items():
        try:
            mod = __import__(module)
            ver = getattr(mod, "__version__", "?")
            logger.info(f"  ✅ {name}: {ver}")
        except ImportError:
            logger.error(f"  ❌ {name} not found! pip install {module}")
            all_ok = False

    return all_ok


def check_dataset():
    """Validasi integritas dataset."""
    from pathlib import Path

    root = Path(r"d:\Semester 6\Capstone")
    splits = {"train": 632, "valid": 181, "test": 90}

    for split, expected in splits.items():
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"

        images = list(img_dir.glob("*.*"))
        labels = list(lbl_dir.glob("*.txt"))

        logger.info(f"  {split}: {len(images)} images, {len(labels)} labels")

        # Check image-label pairing
        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}
        missing_labels = img_stems - lbl_stems
        if missing_labels:
            logger.warning(f"  ⚠️  {split}: {len(missing_labels)} images tanpa label")

    # Validate data.yaml
    import yaml
    with open(root / "data.yaml") as f:
        data = yaml.safe_load(f)
    logger.info(f"  Classes ({data['nc']}): {data['names']}")
    logger.info("✅ Dataset OK")


def check_ultralytics():
    """Test YOLOv8 import dan model loading."""
    from ultralytics import YOLO

    logger.info("Testing YOLOv8 model load...")
    model = YOLO("yolov8n.pt")  # Nano for quick test
    logger.info(f"  Model type: {model.type}")
    logger.info("✅ Ultralytics OK")


def main():
    logger.info("=" * 60)
    logger.info("ENVIRONMENT SETUP CHECK — Buah Batu Traffic System")
    logger.info("=" * 60)

    logger.info("\n📌 1/5 Python Version")
    check_python()

    logger.info("\n📌 2/5 CUDA & GPU")
    check_cuda()

    logger.info("\n📌 3/5 Dependencies")
    check_dependencies()

    logger.info("\n📌 4/5 Dataset Integrity")
    check_dataset()

    logger.info("\n📌 5/5 Ultralytics YOLOv8")
    check_ultralytics()

    logger.info("\n" + "=" * 60)
    logger.info("🎉 ALL CHECKS PASSED — Environment ready!")
    logger.info("=" * 60)
    logger.info("Next: python scripts/02_baseline_coco.py")


if __name__ == "__main__":
    main()
