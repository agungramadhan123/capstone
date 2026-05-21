"""
=============================================================
 FASE 1 — CLEANING DATASET & RE-VALIDATION (HULU)
 
 Skrip ini melakukan 3 langkah otomatis:
   1. Bersihkan kelas bias 'Labelling-data-lalu-lintas' dari data.yaml
   2. Re-mapping indeks label di seluruh file annotation
   3. Validasi ulang model best.pt dengan dataset yang sudah bersih
   
 Cara menjalankan:
   python clean_and_validate.py
   python clean_and_validate.py --skip-validate   # hanya clean, tanpa validasi
   python clean_and_validate.py --dry-run          # preview tanpa mengubah file
=============================================================
"""

import os
import sys
import glob
import yaml
import argparse
import shutil
from pathlib import Path
from datetime import datetime


# ── KONFIGURASI ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# Kelas bias yang harus dihapus
BIAS_CLASS = "Labelling-data-lalu-lintas"

# Kelas valid yang diharapkan (urutan final)
VALID_CLASSES = ["Bis", "Mobil", "Motor", "Truk"]

# Daftar data.yaml yang perlu dibersihkan
DATA_YAML_PATHS = [
    PROJECT_ROOT / "valid" / "data.yaml",
    # Tambahkan path lain jika ada data.yaml yang masih kotor
]

# Daftar direktori label yang perlu di-remap
LABEL_DIRS = [
    PROJECT_ROOT / "valid" / "labels",
]

# Path ke model terbaik
BEST_MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "cctv_bubat" / "finetune_v1-9" / "weights" / "best.pt"


# ── STEP 1: CLEAN DATA.YAML ─────────────────────────────────
def clean_data_yaml(yaml_path: Path, dry_run: bool = False) -> dict:
    """
    Bersihkan kelas bias dari file data.yaml.
    
    Returns:
        dict: Mapping indeks lama → indeks baru untuk re-mapping label
    """
    if not yaml_path.exists():
        print(f"  ⚠️  File tidak ditemukan: {yaml_path}")
        return {}

    # Baca konfigurasi YAML
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    names = config.get("names", [])
    nc = config.get("nc", len(names))

    print(f"\n  📄 File  : {yaml_path}")
    print(f"  📊 Kelas : {nc} → {names}")

    # Cek apakah kelas bias ada
    if BIAS_CLASS not in names:
        print(f"  ✅ Sudah bersih! Kelas '{BIAS_CLASS}' tidak ditemukan.")
        # Tetap buat mapping identitas untuk label dirs yang terkait
        return {i: i for i in range(len(names))}

    # Catat indeks kelas bias
    bias_idx = names.index(BIAS_CLASS)
    print(f"  🎯 Kelas bias ditemukan di indeks: {bias_idx}")

    # Buat mapping indeks: lama → baru
    # Contoh: jika bias_idx=1 dan names=[Bis, BIAS, Mobil, Motor, Truk]
    # Mapping: 0→0, 1→HAPUS, 2→1, 3→2, 4→3
    index_map = {}
    new_idx = 0
    for old_idx in range(len(names)):
        if old_idx == bias_idx:
            index_map[old_idx] = None  # Tandai untuk dihapus
        else:
            index_map[old_idx] = new_idx
            new_idx += 1

    print(f"  🔄 Mapping indeks: {index_map}")

    # Hapus kelas bias
    new_names = [n for n in names if n != BIAS_CLASS]
    new_nc = len(new_names)

    print(f"  📊 Hasil : {new_nc} → {new_names}")

    # Validasi hasil
    if new_names != VALID_CLASSES:
        print(f"  ⚠️  PERINGATAN: Kelas tidak sesuai ekspektasi!")
        print(f"      Ekspektasi : {VALID_CLASSES}")
        print(f"      Aktual     : {new_names}")

    if dry_run:
        print(f"  🔍 [DRY-RUN] Tidak ada perubahan yang ditulis.")
        return index_map

    # Backup file asli
    backup_path = yaml_path.with_suffix(".yaml.bak")
    shutil.copy2(yaml_path, backup_path)
    print(f"  💾 Backup : {backup_path}")

    # Tulis file bersih
    config["names"] = new_names
    config["nc"] = new_nc

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"  ✅ data.yaml berhasil dibersihkan!")
    return index_map


# ── STEP 2: REMAP LABEL FILES ───────────────────────────────
def remap_labels(label_dir: Path, index_map: dict, dry_run: bool = False):
    """
    Re-mapping indeks kelas di seluruh file label (.txt).
    Menghapus baris dengan kelas bias dan menggeser indeks kelas lainnya.
    """
    if not label_dir.exists():
        print(f"  ⚠️  Direktori label tidak ditemukan: {label_dir}")
        return

    label_files = sorted(label_dir.glob("*.txt"))
    if not label_files:
        print(f"  ⚠️  Tidak ada file label di: {label_dir}")
        return

    print(f"\n  📁 Label dir  : {label_dir}")
    print(f"  📄 Total file : {len(label_files)}")

    # Cek apakah perlu remap (ada indeks yang berubah?)
    needs_remap = any(
        old != new for old, new in index_map.items() if new is not None
    ) or any(new is None for new in index_map.values())

    if not needs_remap:
        print(f"  ✅ Tidak perlu re-mapping (indeks sudah sesuai).")
        return

    stats = {
        "files_modified": 0,
        "lines_removed": 0,
        "lines_remapped": 0,
        "files_unchanged": 0,
    }

    for label_file in label_files:
        with open(label_file, "r") as f:
            lines = f.readlines()

        new_lines = []
        modified = False

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            try:
                old_class_id = int(parts[0])
            except ValueError:
                new_lines.append(line)
                continue

            # Cek apakah kelas ini perlu dihapus
            new_class_id = index_map.get(old_class_id)

            if new_class_id is None:
                # Kelas bias → hapus baris ini
                stats["lines_removed"] += 1
                modified = True
                continue

            if new_class_id != old_class_id:
                # Re-map indeks
                parts[0] = str(new_class_id)
                new_lines.append(" ".join(parts) + "\n")
                stats["lines_remapped"] += 1
                modified = True
            else:
                new_lines.append(line)

        if modified:
            stats["files_modified"] += 1
            if not dry_run:
                with open(label_file, "w") as f:
                    f.writelines(new_lines)
        else:
            stats["files_unchanged"] += 1

    prefix = "[DRY-RUN] " if dry_run else ""
    print(f"  {prefix}📊 Hasil re-mapping:")
    print(f"      File dimodifikasi : {stats['files_modified']}")
    print(f"      File tidak berubah: {stats['files_unchanged']}")
    print(f"      Baris dihapus     : {stats['lines_removed']}")
    print(f"      Baris di-remap    : {stats['lines_remapped']}")


# ── STEP 3: RE-VALIDASI MODEL ───────────────────────────────
def validate_model(data_yaml_path: Path, model_path: Path):
    """
    Jalankan validasi ulang model best.pt dengan data.yaml yang sudah bersih.
    """
    if not model_path.exists():
        print(f"\n  ❌ Model tidak ditemukan: {model_path}")
        print(f"     Pastikan file best.pt ada di lokasi yang benar.")
        return None

    if not data_yaml_path.exists():
        print(f"\n  ❌ data.yaml tidak ditemukan: {data_yaml_path}")
        return None

    print(f"\n  🤖 Model    : {model_path}")
    print(f"  📄 Data YAML: {data_yaml_path}")
    print(f"  ⏳ Memulai validasi...\n")

    try:
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        results = model.val(
            data=str(data_yaml_path),
            imgsz=640,
            batch=16,
            device=0,
            verbose=True,
        )

        # Tampilkan hasil
        print(f"\n{'='*55}")
        print(f"  📊 HASIL VALIDASI ULANG (CLEAN DATASET)")
        print(f"{'='*55}")
        print(f"  mAP@0.5      : {results.box.map50:.4f}")
        print(f"  mAP@0.5:0.95 : {results.box.map:.4f}")
        print(f"  Precision     : {results.box.mp:.4f}")
        print(f"  Recall        : {results.box.mr:.4f}")

        # Per-class metrics
        if hasattr(results.box, "ap50") and results.box.ap50 is not None:
            print(f"\n  Per-Kelas AP@0.5:")
            for i, cls_name in enumerate(VALID_CLASSES):
                if i < len(results.box.ap50):
                    print(f"    {cls_name:20s}: {results.box.ap50[i]:.4f}")

        print(f"{'='*55}")
        return results

    except ImportError:
        print("  ❌ Ultralytics belum terinstall. Jalankan: pip install ultralytics")
        return None
    except Exception as e:
        print(f"  ❌ Error saat validasi: {e}")
        return None


# ── MAIN ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fase 1: Cleaning Dataset & Re-Validation",
    )
    parser.add_argument(
        "--skip-validate", action="store_true",
        help="Lewati langkah validasi model (hanya cleaning)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview perubahan tanpa menulis file"
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*55}")
    print(f"  FASE 1 — CLEANING DATASET & RE-VALIDATION")
    print(f"  Waktu : {timestamp}")
    print(f"  Mode  : {'DRY-RUN (preview)' if args.dry_run else 'LIVE (modifikasi file)'}")
    print(f"{'='*55}")

    # ── Step 1: Clean data.yaml ──
    print(f"\n{'─'*55}")
    print(f"  STEP 1: Membersihkan data.yaml")
    print(f"{'─'*55}")

    combined_index_map = {}
    for yaml_path in DATA_YAML_PATHS:
        index_map = clean_data_yaml(yaml_path, dry_run=args.dry_run)
        if index_map:
            combined_index_map = index_map

    # ── Step 2: Remap labels ──
    print(f"\n{'─'*55}")
    print(f"  STEP 2: Re-mapping indeks label")
    print(f"{'─'*55}")

    if combined_index_map:
        for label_dir in LABEL_DIRS:
            remap_labels(label_dir, combined_index_map, dry_run=args.dry_run)
    else:
        print("  ⚠️  Tidak ada mapping indeks. Lewati re-mapping.")

    # ── Step 3: Validate ──
    if not args.skip_validate and not args.dry_run:
        print(f"\n{'─'*55}")
        print(f"  STEP 3: Validasi ulang model")
        print(f"{'─'*55}")

        # Gunakan data.yaml pertama yang ditemukan
        val_yaml = DATA_YAML_PATHS[0] if DATA_YAML_PATHS else None
        if val_yaml:
            validate_model(val_yaml, BEST_MODEL_PATH)
        else:
            print("  ⚠️  Tidak ada data.yaml untuk validasi.")
    elif args.skip_validate:
        print(f"\n  ⏭️  Validasi dilewati (--skip-validate)")
    elif args.dry_run:
        print(f"\n  ⏭️  Validasi dilewati (--dry-run mode)")

    print(f"\n{'='*55}")
    print(f"  ✅ Fase 1 selesai!")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
