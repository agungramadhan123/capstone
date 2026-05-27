"""
FASE 1 - CLEANING DATASET & RE-VALIDATION (HULU)

Skrip ini otomatis membersihkan kelas bias 'Labelling-data-lalu-lintas' HANYA jika diperlukan.
Setiap folder diperiksa secara independen agar tidak merusak data yang sudah bersih.
"""

import os
import glob
import yaml
import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Naik ke root project (Capstone/)
BIAS_CLASS = "Labelling-data-lalu-lintas"
STANDARD_CLASSES = ["Bis", "Mobil", "Motor", "Truk"]

# Pemetaan folder dataset (data.yaml dan folder labels-nya)
DATASETS = [
    {
        "yaml_path": PROJECT_ROOT / "valid" / "data.yaml",
        "label_dir": PROJECT_ROOT / "valid" / "labels"
    },
    {
        "yaml_path": PROJECT_ROOT / "data bubat barat" / "data.yaml",
        "label_dir": PROJECT_ROOT / "data bubat barat" / "train" / "labels"
    },
    {
        "yaml_path": PROJECT_ROOT / "data bubat timur" / "data.yaml",
        "label_dir": PROJECT_ROOT / "data bubat timur" / "train" / "labels"
    }
]

def clean_dataset(dataset, dry_run=False):
    yaml_path = dataset["yaml_path"]
    label_dir = dataset["label_dir"]
    
    if not yaml_path.exists():
        print(f"File YAML tidak ditemukan: {yaml_path}")
        return

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    names = config.get("names", [])
    print(f"\n Memproses Dataset: {yaml_path.parent.name}")
    print(f"   Kelas asli: {names}")

    bias_idx = None
    if BIAS_CLASS in names:
        bias_idx = names.index(BIAS_CLASS)
        print(f"Kelas bias '{BIAS_CLASS}' ditemukan di index: {bias_idx}")
    else:
        print(f" Kelas bias tidak ditemukan. File .txt aman (tidak perlu di-remap).")

    # Update YAML agar seragam menggunakan STANDARD_CLASSES
    if not dry_run:
        backup_path = yaml_path.with_suffix(".yaml.bak")
        shutil.copy2(yaml_path, backup_path)
        
        config["names"] = STANDARD_CLASSES
        config["nc"] = len(STANDARD_CLASSES)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"{yaml_path.name} distandarisasi menjadi: {STANDARD_CLASSES}")

    # Jika tidak ada kelas bias, kita tidak perlu memodifikasi file .txt
    if bias_idx is None:
        return

    # Proses modifikasi file .txt
    if not label_dir.exists():
        print(f" Folder label tidak ditemukan: {label_dir}")
        return

    label_files = list(label_dir.glob("*.txt"))
    files_modified = 0
    lines_removed = 0
    lines_remapped = 0

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
                class_id = int(parts[0])
                if class_id == bias_idx:
                    # Hapus baris ini
                    lines_removed += 1
                    modified = True
                elif class_id > bias_idx:
                    # Geser index
                    parts[0] = str(class_id - 1)
                    new_lines.append(" ".join(parts) + "\n")
                    lines_remapped += 1
                    modified = True
                else:
                    new_lines.append(line)
            except ValueError:
                new_lines.append(line)

        if modified:
            files_modified += 1
            if not dry_run:
                with open(label_file, "w") as f:
                    f.writelines(new_lines)

    print(f"     Re-mapping selesai:")
    print(f"      File diubah   : {files_modified}")
    print(f"      Baris dihapus : {lines_removed}")
    print(f"      Baris digeser : {lines_remapped}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Preview tanpa modifikasi")
    args = parser.parse_args()

    print(" MEMULAI CLEANING & VALIDASI DATASET")
    for dataset in DATASETS:
        clean_dataset(dataset, dry_run=args.dry_run)
    print("\nProses Selesai! Semua dataset telah distandarisasi dan dibersihkan.")

if __name__ == "__main__":
    main()
