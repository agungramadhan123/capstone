"""
EDA Dataset Vehicle Tracking, Counting, and Analytics
Struktur disesuaikan dengan folder capstone-agung.

Fungsi:
1. EDA data mentah CCTV di folder Data/
2. EDA dataset YOLO hasil Roboflow
3. Hitung jumlah gambar
4. Hitung jumlah bounding box per kelas
5. Deteksi label kosong
6. Deteksi class ID invalid
7. Deteksi gambar tanpa label
8. Export hasil EDA ke folder eda_outputs/

Cara menjalankan:
    python eda_dataset.py
"""

import os
import cv2
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict


PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DATA_DIRS = [
    PROJECT_ROOT / "Data" / "bubat barat" / "pagi",
    PROJECT_ROOT / "Data" / "bubat barat" / "malam",
]

YOLO_DATASET_DIRS = {
    "train_bubat_barat": PROJECT_ROOT / "data bubat barat" / "train",
    "train_bubat_timur": PROJECT_ROOT / "data bubat timur" / "train",
    "valid": PROJECT_ROOT / "valid",
}

DATA_YAML_CANDIDATES = [
    PROJECT_ROOT / "valid" / "data.yaml",
    PROJECT_ROOT / "data bubat barat" / "data.yaml",
    PROJECT_ROOT / "data bubat timur" / "data.yaml",
]

OUTPUT_DIR = PROJECT_ROOT / "eda_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_CLASSES = ["Bis", "Mobil", "Motor", "Truk"]


def load_class_names():
    """
    Ambil nama kelas dari data.yaml.
    Kalau gagal, pakai default 4 kelas kendaraan.
    """
    for yaml_path in DATA_YAML_CANDIDATES:
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            names = data.get("names", DEFAULT_CLASSES)

            if isinstance(names, dict):
                names = [names[i] for i in sorted(names.keys())]

            print(f"Class names diambil dari: {yaml_path}")
            print(f"Classes: {names}")
            return names

    print("data.yaml tidak ditemukan. Pakai default classes.")
    return DEFAULT_CLASSES


def get_images(folder: Path):
    if not folder.exists():
        return []

    return sorted([
        file for file in folder.rglob("*")
        if file.suffix.lower() in IMAGE_EXTENSIONS
    ])


def read_yolo_label(label_path: Path):
    """
    Format label YOLO:
    class_id x_center y_center width height
    """
    if not label_path.exists():
        return []

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    labels = []

    for line_no, line in enumerate(lines, start=1):
        parts = line.strip().split()

        if len(parts) == 0:
            continue

        if len(parts) != 5:
            labels.append({
                "valid": False,
                "line_no": line_no,
                "error": "format_bukan_5_kolom",
                "raw": line.strip(),
            })
            continue

        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            labels.append({
                "valid": True,
                "line_no": line_no,
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
                "raw": line.strip(),
            })

        except ValueError:
            labels.append({
                "valid": False,
                "line_no": line_no,
                "error": "nilai_tidak_valid",
                "raw": line.strip(),
            })

    return labels


def find_images_labels_dirs(dataset_root: Path):
    """
    Mendukung struktur:
    train/images + train/labels
    valid/images + valid/labels
    atau langsung folder yang berisi images/labels.
    """
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"

    if images_dir.exists() and labels_dir.exists():
        return images_dir, labels_dir

    # fallback jika dataset_root langsung berisi gambar dan labels
    if labels_dir.exists():
        return dataset_root, labels_dir

    return None, None


def analyze_raw_data():
    """
    EDA data mentah sebelum labeling.
    """
    rows = []

    print("\n" + "=" * 70)
    print("EDA DATA MENTAH CCTV")
    print("=" * 70)

    for folder in RAW_DATA_DIRS:
        images = get_images(folder)

        print(f"\nFolder: {folder}")
        print(f"Jumlah gambar: {len(images)}")

        for image_path in images:
            img = cv2.imread(str(image_path))

            if img is None:
                rows.append({
                    "folder": str(folder),
                    "image_path": str(image_path),
                    "status": "gagal_dibaca",
                    "width": None,
                    "height": None,
                })
                continue

            h, w = img.shape[:2]

            rows.append({
                "folder": str(folder),
                "image_path": str(image_path),
                "status": "ok",
                "width": w,
                "height": h,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "raw_data_summary.csv", index=False)

    if not df.empty:
        summary = (
            df.groupby(["folder", "width", "height"])
            .size()
            .reset_index(name="jumlah_gambar")
        )
        summary.to_csv(OUTPUT_DIR / "raw_image_size_distribution.csv", index=False)

    return df


def analyze_yolo_dataset(class_names):
    """
    EDA dataset YOLO hasil Roboflow.
    """
    summary_rows = []
    bbox_rows = []
    invalid_rows = []
    image_rows = []

    print("\n" + "=" * 70)
    print("EDA DATASET YOLO / ROBOFLOW")
    print("=" * 70)

    for split_name, dataset_root in YOLO_DATASET_DIRS.items():
        print(f"\nSplit: {split_name}")
        print(f"Path : {dataset_root}")

        images_dir, labels_dir = find_images_labels_dirs(dataset_root)

        if images_dir is None or labels_dir is None:
            print("Folder images/labels tidak ditemukan. Dilewati.")
            continue

        images = get_images(images_dir)
        labels = sorted(labels_dir.rglob("*.txt")) if labels_dir.exists() else []

        image_stems = {img.stem for img in images}
        label_stems = {lbl.stem for lbl in labels}

        gambar_tanpa_label = image_stems - label_stems
        label_tanpa_gambar = label_stems - image_stems

        class_counter = Counter()
        empty_label_count = 0
        missing_label_count = 0
        invalid_count = 0
        total_bbox = 0

        for image_path in images:
            img = cv2.imread(str(image_path))

            if img is None:
                image_rows.append({
                    "split": split_name,
                    "image_path": str(image_path),
                    "status": "gagal_dibaca",
                    "width": None,
                    "height": None,
                })
                continue

            h, w = img.shape[:2]

            image_rows.append({
                "split": split_name,
                "image_path": str(image_path),
                "status": "ok",
                "width": w,
                "height": h,
            })

            label_path = labels_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                missing_label_count += 1
                continue

            label_data = read_yolo_label(label_path)

            if len(label_data) == 0:
                empty_label_count += 1
                continue

            for label in label_data:
                if not label.get("valid"):
                    invalid_count += 1
                    invalid_rows.append({
                        "split": split_name,
                        "label_path": str(label_path),
                        "line_no": label.get("line_no"),
                        "error": label.get("error"),
                        "raw": label.get("raw"),
                    })
                    continue

                class_id = label["class_id"]

                if class_id < 0 or class_id >= len(class_names):
                    invalid_count += 1
                    invalid_rows.append({
                        "split": split_name,
                        "label_path": str(label_path),
                        "line_no": label.get("line_no"),
                        "error": "class_id_invalid",
                        "raw": label.get("raw"),
                    })
                    continue

                class_name = class_names[class_id]
                class_counter[class_name] += 1
                total_bbox += 1

                bbox_w_px = label["width"] * w
                bbox_h_px = label["height"] * h
                bbox_area_norm = label["width"] * label["height"]

                bbox_rows.append({
                    "split": split_name,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "class_id": class_id,
                    "class_name": class_name,
                    "x_center": label["x_center"],
                    "y_center": label["y_center"],
                    "bbox_width_norm": label["width"],
                    "bbox_height_norm": label["height"],
                    "bbox_area_norm": bbox_area_norm,
                    "bbox_width_px": bbox_w_px,
                    "bbox_height_px": bbox_h_px,
                    "image_width": w,
                    "image_height": h,
                })

        summary_rows.append({
            "split": split_name,
            "dataset_root": str(dataset_root),
            "jumlah_gambar": len(images),
            "jumlah_file_label": len(labels),
            "jumlah_bbox": total_bbox,
            "gambar_tanpa_label": len(gambar_tanpa_label),
            "label_tanpa_gambar": len(label_tanpa_gambar),
            "label_kosong": empty_label_count,
            "label_hilang": missing_label_count,
            "label_invalid": invalid_count,
        })

        print(f"Jumlah gambar       : {len(images)}")
        print(f"Jumlah file label   : {len(labels)}")
        print(f"Jumlah bbox valid   : {total_bbox}")
        print(f"Gambar tanpa label  : {len(gambar_tanpa_label)}")
        print(f"Label tanpa gambar  : {len(label_tanpa_gambar)}")
        print(f"Label kosong        : {empty_label_count}")
        print(f"Label invalid       : {invalid_count}")

        print("Distribusi kelas:")
        for cls in class_names:
            print(f"  {cls:10s}: {class_counter.get(cls, 0)}")

    summary_df = pd.DataFrame(summary_rows)
    bbox_df = pd.DataFrame(bbox_rows)
    invalid_df = pd.DataFrame(invalid_rows)
    image_df = pd.DataFrame(image_rows)

    summary_df.to_csv(OUTPUT_DIR / "yolo_summary_per_split.csv", index=False)
    bbox_df.to_csv(OUTPUT_DIR / "yolo_bbox_detail.csv", index=False)
    invalid_df.to_csv(OUTPUT_DIR / "yolo_invalid_labels.csv", index=False)
    image_df.to_csv(OUTPUT_DIR / "yolo_image_detail.csv", index=False)

    return summary_df, bbox_df, invalid_df, image_df


def make_plots(summary_df, bbox_df, image_df):
    """
    Membuat grafik sederhana untuk laporan EDA.
    """
    if not summary_df.empty:
        plt.figure(figsize=(10, 5))
        plt.bar(summary_df["split"], summary_df["jumlah_gambar"])
        plt.title("Jumlah Gambar per Split")
        plt.xlabel("Split")
        plt.ylabel("Jumlah Gambar")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "plot_jumlah_gambar_per_split.png", dpi=200)
        plt.close()

    if not bbox_df.empty:
        class_dist = (
            bbox_df.groupby("class_name")
            .size()
            .reset_index(name="jumlah_bbox")
            .sort_values("jumlah_bbox", ascending=False)
        )

        class_dist.to_csv(OUTPUT_DIR / "class_distribution_total.csv", index=False)

        plt.figure(figsize=(10, 5))
        plt.bar(class_dist["class_name"], class_dist["jumlah_bbox"])
        plt.title("Distribusi Bounding Box per Kelas")
        plt.xlabel("Kelas Kendaraan")
        plt.ylabel("Jumlah Bounding Box")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "plot_distribusi_kelas.png", dpi=200)
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.hist(bbox_df["bbox_area_norm"], bins=30)
        plt.title("Distribusi Luas Bounding Box")
        plt.xlabel("Luas Bounding Box Normalized")
        plt.ylabel("Frekuensi")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "plot_bbox_area.png", dpi=200)
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.hist(bbox_df["bbox_width_px"], bins=30)
        plt.title("Distribusi Lebar Bounding Box Pixel")
        plt.xlabel("Lebar Bounding Box Pixel")
        plt.ylabel("Frekuensi")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "plot_bbox_width.png", dpi=200)
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.hist(bbox_df["bbox_height_px"], bins=30)
        plt.title("Distribusi Tinggi Bounding Box Pixel")
        plt.xlabel("Tinggi Bounding Box Pixel")
        plt.ylabel("Frekuensi")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "plot_bbox_height.png", dpi=200)
        plt.close()

    if not image_df.empty:
        image_size_dist = (
            image_df.dropna(subset=["width", "height"])
            .groupby(["width", "height"])
            .size()
            .reset_index(name="jumlah_gambar")
            .sort_values("jumlah_gambar", ascending=False)
        )

        image_size_dist.to_csv(OUTPUT_DIR / "image_size_distribution.csv", index=False)

        if not image_size_dist.empty:
            image_size_dist["size"] = (
                image_size_dist["width"].astype(int).astype(str)
                + "x"
                + image_size_dist["height"].astype(int).astype(str)
            )

            plt.figure(figsize=(10, 5))
            plt.bar(image_size_dist["size"], image_size_dist["jumlah_gambar"])
            plt.title("Distribusi Ukuran Gambar")
            plt.xlabel("Ukuran Gambar")
            plt.ylabel("Jumlah")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "plot_image_size.png", dpi=200)
            plt.close()


def main():
    print("\nEDA DATASET CAPSTONE VEHICLE TRACKING")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output folder: {OUTPUT_DIR}")

    class_names = load_class_names()

    raw_df = analyze_raw_data()
    summary_df, bbox_df, invalid_df, image_df = analyze_yolo_dataset(class_names)

    make_plots(summary_df, bbox_df, image_df)

    print("\n" + "=" * 70)
    print("EDA SELESAI")
    print("=" * 70)
    print(f"Output disimpan di: {OUTPUT_DIR}")
    print("\nFile penting yang dihasilkan:")
    print("- raw_data_summary.csv")
    print("- raw_image_size_distribution.csv")
    print("- yolo_summary_per_split.csv")
    print("- yolo_bbox_detail.csv")
    print("- yolo_invalid_labels.csv")
    print("- yolo_image_detail.csv")
    print("- class_distribution_total.csv")
    print("- image_size_distribution.csv")
    print("- plot_jumlah_gambar_per_split.png")
    print("- plot_distribusi_kelas.png")
    print("- plot_bbox_area.png")
    print("- plot_bbox_width.png")
    print("- plot_bbox_height.png")
    print("- plot_image_size.png")


if __name__ == "__main__":
    main()