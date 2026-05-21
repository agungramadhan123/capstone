"""
TRAINING SCRIPT - YOLOv8 Fine-Tuning CCTV Lalu Lintas
GPU    : NVIDIA RTX 5060 8GB
Model  : yolov8s fine-tuning dari best.pt
Dataset: Bubat Barat + Bubat Timur (~1831 gambar)
Kondisi: Pagi + Malam | Format: YOLOv8 (Roboflow)
"""

import os
import glob
from multiprocessing import freeze_support
MODEL_PRETRAINED = "best.pt"       
DATA_YAML        = "data.yaml"     
HYP_YAML         = "train_cctv.yaml"
PROJECT_NAME     = "cctv_bubat"
RUN_NAME         = "finetune_v1"

def _find_data_yaml(path="data.yaml"):
    if os.path.exists(path):
        return path
    matches = glob.glob("**/data.yaml", recursive=True)
    return matches[0] if matches else None

_detected_data = _find_data_yaml(DATA_YAML)
if _detected_data is None:
    raise FileNotFoundError(
        "Tidak menemukan file data.yaml. Set `DATA_YAML` di train.py ke path yang benar."
    )
DATA_YAML = _detected_data
if not os.path.exists(MODEL_PRETRAINED):
    MODEL_PRETRAINED = "yolov8s.pt"

def main():
    from ultralytics import YOLO
    model = YOLO(MODEL_PRETRAINED)
    results = model.train(
        data = DATA_YAML,

        imgsz       = 640,
        epochs      = 300,
        batch       = 16,              

        device      = 0,               
        amp         = True,          
        workers     = 4,             
        cache       = "ram",                                    
        dropout     = 0.1,
        weight_decay= 0.0005,
        lr0         = 0.001,         
        lrf         = 0.01,
        warmup_epochs = 3,
        patience    = 50,             
        save        = True,
        save_period = 20,             
        project     = PROJECT_NAME,
        name        = RUN_NAME,

        # Augmentasi (override dari YAML jika diperlukan)
        # Sudah didefinisikan di train_cctv.yaml
        # Uncomment baris di bawah untuk override langsung:

        # hsv_h      = 0.010,
        # hsv_s      = 0.5,
        # hsv_v      = 0.4,
        # degrees    = 0.0,
        # translate  = 0.1,
        # scale      = 0.4,
        # shear      = 0.0,
        # perspective= 0.0,
        # flipud     = 0.0,
        # fliplr     = 0.3,
        # mosaic     = 0.8,
        # mixup      = 0.05,
        # copy_paste = 0.1,
        # erasing    = 0.3,
        # close_mosaic = 20,

        # Verbose
        verbose     = True,
    )

    print("\nTraining selesai!")
    print(f"Best model: {results.save_dir}/weights/best.pt")


if __name__ == '__main__':
    freeze_support()
    main()
