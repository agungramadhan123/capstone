"""
Central Configuration — Single Source of Truth
Semua script import dari sini. Ubah value di sini, efek ke seluruh pipeline.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class PathConfig:
    project_root: Path = Path(r"d:\Semester 6\Capstone")

    @property
    def data_yaml(self) -> Path:
        return self.project_root / "data.yaml"

    @property
    def output_dir(self) -> Path:
        d = self.project_root / "outputs"
        d.mkdir(exist_ok=True)
        return d

    @property
    def runs_dir(self) -> Path:
        d = self.project_root / "runs"
        d.mkdir(exist_ok=True)
        return d

    @property
    def tracker_config(self) -> Path:
        return self.project_root / "configs" / "bytetrack.yaml"


@dataclass
class ModelConfig:
    weights: str = "yolov8m.pt"       # Medium model: balance speed/accuracy
    img_size: int = 880               # Match dataset resolution
    conf_threshold: float = 0.25
    iou_threshold: float = 0.6        # Tuned from EDA overlap analysis
    device: str = "0"
    half: bool = True                 # FP16 for faster inference


@dataclass
class TrainConfig:
    epochs: int = 150
    batch_size: int = 8               # RTX 5060 8GB safe value
    patience: int = 30
    optimizer: str = "AdamW"
    lr0: float = 0.001
    lrf: float = 0.01
    weight_decay: float = 0.0005
    warmup_epochs: int = 5
    # Augmentation — tuned for tilted camera + small objects
    degrees: float = 10.0
    translate: float = 0.15
    scale: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.1
    copy_paste: float = 0.1
    # Loss weights — box loss higher for small object precision
    cls_loss: float = 1.0
    box_loss: float = 7.5
    dfl_loss: float = 1.5


@dataclass
class TrackConfig:
    track_high_thresh: float = 0.5    # High-conf detection threshold
    track_low_thresh: float = 0.1     # Low-conf for occluded objects
    new_track_thresh: float = 0.6
    track_buffer: int = 30            # Frames to keep lost tracks
    match_thresh: float = 0.8


@dataclass
class CountConfig:
    # Virtual line coordinates (will be set per-video)
    line_start: Tuple[int, int] = (0, 528)
    line_end: Tuple[int, int] = (880, 528)
    direction: str = "down"           # Arah kendaraan di Buah Batu
    min_track_length: int = 5         # Min frames sebelum track dihitung
    max_lost_frames: int = 30         # Max frames track bisa hilang


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    count: CountConfig = field(default_factory=CountConfig)

    class_names: List[str] = field(default_factory=lambda: [
        'Bis mini', 'Manusia', 'Mobil', 'Motor',
        'Truk berat', 'Truk menengah', 'Truk ringan', 'bus'
    ])

    # Indeks kelas kendaraan (exclude Manusia idx=1)
    vehicle_class_ids: List[int] = field(default_factory=lambda: [0, 2, 3, 4, 5, 6, 7])


# Singleton — import ini dari mana saja
cfg = Config()
