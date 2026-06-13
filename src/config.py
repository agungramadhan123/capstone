"""
Konfigurasi default dan konstanta untuk Traffic Monitor.
Semua path, warna, dan parameter global didefinisikan di sini.
"""

import logging
from pathlib import Path

import supervision as sv

# ── PATH KONFIGURASI ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL = str(
    PROJECT_ROOT / "models" / "best_bubat.pt"
)
DEFAULT_RTSP = "rtsp://username:password@ip:port/stream"
DEFAULT_VIDEO = str(PROJECT_ROOT / "video_buahbatu.mp4")
DEFAULT_CSV = str(PROJECT_ROOT / "logs" / "traffic_logs_buahbatu.csv")
TRACKER_CONFIG = str(PROJECT_ROOT / "config" / "custom_bytetrack.yaml")
DEFAULT_ROI_CONFIG = str(PROJECT_ROOT / "config" / "roi_zones.yaml")

# ── KELAS KENDARAAN ──────────────────────────────────────────────────────────
# Akan disinkronkan secara dinamis saat model dimuat
CLASS_NAMES = {0: "Bis", 1: "Mobil", 2: "Motor", 3: "Truk"}
CLASS_EMOJIS = {"Bis": "", "Mobil": "", "Motor": "", "Truk": ""}

# ── PARAMETER DETEKSI ────────────────────────────────────────────────────────
# Displacement minimal (piksel) untuk mencegah double-count saat macet
MIN_DISPLACEMENT_PX = 5

# ── WARNA HUD (BGR) ──────────────────────────────────────────────────────────
COLOR_HUD_BG = (30, 30, 30)          # Dark background
COLOR_HUD_TEXT = (255, 255, 255)      # White text
COLOR_HUD_ACCENT = (0, 200, 150)     # Teal accent

# Warna default zona ROI (BGR)
ROI_ZONE_COLORS = {
    "Utara":   (0, 255, 0),      # Hijau
    "Selatan": (0, 0, 255),      # Merah
    "Timur":   (255, 0, 0),      # Biru
    "Barat":   (0, 255, 255),    # Kuning
}

# ── LOGGING SETUP ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")
