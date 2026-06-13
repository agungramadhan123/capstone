"""
FASE 2 - CORE MONITORING APPLICATION (PRODUKSI & DEMO)

Vehicle Tracking, Counting & Analytics - Jalan Buah Batu
Arsitektur: Fail-Safe + Polygon ROI Multi-Directional

Fitur:
  1. Fail-Safe RTSP/HTTP -> MP4 fallback (anti-crash)
  2. YOLOv8 + ByteTrack dengan occlusion handling
  3. Polygon ROI multi-directional counting (4 arah)
  4. Anti-memory-leak CSV logging (TrafficLogger)
  5. Premium Visual HUD + trace + OD matrix
  6. Tool kalibrasi interaktif untuk polygon zone

Cara menjalankan:
  python src/main.py --source https://atcs-dishub.bandung.go.id:1990/Cikutra/main_stream.m3u8
  https://atcs-dishub.bandung.go.id:1990/MerdekaAceh/main_stream.m3u8
  python src/main.py --calibrate --source video.mp4
"""

import sys
import os
import argparse
from multiprocessing import freeze_support

# ── IMPORT HANDLING ───────────────────────────────────────────────────────────
# Mendukung dua cara menjalankan:
#   1. python src/main.py          (direct script)
#   2. python -m src.main          (module mode)

if __name__ == "__main__":
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

from src.config import (
    DEFAULT_MODEL, DEFAULT_VIDEO, DEFAULT_CSV, logger
)
from src.traffic_monitor import TrafficMonitorApp


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Traffic Monitor - Jalan Buah Batu, Bandung (Single Polygon Mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  python src/main.py video.mp4 --show
  python src/main.py --source "video dengan spasi.mp4" --show
  python src/main.py --source rtsp://user:pass@ip:port/stream --show
  python src/main.py video.mp4 --save-video
        """,
    )
    parser.add_argument(
        "source_positional", nargs="*", default=None,
        help="Sumber video (drag-and-drop atau ketik path langsung)"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help=f"Sumber video: path ke .mp4 atau URL RTSP (default: {DEFAULT_VIDEO})"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help="Path ke model YOLO (.pt)"
    )
    parser.add_argument(
        "--csv-output", type=str, default=DEFAULT_CSV,
        help=f"Path output CSV log (default: {DEFAULT_CSV})"
    )
    parser.add_argument(
        "--show", action="store_true", default=True,
        help="Tampilkan jendela OpenCV (default: True)"
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Jangan tampilkan jendela OpenCV"
    )
    parser.add_argument(
        "--save-video", action="store_true",
        help="Simpan video output ke file"
    )

    args = parser.parse_args()

    # Resolve sumber video: prioritas --source > positional > default
    if args.source:
        pass  # Gunakan --source langsung
    elif args.source_positional:
        args.source = " ".join(args.source_positional)
    else:
        args.source = DEFAULT_VIDEO

    # Bersihkan path: hapus kutip ganda jika ada (dari drag-drop Windows)
    args.source = args.source.strip().strip('"').strip("'")

    if args.no_show:
        args.show = False

    return args


def main():
    args = parse_args()

    # Mode monitoring: jalankan pipeline utama
    app = TrafficMonitorApp(args)
    app.run()


if __name__ == "__main__":
    freeze_support()
    main()
