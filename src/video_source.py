"""
Video Source Manager - Fail-Safe RTSP → MP4 Fallback

Mengelola input video dengan mekanisme fail-safe:
- Coba RTSP/HTTP stream terlebih dahulu
- Jika gagal/terputus → otomatis fallback ke video lokal
- Reconnect berkala ke RTSP setiap N detik
"""

import os
import time

import cv2
from pathlib import Path

from .config import logger


class VideoSourceManager:
    """
    Mengelola input video dengan mekanisme fail-safe:
    - Coba RTSP stream terlebih dahulu
    - Jika gagal/terputus -> otomatis fallback ke video lokal
    - Reconnect berkala ke RTSP setiap N detik
    """

    def __init__(self, rtsp_url: str, fallback_path: str, reconnect_interval: int = 30):
        self.rtsp_url = rtsp_url
        self.fallback_path = fallback_path
        self.reconnect_interval = reconnect_interval

        self._cap = None
        self._using_rtsp = False
        self._last_reconnect_attempt = 0
        self._frame_count = 0
        self._source_label = "NONE"

    @property
    def is_rtsp(self) -> bool:
        return self._using_rtsp

    @property
    def source_label(self) -> str:
        return self._source_label

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def open(self) -> bool:
        """Buka sumber video. Coba RTSP dulu, lalu fallback."""
        if self._try_open_rtsp():
            return True

        return self._try_open_fallback()

    def _try_open_rtsp(self) -> bool:
        """Coba membuka stream jaringan utama (RTSP/HTTP/HTTPS)."""
        if not self.rtsp_url:
            return False
        
        is_rtsp = self.rtsp_url.startswith("rtsp://")
        label = "RTSP LIVE" if is_rtsp else "LIVE STREAM"
        
        try:
            logger.info(f"Mencoba koneksi {label}: {self.rtsp_url[:50]}...")
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            
            # Set timeout pendek untuk live stream
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self._release_current()
                    self._cap = cap
                    self._using_rtsp = True
                    self._source_label = label
                    logger.info(f"{label} berhasil terhubung!")
                    return True
                else:
                    cap.release()
                    logger.warning(f"{label} terbuka tapi tidak bisa membaca frame.")
            else:
                logger.warning(f"{label} gagal dibuka.")

        except Exception as e:
            logger.warning(f"Exception saat koneksi {label}: {e}")

        self._last_reconnect_attempt = time.time()
        return False

    def _try_open_fallback(self) -> bool:
        """Buka video lokal sebagai fallback."""
        if not os.path.exists(self.fallback_path):
            logger.error(f"Video fallback tidak ditemukan: {self.fallback_path}")
            return False

        try:
            cap = cv2.VideoCapture(self.fallback_path)
            if cap.isOpened():
                self._release_current()
                self._cap = cap
                self._using_rtsp = False
                self._source_label = f"LOCAL ({Path(self.fallback_path).name})"
                logger.info(f"Fallback ke video lokal: {self.fallback_path}")
                return True
        except Exception as e:
            logger.error(f"Gagal membuka video fallback: {e}")

        return False

    def read(self):
        """
        Baca frame berikutnya. Jika gagal, coba reconnect atau fallback.
        Returns: (success: bool, frame: np.ndarray | None)
        """
        if self._cap is None:
            return False, None

        try:
            ret, frame = self._cap.read()

            if ret and frame is not None:
                self._frame_count += 1

                # Periodically coba reconnect ke RTSP jika sedang di fallback
                if not self._using_rtsp:
                    elapsed = time.time() - self._last_reconnect_attempt
                    if elapsed >= self.reconnect_interval:
                        self._try_rtsp_reconnect()

                return True, frame

            else:
                # Frame gagal dibaca
                if self._using_rtsp:
                    # RTSP terputus mid-stream -> switch ke fallback
                    logger.warning("RTSP stream terputus! Switching ke fallback...")
                    self._last_reconnect_attempt = time.time()
                    if self._try_open_fallback():
                        return self.read()  # Baca frame pertama dari fallback
                    return False, None
                else:
                    # Video lokal habis
                    logger.info("Video lokal selesai (end of file).")
                    return False, None

        except Exception as e:
            logger.error(f"Exception saat membaca frame: {e}")
            if self._using_rtsp:
                logger.warning("Switching ke fallback...")
                if self._try_open_fallback():
                    return self.read()
            return False, None

    def _try_rtsp_reconnect(self):
        """Coba reconnect ke RTSP stream di background."""
        self._last_reconnect_attempt = time.time()
        logger.info("Mencoba reconnect ke RTSP...")
        if self._try_open_rtsp():
            logger.info("Reconnect RTSP berhasil! Beralih ke live stream.")

    def get_fps(self) -> float:
        """Ambil FPS dari sumber video."""
        if self._cap:
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            return fps if fps > 0 else 30.0
        return 30.0

    def get_frame_size(self) -> tuple:
        """Ambil ukuran frame (width, height)."""
        if self._cap:
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        return (640, 480)

    def _release_current(self):
        """Release VideoCapture yang sedang aktif."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def release(self):
        """Release semua resource."""
        self._release_current()
        logger.info("Video source ditutup.")
