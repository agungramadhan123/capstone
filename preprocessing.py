"""
Tahapan preprocessing:
    1. Masking teks overlay (timestamp, Fase, Skor, DISHUB, nama lokasi)
    2. Penyesuaian kecerahan (pagi: turunkan, malam: naikkan)
    3. Penyesuaian kontras CLAHE (pagi: tingkatkan, malam: turunkan)
    4. Penyesuaian resolusi / denoising (malam: kurangi blur dari lampu)
    5. Resize ke ukuran seragam (640x640 letterbox)

Cara menjalankan:
    python preprocessing.py                              # semua lokasi
    python preprocessing.py --lokasi bubat_barat         # lokasi spesifik
    python preprocessing.py --lokasi bubat_barat --waktu malam  # waktu spesifik
    python preprocessing.py --lokasi .....               # Sesuaikan dengan titik masing2
"""

import cv2
import numpy as np
import argparse
import time
from pathlib import Path
from abc import ABC, abstractmethod

#  BASE CLASS — Template preprocessing untuk semua titik CCTV
class BaseCCTVPreprocessor(ABC):
    # ── Path dasar (bisa di-override) ────────────────────────────
    PROJECT_ROOT = Path(__file__).resolve().parent
    DATA_ROOT = PROJECT_ROOT / "Data"
    OUTPUT_ROOT = PROJECT_ROOT / "dataset_preprocessing"

    def __init__(self):
        """Inisialisasi preprocessor."""
        self.stats = {"processed": 0, "errors": 0, "skipped": 0}

    # ── Property yang wajib di-override ──────────────────────────
    @property
    @abstractmethod
    def nama(self) -> str:
        """Nama tampilan lokasi CCTV."""
        pass

    @property
    @abstractmethod
    def folder_data(self) -> str:
        """Nama folder di Data/ (bisa pakai spasi)."""
        pass

    @property
    @abstractmethod
    def waktu_tersedia(self) -> list:
        """Daftar waktu yang tersedia: ['pagi', 'malam'] atau ['pagi', 'siang', 'malam']."""
        pass

    @property
    @abstractmethod
    def folder_output(self) -> str:
        """Nama folder output di dataset_preprocessing/."""
        pass

    # ── Parameter preprocessing (bisa di-override per lokasi) ────

    # Region masking teks overlay: [(x1, y1, x2, y2), ...]
    # Default untuk kamera Dishub Bandung 640x480
    @property
    def mask_regions_atas(self):
        """Area teks overlay atas (Fase, Skor, timestamp, DISHUB KOTA BANDUNG)."""
        return [(0, 0, 640, 75)]

    @property
    def mask_regions_bawah(self):
        """Area teks overlay bawah (VID BUAH BATU xxx, DARI ARAH xxx)."""
        return [(230, 330, 640, 480)]

    @property
    def inpaint_radius(self):
        return 3

    # Brightness: (alpha, beta) per waktu
    @property
    def gamma_malam(self):
        """Nilai gamma mencerahkan gambar malam. 1.0 = tidak ada perubahan."""
        return 1.0

    @property
    def brightness_pagi(self):
        """(alpha, beta) — pagi: turunkan kecerahan."""
        return (1.0, 0)

    @property
    def brightness_malam(self):
        """(alpha, beta) — malam: naikkan kecerahan untuk visibilitas."""
        return (1.0, 0)

    @property
    def brightness_siang(self):
        """(alpha, beta) — siang: netral."""
        return (1.0, 0)

    # CLAHE kontras per waktu
    @property
    def kontras_pagi(self):
        """(clip_limit, grid_size) — pagi: tingkatkan kontras."""
        return (1.3, (8, 8))

    @property
    def kontras_malam(self):
        """(clip_limit, grid_size) — malam: turunkan kontras."""
        return (1.1, (8, 8))

    @property
    def kontras_siang(self):
        """(clip_limit, grid_size) — siang: moderat."""
        return (1.5, (8, 8))

    # Resolusi & Denoising
    @property
    def unsharp_mask_pagi(self):
        """(weight_original, weight_blur) untuk penajaman gambar pagi. (1.0, 0.0) = tidak ada efek."""
        return (1.0, 0.0)

    @property
    def denoise_strength(self):
        """Strength denoising untuk malam hari (cv2.fastNlMeansDenoisingColored). 0 = off."""
        return 0

    @property
    def bilateral_filter_params(self):
        """Parameter bilateral filter malam hari (d, sigmaColor, sigmaSpace). 0 = off."""
        return (0, 0, 0)

    @property
    def highlight_params(self):
        """(threshold, strength) untuk meredam silau lampu malam. strength 0 = off."""
        return (220, 0.0)

    # Resize
    @property
    def target_size(self):
        """Target ukuran gambar (width, height)."""
        return (640, 480)
        
    @property
    def letterbox_color(self):
        """Warna latar belakang letterbox (B, G, R)."""
        return (114, 114, 114)

    #  TAHAP 1: Masking Teks Overlay (Timestamp, Fase, dsb
    def masking_teks(self, image: np.ndarray) -> np.ndarray:
        """
        Menghilangkan teks overlay CCTV (Putih, Cyan, dan Hitam) 
        menggunakan teknik Multi-Color Thresholding & Inpainting.
        """
        result = image.copy()
        all_regions = self.mask_regions_atas + self.mask_regions_bawah
        
        # Buat kanvas masker kosong
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Konversi ke HSV untuk deteksi warna Cyan yang akurat
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        for (x1, y1, x2, y2) in all_regions:
            # Isolasi area teks agar tidak mengganggu piksel jalan raya di luar kotak
            reg_gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            reg_hsv = hsv[y1:y2, x1:x2]
            
            # 1. Masker untuk Teks Putih/Terang (Timestamp & DISHUB)
            _, mask_putih = cv2.threshold(reg_gray, 200, 255, cv2.THRESH_BINARY)
            
            # 2. Masker untuk Teks Hitam (Tulisan "VID BUAH BATU", dll)
            _, mask_hitam = cv2.threshold(reg_gray, 50, 255, cv2.THRESH_BINARY_INV)
            
            # 3. Masker untuk Teks Cyan (Fase: GERAK, Skor)
            lower_cyan = np.array([80, 100, 100])
            upper_cyan = np.array([100, 255, 255])
            mask_cyan = cv2.inRange(reg_hsv, lower_cyan, upper_cyan)
            
            # Gabungkan ketiga masker warna tersebut (Logika OR)
            mask_gabungan = cv2.bitwise_or(mask_putih, mask_hitam)
            mask_gabungan = cv2.bitwise_or(mask_gabungan, mask_cyan)
            
            # Lakukan dilasi (penebalan) tipis agar pinggiran huruf ikut terhapus bersih
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask_gabungan = cv2.dilate(mask_gabungan, kernel, iterations=1)
            
            # Tempelkan hasil masker region ke masker utama gambar
            mask[y1:y2, x1:x2] = mask_gabungan

        # Inpaint: Mengisi bekas teks dengan piksel tiruan dari aspal/latar sekitarnya
        # Menggunakan radius kecil (3) agar hasil tambalan tidak terlihat blur/meleleh
        result = cv2.inpaint(result, mask, self.inpaint_radius, cv2.INPAINT_TELEA)
        return result

    #  TAHAP 2: Penyesuaian Kecerahan
    def sesuaikan_kecerahan(self, image: np.ndarray, waktu: str) -> np.ndarray:
        if waktu == "malam":
            gamma = self.gamma_malam
            if gamma != 1.0:
                invGamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                image = cv2.LUT(image, table)
        
        params = {
            "pagi": self.brightness_pagi,
            "malam": self.brightness_malam,
            "siang": self.brightness_siang,
        }
        alpha, beta = params.get(waktu, (1.0, 0))
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    #  TAHAP 3: Penyesuaian Kontras (CLAHE)
    def sesuaikan_kontras(self, image: np.ndarray, waktu: str) -> np.ndarray:
        """
        Menyesuaikan kontras menggunakan CLAHE pada channel L (LAB).
        - Pagi: tingkatkan kontras (clip_limit tinggi)
        - Malam: turunkan kontras (mengurangi glare lampu kendaraan)
        """
        params = {
            "pagi": self.kontras_pagi,
            "malam": self.kontras_malam,
            "siang": self.kontras_siang,
        }
        clip_limit, grid_size = params.get(waktu, (2.0, (8, 8)))

        # Konversi ke LAB → CLAHE pada channel L → konversi balik
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    #  TAHAP 4: Penyesuaian Resolusi / Denoising (Malam)
    def sesuaikan_resolusi(self, image: np.ndarray, waktu: str) -> np.ndarray:
        if waktu == "pagi":
            w_orig, w_blur = self.unsharp_mask_pagi
            if w_orig != 1.0 or w_blur != 0.0:
                blur = cv2.GaussianBlur(image, (0, 0), 2.0)
                return cv2.addWeighted(image, w_orig, blur, w_blur, 0)
            return image

        if waktu == "malam":
            h = self.denoise_strength
            denoised = image
            if h > 0:
                denoised = cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
            
            d, sc, ss = self.bilateral_filter_params
            if d > 0:
                return cv2.bilateralFilter(denoised, d=d, sigmaColor=sc, sigmaSpace=ss)
            return denoised

        return image

    #  TAHAP 5: Resize ke Ukuran Seragam (Letterbox)
    def resize_gambar(self, image: np.ndarray) -> np.ndarray:
        """
        Resize gambar ke ukuran target dengan letterboxing.
        Letterbox = pertahankan aspek rasio + padding abu-abu.
        Ini standar YOLO agar objek tidak terdistorsi.
        """
        target_w, target_h = self.target_size
        h, w = image.shape[:2]

        # Hitung skala yang mempertahankan aspek rasio
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Buat canvas sesuai warna letterbox
        color = self.letterbox_color
        canvas = np.full((target_h, target_w, 3), color, dtype=np.uint8)
        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

        return canvas

    def suppress_highlights(self, image: np.ndarray) -> np.ndarray:
        threshold, strength = self.highlight_params
        if strength <= 0:
            return image
            
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Ambil area lampu
        _, mask = cv2.threshold(v, threshold, 255, cv2.THRESH_BINARY)
        
        # Buat transisi pendaran lampu menjadi halus dengan Gaussian Blur
        mask_blurred = cv2.GaussianBlur(mask, (21, 21), 0) / 255.0

        # Tekan intensitas secara bertahap
        v_new = v.astype(np.float32)
        v_new = v_new * (1.0 - (strength * mask_blurred))
        v_new = np.clip(v_new, 0, 255).astype(np.uint8)

        hsv_new = cv2.merge([h, s, v_new])
        return cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)
    
    #  PIPELINE LENGKAP
    def preprocess(self, image: np.ndarray, waktu: str) -> np.ndarray:
        """Jalankan seluruh tahapan preprocessing pada satu gambar."""
        result = self.masking_teks(image)
        if waktu == "malam":
            result = self.suppress_highlights(result)
        result = self.sesuaikan_kecerahan(result, waktu)
        result = self.sesuaikan_kontras(result, waktu)
        result = self.sesuaikan_resolusi(result, waktu)
        result = self.resize_gambar(result)
        return result

    #  PROSES BATCH — Proses semua gambar di folder
    def proses_semua(self, waktu_filter: str = None):
        """
        Proses semua gambar untuk lokasi ini.
        Hasil disimpan di: dataset_preprocessing/<folder_output>/<waktu>/

        Args:
            waktu_filter: Jika diisi, hanya proses waktu tersebut
        """
        data_path = self.DATA_ROOT / self.folder_data
        if not data_path.exists():
            print(f"  ⚠  Folder data tidak ditemukan: {data_path}")
            print(f"      Data untuk {self.nama} belum tersedia. Dilewati.")
            self.stats["skipped"] += 1
            return

        print(f"\n{'='*55}")
        print(f"  📍 {self.nama}")
        print(f"  📂 Input : {data_path}")
        print(f"{'='*55}")

        waktu_list = [waktu_filter] if waktu_filter else self.waktu_tersedia

        for waktu in waktu_list:
            input_dir = data_path / waktu
            if not input_dir.exists():
                print(f"  ⚠  Subfolder '{waktu}' tidak ada. Dilewati.")
                continue

            # Kumpulkan file gambar
            ekstensi = {".jpg", ".jpeg", ".png", ".bmp"}
            files = sorted([
                f for f in input_dir.iterdir()
                if f.suffix.lower() in ekstensi
            ])[:4]

            if not files:
                print(f"  ⚠  Tidak ada gambar di {input_dir}.")
                continue

            # Buat folder output
            output_dir = self.OUTPUT_ROOT / self.folder_output / waktu
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n  🕐 {waktu.upper()} — {len(files)} gambar")
            print(f"     Output: {output_dir}")

            start = time.time()
            for idx, filepath in enumerate(files):
                try:
                    img = cv2.imread(str(filepath))
                    if img is None:
                        print(f"     ❌ Gagal baca: {filepath.name}")
                        self.stats["errors"] += 1
                        continue

                    # Jalankan pipeline preprocessing
                    hasil = self.preprocess(img, waktu)

                    # Simpan hasil
                    cv2.imwrite(str(output_dir / filepath.name), hasil)
                    self.stats["processed"] += 1

                    # Log progress setiap 100 gambar
                    if (idx + 1) % 100 == 0 or (idx + 1) == len(files):
                        pct = ((idx + 1) / len(files)) * 100
                        print(f"     ✅ {idx+1}/{len(files)} ({pct:.0f}%)")

                except Exception as e:
                    print(f"     ❌ Error {filepath.name}: {e}")
                    self.stats["errors"] += 1

            elapsed = time.time() - start
            print(f"     ⏱  Selesai dalam {elapsed:.1f} detik")


#  CLASS PER-LOKASI CCTV
class BubatBarat(BaseCCTVPreprocessor):
    """
    Preprocessing untuk titik CCTV Buah Batu Barat.
    Kamera: 640x480, kualitas menengah.
    Data: pagi + malam.
    """
    @property
    def nama(self):
        return "Bubat Barat"

    @property
    def folder_data(self):
        return "bubat barat"

    @property
    def waktu_tersedia(self):
        return ["pagi", "malam"]

    @property
    def folder_output(self):
        return "bubat_barat"

    # Parameter spesifik Bubat Barat
    @property
    def mask_regions_atas(self):
        return [(0, 0, 310, 65),
                (335,35,520,60)]

    @property
    def mask_regions_bawah(self):
        return [(360, 355, 585, 390),
                (360, 395, 610, 430)]

    @property
    def inpaint_radius(self):
        return 4

    @property
    def gamma_malam(self):
        return 0.6

    @property
    def brightness_pagi(self):
        return (0.95, 0)

    @property
    def brightness_malam(self):
        return (1.0, 0)

    @property
    def kontras_pagi(self):
        return (1.8, (8, 8))

    @property
    def kontras_malam(self):
        return (1.2, (8, 8))

    @property
    def unsharp_mask_pagi(self):
        return (1.5, -0.5)

    @property
    def denoise_strength(self):
        """Strength denoising untuk malam hari."""
        return 5  # Dikembalikan ke 5 agar tidak terlalu blur

    @property
    def bilateral_filter_params(self):
        return (7, 50, 50)

    @property
    def highlight_params(self):
        return (220, 0.3)

    @property
    def target_size(self):
        return (640, 480)



class BubatTimur(BaseCCTVPreprocessor):
    """
    Preprocessing untuk titik CCTV Buah Batu Timur.
    Data: pagi + siang + malam.
    (Parameter akan di-tune saat data terkumpul)
    """
    @property
    def nama(self):
        return "Bubat Timur"

    @property
    def folder_data(self):
        return "bubat timur"

    @property
    def waktu_tersedia(self):
        return ["pagi", "siang", "malam"]

    @property
    def folder_output(self):
        return "bubat_timur"


class BubatLingkar(BaseCCTVPreprocessor):
    """
    Preprocessing untuk titik CCTV Buah Batu Lingkar.
    Data: pagi + siang + malam.
    (Parameter akan di-tune saat data terkumpul)
    """
    @property
    def nama(self):
        return "Bubat Lingkar"

    @property
    def folder_data(self):
        return "bubat lingkar"

    @property
    def waktu_tersedia(self):
        return ["pagi", "siang", "malam"]

    @property
    def folder_output(self):
        return "bubat_lingkar"


class BubatSelatan(BaseCCTVPreprocessor):
    """
    Preprocessing untuk titik CCTV Buah Batu Selatan.
    Data: pagi + malam.
    (Parameter akan di-tune saat data terkumpul)
    """
    @property
    def nama(self):
        return "Bubat Selatan"

    @property
    def folder_data(self):
        return "bubat selatan"

    @property
    def waktu_tersedia(self):
        return ["pagi", "malam"]

    @property
    def folder_output(self):
        return "bubat_selatan"


class SpBuahBatu(BaseCCTVPreprocessor):
    """
    Preprocessing untuk titik CCTV Simpang Buah Batu.
    Data: pagi + malam.
    (Parameter akan di-tune saat data terkumpul)
    """
    @property
    def nama(self):
        return "Simpang Buah Batu"

    @property
    def folder_data(self):
        return "sp buah batu"

    @property
    def waktu_tersedia(self):
        return ["pagi", "malam"]

    @property
    def folder_output(self):
        return "sp_buah_batu"

#  DAFTAR SEMUA LOKASI
LOKASI_CCTV = {
    "bubat_barat": BubatBarat,
    "bubat_timur": BubatTimur,
    "bubat_lingkar": BubatLingkar,
    "bubat_selatan": BubatSelatan,
    "sp_buah_batu": SpBuahBatu,
}

#  MAIN — Entry Point
def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing Dataset CCTV Buah Batu Per-Lokasi",
    )
    parser.add_argument(
        "--lokasi", type=str, default=None,
        choices=list(LOKASI_CCTV.keys()),
        help="Lokasi CCTV yang ingin diproses (default: semua)",
    )
    parser.add_argument(
        "--waktu", type=str, default=None,
        choices=["pagi", "siang", "malam"],
        help="Waktu spesifik (default: semua waktu yang tersedia)",
    )
    args = parser.parse_args()

    print("╔═══════════════════════════════════════════════════════╗")
    print("║   PREPROCESSING DATASET CCTV BUAH BATU BANDUNG      ║")
    print("╚═══════════════════════════════════════════════════════╝")

    # Tentukan lokasi yang akan diproses
    if args.lokasi:
        lokasi_proses = {args.lokasi: LOKASI_CCTV[args.lokasi]}
    else:
        lokasi_proses = LOKASI_CCTV

    total_processed = 0
    total_errors = 0
    start_all = time.time()

    for key, kelas in lokasi_proses.items():
        preprocessor = kelas()
        preprocessor.proses_semua(waktu_filter=args.waktu)
        total_processed += preprocessor.stats["processed"]
        total_errors += preprocessor.stats["errors"]

    elapsed = time.time() - start_all

    # Ringkasan
    print(f"\n{'='*55}")
    print(f" RINGKASAN")
    print(f"{'='*55}")
    print(f"  Total gambar diproses : {total_processed}")
    print(f"  Total error           : {total_errors}")
    print(f"  Waktu total           : {elapsed:.1f} detik")
    print(f"  Output                : dataset_preprocessing/")
    print(f"\n  ✨ Selesai! Hasil siap untuk tahapan pemodelan.")


if __name__ == "__main__":
    main()
