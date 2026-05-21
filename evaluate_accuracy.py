"""
=============================================================
 FASE 3 — VALIDASI LAPANGAN & EVALUASI JAM SIBUK (1 JAM)
 
 Skrip analisis untuk menguji performa sistem setelah 
 merekam video 1 jam di jam sibuk Buah Batu.
 
 Fitur:
   1. Agregasi interval time-series (5 menit)
   2. Kalkulasi MAE per kelas kendaraan
   3. Tabel komparasi Sistem vs Manual (Ground Truth)
   4. Persentase akurasi akhir per kelas
 
 Cara menjalankan:
   python evaluate_accuracy.py
   python evaluate_accuracy.py --csv traffic_logs_buahbatu.csv
   python evaluate_accuracy.py --interval 5
=============================================================
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


# ── KONFIGURASI ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = str(PROJECT_ROOT / "traffic_logs_buahbatu.csv")

# Kelas kendaraan yang dianalisis
VEHICLE_CLASSES = ["Bis", "Mobil", "Motor", "Truk"]

# ══════════════════════════════════════════════════════════════
#  GROUND TRUTH — Data Hitungan Manual Manusia
# ══════════════════════════════════════════════════════════════
# 
# INSTRUKSI PENGISIAN:
#   Isi tabel di bawah dengan hitungan manual per interval 5 menit.
#   Setiap list berisi 12 angka (12 × 5 menit = 60 menit = 1 jam).
#   
#   Interval:  [0-5, 5-10, 10-15, 15-20, 20-25, 25-30,
#               30-35, 35-40, 40-45, 45-50, 50-55, 55-60]
#
#   Jika belum ada data, biarkan dengan angka 0.
#   Ganti angka-angka di bawah dengan hasil hitungan manual Anda.
#
GROUND_TRUTH = {
    "Bis":   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Mobil": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Motor": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Truk":  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}


# ══════════════════════════════════════════════════════════════
#  1. MEMBACA & MEMPROSES LOG CSV
# ══════════════════════════════════════════════════════════════
def load_traffic_log(csv_path: str) -> pd.DataFrame:
    """
    Baca file traffic_logs_buahbatu.csv dan siapkan untuk analisis.
    """
    if not os.path.exists(csv_path):
        print(f"❌ File CSV tidak ditemukan: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Validasi kolom
    required_cols = ["timestamp", "frame_id", "vehicle_id", "class_name", "confidence", "direction"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"❌ Kolom tidak ditemukan dalam CSV: {missing}")
        sys.exit(1)

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"✅ CSV dimuat: {len(df)} event")
    print(f"   Rentang waktu: {df['timestamp'].min()} — {df['timestamp'].max()}")
    print(f"   Kelas terdeteksi: {df['class_name'].unique().tolist()}")

    return df


# ══════════════════════════════════════════════════════════════
#  2. AGREGASI INTERVAL TIME-SERIES (5 MENIT)
# ══════════════════════════════════════════════════════════════
def aggregate_intervals(df: pd.DataFrame, interval_minutes: int = 5) -> pd.DataFrame:
    """
    Resample data otomatis per interval N menit.
    Mengelompokkan total hitungan kendaraan per kelas per interval.
    """
    # Set timestamp sebagai index
    df_indexed = df.set_index("timestamp")

    # Resample per interval
    interval_str = f"{interval_minutes}min"

    # Pivot: hitung jumlah event per kelas per interval
    resampled = (
        df_indexed
        .groupby([pd.Grouper(freq=interval_str), "class_name"])
        .size()
        .unstack(fill_value=0)
    )

    # Pastikan semua kelas ada (bahkan jika 0)
    for cls in VEHICLE_CLASSES:
        if cls not in resampled.columns:
            resampled[cls] = 0

    resampled = resampled[VEHICLE_CLASSES]  # Urutkan sesuai VEHICLE_CLASSES

    print(f"\n📊 Agregasi per {interval_minutes} menit ({len(resampled)} interval):")
    print(resampled.to_string())

    return resampled


# ══════════════════════════════════════════════════════════════
#  3. KALKULASI MAE PER KELAS
# ══════════════════════════════════════════════════════════════
def calculate_mae(system_counts: list, manual_counts: list) -> float:
    """
    Hitung Mean Absolute Error (MAE).
    
    Rumus: MAE = (1/n) × Σ|yᵢ - ŷᵢ|
    
    Di mana:
      yᵢ  = hitungan manual (ground truth) pada interval ke-i
      ŷᵢ  = hitungan sistem pada interval ke-i
      n   = jumlah interval
    """
    n = min(len(system_counts), len(manual_counts))
    if n == 0:
        return 0.0

    system = np.array(system_counts[:n], dtype=float)
    manual = np.array(manual_counts[:n], dtype=float)

    mae = np.mean(np.abs(manual - system))
    return mae


def calculate_accuracy(system_total: int, manual_total: int) -> float:
    """
    Hitung persentase akurasi.
    
    Rumus: Accuracy = max(0, (1 - |manual - system| / manual)) × 100%
    
    Jika manual_total == 0, return 100% (tidak ada yang perlu dideteksi).
    """
    if manual_total == 0:
        return 100.0 if system_total == 0 else 0.0

    error_ratio = abs(manual_total - system_total) / manual_total
    accuracy = max(0.0, (1.0 - error_ratio)) * 100.0
    return accuracy


# ══════════════════════════════════════════════════════════════
#  4. TABEL KOMPARASI & LAPORAN
# ══════════════════════════════════════════════════════════════
def generate_comparison_report(resampled: pd.DataFrame, ground_truth: dict,
                                interval_minutes: int):
    """
    Buat tabel komparasi lengkap antara hitungan sistem vs manual.
    """
    n_intervals = len(resampled)

    print(f"\n{'='*70}")
    print(f"  📊 LAPORAN EVALUASI AKURASI — JAM SIBUK BUAH BATU")
    print(f"  Tanggal   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Interval  : {interval_minutes} menit ({n_intervals} blok)")
    print(f"{'='*70}")

    # ── Tabel Detail per Interval ──
    print(f"\n{'─'*70}")
    print(f"  DETAIL PER INTERVAL ({interval_minutes} MENIT)")
    print(f"{'─'*70}")

    for cls in VEHICLE_CLASSES:
        system_vals = resampled[cls].values.tolist()
        manual_vals = ground_truth.get(cls, [0] * n_intervals)

        # Pastikan panjang sama
        n = min(len(system_vals), len(manual_vals))

        print(f"\n  📌 {cls}:")
        print(f"  {'Interval':<12} {'Sistem':>8} {'Manual':>8} {'Selisih':>8}")
        print(f"  {'─'*40}")

        for i in range(n):
            start_min = i * interval_minutes
            end_min = (i + 1) * interval_minutes
            interval_label = f"{start_min:02d}-{end_min:02d} min"
            diff = system_vals[i] - manual_vals[i]
            diff_str = f"+{diff}" if diff > 0 else str(diff)
            print(f"  {interval_label:<12} {system_vals[i]:>8} {manual_vals[i]:>8} {diff_str:>8}")

    # ── Tabel Ringkasan Akhir ──
    print(f"\n{'='*70}")
    print(f"  📊 RINGKASAN AKHIR")
    print(f"{'='*70}")

    header = f"  {'Kelas':<10} {'Sistem':>10} {'Manual':>10} {'MAE':>10} {'Akurasi':>10}"
    print(header)
    print(f"  {'─'*52}")

    total_system = 0
    total_manual = 0
    mae_values = []

    for cls in VEHICLE_CLASSES:
        system_vals = resampled[cls].values.tolist()
        manual_vals = ground_truth.get(cls, [0] * n_intervals)

        n = min(len(system_vals), len(manual_vals))

        system_total = sum(system_vals[:n])
        manual_total = sum(manual_vals[:n])

        mae = calculate_mae(system_vals, manual_vals)
        accuracy = calculate_accuracy(system_total, manual_total)

        total_system += system_total
        total_manual += manual_total
        mae_values.append(mae)

        print(f"  {cls:<10} {system_total:>10} {manual_total:>10} {mae:>10.2f} {accuracy:>9.1f}%")

    # Total
    print(f"  {'─'*52}")
    overall_mae = np.mean(mae_values) if mae_values else 0.0
    overall_accuracy = calculate_accuracy(total_system, total_manual)
    print(f"  {'TOTAL':<10} {total_system:>10} {total_manual:>10} {overall_mae:>10.2f} {overall_accuracy:>9.1f}%")

    print(f"\n{'='*70}")

    # ── Interpretasi ──
    print(f"\n  💡 INTERPRETASI:")
    if overall_accuracy >= 90:
        print(f"     ✅ Akurasi keseluruhan {overall_accuracy:.1f}% — SANGAT BAIK")
    elif overall_accuracy >= 80:
        print(f"     ⚠️ Akurasi keseluruhan {overall_accuracy:.1f}% — CUKUP BAIK")
    elif overall_accuracy >= 70:
        print(f"     ⚠️ Akurasi keseluruhan {overall_accuracy:.1f}% — PERLU PENINGKATAN")
    else:
        print(f"     ❌ Akurasi keseluruhan {overall_accuracy:.1f}% — PERLU EVALUASI MENDALAM")

    if overall_mae <= 2.0:
        print(f"     ✅ MAE rata-rata {overall_mae:.2f} — Error sangat rendah")
    elif overall_mae <= 5.0:
        print(f"     ⚠️ MAE rata-rata {overall_mae:.2f} — Error moderat")
    else:
        print(f"     ❌ MAE rata-rata {overall_mae:.2f} — Error tinggi, periksa konfigurasi tracker")

    # Identifikasi kelas terburuk
    if mae_values:
        worst_idx = np.argmax(mae_values)
        worst_cls = VEHICLE_CLASSES[worst_idx]
        worst_mae = mae_values[worst_idx]
        print(f"     📌 Kelas dengan MAE tertinggi: {worst_cls} (MAE={worst_mae:.2f})")

    print()

    return {
        "total_system": total_system,
        "total_manual": total_manual,
        "overall_mae": overall_mae,
        "overall_accuracy": overall_accuracy,
    }


# ══════════════════════════════════════════════════════════════
#  5. EXPORT KE CSV (OPSIONAL)
# ══════════════════════════════════════════════════════════════
def export_report_csv(resampled: pd.DataFrame, ground_truth: dict,
                      output_path: str, interval_minutes: int):
    """
    Export tabel komparasi ke file CSV untuk dokumentasi laporan.
    """
    n_intervals = len(resampled)
    rows = []

    for cls in VEHICLE_CLASSES:
        system_vals = resampled[cls].values.tolist()
        manual_vals = ground_truth.get(cls, [0] * n_intervals)
        n = min(len(system_vals), len(manual_vals))

        for i in range(n):
            start_min = i * interval_minutes
            end_min = (i + 1) * interval_minutes
            rows.append({
                "kelas": cls,
                "interval": f"{start_min:02d}-{end_min:02d}",
                "hitungan_sistem": system_vals[i],
                "hitungan_manual": manual_vals[i],
                "selisih": system_vals[i] - manual_vals[i],
            })

    report_df = pd.DataFrame(rows)
    report_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"📄 Laporan CSV di-export ke: {output_path}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Fase 3: Evaluasi Akurasi Monitoring Lalu Lintas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  python evaluate_accuracy.py
  python evaluate_accuracy.py --csv traffic_logs_buahbatu.csv
  python evaluate_accuracy.py --interval 5 --export
        """,
    )
    parser.add_argument(
        "--csv", type=str, default=DEFAULT_CSV,
        help=f"Path ke file CSV log (default: {DEFAULT_CSV})"
    )
    parser.add_argument(
        "--interval", type=int, default=5,
        help="Interval agregasi dalam menit (default: 5)"
    )
    parser.add_argument(
        "--export", action="store_true",
        help="Export laporan ke CSV"
    )
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  FASE 3 — EVALUASI AKURASI JAM SIBUK BUAH BATU")
    print(f"  Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    # ── 1. Baca CSV ──
    print(f"\n📥 Membaca log CSV: {args.csv}")
    df = load_traffic_log(args.csv)

    # ── 2. Agregasi per interval ──
    resampled = aggregate_intervals(df, interval_minutes=args.interval)

    # ── 3. Validasi ground truth ──
    n_intervals = len(resampled)
    gt_valid = True
    for cls in VEHICLE_CLASSES:
        gt = GROUND_TRUTH.get(cls, [])
        if len(gt) < n_intervals:
            print(f"  ⚠️ Ground truth '{cls}' hanya {len(gt)} interval, "
                  f"diperlukan {n_intervals}. Padding dengan 0.")
            GROUND_TRUTH[cls] = gt + [0] * (n_intervals - len(gt))

        # Cek apakah semua 0 (belum diisi)
        if all(v == 0 for v in GROUND_TRUTH[cls]):
            gt_valid = False

    if not gt_valid:
        print(f"\n  ⚠️  PERINGATAN: Ground truth masih berisi angka 0 (belum diisi)!")
        print(f"     Buka file evaluate_accuracy.py dan isi dictionary GROUND_TRUTH")
        print(f"     dengan data hitungan manual Anda per {args.interval} menit.")
        print(f"     Lanjutkan dengan data kosong untuk preview format laporan.\n")

    # ── 4. Buat laporan ──
    results = generate_comparison_report(
        resampled=resampled,
        ground_truth=GROUND_TRUTH,
        interval_minutes=args.interval,
    )

    # ── 5. Export (opsional) ──
    if args.export:
        export_path = str(PROJECT_ROOT / "evaluation_report.csv")
        export_report_csv(resampled, GROUND_TRUTH, export_path, args.interval)

    print("✅ Evaluasi selesai!")


if __name__ == "__main__":
    main()
