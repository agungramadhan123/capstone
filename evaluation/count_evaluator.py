"""
TAHAP 5A — Evaluasi Counting: MAE (Mean Absolute Error)
=========================================================
Bandingkan hasil counting sistem vs ground truth manual.
Jalankan: python evaluation/count_evaluator.py --predicted results.csv --ground-truth gt.csv
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger

logger = setup_logger("eval_count", log_dir="outputs/logs")


def compute_mae(predicted: dict, ground_truth: dict) -> dict:
    """
    Hitung MAE per kelas dan total.

    Args:
        predicted: {"Mobil": 45, "Motor": 120, ...}
        ground_truth: {"Mobil": 50, "Motor": 115, ...}

    Returns:
        {"per_class": {...}, "total_mae": float, "total_mape": float}
    """
    all_classes = set(list(predicted.keys()) + list(ground_truth.keys()))
    all_classes.discard("TOTAL")

    per_class = {}
    errors = []

    for cls in sorted(all_classes):
        pred = predicted.get(cls, 0)
        gt = ground_truth.get(cls, 0)
        error = abs(pred - gt)
        pct_error = (error / gt * 100) if gt > 0 else 0

        per_class[cls] = {
            "predicted": pred,
            "ground_truth": gt,
            "abs_error": error,
            "pct_error": round(pct_error, 1),
        }
        errors.append(error)

    total_pred = sum(predicted.get(c, 0) for c in all_classes)
    total_gt = sum(ground_truth.get(c, 0) for c in all_classes)

    return {
        "per_class": per_class,
        "total_mae": round(np.mean(errors), 2) if errors else 0,
        "total_predicted": total_pred,
        "total_ground_truth": total_gt,
        "total_abs_error": abs(total_pred - total_gt),
        "total_pct_error": round(abs(total_pred - total_gt) / total_gt * 100, 1) if total_gt > 0 else 0,
    }


def evaluate_from_csvs(predicted_csv: str, ground_truth_csv: str):
    """
    Evaluasi dari file CSV.

    Format CSV (both):
        class,count
        Mobil,45
        Motor,120
        ...
    """
    pred_df = pd.read_csv(predicted_csv)
    gt_df = pd.read_csv(ground_truth_csv)

    predicted = dict(zip(pred_df["class"], pred_df["count"]))
    ground_truth = dict(zip(gt_df["class"], gt_df["count"]))

    results = compute_mae(predicted, ground_truth)

    # Display
    logger.info("=" * 60)
    logger.info("COUNTING EVALUATION — MAE Analysis")
    logger.info("=" * 60)

    logger.info(f"\n{'Class':<18} {'Predicted':>10} {'GT':>10} {'Error':>8} {'%':>7}")
    logger.info("-" * 55)

    for cls, data in results["per_class"].items():
        status = "✅" if data["pct_error"] < 10 else "⚠️" if data["pct_error"] < 20 else "❌"
        logger.info(
            f"  {status} {cls:<16} {data['predicted']:>8} {data['ground_truth']:>8} "
            f"{data['abs_error']:>8} {data['pct_error']:>6.1f}%"
        )

    logger.info("-" * 55)
    logger.info(
        f"  TOTAL{'':<12} {results['total_predicted']:>8} "
        f"{results['total_ground_truth']:>8} "
        f"{results['total_abs_error']:>8} "
        f"{results['total_pct_error']:>6.1f}%"
    )
    logger.info(f"\n  MAE (rata-rata per kelas): {results['total_mae']}")

    # Recommendations
    logger.info("\n💡 Analisis:")
    for cls, data in results["per_class"].items():
        if data["pct_error"] > 15:
            if data["predicted"] > data["ground_truth"]:
                logger.info(f"  {cls}: Over-counting → naikkan conf threshold")
            else:
                logger.info(f"  {cls}: Under-counting → turunkan conf atau cek occlusion")

    return results


def create_ground_truth_template(output_path: str):
    """Buat template CSV untuk ground truth manual."""
    template = pd.DataFrame({
        "class": ['Bis mini', 'Manusia', 'Mobil', 'Motor',
                  'Truk berat', 'Truk menengah', 'Truk ringan', 'bus'],
        "count": [0, 0, 0, 0, 0, 0, 0, 0],
    })
    template.to_csv(output_path, index=False)
    logger.info(f"Template created: {output_path}")
    logger.info("→ Isi kolom 'count' dengan hitungan manual dari video")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted", help="Predicted counts CSV")
    parser.add_argument("--ground-truth", help="Ground truth counts CSV")
    parser.add_argument("--create-template", help="Create GT template CSV")
    args = parser.parse_args()

    if args.create_template:
        create_ground_truth_template(args.create_template)
    elif args.predicted and args.ground_truth:
        evaluate_from_csvs(args.predicted, args.ground_truth)
    else:
        logger.info("Usage:")
        logger.info("  Create GT template:")
        logger.info("    python evaluation/count_evaluator.py --create-template gt.csv")
        logger.info("  Run evaluation:")
        logger.info("    python evaluation/count_evaluator.py --predicted pred.csv --ground-truth gt.csv")


if __name__ == "__main__":
    main()
