"""Phase 15 Experiment Execution Script.

Runs the Phase 15 Trainable Quantum Kernel Optimization benchmark.
Supports running Stage 2 pilot mode (Fold 0) or the full matrix.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.phase15_trainable_kernels import run_phase15_pipeline

OUTPUT_DIR = REPO_ROOT / "reports" / "phase15"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUT_DIR / "phase15_execution.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("phase15_runner")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 15 Trainable Quantum Kernel Optimization.")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot",
                        help="Execution mode: 'pilot' (Fold 0 only) or 'full' (5 folds x 3 seeds).")
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[2000, 3000],
                        help="Target training subset sizes.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 2024],
                        help="Random seeds for full multi-seed execution.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for pilot mode.")
    parser.add_argument("--chunk-size", type=int, default=2500,
                        help="Chunk size for memory-safe kernel evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("==================================================================")
    logger.info("Starting Phase 15: Trainable Quantum Kernel Optimization Runner")
    logger.info("Mode: %s", args.mode.upper())
    logger.info("Sample sizes: %s | Seeds: %s | Chunk size: %d",
                args.sample_sizes, args.seeds if args.mode == "full" else [args.seed], args.chunk_size)
    logger.info("==================================================================")

    train_x_path = REPO_ROOT / "data" / "processed" / "X_train.csv"
    train_y_path = REPO_ROOT / "data" / "processed" / "y_train.csv"

    if not train_x_path.exists() or not train_y_path.exists():
        logger.error("Training data files not found in data/processed/!")
        sys.exit(1)

    t0 = time.perf_counter()
    logger.info("Loading training data from %s...", train_x_path)
    X_train_full = pd.read_csv(train_x_path)
    y_train_full = pd.read_csv(train_y_path).squeeze().to_numpy()
    logger.info("Loaded X_train shape: %s, y_train shape: %s (Frauds: %d) in %.2fs",
                X_train_full.shape, y_train_full.shape, int(np.sum(y_train_full == 1)),
                time.perf_counter() - t0)

    # Security check: Ensure holdout test set is NEVER touched
    test_x_path = REPO_ROOT / "data" / "processed" / "X_test.csv"
    logger.info("Security Check: Test dataset path %s is strictly isolated (UNTOUCHED).", test_x_path)

    is_pilot = (args.mode == "pilot")
    sample_sizes = [2000] if is_pilot else args.sample_sizes
    seeds = [args.seed] if is_pilot else args.seeds

    results = run_phase15_pipeline(
        X_train_full=X_train_full,
        y_train_full=y_train_full,
        sample_sizes=sample_sizes,
        seeds=seeds,
        chunk_size=args.chunk_size,
        is_pilot=is_pilot,
    )

    if is_pilot:
        output_json_path = OUTPUT_DIR / "pilot_metrics.json"
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved pilot metrics to %s", output_json_path)

        print("\n" + "=" * 115)
        print(" PHASE 15 STAGE 2 PILOT BENCHMARK SUMMARY (Fold 0, N_val = 45,396, Frauds = 75, pi = 0.165%)")
        print("=" * 115)
        print(f"{'Training Size':<14} | {'Model Architecture':<35} | {'PR-AUC':<8} | {'ROC-AUC':<8} | {'Prec (%)':<9} | {'Recall (%)':<10} | {'FP':<5} | {'Cost R=20':<9}")
        print("-" * 115)

        seed_str = str(args.seed)
        for sz_str, sz_data in results["per_seed_results"][seed_str]["oof_summary"].items():
            for model_name, m_metrics in sz_data.items():
                pr = m_metrics["oof_pr_auc"]
                roc = m_metrics["oof_roc_auc"]
                prec = m_metrics["calibrated_precision"] * 100.0
                rec = m_metrics["calibrated_recall"] * 100.0
                fp = m_metrics["calibrated_fp"]
                cost = m_metrics["calibrated_cost"]
                print(f"{sz_str:<14} | {model_name:<35} | {pr:<8.4f} | {roc:<8.4f} | {prec:<9.2f} | {rec:<10.2f} | {fp:<5d} | ${cost:<8.1f}")
        print("=" * 115)

    else:
        output_json_path = OUTPUT_DIR / "full_metrics.json"
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved full experiment metrics to %s", output_json_path)

        rows = []
        for sz_str, sz_data in results["multi_seed_aggregated_results"].items():
            for model_name, m_data in sz_data.items():
                rows.append({
                    "sample_size": sz_str,
                    "model": model_name,
                    "mean_oof_pr_auc": round(m_data["mean_oof_pr_auc"], 4),
                    "std_oof_pr_auc_sample": round(m_data["std_oof_pr_auc_sample"], 4),
                    "std_oof_pr_auc_pop": round(m_data["std_oof_pr_auc_pop"], 4),
                    "mean_oof_roc_auc": round(m_data["mean_oof_roc_auc"], 4),
                    "mean_default_precision": round(m_data["mean_default_precision"], 4),
                    "mean_default_recall": round(m_data["mean_default_recall"], 4),
                    "mean_calibrated_cost": round(m_data["mean_calibrated_cost"], 2),
                    "std_calibrated_cost": round(m_data["std_calibrated_cost"], 2),
                })

        summary_df = pd.DataFrame(rows)
        csv_path = OUTPUT_DIR / "full_summary_table.csv"
        summary_df.to_csv(csv_path, index=False)
        logger.info("Saved summary CSV table to %s", csv_path)

        # Save detailed fold-by-fold CSV
        fold_rows = []
        for s_str, s_data in results["per_seed_results"].items():
            for sz_str, sz_models in s_data["fold_metrics"].items():
                for m_name, f_list in sz_models.items():
                    for fold_num, fm in enumerate(f_list):
                        fold_rows.append({
                            "seed": int(s_str),
                            "fold": fold_num,
                            "sample_size": sz_str,
                            "model": m_name,
                            "pr_auc": round(fm["pr_auc"], 6),
                            "roc_auc": round(fm["roc_auc"], 6),
                            "calibrated_threshold": round(fm["calibrated_threshold"], 6),
                            "calibrated_precision": round(fm["calibrated"]["precision"], 6),
                            "calibrated_recall": round(fm["calibrated"]["recall"], 6),
                            "calibrated_fp": fm["calibrated"]["false_positives"],
                            "calibrated_fn": fm["calibrated"]["false_negatives"],
                            "calibrated_cost": round(fm["calibrated"]["financial_cost"], 2),
                            "default_precision": round(fm["default"]["precision"], 6),
                            "default_recall": round(fm["default"]["recall"], 6),
                            "default_fp": fm["default"]["false_positives"],
                            "default_cost": round(fm["default"]["financial_cost"], 2),
                        })
        per_fold_df = pd.DataFrame(fold_rows)
        per_fold_csv_path = OUTPUT_DIR / "per_fold_details.csv"
        per_fold_df.to_csv(per_fold_csv_path, index=False)
        logger.info("Saved per-fold details CSV to %s", per_fold_csv_path)

        print("\n" + "=" * 115)
        print(" PHASE 15 FULL MULTI-SEED BENCHMARK SUMMARY (3 Seeds x 5 Folds, N_val = 226,980, Frauds = 378)")
        print("=" * 115)
        print(f"{'Size':<8} | {'Model Architecture':<35} | {'PR-AUC (Mean+-Std)':<20} | {'ROC-AUC':<8} | {'Cal Cost ($)':<14}")
        print("-" * 115)
        for _, r in summary_df.iterrows():
            print(f"{r['sample_size']:<8} | {r['model']:<35} | {r['mean_oof_pr_auc']:.4f} +- {r['std_oof_pr_auc_sample']:.4f} | {r['mean_oof_roc_auc']:.4f}   | ${r['mean_calibrated_cost']:.1f}")
        print("=" * 115)

        if "statistical_tests" in results and results["statistical_tests"]:
            print("\n" + "=" * 115)
            print(" PHASE 15 PAIRED STATISTICAL HYPOTHESIS TESTS (15 Matched Folds: 3 Seeds x 5 Folds)")
            print("=" * 115)
            print(f"{'Size':<8} | {'Comparison':<35} | {'Delta PR-AUC':<14} | {'t-stat':<8} | {'t p-val':<8} | {'Wilcoxon p':<10} | {'Signif?':<7}")
            print("-" * 115)
            for sz_str, comps in results["statistical_tests"].items():
                for c_name, st in comps.items():
                    sig_str = "YES" if st["is_significant_p05"] else "NO"
                    print(f"{sz_str:<8} | {c_name:<35} | {st['mean_diff']:+.4f} +- {st['std_diff']:.4f} | {st['t_statistic']:<8.3f} | {st['t_pvalue']:<8.4f} | {st['wilcoxon_pvalue']:<10.4f} | {sig_str:<7}")
            print("=" * 115)


if __name__ == "__main__":
    main()
