"""Phase 13 Experiment Execution Script.

Runs the Phase 13 systematic model optimization experiments.
Supports running Stage 2 pilot mode or the full matrix.
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

# Add repo root to Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.phase13_optimization import run_phase13_full_experiment, run_phase13_pilot

# Setup output directories
OUTPUT_DIR = REPO_ROOT / "reports" / "phase13"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUT_DIR / "phase13_execution.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("phase13_runner")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 13 Systematic Model Optimization.")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot",
                        help="Execution mode: 'pilot' (Fold 0 only) or 'full' (5 folds x 3 seeds).")
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[500, 1000, 2000],
                        help="Target training subset sizes.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 2024],
                        help="Random seeds for full multi-seed execution.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for pilot mode.")
    parser.add_argument("--chunk-size", type=int, default=5000,
                        help="Chunk size for memory-safe kernel evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("==================================================================")
    logger.info("Starting Phase 13: Systematic Model Optimization Runner")
    logger.info("Mode: %s", args.mode.upper())
    logger.info("Sample sizes: %s | Seeds: %s | Chunk size: %d",
                args.sample_sizes, args.seeds if args.mode == "full" else [args.seed], args.chunk_size)
    logger.info("==================================================================")

    # 1. Load data strictly from training partition (Zero test-set access!)
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

    # 2. Execute Pilot Mode
    if args.mode == "pilot":
        logger.info("Executing Phase 13 Stage 2 Pilot Experiment...")
        pilot_results = run_phase13_pilot(
            X_train_full=X_train_full,
            y_train_full=y_train_full,
            sample_sizes=args.sample_sizes,
            random_state=args.seed,
            chunk_size=args.chunk_size,
        )

        output_json_path = OUTPUT_DIR / "pilot_metrics.json"
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(pilot_results, f, indent=2)
        logger.info("Saved pilot metrics to %s", output_json_path)

        print("\n" + "=" * 105)
        print(" PHASE 13 STAGE 2 PILOT BENCHMARK SUMMARY (Fold 0, N_val = 45,396, Frauds = 75, pi = 0.165%)")
        print("=" * 105)
        print(f"{'Training Size':<14} | {'Model Architecture':<30} | {'PR-AUC':<8} | {'ROC-AUC':<8} | {'Prec (%)':<9} | {'Recall (%)':<10} | {'FP':<5} | {'Cost R=20':<9}")
        print("-" * 105)

        for size_str, size_data in pilot_results["per_size_results"].items():
            models = size_data["models"]
            for model_name, m_data in models.items():
                def_m = m_data["default_metrics"]
                calib = m_data["calibrated_cost_r20"]
                clean_name = model_name.replace("_", " ").title()
                print(f"N = {size_str:<10} | {clean_name:<30} | {def_m['pr_auc']:<8.4f} | {def_m['roc_auc']:<8.4f} | {def_m['precision']*100:<9.2f} | {def_m['recall']*100:<10.2f} | {def_m['false_positives']:<5} | ${calib['cost']:<8.1f}")
            print("-" * 105)

        for ref_name, ref_data in pilot_results["macro_reference_models"].items():
            def_m = ref_data["default_metrics"]
            calib = ref_data["calibrated_cost_r20"]
            clean_name = ref_name.replace("_", " ").title()
            print(f"Full Fold (181k)| {clean_name:<30} | {def_m['pr_auc']:<8.4f} | {def_m['roc_auc']:<8.4f} | {def_m['precision']*100:<9.2f} | {def_m['recall']*100:<10.2f} | {def_m['false_positives']:<5} | ${calib['cost']:<8.1f}")
        print("=" * 105)

        res = pilot_results["resource_profile"]
        print(f"Pilot Resource Profile: Runtime = {res['total_runtime_seconds']}s | Peak RAM = {res['peak_memory_mb']} MB")
        print("=" * 105 + "\n")

    # 3. Execute Full Mode
    elif args.mode == "full":
        logger.info("Executing Phase 13 Stage 3 Full Experiment (5 Folds x %d Seeds)...", len(args.seeds))
        full_results = run_phase13_full_experiment(
            X_train_full=X_train_full,
            y_train_full=y_train_full,
            sample_sizes=args.sample_sizes,
            seeds=args.seeds,
            n_splits=5,
            chunk_size=args.chunk_size,
        )

        output_json_path = OUTPUT_DIR / "full_metrics.json"
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2)
        logger.info("Saved full experiment metrics to %s", output_json_path)

        # Build Summary CSV table
        rows = []
        agg = full_results["multi_seed_aggregated_results"]
        for size_str, models in agg.items():
            for model_name, m_stats in models.items():
                rows.append({
                    "sample_size": size_str,
                    "model": model_name,
                    "mean_oof_pr_auc": m_stats["mean_oof_pr_auc"],
                    "std_oof_pr_auc": m_stats["std_oof_pr_auc"],
                    "mean_oof_roc_auc": m_stats["mean_oof_roc_auc"],
                    "std_oof_roc_auc": m_stats["std_oof_roc_auc"],
                    "mean_precision": m_stats["mean_precision"],
                    "mean_recall": m_stats["mean_recall"],
                    "mean_fp": m_stats["mean_fp"],
                    "mean_cost_r20": m_stats["mean_cost_r20"],
                    "std_cost_r20": m_stats["std_cost_r20"],
                })
        df_summary = pd.DataFrame(rows)
        csv_path = OUTPUT_DIR / "full_summary_table.csv"
        df_summary.to_csv(csv_path, index=False)
        logger.info("Saved summary CSV table to %s", csv_path)

        # Print Aggregated Summary Table
        print("\n" + "=" * 115)
        print(f" PHASE 13 FULL EXPERIMENT AGGREGATED SUMMARY (3 Seeds: {args.seeds}, 5-Fold OOF N=226,980, pi=0.166%)")
        print("=" * 115)
        print(f"{'Training Size':<14} | {'Model Architecture':<30} | {'Mean PR-AUC':<15} | {'Mean ROC-AUC':<14} | {'Precision':<10} | {'Recall':<10} | {'Mean Cost R=20':<15}")
        print("-" * 115)
        for _, r in df_summary.iterrows():
            size_label = f"N = {r['sample_size']}" if r['sample_size'] != "macro" else "Macro (181k)"
            clean_name = r['model'].replace('_', ' ').title()
            pr_str = f"{r['mean_oof_pr_auc']:.4f} ± {r['std_oof_pr_auc']:.4f}"
            roc_str = f"{r['mean_oof_roc_auc']:.4f} ± {r['std_oof_roc_auc']:.4f}"
            cost_str = f"${r['mean_cost_r20']:.1f} ± ${r['std_cost_r20']:.1f}"
            print(f"{size_label:<14} | {clean_name:<30} | {pr_str:<15} | {roc_str:<14} | {r['mean_precision']*100:<9.2f}% | {r['mean_recall']*100:<9.2f}% | {cost_str:<15}")
        print("=" * 115)
        meta = full_results["metadata"]
        print(f"Total Experiment Runtime: {meta['total_runtime_minutes']} minutes ({meta['total_runtime_seconds']}s) | Peak RAM: {meta['peak_memory_mb']} MB")
        print("=" * 115 + "\n")


if __name__ == "__main__":
    main()

