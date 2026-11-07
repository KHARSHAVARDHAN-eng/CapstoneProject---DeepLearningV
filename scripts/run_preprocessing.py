#!/usr/bin/env python3
"""
Phase 2 Preprocessing Pipeline Runner.

Executes data loading, duplicate removal, feature-target separation,
stratified train/test splitting, leak-free feature scaling, and artifact persistence.

Usage:
    python scripts/run_preprocessing.py [--data-path path/to/creditcard.csv] [--output-dir data/processed]
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import run_preprocessing_pipeline
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2: Data Preprocessing and Classical Baseline Preparation"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "creditcard.csv",
        help="Path to the raw creditcard.csv dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Directory to save processed splits and artifacts",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Test split fraction (default: 0.20 for 80/20 split)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible train/test split (default: 42)",
    )
    parser.add_argument(
        "--scaler-type",
        type=str,
        default="robust",
        choices=["robust", "standard", "minmax"],
        help="Feature scaling method (default: robust)",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="If specified, retain duplicate rows rather than dropping them",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger("quantum_fraud.run_preprocessing")
    logger.info("============================================================")
    logger.info("Starting Phase 2: Data Preprocessing Pipeline")
    logger.info("============================================================")
    logger.info("Input Dataset Path: %s", args.data_path)
    logger.info("Output Directory:   %s", args.output_dir)
    logger.info("Test Size:          %.2f (Stratified)", args.test_size)
    logger.info("Random Seed:        %d", args.random_state)
    logger.info("Feature Scaler:     %s", args.scaler_type)
    logger.info("Drop Duplicates:    %s", not args.keep_duplicates)

    try:
        results = run_preprocessing_pipeline(
            data_path=args.data_path,
            output_dir=args.output_dir,
            drop_duplicates=not args.keep_duplicates,
            test_size=args.test_size,
            random_state=args.random_state,
            scaler_type=args.scaler_type,
        )

        metadata = results["metadata"]
        splits = metadata["splits"]

        logger.info("------------------------------------------------------------")
        logger.info("Pipeline Execution Summary:")
        logger.info("  Initial Rows:    %d", metadata["initial_rows"])
        logger.info("  Cleaned Rows:    %d", metadata["final_rows"])
        logger.info(
            "  Duplicates:      %d dropped (%d legit, %d fraud)",
            metadata["duplicate_handling"]["total_duplicates"],
            metadata["duplicate_handling"]["duplicates_by_class"]["legitimate (0)"],
            metadata["duplicate_handling"]["duplicates_by_class"]["fraudulent (1)"],
        )
        logger.info("  Train Set:       %d samples (Fraud: %d, %.4f%%)",
                    splits["train"]["total_samples"],
                    splits["train"]["fraud_count"],
                    splits["train"]["fraud_percentage"])
        logger.info("  Test Set:        %d samples (Fraud: %d, %.4f%%)",
                    splits["test"]["total_samples"],
                    splits["test"]["fraud_count"],
                    splits["test"]["fraud_percentage"])
        logger.info("------------------------------------------------------------")
        logger.info("Saved Artifacts:")
        for name, path in results["saved_files"].items():
            logger.info("  %s: %s", name, path)
        logger.info("============================================================")
        logger.info("Phase 2 Preprocessing Pipeline completed successfully!")
        return 0

    except Exception as exc:
        logger.error("Pipeline failed with exception: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
