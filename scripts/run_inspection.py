#!/usr/bin/env python3
"""
Reproducible execution script for Phase 1: Dataset Inspection.

Usage:
    python scripts/run_inspection.py [--data-path path/to/creditcard.csv] [--reports-dir path/to/reports]
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.inspect_data import run_full_inspection
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1: Dataset Inspection for Quantum Fraud Detection Capstone Project"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "creditcard.csv",
        help="Path to the raw creditcard.csv dataset",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=PROJECT_ROOT / "reports",
        help="Directory to store reports and generated figures",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger("quantum_fraud.run_inspection")
    logger.info("Starting Phase 1: Project Setup & Dataset Inspection...")
    logger.info("Target dataset path: %s", args.data_path)
    logger.info("Target reports directory: %s", args.reports_dir)

    result = run_full_inspection(args.data_path, args.reports_dir)

    if result["status"] == "DATASET_MISSING":
        logger.warning("============================================================")
        logger.warning(result["message"])
        logger.warning("============================================================")
        return 1

    logger.info("Dataset inspection completed successfully!")
    logger.info("Report generated at: %s", result["report_file"])
    for fig in result.get("figures", []):
        logger.info("Figure saved: %s", fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
