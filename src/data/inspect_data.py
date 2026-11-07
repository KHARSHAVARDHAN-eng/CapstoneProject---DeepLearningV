"""
Dataset inspection module for financial fraud detection.

Performs exploratory data inspection on the ULB credit card fraud dataset,
evaluating dimensions, schemas, missing values, duplicates, class distributions,
and feature statistics without altering the source data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger("quantum_fraud.inspect_data")


def check_dataset_exists(data_path: Path) -> bool:
    """Checks whether the raw dataset exists at the expected location.

    Args:
        data_path: Path to the expected CSV file.

    Returns:
        bool: True if the file exists and is a file, False otherwise.
    """
    return data_path.is_file()


def get_missing_dataset_instructions(data_path: Path) -> str:
    """Returns clear instructions on how to obtain and place the missing dataset.

    Args:
        data_path: Expected file destination path.

    Returns:
        str: User instructions.
    """
    return (
        f"DATASET MISSING: The required dataset was not found at '{data_path.resolve()}'.\n"
        "To resolve this:\n"
        "  1. Visit Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
        "  2. Download 'creditcard.csv.zip' and extract 'creditcard.csv'.\n"
        f"  3. Place the file at: '{data_path.resolve()}'.\n"
        "  4. Re-run this inspection script.\n"
        "Note: No synthetic or fabricated data has been generated."
    )


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Loads the ULB credit card fraud dataset in read-only fashion.

    Args:
        data_path: Path to the creditcard.csv file.

    Raises:
        FileNotFoundError: If the file is not found.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not check_dataset_exists(data_path):
        error_msg = get_missing_dataset_instructions(data_path)
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info("Loading dataset from: %s", data_path)
    df = pd.read_csv(data_path)
    logger.info("Dataset successfully loaded: %d rows, %d columns.", df.shape[0], df.shape[1])
    return df


def inspect_dimensions(df: pd.DataFrame) -> Dict[str, int]:
    """Inspects dataset dimensions.

    Args:
        df: Input DataFrame.

    Returns:
        Dict[str, int]: Number of rows, columns, and total memory usage in MB.
    """
    mem_mb = int(df.memory_usage(deep=True).sum() / (1024 * 1024))
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "memory_mb": mem_mb,
    }


def inspect_columns_and_types(df: pd.DataFrame) -> pd.DataFrame:
    """Inspects column names, non-null counts, and data types.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Summary table of columns, non-null counts, and dtypes.
    """
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Dtype": [str(t) for t in df.dtypes],
        "Non-Null Count": df.notnull().sum().values,
        "Null Count": df.isnull().sum().values,
        "Null %": (df.isnull().sum().values / len(df) * 100).round(4),
    })
    return info_df


def inspect_missing_and_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Inspects missing values and duplicate rows.

    Args:
        df: Input DataFrame.

    Returns:
        Dict[str, Any]: Summary of missing and duplicate counts.
    """
    missing_total = int(df.isnull().sum().sum())
    missing_by_col = df.isnull().sum()[df.isnull().sum() > 0].to_dict()
    duplicate_count = int(df.duplicated().sum())
    duplicate_percentage = (duplicate_count / len(df)) * 100

    return {
        "total_missing_values": missing_total,
        "columns_with_missing": missing_by_col,
        "duplicate_rows": duplicate_count,
        "duplicate_percentage": round(duplicate_percentage, 4),
    }


def inspect_class_distribution(df: pd.DataFrame, target_col: str = "Class") -> Dict[str, Any]:
    """Inspects target class distribution and imbalance metrics.

    Args:
        df: Input DataFrame.
        target_col: Target column name.

    Returns:
        Dict[str, Any]: Counts, proportions, and fraud percentage.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not present in dataset.")

    counts = df[target_col].value_counts().to_dict()
    total = len(df)
    legit_count = int(counts.get(0, 0))
    fraud_count = int(counts.get(1, 0))
    fraud_pct = (fraud_count / total) * 100 if total > 0 else 0.0
    imbalance_ratio = (legit_count / fraud_count) if fraud_count > 0 else float("inf")

    return {
        "total_records": total,
        "legitimate_count (0)": legit_count,
        "fraudulent_count (1)": fraud_count,
        "fraud_percentage": round(fraud_pct, 4),
        "imbalance_ratio": round(imbalance_ratio, 2),
    }


def inspect_feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates descriptive statistics for all numerical features.

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Transposed descriptive statistics (count, mean, std, min, 25%, 50%, 75%, max).
    """
    stats = df.describe().T
    stats["skewness"] = df.skew(numeric_only=True)
    return stats


def generate_visualizations(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Generates visualizations for class imbalance and transaction amounts.

    Args:
        df: Input DataFrame.
        output_dir: Directory to store figures.

    Returns:
        list[Path]: Paths to generated image files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    generated_figures: list[Path] = []

    # 1. Class Distribution Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    class_counts = df["Class"].value_counts().sort_index()
    legit_cnt, fraud_cnt = class_counts.get(0, 0), class_counts.get(1, 0)
    fraud_pct = (fraud_cnt / len(df)) * 100

    # Linear scale with counts and percentages
    bars = axes[0].bar(
        ["Legitimate (0)", "Fraud (1)"],
        [legit_cnt, fraud_cnt],
        color=["#2b5c8f", "#d9534f"],
    )
    axes[0].set_ylim(0, legit_cnt * 1.18)
    axes[0].set_title("Class Distribution (Linear Scale)", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_ylabel("Transaction Count")
    for bar, count in zip(bars, [legit_cnt, fraud_cnt]):
        height = bar.get_height()
        axes[0].annotate(
            f"{count:,}\n({count/len(df)*100:.2f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Logarithmic scale to visualize the minority class clearly
    axes[1].bar(
        ["Legitimate (0)", "Fraud (1)"],
        [legit_cnt, fraud_cnt],
        color=["#2b5c8f", "#d9534f"],
    )
    axes[1].set_yscale("log")
    axes[1].set_ylim(1, legit_cnt * 6)
    axes[1].set_title("Class Distribution (Log Scale)", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_ylabel("Transaction Count (Log Scale)")
    for i, count in enumerate([legit_cnt, fraud_cnt]):
        axes[1].annotate(
            f"{count:,}",
            xy=(i, count),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.suptitle(
        f"ULB Credit Card Fraud Imbalance (Fraud: {fraud_pct:.3f}%)",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    fig1_path = output_dir / "class_imbalance.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()
    generated_figures.append(fig1_path)

    # 2. Transaction Amount Distribution (Overall & Log-transformed)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Clipped / zoom on Amount <= 500 (filter before computing histogram bins)
    amount_sub_500 = df.loc[df["Amount"] <= 500, "Amount"]
    sns.histplot(
        amount_sub_500,
        bins=50,
        kde=True,
        ax=axes[0],
        color="#2b5c8f",
    )
    axes[0].set_xlim(0, 500)
    axes[0].set_title(
        f"Amount Distribution (Zoomed $0 - $500, {len(amount_sub_500)/len(df)*100:.1f}% of data)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    axes[0].set_xlabel("Transaction Amount ($)")
    axes[0].set_ylabel("Frequency")

    # Log10-transformed Amount
    log_amount = np.log10(df["Amount"] + 1)
    sns.histplot(
        log_amount,
        bins=50,
        kde=True,
        ax=axes[1],
        color="#2a9d8f",
    )
    axes[1].set_title("Log10(Amount + 1) Distribution (All Records)", fontsize=12, fontweight="bold", pad=10)
    axes[1].set_xlabel("log10(Amount + 1)")
    axes[1].set_ylabel("Frequency")

    plt.suptitle("Transaction Amount Distribution", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig2_path = output_dir / "amount_distribution.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()
    generated_figures.append(fig2_path)

    # 3. Transaction Amount Comparison by Class (Boxplot & Statistics)
    fig, ax = plt.subplots(figsize=(8, 6))
    df_plot = df[["Amount", "Class"]].copy()
    df_plot["Class_Label"] = df_plot["Class"].map({0: "Legitimate", 1: "Fraud"})
    df_plot["Log_Amount"] = np.log10(df_plot["Amount"] + 1)

    sns.boxplot(
        x="Class_Label",
        y="Log_Amount",
        hue="Class_Label",
        data=df_plot,
        palette=["#2b5c8f", "#d9534f"],
        ax=ax,
        width=0.4,
        legend=False,
    )
    ax.set_title("Log10(Amount + 1) Comparison by Class", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Transaction Class")
    ax.set_ylabel("log10(Amount + 1)")

    # Add summary text box inside plot in the upper right
    legit_mean = df[df["Class"] == 0]["Amount"].mean()
    fraud_mean = df[df["Class"] == 1]["Amount"].mean()
    legit_median = df[df["Class"] == 0]["Amount"].median()
    fraud_median = df[df["Class"] == 1]["Amount"].median()

    stats_text = (
        f"Legitimate: Mean = ${legit_mean:.2f}, Median = ${legit_median:.2f}\n"
        f"Fraudulent: Mean = ${fraud_mean:.2f}, Median = ${fraud_median:.2f}"
    ).replace("$", r"\$")

    ax.text(
        0.96,
        0.96,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#ffffff", edgecolor="#ced4da", alpha=0.95),
    )

    plt.tight_layout()
    fig3_path = output_dir / "amount_by_class_boxplot.png"
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close()
    generated_figures.append(fig3_path)

    logger.info("Generated %d visualization figures in %s", len(generated_figures), output_dir)
    return generated_figures


def df_to_markdown(df: pd.DataFrame, include_index: bool = False, index_col_name: str = "Feature") -> str:
    """Formats a pandas DataFrame into a GitHub Flavored Markdown table without third-party dependencies."""
    table_df = df.copy()
    if include_index:
        table_df = table_df.reset_index()
        table_df.rename(columns={table_df.columns[0]: index_col_name}, inplace=True)
    headers = [str(col) for col in table_df.columns]
    rows = [[str(val) for val in row] for row in table_df.values]
    col_widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]

    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    separator_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    row_lines = ["| " + " | ".join(r[i].ljust(col_widths[i]) for i in range(len(headers))) + " |" for r in rows]
    return "\n".join([header_line, separator_line] + row_lines)


def generate_markdown_report(
    dimensions: Dict[str, Any],
    cols_df: pd.DataFrame,
    missing_duplicates: Dict[str, Any],
    class_dist: Dict[str, Any],
    stats_df: pd.DataFrame,
    report_path: Path,
    figures: list[Path] | None = None,
) -> Path:
    """Writes a comprehensive Markdown inspection report under reports/.

    Args:
        dimensions: Output of inspect_dimensions.
        cols_df: Output of inspect_columns_and_types.
        missing_duplicates: Output of inspect_missing_and_duplicates.
        class_dist: Output of inspect_class_distribution.
        stats_df: Output of inspect_feature_statistics.
        report_path: Filepath for the generated report.
        figures: Optional list of generated figure paths.

    Returns:
        Path: Path to the generated report file.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Key stats for Time and Amount
    time_stats = stats_df.loc["Time"] if "Time" in stats_df.index else None
    amount_stats = stats_df.loc["Amount"] if "Amount" in stats_df.index else None

    # Format columns table
    cols_table = df_to_markdown(cols_df, include_index=False)

    # Key numerical features summary
    key_features = ["Time", "Amount", "V1", "V2", "V3", "V4", "V17", "V28"]
    avail_features = [f for f in key_features if f in stats_df.index]
    subset_stats = stats_df.loc[avail_features, ["mean", "std", "min", "50%", "max", "skewness"]].round(4)
    subset_stats_table = df_to_markdown(subset_stats, include_index=True, index_col_name="Feature")


    lines = [
        "# Phase 1: Dataset Inspection Report",
        "",
        "**Project Title**: Quantum Machine Learning for Financial Fraud Detection using Quantum Kernel Methods  ",
        "**Dataset**: ULB Credit Card Fraud Detection (`data/raw/creditcard.csv`)  ",
        "",
        "---",
        "",
        "## 1. Dataset Dimensions & Memory Footprint",
        "",
        f"- **Total Rows (Observations)**: {dimensions['n_rows']:,}",
        f"- **Total Columns (Features + Target)**: {dimensions['n_columns']}",
        f"- **In-Memory Size**: ~{dimensions['memory_mb']} MB",
        "",
        "## 2. Missing Values & Duplicate Records",
        "",
        f"- **Total Missing (Null) Values**: {missing_duplicates['total_missing_values']}",
        f"- **Columns with Missing Values**: {len(missing_duplicates['columns_with_missing'])}",
        f"- **Duplicate Rows**: {missing_duplicates['duplicate_rows']:,} ({missing_duplicates['duplicate_percentage']}%)",
        "",
        "## 3. Class Imbalance Analysis",
        "",
        f"- **Total Transactions**: {class_dist['total_records']:,}",
        f"- **Legitimate Transactions (`Class = 0`)**: {class_dist['legitimate_count (0)']:,} ({100 - class_dist['fraud_percentage']:.4f}%)",
        f"- **Fraudulent Transactions (`Class = 1`)**: {class_dist['fraudulent_count (1)']:,} ({class_dist['fraud_percentage']}%)",
        f"- **Imbalance Ratio**: {class_dist['imbalance_ratio']}:1 (Legitimate to Fraud)",
        "",
        "## 4. Key Feature Statistics",
        "",
        "### Transaction Amount (`Amount`)",
        f"- **Mean**: ${amount_stats['mean']:.2f}" if amount_stats is not None else "- N/A",
        f"- **Standard Deviation**: ${amount_stats['std']:.2f}" if amount_stats is not None else "- N/A",
        f"- **Min / Max**: ${amount_stats['min']:.2f} / ${amount_stats['max']:.2f}" if amount_stats is not None else "- N/A",
        f"- **25% / 50% (Median) / 75%**: ${amount_stats['25%']:.2f} / ${amount_stats['50%']:.2f} / ${amount_stats['75%']:.2f}" if amount_stats is not None else "- N/A",
        f"- **Skewness**: {amount_stats['skewness']:.2f} (High right-skewed tail)" if amount_stats is not None else "- N/A",
        "",
        "### Transaction Elapsed Time (`Time`)",
        f"- **Range**: {time_stats['min']:.1f}s to {time_stats['max']:.1f}s (~{time_stats['max']/3600:.2f} hours / 2 days)" if time_stats is not None else "- N/A",
        "",
        "### Sample Numerical Feature Statistics (Subset)",
        "",
        subset_stats_table,
        "",
        "## 5. Column Schema & Data Types",
        "",
        cols_table,
        "",
        "## 6. Visualizations Saved",
        "",
    ]

    if figures:
        for fig in figures:
            rel_fig = fig.relative_to(report_path.parent)
            lines.append(f"- `reports/{rel_fig}`")
    else:
        lines.append("- No figures generated.")

    lines.extend([
        "",
        "## 7. Implications for Quantum Kernel Methods (Next Phases)",
        "",
        "1. **Dimensionality Reduction Requirement**:",
        "   - The dataset contains 30 input features (`Time`, `Amount`, `V1`–`V28`).",
        "   - Current Noisy Intermediate-Scale Quantum (NISQ) simulators and hardware have limited qubit capacity (typically 8–16 qubits for practical quantum kernel simulation).",
        "   - Feature selection or further dimensionality reduction (e.g., PCA, autoencoders, or feature importance ranking) will be essential prior to quantum feature mapping.",
        "",
        "2. **Severe Class Imbalance Strategy**:",
        "   - With only 0.17% fraud prevalence, training quantum kernel classifiers directly on the full dataset is computationally prohibitive ($O(N^2)$ kernel matrix complexity) and susceptible to majority class bias.",
        "   - Balanced sub-sampling (e.g., all frauds + sampled legitimate cases) or undersampling/SMOTE techniques will be needed for quantum kernel matrix calculation.",
        "",
        "3. **Data Scaling for Quantum State Encoding**:",
        "   - Quantum feature maps (e.g., angle embedding, ZZ-feature maps) require features bounded in $[0, \pi]$ or $[-\pi, \pi]$ to prevent phase ambiguity.",
        "   - Power-transforming `Amount` (to handle extreme skewness) followed by `MinMaxScaler` or `RobustScaler` will be crucial in Phase 2.",
    ])

    content = "\n".join(lines) + "\n"
    report_path.write_text(content, encoding="utf-8")
    logger.info("Inspection report successfully written to %s", report_path)
    return report_path


def run_full_inspection(data_path: Path, reports_dir: Path) -> Dict[str, Any]:
    """Executes the complete Phase 1 inspection pipeline.

    Args:
        data_path: Path to raw dataset.
        reports_dir: Directory where reports and figures should be saved.

    Returns:
        Dict[str, Any]: Consolidated inspection metrics and status.
    """
    if not check_dataset_exists(data_path):
        instructions = get_missing_dataset_instructions(data_path)
        logger.warning(instructions)
        return {
            "status": "DATASET_MISSING",
            "message": instructions,
            "data_path": str(data_path),
        }

    df = load_dataset(data_path)
    dims = inspect_dimensions(df)
    cols_df = inspect_columns_and_types(df)
    missing_dups = inspect_missing_and_duplicates(df)
    class_dist = inspect_class_distribution(df)
    stats_df = inspect_feature_statistics(df)

    figures_dir = reports_dir / "figures"
    figures = generate_visualizations(df, figures_dir)

    report_file = reports_dir / "dataset_inspection_report.md"
    generate_markdown_report(
        dimensions=dims,
        cols_df=cols_df,
        missing_duplicates=missing_dups,
        class_dist=class_dist,
        stats_df=stats_df,
        report_path=report_file,
        figures=figures,
    )

    return {
        "status": "SUCCESS",
        "dimensions": dims,
        "columns_summary": cols_df.to_dict(orient="records"),
        "missing_duplicates": missing_dups,
        "class_distribution": class_dist,
        "feature_statistics": stats_df,
        "figures": [str(f) for f in figures],
        "report_file": str(report_file),
    }

