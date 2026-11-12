"""Generate publication-quality visual diagrams for the Model Performance Report:
1. Decision Score Distribution Histograms (Fraud vs. Legitimate separation)
2. Performance Metrics Histogram / Grouped Bar Distribution
3. Multi-Model Confusion Matrix Heatmaps (Default tau = 0.0)
4. Multi-Model Confusion Matrix Heatmaps (Calibrated tau*)
5. Holdout Test Set Confusion Matrix Comparison (Default vs Cost-Optimal)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DEMO_DIR = REPO_ROOT / "reports" / "mentor_demo"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

# Set high-quality styling
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8


def plot_score_distribution_histograms():
    """Generates 4-panel histogram of model decision scores for Legit vs Fraud transactions."""
    test_preds_path = REPO_ROOT / "data" / "processed" / "quantum" / "predictions_test.csv"
    if not test_preds_path.exists():
        print(f"File not found: {test_preds_path}")
        return

    df = pd.read_csv(test_preds_path)
    y_true = df["y_true"].values

    models_config = [
        {
            "id": "score_classical_svm_30f",
            "name": "Historical Classical SVM (30 Features, N=226k)",
            "th_def": 0.0,
            "th_cost": None,
            "prauc": 0.7052,
            "color_legit": "#2b5c8f",
            "color_fraud": "#d62728",
        },
        {
            "id": "score_linear_svm_6f",
            "name": "Tuned Classical Linear SVM (6 Features, N=2k)",
            "th_def": 0.0,
            "th_cost": -0.3833,
            "prauc": 0.6723,
            "color_legit": "#4a7c59",
            "color_fraud": "#e65100",
        },
        {
            "id": "score_rbf_svm_6f",
            "name": "Tuned Classical RBF SVM (6 Features, N=2k)",
            "th_def": 0.0,
            "th_cost": -0.8324,
            "prauc": 0.6713,
            "color_legit": "#1976d2",
            "color_fraud": "#c2185b",
        },
        {
            "id": "score_qsvm_6f",
            "name": "Tuned Quantum Kernel SVM (6 Features, N=2k)",
            "th_def": 0.0,
            "th_cost": -0.8670,
            "prauc": 0.4966,
            "color_legit": "#512da8",
            "color_fraud": "#d32f2f",
        },
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    axes = axes.flatten()

    for idx, cfg in enumerate(models_config):
        ax = axes[idx]
        scores = df[cfg["id"]].values

        scores_legit = scores[y_true == 0]
        scores_fraud = scores[y_true == 1]

        mu_legit = np.mean(scores_legit)
        std_legit = np.std(scores_legit)
        mu_fraud = np.mean(scores_fraud)
        std_fraud = np.std(scores_fraud)
        delta_mu = mu_fraud - mu_legit

        # Determine reasonable x limits (avoid extreme outliers clipping the view)
        q_low = np.percentile(scores_legit, 0.5)
        q_high = max(np.percentile(scores_fraud, 98.0), np.percentile(scores_legit, 99.5))
        bins = np.linspace(q_low, q_high, 75)

        # Plot histograms with density normalization
        ax.hist(
            scores_legit,
            bins=bins,
            density=True,
            alpha=0.55,
            color=cfg["color_legit"],
            label=f"Legitimate ({len(scores_legit):,} txns)",
            edgecolor="white",
            linewidth=0.5,
        )
        ax.hist(
            scores_fraud,
            bins=bins,
            density=True,
            alpha=0.75,
            color=cfg["color_fraud"],
            label=f"Fraud ({len(scores_fraud):,} txns)",
            edgecolor="black",
            linewidth=0.7,
        )

        # Plot vertical threshold lines
        ax.axvline(
            cfg["th_def"],
            color="#b71c1c",
            linestyle="--",
            linewidth=2.2,
            label=f"Default Threshold ($\\tau = {cfg['th_def']:.1f}$)",
        )
        if cfg["th_cost"] is not None:
            ax.axvline(
                cfg["th_cost"],
                color="#1b5e20",
                linestyle="-.",
                linewidth=2.2,
                label=f"Calibrated Threshold ($\\tau^* = {cfg['th_cost']:.2f}$)",
            )

        # Annotations
        stats_text = (
            f"PR-AUC: {cfg['prauc']:.4f}\n"
            f"Legit Mean: {mu_legit:.2f} (±{std_legit:.2f})\n"
            f"Fraud Mean: {mu_fraud:.2f} (±{std_fraud:.2f})\n"
            f"Separation (Δμ): {delta_mu:+.2f}"
        )
        ax.text(
            0.03,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.92),
        )

        ax.set_title(cfg["name"], fontsize=12.5, fontweight="bold", pad=10)
        ax.set_xlabel("SVM Decision Score $f(x)$", fontsize=11, fontweight="bold")
        ax.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
        ax.set_xlim([q_low, q_high])
        ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=9.5)

    fig.suptitle(
        "Decision Score Distribution Histograms: Fraud vs. Legitimate Separation\n"
        "(Holdout Test Set: 56,746 Transactions, 95 Frauds, Natural Prevalence $\\pi = 0.1674\%$)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = FIGURES_DIR / "model_score_histograms.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_performance_metrics_histogram():
    """Generates grouped bar / histogram visual comparing performance metrics across models."""
    csv_path = DEMO_DIR / "mentor_demo_results.csv"
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    short_names = [
        "Classical Linear",
        "Fixed Classical RBF",
        "Fixed Separable QSVM",
        "Trainable KTA QSVM",
        "Trainable RBF",
        "Macro Nystroem (181k)",
    ]
    df["display_name"] = short_names

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)

    # 1. PR-AUC & ROC-AUC
    ax1 = axes[0, 0]
    x = np.arange(len(df))
    w = 0.36
    b1 = ax1.bar(x - w / 2, df["pr_auc"], w, label="PR-AUC (Primary Metric)", color="#1f77b4", edgecolor="black", linewidth=0.6)
    b2 = ax1.bar(x + w / 2, df["roc_auc"], w, label="ROC-AUC", color="#aec7e8", edgecolor="black", linewidth=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["display_name"], rotation=20, ha="right", fontsize=9.5, fontweight="bold")
    ax1.set_ylim([0.0, 1.08])
    ax1.set_ylabel("Metric Score", fontsize=11, fontweight="bold")
    ax1.set_title("Ranking Capacity: PR-AUC vs. ROC-AUC", fontsize=12.5, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True)
    for b in b1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, h + 0.02, f"{h:.3f}", ha="center", fontsize=8.5, fontweight="bold")

    # 2. Precision & Recall at Default Threshold
    ax2 = axes[0, 1]
    b3 = ax2.bar(x - w / 2, df["default_precision_pct"], w, label="Precision (%)", color="#2ca02c", edgecolor="black", linewidth=0.6)
    b4 = ax2.bar(x + w / 2, df["default_recall_pct"], w, label="Recall (%)", color="#ff7f0e", edgecolor="black", linewidth=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["display_name"], rotation=20, ha="right", fontsize=9.5, fontweight="bold")
    ax2.set_ylim([0, 105])
    ax2.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Operational Detection: Precision vs. Recall (Default $\\tau=0.0$)", fontsize=12.5, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True)
    for b in b3:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
    for b in b4:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", fontsize=8.5)

    # 3. False Alarm Burden (FP Count on Log Scale)
    ax3 = axes[1, 0]
    colors_fp = ["#d95f02" if fp > 100 else "#1b9e77" for fp in df["default_fp"]]
    bars_fp = ax3.bar(x, df["default_fp"], width=0.55, color=colors_fp, edgecolor="black", linewidth=0.8)
    ax3.set_yscale("log")
    ax3.set_xticks(x)
    ax3.set_xticklabels(df["display_name"], rotation=20, ha="right", fontsize=9.5, fontweight="bold")
    ax3.set_ylabel("False Positive Count (Log Scale)", fontsize=11, fontweight="bold")
    ax3.set_title("False Alarm Operational Burden (Lower is Better)", fontsize=12.5, fontweight="bold")
    ax3.axhline(50, color="gray", linestyle=":", label="50 Alert Threshold")
    for b, val in zip(bars_fp, df["default_fp"]):
        ax3.text(b.get_x() + b.get_width() / 2, val * 1.15, f"{val}", ha="center", fontsize=9, fontweight="bold")
    ax3.legend(loc="upper left", frameon=True)

    # 4. Financial Cost ($10 FP + $200 FN)
    ax4 = axes[1, 1]
    b5 = ax4.bar(x - w / 2, df["default_cost_dollars"], w, label="Default Cost ($)", color="#d62728", edgecolor="black", linewidth=0.6)
    b6 = ax4.bar(x + w / 2, df["calibrated_cost_dollars"], w, label="Calibrated Cost ($)", color="#9467bd", edgecolor="black", linewidth=0.6)
    ax4.set_xticks(x)
    ax4.set_xticklabels(df["display_name"], rotation=20, ha="right", fontsize=9.5, fontweight="bold")
    ax4.set_ylabel("Institutional Loss ($)", fontsize=11, fontweight="bold")
    ax4.set_title("Total Operational Financial Loss ($C = 10\\text{FP} + 200\\text{FN}$)", fontsize=12.5, fontweight="bold")
    ax4.legend(loc="upper right", frameon=True)
    for b in b5:
        h = b.get_height()
        ax4.text(b.get_x() + b.get_width() / 2, h + 300, f"${int(h):,}", ha="center", fontsize=7.5, fontweight="bold", rotation=40)

    fig.suptitle(
        "Model Performance Metrics Histogram & Comparative Distribution\n"
        "(Evaluated on 45,396 Validation Transactions, 75 Frauds, Natural Prevalence $\\pi = 0.1652\%$)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = FIGURES_DIR / "model_performance_histogram.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_detailed_confusion_matrices(threshold_type: str = "default"):
    """Plots 2x3 grid of confusion matrices with counts, percentages, and performance stats."""
    csv_path = DEMO_DIR / "mentor_demo_results.csv"
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5), dpi=300)
    axes = axes.flatten()

    short_names = [
        "Classical Linear SVM (K=6)",
        "Fixed Classical RBF SVM (K=16)",
        "Fixed Separable QSVM (K=16)",
        "Trainable KTA QSVM (K=16)",
        "Trainable Classical RBF (K=16)",
        "Macro Nystroem RBF (181k Ref)",
    ]

    for idx, row in df.iterrows():
        ax = axes[idx]
        if threshold_type == "default":
            tp = int(row["default_tp"])
            fp = int(row["default_fp"])
            tn = int(row["default_tn"])
            fn = int(row["default_fn"])
            prec = row["default_precision_pct"]
            rec = row["default_recall_pct"]
            f1 = row["default_f1"]
            cost = row["default_cost_dollars"]
            th_str = f"Default $\\tau = 0.0$"
        else:
            tp = int(row["calibrated_tp"])
            fp = int(row["calibrated_fp"])
            tn = int(row["calibrated_tn"])
            fn = int(row["calibrated_fn"])
            prec = row["calibrated_precision_pct"]
            rec = row["calibrated_recall_pct"]
            f1 = row["calibrated_f1"]
            cost = row["calibrated_cost_dollars"]
            th_val = row["calibrated_threshold"]
            th_str = f"Calibrated $\\tau^* = {th_val:.2f}$"

        cm = np.array([[tn, fp], [fn, tp]])
        total_legit = tn + fp
        total_fraud = fn + tp

        # Create annotation text with counts and rates
        annot = np.array([
            [
                f"TN\n{tn:,}\n({tn/total_legit*100:.2f}%)",
                f"FP (Alerts)\n{fp:,}\n({fp/total_legit*100:.2f}%)",
            ],
            [
                f"FN (Missed)\n{fn:,}\n({fn/total_fraud*100:.2f}%)",
                f"TP (Caught)\n{tp:,}\n({tp/total_fraud*100:.2f}%)",
            ],
        ])

        # Plot heatmap
        cmap = "Blues" if "QSVM" not in short_names[idx] else "Purples"
        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap=cmap,
            cbar=False,
            ax=ax,
            annot_kws={"size": 11, "weight": "bold", "va": "center"},
            linewidths=1.2,
            linecolor="white",
        )

        title_text = f"{short_names[idx]}\n{th_str}\nPrec: {prec:.1f}% | Rec: {rec:.1f}% | Cost: ${int(cost):,}"
        ax.set_title(title_text, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
        ax.set_ylabel("Actual Label", fontsize=10, fontweight="bold")
        ax.set_xticklabels(["Legitimate (0)", "Fraudulent (1)"], fontsize=10, fontweight="bold")
        ax.set_yticklabels(["Legitimate (0)", "Fraudulent (1)"], fontsize=10, fontweight="bold")

    prefix = "Default ($\\tau = 0.0$)" if threshold_type == "default" else "Prevalence-Calibrated ($\\tau^*$)"
    fig.suptitle(
        f"Multi-Model Confusion Matrices under {prefix} Operating Thresholds\n"
        f"(Validation Set: 45,396 Transactions, 75 Frauds, 45,321 Legitimate, $\\pi = 0.1652\%$)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    filename = f"confusion_matrices_{threshold_type}.png"
    out_path = FIGURES_DIR / filename
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_holdout_confusion_matrices():
    """Generates confusion matrices for the 56,746-sample locked holdout test set."""
    test_preds_path = REPO_ROOT / "data" / "processed" / "quantum" / "predictions_test.csv"
    if not test_preds_path.exists():
        return

    df = pd.read_csv(test_preds_path)
    y_true = df["y_true"].values

    models = [
        ("pred_classical_svm_30f_default", "Classical SVM (30 Features, Default)", 0.7052),
        ("pred_linear_svm_6f_default", "Tuned Linear SVM (6 Features, Default)", 0.6723),
        ("pred_linear_svm_6f_cost_opt_R20", "Tuned Linear SVM (6 Features, Cost-Opt R=20)", 0.6723),
        ("pred_rbf_svm_6f_default", "Tuned Classical RBF (6 Features, Default)", 0.6713),
        ("pred_rbf_svm_6f_cost_opt_R20", "Tuned Classical RBF (6 Features, Cost-Opt R=20)", 0.6713),
        ("pred_qsvm_6f_default", "Tuned QSVM (6 Features, Default)", 0.4966),
        ("pred_qsvm_6f_cost_opt_R20", "Tuned QSVM (6 Features, Cost-Opt R=20)", 0.4966),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9.5), dpi=300)
    axes = axes.flatten()

    for idx, (col, name, prauc) in enumerate(models):
        ax = axes[idx]
        preds = df[col].values
        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = cm.ravel()
        total_legit = tn + fp
        total_fraud = fn + tp

        prec = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
        rec = tp / total_fraud * 100.0
        cost = 10 * fp + 200 * fn

        annot = np.array([
            [f"TN\n{tn:,}\n({tn/total_legit*100:.2f}%)", f"FP\n{fp:,}\n({fp/total_legit*100:.2f}%)"],
            [f"FN\n{fn:,}\n({fn/total_fraud*100:.2f}%)", f"TP\n{tp:,}\n({tp/total_fraud*100:.2f}%)"],
        ])

        cmap = "Purples" if "QSVM" in name else "Blues"
        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap=cmap,
            cbar=False,
            ax=ax,
            annot_kws={"size": 10, "weight": "bold", "va": "center"},
            linewidths=1.0,
            linecolor="white",
        )
        ax.set_title(f"{name}\nPrec: {prec:.1f}% | Rec: {rec:.1f}% | Cost: ${cost:,}", fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel("Predicted", fontsize=9, fontweight="bold")
        ax.set_ylabel("Actual", fontsize=9, fontweight="bold")
        ax.set_xticklabels(["Legit (0)", "Fraud (1)"], fontsize=9)
        ax.set_yticklabels(["Legit (0)", "Fraud (1)"], fontsize=9)

    # Hide the 8th unused subplot
    axes[7].axis("off")

    fig.suptitle(
        "Holdout Test Set Confusion Matrices: Default vs. Cost-Optimal ($R=20$) Operating Points\n"
        "(56,746 Transactions, 95 Frauds, 56,651 Legitimate, Natural Prevalence $\\pi = 0.1674\%$)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = FIGURES_DIR / "holdout_confusion_matrices_detailed.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    print("Generating visual diagrams for model report...")
    plot_score_distribution_histograms()
    plot_performance_metrics_histogram()
    plot_detailed_confusion_matrices(threshold_type="default")
    plot_detailed_confusion_matrices(threshold_type="calibrated")
    plot_holdout_confusion_matrices()
    print("All diagrams generated successfully!")
