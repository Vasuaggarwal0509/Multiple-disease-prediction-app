"""
Generate visualization plots from evaluation_results.json.

Produces:
  - <disease>_confusion_matrix.png    : Heatmap (one per disease)
  - <disease>_metrics.png              : Per-disease 5-metric bar chart
  - <disease>_per_class.png            : Per-class precision/recall/F1 bar chart
  - summary_accuracy.png               : Cross-disease accuracy comparison
  - summary_metrics.png                : Cross-disease 5-metric comparison

All plots saved to test_data/plots/ .

Usage:
    python scripts/visualize_results.py
"""

import os
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_JSON = os.path.join(BASE_DIR, "test_data", "evaluation_results.json")
PLOTS_DIR = os.path.join(BASE_DIR, "test_data", "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_style("whitegrid")
COLORS = {
    "eye_disease": "#4361ee",
    "pneumonia": "#06d6a0",
    "malaria": "#f72585",
}
DISEASE_TITLE = {
    "eye_disease": "Retinal Disease (OCT)",
    "pneumonia": "Pneumonia (Chest X-Ray)",
    "malaria": "Malaria (Cell Images)",
}


def load_results():
    if not os.path.exists(RESULTS_JSON):
        print(f"ERROR: {RESULTS_JSON} not found.")
        print("Run: python scripts/evaluate.py")
        sys.exit(1)
    with open(RESULTS_JSON) as f:
        return json.load(f)


def plot_confusion_matrix(disease_key, metrics):
    cm = np.array(metrics["confusion_matrix"])
    classes = metrics["class_labels"]

    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(classes)), max(4, 1.2 * len(classes))))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={"label": "Count"},
        ax=ax,
        annot_kws={"size": 12, "weight": "bold"},
    )
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax.set_title(
        f"{DISEASE_TITLE[disease_key]}\nConfusion Matrix "
        f"(Accuracy: {metrics['accuracy']*100:.2f}%)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"{disease_key}_confusion_matrix.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out


def plot_disease_metrics(disease_key, metrics):
    names = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]
    vals = [
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        metrics.get("auc_roc") if metrics.get("auc_roc") is not None else 0.0,
    ]
    vals_pct = [v * 100 for v in vals]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, vals_pct, color=COLORS[disease_key], edgecolor="black", linewidth=0.5)

    for bar, v in zip(bars, vals_pct):
        y_top = bar.get_height()
        # Place label inside the bar when close to the top, above otherwise
        if y_top >= 98:
            y_pos = y_top - 0.6
            va, color = "top", "white"
        else:
            y_pos = y_top + 0.3
            va, color = "bottom", "black"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"{v:.2f}%",
            ha="center", va=va, color=color,
            fontsize=10, fontweight="bold",
        )

    ax.set_ylim(70, 100)
    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"{DISEASE_TITLE[disease_key]} — Evaluation Metrics",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"{disease_key}_metrics.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out


def plot_per_class_metrics(disease_key, metrics):
    per_class = metrics.get("per_class_metrics", {})
    if not per_class:
        return None

    classes = list(per_class.keys())
    precision = [per_class[c]["precision"] * 100 for c in classes]
    recall = [per_class[c]["recall"] * 100 for c in classes]
    f1 = [per_class[c]["f1"] * 100 for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(classes)), 5))
    ax.bar(x - width, precision, width, label="Precision", color="#4361ee", edgecolor="black", linewidth=0.3)
    ax.bar(x, recall, width, label="Recall", color="#06d6a0", edgecolor="black", linewidth=0.3)
    ax.bar(x + width, f1, width, label="F1 Score", color="#f72585", edgecolor="black", linewidth=0.3)

    # Value labels — inside bar if near top, above otherwise
    def _place_label(ax, xv, yv, text):
        if yv >= 98:
            ax.text(xv, yv - 0.6, text, ha="center", va="top", fontsize=8, color="white", fontweight="bold")
        else:
            ax.text(xv, yv + 0.3, text, ha="center", va="bottom", fontsize=8)

    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        _place_label(ax, i - width, p, f"{p:.1f}")
        _place_label(ax, i, r, f"{r:.1f}")
        _place_label(ax, i + width, f, f"{f:.1f}")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Class", fontsize=11, fontweight="bold")
    ax.set_title(
        f"{DISEASE_TITLE[disease_key]} — Per-Class Metrics",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylim(70, 100)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"{disease_key}_per_class.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out


def plot_summary_accuracy(results):
    diseases = list(results.keys())
    labels = [DISEASE_TITLE[d] for d in diseases]
    accuracies = [results[d]["accuracy"] * 100 for d in diseases]
    colors = [COLORS[d] for d in diseases]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, accuracies, color=colors, edgecolor="black", linewidth=0.5)

    for bar, v in zip(bars, accuracies):
        y_top = bar.get_height()
        if y_top >= 98:
            y_pos, va, color = y_top - 0.6, "top", "white"
        else:
            y_pos, va, color = y_top + 0.3, "bottom", "black"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"{v:.2f}%",
            ha="center", va=va, color=color,
            fontsize=11, fontweight="bold",
        )

    ax.set_ylim(70, 100)
    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Cross-Disease Accuracy Comparison",
        fontsize=13, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "summary_accuracy.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out


def plot_summary_metrics(results):
    diseases = list(results.keys())
    labels = [DISEASE_TITLE[d] for d in diseases]
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]
    metric_keys = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]

    data = np.zeros((len(diseases), len(metric_keys)))
    for i, d in enumerate(diseases):
        for j, mk in enumerate(metric_keys):
            v = results[d].get(mk)
            data[i, j] = (v if v is not None else 0.0) * 100

    x = np.arange(len(metric_names))
    width = 0.8 / len(diseases)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, d in enumerate(diseases):
        offset = (i - (len(diseases) - 1) / 2) * width
        bars = ax.bar(x + offset, data[i], width, label=labels[i], color=COLORS[d], edgecolor="black", linewidth=0.3)
        for bar, v in zip(bars, data[i]):
            y_top = bar.get_height()
            if y_top >= 98:
                y_pos, va, txt_color = y_top - 0.5, "top", "white"
            else:
                y_pos, va, txt_color = y_top + 0.3, "bottom", "black"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                f"{v:.1f}",
                ha="center", va=va, color=txt_color,
                fontsize=8, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Cross-Disease Comparison — All Metrics",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylim(70, 100)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "summary_metrics.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out


def main():
    print("=" * 60)
    print("  Generating Evaluation Visualizations")
    print("=" * 60)

    results = load_results()
    print(f"Loaded metrics for {len(results)} diseases: {list(results.keys())}\n")

    generated = []

    for disease_key, metrics in results.items():
        print(f"[{disease_key}]")
        p = plot_confusion_matrix(disease_key, metrics)
        print(f"  Saved: {os.path.basename(p)}")
        generated.append(p)

        p = plot_disease_metrics(disease_key, metrics)
        print(f"  Saved: {os.path.basename(p)}")
        generated.append(p)

        p = plot_per_class_metrics(disease_key, metrics)
        if p:
            print(f"  Saved: {os.path.basename(p)}")
            generated.append(p)

    print("\n[summary]")
    p = plot_summary_accuracy(results)
    print(f"  Saved: {os.path.basename(p)}")
    generated.append(p)

    p = plot_summary_metrics(results)
    print(f"  Saved: {os.path.basename(p)}")
    generated.append(p)

    print(f"\n{len(generated)} plots saved to: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
