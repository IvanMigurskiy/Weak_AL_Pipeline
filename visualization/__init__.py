"""
Visualization module — plots and tables for thesis.

Generates:
1. Accuracy vs human labels curve (all methods compared)
2. F1 vs human labels curve
3. Final accuracy bar chart comparison
4. WS contribution analysis
5. Summary table (LaTeX-ready)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from ..pipeline import PipelineResult


# =========================================================================
# STYLE
# =========================================================================

COLORS = {
    "baseline": "#2d3436",
    "random_labels": "#b2bec3",
    "al_only": "#e17055",
    "ws_only": "#00b894",
    "hybrid": "#6c5ce7",
}

LABELS = {
    "baseline": "Full Supervision",
    "random_labels": "Random Labels",
    "al_only": "AL Only",
    "ws_only": "WS Only",
    "hybrid": "Hybrid (AL + WS)",
}

MARKERS = {
    "baseline": "s",
    "random_labels": "D",
    "al_only": "^",
    "ws_only": "v",
    "hybrid": "o",
}


def _setup_plot():
    """Configure matplotlib for publication-quality plots."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def _align_curves(all_curves, metric_key="accuracy"):
    """
    Align curves from multiple runs onto a common x-axis (human_labels_used).
    Returns (common_x, mean_y, std_y) arrays.
    """
    if not all_curves:
        return None, None, None

    # Collect all (human_labels, metric) pairs
    all_points = []
    for human, metric in all_curves:
        all_points.extend(zip(human, metric))

    # Create common x-axis from union of all human label counts
    all_x = sorted(set(p[0] for p in all_points))

    if not all_x:
        return None, None, None

    # Interpolate each curve onto common x
    interp_curves = []
    for human, metric in all_curves:
        interp = np.interp(all_x, human, metric, left=metric[0], right=metric[-1])
        interp_curves.append(interp)

    mean_y = np.mean(interp_curves, axis=0)
    std_y = np.std(interp_curves, axis=0) if len(interp_curves) > 1 else np.zeros_like(mean_y)

    return np.array(all_x), mean_y, std_y


# =========================================================================
# PLOTTING FUNCTIONS
# =========================================================================

def plot_accuracy_vs_labels(
    results: dict[str, list[PipelineResult]],
    output_path: str | Path | None = None,
    title: str = "Classification Accuracy vs. Human Labels",
) -> None:
    """Plot accuracy vs number of human labels for all methods."""
    if not HAS_MPL:
        print("matplotlib not available, skipping plot")
        return

    _setup_plot()
    fig, ax = plt.subplots()

    for mode in ["baseline", "random_labels", "al_only", "ws_only", "hybrid"]:
        if mode not in results or not results[mode]:
            continue

        mode_results = results[mode]

        if mode == "baseline":
            accs = [r.final_accuracy for r in mode_results]
            mean_acc = np.mean(accs)
            ax.axhline(
                y=mean_acc,
                color=COLORS[mode],
                linestyle="--",
                linewidth=1.5,
                label=f"{LABELS[mode]} ({mean_acc:.3f})",
            )
            continue

        if mode == "random_labels":
            accs = [r.final_accuracy for r in mode_results]
            human_counts = [r.total_human_labels for r in mode_results]
            mean_acc = np.mean(accs)
            mean_human = np.mean(human_counts)
            ax.scatter(
                [mean_human], [mean_acc],
                color=COLORS[mode],
                marker=MARKERS[mode],
                s=100,
                zorder=5,
                label=f"{LABELS[mode]} ({mean_acc:.3f})",
            )
            continue

        # Methods with history: plot curve
        all_curves = []
        for r in mode_results:
            if not r.history:
                continue
            human_labels = [h["human_labels_used"] for h in r.history]
            accuracies = [h["accuracy"] for h in r.history]
            all_curves.append((human_labels, accuracies))

        if not all_curves:
            continue

        common_x, mean_y, std_y = _align_curves(all_curves)
        if common_x is None:
            continue

        ax.plot(
            common_x,
            mean_y,
            color=COLORS[mode],
            marker=MARKERS[mode],
            markersize=4,
            markevery=max(1, len(common_x) // 10),
            linewidth=2,
            label=f"{LABELS[mode]} ({mean_y[-1]:.3f})",
        )
        ax.fill_between(
            common_x,
            mean_y - std_y,
            mean_y + std_y,
            color=COLORS[mode],
            alpha=0.15,
        )

    ax.set_xlabel("Number of Human Labels")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_ylim(bottom=max(0, ax.get_ylim()[0]))

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f"Saved: {output_path}")
    else:
        fig.savefig("accuracy_vs_labels.png")

    plt.close(fig)


def plot_f1_vs_labels(
    results: dict[str, list[PipelineResult]],
    output_path: str | Path | None = None,
    title: str = "F1 Macro vs. Human Labels",
) -> None:
    """Plot F1 macro vs human labels."""
    if not HAS_MPL:
        print("matplotlib not available, skipping plot")
        return

    _setup_plot()
    fig, ax = plt.subplots()

    for mode in ["baseline", "random_labels", "al_only", "ws_only", "hybrid"]:
        if mode not in results or not results[mode]:
            continue

        mode_results = results[mode]

        if mode == "baseline":
            f1s = [r.final_f1_macro for r in mode_results]
            mean_f1 = np.mean(f1s)
            ax.axhline(
                y=mean_f1,
                color=COLORS[mode],
                linestyle="--",
                linewidth=1.5,
                label=f"{LABELS[mode]} ({mean_f1:.3f})",
            )
            continue

        if mode == "random_labels":
            f1s = [r.final_f1_macro for r in mode_results]
            human_counts = [r.total_human_labels for r in mode_results]
            mean_f1 = np.mean(f1s)
            mean_human = np.mean(human_counts)
            ax.scatter(
                [mean_human], [mean_f1],
                color=COLORS[mode],
                marker=MARKERS[mode],
                s=100,
                zorder=5,
                label=f"{LABELS[mode]} ({mean_f1:.3f})",
            )
            continue

        all_curves = []
        for r in mode_results:
            if not r.history:
                continue
            human_labels = [h["human_labels_used"] for h in r.history]
            f1s = [h.get("f1_macro", 0) for h in r.history]
            all_curves.append((human_labels, f1s))

        if not all_curves:
            continue

        common_x, mean_y, std_y = _align_curves(all_curves)
        if common_x is None:
            continue

        ax.plot(
            common_x,
            mean_y,
            color=COLORS[mode],
            marker=MARKERS[mode],
            markersize=4,
            markevery=max(1, len(common_x) // 10),
            linewidth=2,
            label=f"{LABELS[mode]} ({mean_y[-1]:.3f})",
        )
        ax.fill_between(
            common_x,
            mean_y - std_y,
            mean_y + std_y,
            color=COLORS[mode],
            alpha=0.15,
        )

    ax.set_xlabel("Number of Human Labels")
    ax.set_ylabel("F1 Macro")
    ax.set_title(title)
    ax.legend(loc="lower right")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f"Saved: {output_path}")
    else:
        fig.savefig("f1_vs_labels.png")

    plt.close(fig)


def plot_final_comparison_bar(
    results: dict[str, list[PipelineResult]],
    output_path: str | Path | None = None,
    title: str = "Final Performance Comparison",
) -> None:
    """Bar chart comparing final accuracy and F1 across methods."""
    if not HAS_MPL:
        print("matplotlib not available, skipping plot")
        return

    _setup_plot()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    modes = ["baseline", "random_labels", "al_only", "ws_only", "hybrid"]
    mode_order = [m for m in modes if m in results and results[m]]

    acc_means = []
    acc_stds = []
    f1_means = []
    f1_stds = []
    labels = []

    for mode in mode_order:
        mode_results = results[mode]
        accs = [r.final_accuracy for r in mode_results]
        f1s = [r.final_f1_macro for r in mode_results]
        acc_means.append(np.mean(accs))
        acc_stds.append(np.std(accs) if len(accs) > 1 else 0)
        f1_means.append(np.mean(f1s))
        f1_stds.append(np.std(f1s) if len(f1s) > 1 else 0)
        labels.append(LABELS[mode])

    x = np.arange(len(labels))
    colors = [COLORS[m] for m in mode_order]

    # Accuracy bars
    bars1 = ax1.bar(x, acc_means, yerr=acc_stds, color=colors, capsize=5, edgecolor="white")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.set_ylim(bottom=max(0, min(acc_means) - 0.1))

    for bar, val in zip(bars1, acc_means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    # F1 bars
    bars2 = ax2.bar(x, f1_means, yerr=f1_stds, color=colors, capsize=5, edgecolor="white")
    ax2.set_ylabel("F1 Macro")
    ax2.set_title("F1 Macro")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha="right")
    ax2.set_ylim(bottom=max(0, min(f1_means) - 0.1))

    for bar, val in zip(bars2, f1_means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    fig.suptitle(title, fontsize=15, fontweight="bold")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f"Saved: {output_path}")
    else:
        fig.savefig("final_comparison.png")

    plt.close(fig)


def plot_ws_contribution(
    results: dict[str, list[PipelineResult]],
    output_path: str | Path | None = None,
) -> None:
    """Stacked bar chart: human labels vs WS labels contribution."""
    if not HAS_MPL:
        print("matplotlib not available, skipping plot")
        return

    _setup_plot()
    fig, ax = plt.subplots(figsize=(8, 5))

    modes_with_ws = ["ws_only", "hybrid"]
    mode_order = [m for m in modes_with_ws if m in results and results[m]]

    if not mode_order:
        print("No WS modes to plot")
        plt.close(fig)
        return

    labels = []
    human_means = []
    ws_means = []
    ws_acc_means = []

    for mode in mode_order:
        mode_results = results[mode]
        human_means.append(np.mean([r.total_human_labels for r in mode_results]))
        ws_means.append(np.mean([r.total_ws_labels for r in mode_results]))
        ws_acc_vals = [r.ws_label_accuracy for r in mode_results if r.ws_label_accuracy > 0]
        ws_acc_means.append(np.mean(ws_acc_vals) if ws_acc_vals else 0)
        labels.append(LABELS[mode])

    x = np.arange(len(labels))
    width = 0.5

    bars1 = ax.bar(x, human_means, width, label="Human Labels", color="#e17055")
    bars2 = ax.bar(x, ws_means, width, bottom=human_means, label="WS Labels", color="#00b894")

    ax.set_ylabel("Number of Labels")
    ax.set_title("Label Source Contribution")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for i, (h, w, wa) in enumerate(zip(human_means, ws_means, ws_acc_means)):
        total = h + w
        pct = w / total * 100 if total > 0 else 0
        ax.text(i, total + 10, f"WS acc: {wa:.3f}\nWS%: {pct:.0f}%",
                ha="center", va="bottom", fontsize=10)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f"Saved: {output_path}")
    else:
        fig.savefig("ws_contribution.png")

    plt.close(fig)


# =========================================================================
# TABLE GENERATION
# =========================================================================

def generate_results_table(
    results: dict[str, list[PipelineResult]],
) -> str:
    """Generate LaTeX-ready results table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Comparison of classification methods on customer support tickets}",
        r"\label{tab:results}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & Accuracy & F1 Macro & Human Labels & WS Labels & WS Accuracy & WS \% \\",
        r"\midrule",
    ]

    modes = ["baseline", "random_labels", "al_only", "ws_only", "hybrid"]
    for mode in modes:
        if mode not in results or not results[mode]:
            continue

        mode_results = results[mode]
        acc = np.mean([r.final_accuracy for r in mode_results])
        f1 = np.mean([r.final_f1_macro for r in mode_results])
        human = np.mean([r.total_human_labels for r in mode_results])
        ws = np.mean([r.total_ws_labels for r in mode_results])
        ws_acc_vals = [r.ws_label_accuracy for r in mode_results if r.ws_label_accuracy > 0]
        ws_acc = np.mean(ws_acc_vals) if ws_acc_vals else 0
        total = human + ws
        ws_pct = ws / total * 100 if total > 0 else 0

        label = LABELS[mode]
        lines.append(
            f"{label} & {acc:.4f} & {f1:.4f} & {int(human)} & "
            f"{int(ws)} & {ws_acc:.4f} & {ws_pct:.1f}\\% \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_all_plots(
    results: dict[str, list[PipelineResult]],
    output_dir: str | Path = "results",
) -> None:
    """Generate all plots and save to output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating plots in {output_dir}/...")

    plot_accuracy_vs_labels(
        results,
        output_path=output_dir / "accuracy_vs_labels.png",
    )
    plot_f1_vs_labels(
        results,
        output_path=output_dir / "f1_vs_labels.png",
    )
    plot_final_comparison_bar(
        results,
        output_path=output_dir / "final_comparison.png",
    )
    plot_ws_contribution(
        results,
        output_path=output_dir / "ws_contribution.png",
    )

    # Save LaTeX table
    latex_table = generate_results_table(results)
    table_path = output_dir / "results_table.tex"
    table_path.write_text(latex_table)
    print(f"Saved: {table_path}")

    # Save raw results as JSON
    import json
    json_results = {}
    for mode, mode_results in results.items():
        json_results[mode] = []
        for r in mode_results:
            json_results[mode].append({
                "name": r.name,
                "final_accuracy": r.final_accuracy,
                "final_f1_macro": r.final_f1_macro,
                "total_human_labels": r.total_human_labels,
                "total_ws_labels": r.total_ws_labels,
                "total_labels": r.total_labels,
                "ws_label_accuracy": r.ws_label_accuracy,
                "ws_contribution_pct": r.ws_contribution_pct,
                "human_savings_pct": r.human_savings_pct,
                "baseline_accuracy": r.baseline_accuracy,
                "baseline_f1_macro": r.baseline_f1_macro,
                "n_pool": r.n_pool,
                "n_test": r.n_test,
                "n_classes": r.n_classes,
                "history": r.history,
            })

    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(json_results, indent=2))
    print(f"Saved: {json_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for mode in ["baseline", "random_labels", "al_only", "ws_only", "hybrid"]:
        if mode not in results or not results[mode]:
            continue
        mode_results = results[mode]
        acc = np.mean([r.final_accuracy for r in mode_results])
        f1 = np.mean([r.final_f1_macro for r in mode_results])
        human = int(np.mean([r.total_human_labels for r in mode_results]))
        ws = int(np.mean([r.total_ws_labels for r in mode_results]))
        print(f"  {LABELS[mode]:25s} | Acc: {acc:.4f} | F1: {f1:.4f} | "
              f"Human: {human:5d} | WS: {ws:5d}")
