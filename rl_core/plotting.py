"""
rl_core/plotting.py
────────────────────
Plotting utilities for FYP_RL training and evaluation.

All plots are saved to ./plots/ (or a custom save_dir) with timestamps.
Requires: matplotlib, seaborn, numpy.

Exposed functions
─────────────────
  plot_warmstart_curves(epoch_metrics, save_dir)
      → warm-start training loss, val F1, unsafe_skip_rate, wasteful_retrieve_rate

  plot_training_curves(checkpoint_metrics, save_dir)
      → PPO training: avg_reward, judge_score, retrieval_rate, unsafe_skip_rate,
        wasteful_retrieve_rate, utility vs. env steps

  plot_evaluation_results(all_results, save_dir)
      → multi-strategy bar charts (judge_score, retrieval_rate, utility, F1)
      → Pareto front: judge_score vs retrieval_rate for all 4 strategies
      → per-scenario bar chart (Policy only)
      → confusion matrix heatmap (Policy)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("disaster_chatbot")

# ── Lazy imports so matplotlib is not required at module import time ───────────
def _import_plot_libs():
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend — safe for headless servers
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import seaborn as sns
        sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
        return plt, sns, mpatches
    except ImportError as exc:
        raise ImportError(
            "matplotlib and seaborn are required for plotting. "
            "Install with: pip install matplotlib seaborn"
        ) from exc


def _plots_dir(save_dir: str | Path) -> Path:
    """Return (and create) the ./plots/ sub-directory inside save_dir."""
    plots_dir = Path(save_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def _ts() -> str:
    """Short timestamp string for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Warm-start training curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_warmstart_curves(
    epoch_metrics: list[dict],
    save_dir: str | Path = "policy_checkpoints",
) -> Path:
    """
    Plot warm-start Stage A training curves.

    Parameters
    ----------
    epoch_metrics : list of dicts with keys:
        epoch, train_loss, val_accuracy, val_f1,
        unsafe_skip_rate, wasteful_retrieve_rate
    save_dir : where to write the PNG

    Returns
    -------
    Path to the saved PNG file.
    """
    if not epoch_metrics:
        logger.warning("[plotting] No warm-start epoch metrics to plot.")
        return Path(save_dir)

    plt, sns, _ = _import_plot_libs()

    epochs      = [m["epoch"] for m in epoch_metrics]
    train_loss  = [m["train_loss"] for m in epoch_metrics]
    val_f1      = [m["val_f1"] for m in epoch_metrics]
    val_acc     = [m["val_accuracy"] for m in epoch_metrics]
    unsafe_skip = [m["unsafe_skip_rate"] for m in epoch_metrics]
    wasteful    = [m["wasteful_retrieve_rate"] for m in epoch_metrics]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Stage A — Supervised Warm-Start Training Curves", fontsize=14, fontweight="bold")

    # Train loss
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, color="steelblue", linewidth=2, marker="o", markersize=4)
    ax.set_title("Training Loss (CrossEntropy)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")

    # Val F1 & Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, val_f1,  color="seagreen", linewidth=2, marker="s", markersize=4, label="Val F1")
    ax.plot(epochs, val_acc, color="orange",   linewidth=2, marker="^", markersize=4, label="Val Accuracy")
    ax.set_title("Validation F1 & Accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend()

    # Unsafe skip rate (FN / total) — most dangerous failure
    ax = axes[1, 0]
    ax.plot(epochs, unsafe_skip, color="crimson", linewidth=2, marker="o", markersize=4)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Unsafe Skip Rate  (FN / total)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Rate")
    ax.set_ylim(0, max(max(unsafe_skip) * 1.2, 0.1))

    # Wasteful retrieve rate (FP / total)
    ax = axes[1, 1]
    ax.plot(epochs, wasteful, color="darkorange", linewidth=2, marker="s", markersize=4)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Wasteful Retrieve Rate  (FP / total)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Rate")
    ax.set_ylim(0, max(max(wasteful) * 1.2, 0.1))

    plt.tight_layout()

    out_path = _plots_dir(save_dir) / f"warmstart_curves_{_ts()}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[plotting] Warm-start curves saved → %s", out_path)
    print(f"  📊 Warm-start curves → {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# 2. PPO training curves (from EvaluationCallback checkpoints)
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    checkpoint_metrics: list[dict],
    save_dir: str | Path = "policy_checkpoints",
) -> Path:
    """
    Plot PPO Stage B training curves from EvaluationCallback._checkpoint_metrics.

    Parameters
    ----------
    checkpoint_metrics : list of dicts with keys:
        env_steps, avg_reward, avg_judge_score, retrieval_rate,
        unsafe_skip_rate, wasteful_retrieve_rate, utility, f1
    save_dir : where to write the PNG

    Returns
    -------
    Path to the saved PNG file.
    """
    if not checkpoint_metrics:
        logger.warning("[plotting] No PPO checkpoint metrics to plot.")
        return Path(save_dir)

    plt, sns, _ = _import_plot_libs()

    steps       = [m["env_steps"] for m in checkpoint_metrics]
    avg_reward  = [m.get("avg_reward", 0) for m in checkpoint_metrics]
    judge_score = [m.get("avg_judge_score", 0) for m in checkpoint_metrics]
    ret_rate    = [m.get("retrieval_rate", 0) * 100 for m in checkpoint_metrics]
    unsafe_skip = [m.get("unsafe_skip_rate", 0) for m in checkpoint_metrics]
    wasteful    = [m.get("wasteful_retrieve_rate", 0) for m in checkpoint_metrics]
    utility     = [m.get("utility", 0) for m in checkpoint_metrics]
    f1          = [m.get("f1", 0) for m in checkpoint_metrics]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Stage B — PPO Fine-Tuning: Validation Curves", fontsize=14, fontweight="bold")

    _pairs = [
        (axes[0, 0], avg_reward,  "Avg Reward",              "steelblue"),
        (axes[0, 1], judge_score, "Avg Judge Score (/10)",   "seagreen"),
        (axes[1, 0], ret_rate,    "Retrieval Rate (%)",      "mediumpurple"),
        (axes[1, 1], utility,     "Utility  (Q − λC)",       "darkcyan"),
        (axes[2, 0], unsafe_skip, "Unsafe Skip Rate (FN/N)", "crimson"),
        (axes[2, 1], wasteful,    "Wasteful Retrieve Rate",  "darkorange"),
    ]

    for ax, values, title, color in _pairs:
        ax.plot(steps, values, color=color, linewidth=2, marker="o", markersize=5)
        ax.set_title(title)
        ax.set_xlabel("Env Steps")
        if "Rate" in title or "Skip" in title or "Wasteful" in title:
            ax.set_ylim(bottom=0)
        if "Score" in title:
            ax.set_ylim(0, 10)

    plt.tight_layout()

    out_path = _plots_dir(save_dir) / f"ppo_training_curves_{_ts()}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[plotting] PPO training curves saved → %s", out_path)
    print(f"  📊 PPO training curves → {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# 3. Evaluation results: multi-strategy comparison + Pareto + per-scenario + CM
# ──────────────────────────────────────────────────────────────────────────────

def plot_evaluation_results(
    all_results: dict[str, Any],
    save_dir: str | Path = "policy_checkpoints",
) -> list[Path]:
    """
    Generate all post-evaluation plots and return list of saved paths.

    Plots produced:
      1. Multi-strategy bar chart comparison
      2. Pareto front: judge_score vs retrieval_rate
      3. Per-scenario bar chart (Policy)
      4. Confusion matrix heatmap (Policy)
    """
    saved: list[Path] = []
    saved.append(_plot_strategy_bars(all_results, save_dir))
    saved.append(_plot_pareto_front(all_results, save_dir))
    p = _plot_per_scenario(all_results, save_dir)
    if p:
        saved.append(p)
    saved.append(_plot_confusion_matrix(all_results, save_dir))
    return [p for p in saved if p is not None]


def _plot_strategy_bars(
    all_results: dict[str, Any],
    save_dir: str | Path,
) -> Path:
    """4-strategy grouped bar chart for key metrics."""
    plt, sns, _ = _import_plot_libs()

    strategies = ["Policy", "Always-Retrieve", "Always-Skip", "Heuristic-Router"]
    metrics_to_plot = [
        ("avg_judge_score",       "Judge Score (/10)",       "steelblue"),
        ("retrieval_rate",        "Retrieval Rate",          "mediumpurple"),
        ("f1",                    "F1 (routing)",            "seagreen"),
        ("utility",               "Utility (Q−λC)",          "darkcyan"),
        ("unsafe_skip_rate",      "Unsafe Skip Rate",        "crimson"),
        ("wasteful_retrieve_rate","Wasteful Retrieve Rate",  "darkorange"),
    ]

    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("4-Strategy Evaluation Comparison", fontsize=14, fontweight="bold")
    axes_flat = axes.flatten()

    for idx, (key, title, color) in enumerate(metrics_to_plot):
        ax = axes_flat[idx]
        vals = []
        labels = []
        for strat in strategies:
            if strat in all_results and isinstance(all_results[strat], dict):
                v = all_results[strat].get(key, 0)
                vals.append(float(v) * 100 if key == "retrieval_rate" else float(v))
                labels.append(strat.replace("-", "\n"))

        bars = ax.bar(labels, vals, color=color, alpha=0.8, edgecolor="white", linewidth=1.2)
        ax.set_title(title)
        ax.set_ylabel("%" if key == "retrieval_rate" else "Value")

        # Annotate bars with values
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * max(vals + [0.01]),
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

        if key in ("unsafe_skip_rate", "wasteful_retrieve_rate", "retrieval_rate"):
            ax.set_ylim(0, min(max(vals) * 1.4 + 0.01, 105 if key == "retrieval_rate" else 1.1))
        elif key == "avg_judge_score":
            ax.set_ylim(0, 10)

    plt.tight_layout()
    out_path = _plots_dir(save_dir) / f"strategy_comparison_{_ts()}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[plotting] Strategy comparison saved → %s", out_path)
    print(f"  📊 Strategy comparison → {out_path}")
    return out_path


def _plot_pareto_front(
    all_results: dict[str, Any],
    save_dir: str | Path,
) -> Path:
    """
    Pareto front: judge_score (quality) vs retrieval_rate (cost).
    Higher judge score + lower retrieval rate = Pareto-dominant.
    Also shows tokens_saved vs judge_score as a secondary annotation.
    """
    plt, sns, _ = _import_plot_libs()

    strategies = ["Policy", "Always-Retrieve", "Always-Skip", "Heuristic-Router"]
    colors     = ["crimson", "steelblue", "seagreen", "darkorange"]
    markers    = ["*", "o", "s", "^"]
    sizes      = [300, 120, 120, 120]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("Pareto Front: Quality vs Retrieval Cost", fontsize=13, fontweight="bold")

    for strat, color, marker, size in zip(strategies, colors, markers, sizes):
        if strat not in all_results or not isinstance(all_results[strat], dict):
            continue
        res = all_results[strat]
        x = res.get("retrieval_rate", 0) * 100   # retrieval rate %  (cost)
        y = res.get("avg_judge_score", 0)          # judge score       (quality)
        ax.scatter(x, y, color=color, marker=marker, s=size, zorder=5, label=strat)
        ax.annotate(
            strat,
            (x, y),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9,
            color=color,
            fontweight="bold",
        )

    ax.set_xlabel("Retrieval Rate (%) — lower is cheaper", fontsize=11)
    ax.set_ylabel("Avg Judge Score (/10) — higher is better", fontsize=11)
    ax.set_xlim(-5, 105)
    ax.set_ylim(0, 10)

    # Shade the "ideal" quadrant (low cost, high quality)
    ax.axvline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(7,  color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(2, 9.5, "← Ideal region\n(low cost, high quality)", fontsize=8, color="gray")

    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    out_path = _plots_dir(save_dir) / f"pareto_front_{_ts()}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[plotting] Pareto front saved → %s", out_path)
    print(f"  📊 Pareto front → {out_path}")
    return out_path


def _plot_per_scenario(
    all_results: dict[str, Any],
    save_dir: str | Path,
) -> Path | None:
    """
    Per-scenario bar charts comparing Policy vs Heuristic-Router vs Always-Retrieve.
    Shows avg_judge_score, retrieval_rate, and unsafe_skip_rate per scenario type.
    """
    plt, sns, mpatches = _import_plot_libs()

    scenario_table = all_results.get("scenario_table", {})
    if not scenario_table:
        return None

    scenarios = sorted(scenario_table.keys())
    if not scenarios:
        return None

    compare_strategies = ["Policy", "Heuristic-Router", "Always-Retrieve"]
    strategy_colors    = {"Policy": "crimson", "Heuristic-Router": "darkorange", "Always-Retrieve": "steelblue"}
    metrics_to_plot    = [
        ("avg_judge_score",  "Judge Score (/10)"),
        ("retrieval_rate",   "Retrieval Rate"),
        ("unsafe_skip_rate", "Unsafe Skip Rate"),
    ]

    n_scenarios = len(scenarios)
    n_metrics   = len(metrics_to_plot)
    x           = np.arange(n_scenarios)
    width       = 0.25

    fig, axes = plt.subplots(n_metrics, 1, figsize=(max(10, n_scenarios * 1.5), 4 * n_metrics))
    if n_metrics == 1:
        axes = [axes]
    fig.suptitle("Per-Scenario Breakdown: Policy vs Baselines", fontsize=13, fontweight="bold")

    for ax_idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
        ax = axes[ax_idx]
        for s_idx, (strat, color) in enumerate(strategy_colors.items()):
            vals = []
            for scenario in scenarios:
                sm = scenario_table.get(scenario, {}).get(strat, {})
                v = sm.get(metric_key, 0)
                vals.append(float(v) * 100 if metric_key == "retrieval_rate" else float(v))
            offset = (s_idx - 1) * width
            bars = ax.bar(x + offset, vals, width, label=strat, color=color, alpha=0.8,
                          edgecolor="white", linewidth=0.8)

        ax.set_title(metric_label)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("%" if metric_key == "retrieval_rate" else "Value")
        if metric_key == "avg_judge_score":
            ax.set_ylim(0, 10)
        ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = _plots_dir(save_dir) / f"per_scenario_breakdown_{_ts()}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[plotting] Per-scenario breakdown saved → %s", out_path)
    print(f"  📊 Per-scenario breakdown → {out_path}")
    return out_path


def _plot_confusion_matrix(
    all_results: dict[str, Any],
    save_dir: str | Path,
) -> Path | None:
    """Confusion matrix heatmap for the Policy strategy."""
    plt, sns, _ = _import_plot_libs()

    if "Policy" not in all_results or not isinstance(all_results["Policy"], dict):
        return None

    p = all_results["Policy"]
    tp = p.get("TP", 0)
    fp = p.get("FP", 0)
    fn = p.get("FN", 0)
    tn = p.get("TN", 0)

    cm = np.array([[tp, fp], [fn, tn]])
    labels = [["TP", "FP"], ["FN", "TN"]]
    annot  = np.array([
        [f"TP\n{tp}", f"FP\n{fp}"],
        [f"FN\n{fn}", f"TN\n{tn}"],
    ])

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="RdYlGn",
        linewidths=1.5,
        linecolor="white",
        cbar=True,
        ax=ax,
        xticklabels=["Retrieval Needed", "Skip OK"],
        yticklabels=["Retrieved", "Skipped"],
        annot_kws={"size": 13, "weight": "bold"},
    )
    ax.set_title(
        f"Policy Routing Confusion Matrix\n"
        f"Accuracy={p.get('accuracy', 0):.3f}  "
        f"Precision={p.get('precision', 0):.3f}  "
        f"Recall={p.get('recall', 0):.3f}  "
        f"F1={p.get('f1', 0):.3f}",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("Ground Truth", fontsize=11)
    ax.set_ylabel("Policy Decision", fontsize=11)

    # Annotate dangerous cell (FN) with a red border
    for text in ax.texts:
        text_val = text.get_text()
        if text_val.startswith("FN"):
            text.set_color("red")

    plt.tight_layout()
    out_path = _plots_dir(save_dir) / f"confusion_matrix_{_ts()}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[plotting] Confusion matrix saved → %s", out_path)
    print(f"  📊 Confusion matrix → {out_path}")
    return out_path