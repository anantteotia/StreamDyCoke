"""Plot helpers for benchmark results.

Pulled into its own module so the rest of the package has zero matplotlib
dependency. Import this only from scripts that actually need to render.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # safe for headless / CI environments
import matplotlib.pyplot as plt  # noqa: E402

from streamdycoke.benchmark import TrialResult  # noqa: E402


def plot_cache_occupancy(results: Sequence[TrialResult], out_path: Path) -> None:
    """One subplot per (policy x seed) trial showing active vs DP cache size.

    Useful for spotting whether a policy chronically saturates the DP cache or
    keeps it healthy.
    """
    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.6 * rows), sharex=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, r in zip(axes, results):
        frames = [rec.frame for rec in r.per_frame]
        active = [rec.in_active for rec in r.per_frame]
        dp = [rec.in_dp for rec in r.per_frame]
        ax.plot(frames, active, label="active", linewidth=1.6)
        ax.plot(frames, dp, label="dp cache", linewidth=1.6, linestyle="--")
        ax.axhline(r.active_capacity, color="grey", linewidth=0.8, alpha=0.5)
        ax.set_title(r.name, fontsize=9)
        ax.set_xlabel("frame")
        ax.set_ylabel("tokens")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")

    for ax in axes[len(results):]:
        ax.set_visible(False)

    fig.suptitle("StreamDyCoke cache occupancy over time (synthetic stream)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_policy_summary(summary: dict, out_path: Path) -> None:
    """Bar chart comparing aggregate stats per eviction policy."""
    policies = list(summary.keys())
    metrics = ["dp_inserts_mean", "dp_evictions_mean", "dp_final_size_mean"]
    pretty = {
        "dp_inserts_mean": "DP inserts",
        "dp_evictions_mean": "DP evictions",
        "dp_final_size_mean": "DP final size",
    }

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    width = 0.25
    x = list(range(len(policies)))
    for i, m in enumerate(metrics):
        vals = [summary[p][m] for p in policies]
        offset = (i - 1) * width
        ax.bar([xi + offset for xi in x], vals, width=width, label=pretty[m])
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.set_ylabel("count (mean over seeds)")
    ax.set_title("Eviction policy comparison on synthetic stream")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
