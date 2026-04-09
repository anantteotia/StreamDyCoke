"""Run the eviction-policy ablation on synthetic streams and dump results.

Outputs:
  experiments/synthetic/results.json   - per-trial results, JSON-friendly
  experiments/synthetic/summary.json   - mean stats grouped by policy
  experiments/synthetic/cache_occupancy.png
  experiments/synthetic/policy_summary.png

Run:
  python -m scripts.run_synthetic_benchmark
"""

from __future__ import annotations

import json
from pathlib import Path

from streamdycoke.benchmark import sweep_eviction_policies, summarize_by_policy
from streamdycoke.viz import plot_cache_occupancy, plot_policy_summary


OUTDIR = Path("experiments/synthetic")


def main() -> None:
    print("Running eviction-policy sweep on synthetic streams...")
    results = sweep_eviction_policies(
        capacity=64,
        active_capacity=24,
        num_frames=32,
        num_tokens_per_frame=64,
        hidden_dim=32,
        drift=0.05,
        seeds=(0, 1, 2),
    )
    summary = summarize_by_policy(results)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "results.json").write_text(
        json.dumps([r.to_jsonable() for r in results], indent=2)
    )
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))

    plot_cache_occupancy(results, OUTDIR / "cache_occupancy.png")
    plot_policy_summary(summary, OUTDIR / "policy_summary.png")

    print()
    print(f"Wrote {len(results)} trials -> {OUTDIR / 'results.json'}")
    print(f"Wrote summary           -> {OUTDIR / 'summary.json'}")
    print(f"Wrote cache occupancy   -> {OUTDIR / 'cache_occupancy.png'}")
    print(f"Wrote policy summary    -> {OUTDIR / 'policy_summary.png'}")
    print()
    print("Per-policy summary (means over 3 seeds):")
    for policy, stats in summary.items():
        print(f"  {policy:>6}: " + ", ".join(f"{k}={v:.2f}" for k, v in stats.items()))


if __name__ == "__main__":
    main()
