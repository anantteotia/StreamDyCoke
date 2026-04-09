"""Benchmark harness for sweeping StreamDyCoke configurations.

This module exists so the project has working ablation infrastructure *before*
a real Video LLM is plugged in. Today it runs on synthetic streams; tomorrow
the same harness will be re-pointed at real visual encoder outputs and the
result tables will mean something.

A "trial" is one (config, stream) pair. We record per-frame state and roll up
into a small JSON-friendly summary that downstream visualization scripts and
the eventual report can consume.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Callable, Dict, List, Optional, Sequence

import torch
from torch import Tensor

from streamdycoke.config import (
    DPCacheConfig,
    EvictionPolicy,
    StreamDyCokeConfig,
    TTMConfig,
)
from streamdycoke.streaming import StreamDyCoke
from streamdycoke.utils import make_synthetic_stream


@dataclass
class FrameRecord:
    frame: int
    in_active: int
    in_dp: int
    kept_after_ttm_total: int
    input_tokens_total: int


@dataclass
class TrialResult:
    name: str
    config: Dict
    active_capacity: int
    num_frames: int
    num_tokens_per_frame: int
    hidden_dim: int
    drift: float
    seed: int
    per_frame: List[FrameRecord]
    ttm_total_reduction: float
    dp_inserts: int
    dp_evictions: int
    dp_restores: int
    dp_final_size: int
    # Policy-sensitive: what kind of tokens does this policy *retain* in DP
    # cache at end of stream? DECAY should retain higher-attention tokens;
    # FIFO should retain younger tokens; LRR should retain recently-restored
    # ones. These two metrics make the policy choice visible even though the
    # raw insert/evict counts are workload-driven.
    dp_final_mean_attention: float
    dp_final_mean_age_frames: float
    # Active cache equivalent: what's the mean importance of what we're
    # currently letting the model attend over? Higher = better.
    active_final_mean_attention: float
    wall_seconds: float

    def to_jsonable(self) -> Dict:
        d = asdict(self)
        d["per_frame"] = [asdict(r) for r in self.per_frame]
        return d


def _default_attention_generator(num_tokens: int, frame_idx: int, seed: int) -> Tensor:
    """Per-frame random attention so eviction policies actually differentiate.

    Without this, the streaming module falls back to a constant linspace, which
    means FIFO / LRR / DECAY all see identical importance signals and produce
    identical eviction counts. Real attention from a Video LLM is highly
    non-uniform, so injecting non-uniform synthetic attention here is a closer
    approximation than the constant baseline.
    """
    g = torch.Generator().manual_seed(seed * 10_000 + frame_idx)
    return torch.rand(num_tokens, generator=g)


def run_trial(
    name: str,
    config: StreamDyCokeConfig,
    *,
    active_capacity: int,
    num_frames: int = 32,
    num_tokens_per_frame: int = 64,
    hidden_dim: int = 32,
    drift: float = 0.05,
    seed: int = 0,
    attention_generator: Optional[Callable[[int, int, int], Tensor]] = None,
    refresh_every_k_frames: int = 0,
    refresh_top_k: int = 0,
) -> TrialResult:
    """Run one synthetic-stream trial and capture per-frame state."""
    sd = StreamDyCoke(config, active_capacity=active_capacity)
    stream = make_synthetic_stream(
        num_frames=num_frames,
        num_tokens=num_tokens_per_frame,
        hidden_dim=hidden_dim,
        drift=drift,
        seed=seed,
    )
    gen = attention_generator or _default_attention_generator

    records: List[FrameRecord] = []
    t0 = perf_counter()
    for t, frame in enumerate(stream):
        attn = gen(num_tokens_per_frame, t, seed)
        state = sd.ingest_frame(frame, attention_scores=attn)
        if (
            refresh_every_k_frames > 0
            and refresh_top_k > 0
            and (t + 1) % refresh_every_k_frames == 0
        ):
            sd.refresh_from_dp_cache(k=refresh_top_k)
        records.append(
            FrameRecord(
                frame=t,
                in_active=state.tokens_in_active,
                in_dp=state.tokens_in_dp_cache,
                kept_after_ttm_total=state.total_kept_after_ttm,
                input_tokens_total=state.total_input_tokens,
            )
        )
    wall = perf_counter() - t0

    # Policy-sensitive end-of-stream stats.
    if len(sd.dp_cache) > 0:
        dp_entries = list(sd.dp_cache._entries.values())
        dp_mean_attn = sum(e.base_attention for e in dp_entries) / len(dp_entries)
        clock = sd.dp_cache._clock
        dp_mean_age = sum(clock - e.inserted_at for e in dp_entries) / len(dp_entries)
    else:
        dp_mean_attn = 0.0
        dp_mean_age = 0.0
    if len(sd._active) > 0:
        active_mean_attn = sum(e.base_attention for e in sd._active) / len(sd._active)
    else:
        active_mean_attn = 0.0

    return TrialResult(
        name=name,
        config={
            "ttm": {
                "window_size": config.ttm.window_size,
                "similarity_threshold": config.ttm.similarity_threshold,
                "anchor_every": config.ttm.anchor_every,
            },
            "dp_cache": {
                "capacity": config.dp_cache.capacity,
                "eviction_policy": config.dp_cache.eviction_policy.value,
                "decay_lambda": config.dp_cache.decay_lambda,
            },
            "answer_every_k_frames": config.answer_every_k_frames,
        },
        active_capacity=active_capacity,
        num_frames=num_frames,
        num_tokens_per_frame=num_tokens_per_frame,
        hidden_dim=hidden_dim,
        drift=drift,
        seed=seed,
        per_frame=records,
        ttm_total_reduction=sd.ttm.total_reduction_ratio,
        dp_inserts=sd.dp_cache.stats["num_inserts"],
        dp_evictions=sd.dp_cache.stats["num_evictions"],
        dp_restores=sd.dp_cache.stats["num_restores"],
        dp_final_size=len(sd.dp_cache),
        dp_final_mean_attention=dp_mean_attn,
        dp_final_mean_age_frames=dp_mean_age,
        active_final_mean_attention=active_mean_attn,
        wall_seconds=wall,
    )


def sweep_eviction_policies(
    *,
    capacity: int = 64,
    active_capacity: int = 24,
    num_frames: int = 32,
    num_tokens_per_frame: int = 64,
    hidden_dim: int = 32,
    drift: float = 0.05,
    seeds: Sequence[int] = (0, 1, 2),
    refresh_every_k_frames: int = 4,
    refresh_top_k: int = 6,
) -> List[TrialResult]:
    """Run the headline ablation: FIFO vs LRR vs DECAY at fixed capacity.

    Returns one ``TrialResult`` per (policy, seed) pair.
    """
    results: List[TrialResult] = []
    for policy in (EvictionPolicy.FIFO, EvictionPolicy.LRR, EvictionPolicy.DECAY):
        for seed in seeds:
            cfg = StreamDyCokeConfig(
                ttm=TTMConfig(window_size=4, similarity_threshold=0.92, anchor_every=4),
                dp_cache=DPCacheConfig(
                    capacity=capacity,
                    eviction_policy=policy,
                    decay_lambda=0.05,
                ),
                answer_every_k_frames=4,
            )
            results.append(
                run_trial(
                    name=f"{policy.value}_seed{seed}",
                    config=cfg,
                    active_capacity=active_capacity,
                    num_frames=num_frames,
                    num_tokens_per_frame=num_tokens_per_frame,
                    hidden_dim=hidden_dim,
                    drift=drift,
                    seed=seed,
                    refresh_every_k_frames=refresh_every_k_frames,
                    refresh_top_k=refresh_top_k,
                )
            )
    return results


def summarize_by_policy(results: Sequence[TrialResult]) -> Dict[str, Dict[str, float]]:
    """Group trials by eviction policy and compute mean stats per group."""
    groups: Dict[str, List[TrialResult]] = {}
    for r in results:
        policy = r.config["dp_cache"]["eviction_policy"]
        groups.setdefault(policy, []).append(r)

    summary: Dict[str, Dict[str, float]] = {}
    for policy, trials in groups.items():
        n = float(len(trials))
        summary[policy] = {
            "n_trials": len(trials),
            "ttm_total_reduction_mean": sum(t.ttm_total_reduction for t in trials) / n,
            "dp_inserts_mean": sum(t.dp_inserts for t in trials) / n,
            "dp_evictions_mean": sum(t.dp_evictions for t in trials) / n,
            "dp_restores_mean": sum(t.dp_restores for t in trials) / n,
            "dp_final_size_mean": sum(t.dp_final_size for t in trials) / n,
            "dp_final_mean_attention_mean": sum(t.dp_final_mean_attention for t in trials) / n,
            "dp_final_mean_age_mean": sum(t.dp_final_mean_age_frames for t in trials) / n,
            "active_final_mean_attention_mean": sum(t.active_final_mean_attention for t in trials) / n,
            "wall_seconds_mean": sum(t.wall_seconds for t in trials) / n,
        }
    return summary
