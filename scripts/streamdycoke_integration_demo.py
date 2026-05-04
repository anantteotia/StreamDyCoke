"""
Integration-oriented demo (CPU-friendly).

This script does NOT fully hook StreamDyCoke into HuggingFace decoding internals.
Instead it demonstrates the intended wiring pattern:

  synthetic frame embeddings -> causal TTM -> bounded DP cache policy loop -> anytime hooks

This is the stable bridge artifact while real Video LLM integration requires extracting
real attention scores from model internals (layer hooks), which is hardware and model-version sensitive.

Usage:
  python scripts/streamdycoke_integration_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from streamdycoke.config import DPCacheConfig, EvictionPolicy, StreamDyCokeConfig, TTMConfig
from streamdycoke.streaming import StreamDyCoke


def main() -> None:
    torch.manual_seed(0)

    hidden = 32
    tokens_per_frame = 64
    num_frames = 16

    frames = [torch.randn(tokens_per_frame, hidden) for _ in range(num_frames)]

    cfg = StreamDyCokeConfig(
        ttm=TTMConfig(window_size=4, similarity_threshold=0.92, anchor_every=4),
        dp_cache=DPCacheConfig(
            capacity=64,
            eviction_policy=EvictionPolicy.DECAY,
            decay_lambda=0.05,
        ),
        answer_every_k_frames=4,
    )

    answers: list[str] = []

    def fake_model(active, question):
        return f"frame_answer(tokens={len(active)}, q={question!r})"

    sd = StreamDyCoke(cfg, active_capacity=24, model_callable=fake_model)

    for i, emb in enumerate(frames):
        fake_attn = torch.rand(tokens_per_frame)
        sd.ingest_frame(emb, attention_scores=fake_attn, question="What changed recently?")
        if i % 3 == 0:
            sd.refresh_from_dp_cache(k=6)

    answers = sd.answers
    state = sd.state()

    print("OK: synthetic integration demo finished.")
    print(f"Frames seen: {state.frames_seen}")
    print(f"Active tokens: {state.tokens_in_active}")
    print(f"DP cache tokens: {state.tokens_in_dp_cache}")
    print(f"Anytime answers collected: {len(answers)}")
    print("Note: replace fake_attn with real attention from the Video LLM when integrating.")


if __name__ == "__main__":
    main()
