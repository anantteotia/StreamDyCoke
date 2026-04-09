"""Unit tests for the end-to-end streaming pipeline."""

from __future__ import annotations

from typing import List, Optional

import torch

from streamdycoke.config import (
    DPCacheConfig,
    EvictionPolicy,
    StreamDyCokeConfig,
    TTMConfig,
)
from streamdycoke.dp_cache import CacheEntry
from streamdycoke.streaming import StreamDyCoke
from streamdycoke.utils import make_synthetic_stream


def _make(active_capacity: int = 32, k: int = 4) -> StreamDyCoke:
    cfg = StreamDyCokeConfig(
        ttm=TTMConfig(window_size=4, similarity_threshold=0.9, anchor_every=4),
        dp_cache=DPCacheConfig(capacity=64, eviction_policy=EvictionPolicy.DECAY),
        answer_every_k_frames=k,
    )
    return StreamDyCoke(cfg, active_capacity=active_capacity)


def test_ingest_single_frame_populates_active_cache():
    sd = _make(active_capacity=32)
    stream = make_synthetic_stream(1, num_tokens=16, hidden_dim=8)
    state = sd.ingest_frame(stream[0])
    assert state.frames_seen == 1
    assert state.tokens_in_active == 16
    assert state.tokens_in_dp_cache == 0


def test_active_capacity_overflow_demotes_to_dp_cache():
    sd = _make(active_capacity=8)  # tiny budget
    stream = make_synthetic_stream(2, num_tokens=16, hidden_dim=8, drift=5.0)
    sd.ingest_frame(stream[0])
    state = sd.ingest_frame(stream[1])
    # After the second frame, total tokens >> 8, so DP cache must have entries.
    assert state.tokens_in_active <= 8
    assert state.tokens_in_dp_cache > 0


def test_refresh_pulls_top_k_back_into_active():
    sd = _make(active_capacity=4)
    stream = make_synthetic_stream(3, num_tokens=8, hidden_dim=4, drift=5.0)
    for f in stream:
        sd.ingest_frame(f)
    pre_active = sd.state().tokens_in_active
    pre_dp = sd.state().tokens_in_dp_cache
    restored = sd.refresh_from_dp_cache(k=2)
    assert restored == min(2, pre_dp)
    # Active capacity is enforced after refresh, so total should still be <= cap.
    assert sd.state().tokens_in_active <= 4


def test_anytime_answering_invokes_callback_on_schedule():
    answers_seen: List[str] = []

    def fake_model(active: List[CacheEntry], q: Optional[str]) -> str:
        return f"answer@{len(active)}tokens:{q}"

    cfg = StreamDyCokeConfig(
        ttm=TTMConfig(window_size=2, similarity_threshold=1.0, anchor_every=1),
        dp_cache=DPCacheConfig(capacity=64),
        answer_every_k_frames=2,
    )
    sd = StreamDyCoke(cfg, active_capacity=64, model_callable=fake_model)

    stream = make_synthetic_stream(5, num_tokens=4, hidden_dim=4, drift=5.0)
    for f in stream:
        sd.ingest_frame(f, question="what's happening?")
        answers_seen = sd.answers

    # 5 frames, answering every 2 -> answers at frames 2 and 4 -> 2 answers.
    assert len(answers_seen) == 2
    assert all("what's happening?" in a for a in answers_seen)


def test_reset_clears_everything():
    sd = _make()
    stream = make_synthetic_stream(3, num_tokens=8, hidden_dim=4)
    for f in stream:
        sd.ingest_frame(f)
    sd.reset()
    s = sd.state()
    assert s.frames_seen == 0
    assert s.tokens_in_active == 0
    assert s.tokens_in_dp_cache == 0
    assert s.total_input_tokens == 0


def test_external_attention_scores_are_used():
    sd = _make(active_capacity=2)
    f = torch.randn(4, 8)
    # Token 2 has the highest attention -> should be the one that survives
    # the active-capacity squeeze.
    attn = torch.tensor([0.1, 0.2, 0.9, 0.3])
    sd.ingest_frame(f, attention_scores=attn)
    # active_capacity=2, so 2 tokens should be in active and 2 in DP cache.
    s = sd.state()
    assert s.tokens_in_active == 2
    assert s.tokens_in_dp_cache == 2
    # Highest-attention token should be in active.
    active_attns = sorted(e.base_attention for e in sd._active)
    assert abs(active_attns[-1] - 0.9) < 1e-5
