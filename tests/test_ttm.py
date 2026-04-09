"""Unit tests for the causal sliding-window TTM module."""

from __future__ import annotations

import torch

from streamdycoke.config import TTMConfig
from streamdycoke.ttm import CausalSlidingTTM
from streamdycoke.utils import make_synthetic_frame, make_synthetic_stream


def test_first_frame_is_anchor_and_kept_verbatim():
    ttm = CausalSlidingTTM(TTMConfig(window_size=4, similarity_threshold=0.9))
    frame = make_synthetic_frame(16, 8, seed=1)
    out = ttm.ingest(frame)
    assert out.is_anchor is True
    assert out.num_kept == 16
    assert out.num_input == 16
    assert torch.equal(out.kept_tokens, frame)


def test_anchor_every_n_preserves_periodic_anchors():
    cfg = TTMConfig(window_size=4, similarity_threshold=0.99, anchor_every=4)
    ttm = CausalSlidingTTM(cfg)
    base = make_synthetic_frame(8, 4, seed=2)
    # Feed 8 nearly-identical frames
    for t in range(8):
        out = ttm.ingest(base.clone())
        if t % 4 == 0:
            assert out.is_anchor is True, f"frame {t} should be an anchor"


def test_identical_frames_are_aggressively_merged():
    cfg = TTMConfig(window_size=4, similarity_threshold=0.5, anchor_every=100)
    ttm = CausalSlidingTTM(cfg)
    frame = make_synthetic_frame(32, 16, seed=3)

    out0 = ttm.ingest(frame)
    assert out0.num_kept == 32  # first frame always anchored

    out1 = ttm.ingest(frame.clone())
    # Second identical frame should be merged down to (almost) nothing.
    # Our implementation guarantees at least one token survives.
    assert out1.is_anchor is False
    assert out1.num_kept >= 1
    assert out1.num_kept <= 1  # everything was redundant


def test_completely_dissimilar_frames_keep_all_tokens():
    cfg = TTMConfig(window_size=4, similarity_threshold=0.9, anchor_every=100)
    ttm = CausalSlidingTTM(cfg)
    g = torch.Generator().manual_seed(42)
    f0 = torch.randn(20, 12, generator=g)
    f1 = torch.randn(20, 12, generator=g) * 100  # very different magnitudes/dirs
    ttm.ingest(f0)
    out = ttm.ingest(f1)
    assert out.num_kept == 20  # nothing should match above threshold


def test_total_reduction_ratio_is_in_range():
    cfg = TTMConfig(window_size=4, similarity_threshold=0.95)
    ttm = CausalSlidingTTM(cfg)
    stream = make_synthetic_stream(num_frames=12, num_tokens=24, hidden_dim=16, drift=0.01)
    for f in stream:
        ttm.ingest(f)
    r = ttm.total_reduction_ratio
    assert 0.0 <= r < 1.0


def test_reset_clears_buffer_and_counters():
    ttm = CausalSlidingTTM(TTMConfig())
    f = make_synthetic_frame(8, 4, seed=5)
    ttm.ingest(f)
    ttm.ingest(f)
    assert ttm.frame_idx == 2
    ttm.reset()
    assert ttm.frame_idx == 0
    assert ttm.total_reduction_ratio == 0.0


def test_ingest_rejects_wrong_rank():
    ttm = CausalSlidingTTM(TTMConfig())
    bad = torch.randn(2, 3, 4)
    try:
        ttm.ingest(bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-2D input")
