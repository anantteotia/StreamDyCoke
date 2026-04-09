"""Causal sliding-window Temporal Token Merging.

DyCoke's original TTM uses a symmetric 4-frame window centered on the current
frame, which means it cannot run until ``window_size // 2`` future frames have
arrived. That's a deal-breaker for streaming.

This module replaces it with a strictly **causal** sliding window: each
incoming frame is compared only against the most recent ``window_size - 1``
buffered frames. Tokens at a given spatial position whose maximum cosine
similarity to any past frame's token at that position exceeds
``similarity_threshold`` are dropped (merged into the older anchor token).
The first frame in each window of ``anchor_every`` is always preserved.

Assumptions
-----------
- Each frame is encoded into a fixed number of visual tokens of dimension
  ``hidden_dim``. This matches LLaVA-OneVision / CLIP-style encoders that
  produce a regular spatial grid of patch tokens per frame (e.g. 14x14 = 196).
- Spatial positions are aligned across frames: token ``i`` in frame ``t``
  corresponds to the same patch as token ``i`` in frame ``t-1``. This is the
  same assumption DyCoke makes.
- The module is **stateful** across ``ingest`` calls and is intended to be
  driven by the streaming loop one frame at a time.

The module is pure PyTorch and runs on CPU. No LLM is required to test it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import torch
from torch import Tensor

from streamdycoke.config import TTMConfig


@dataclass
class TTMResult:
    """Output of a single ``ingest`` call.

    Attributes
    ----------
    kept_tokens:
        ``[num_kept, hidden_dim]`` tensor of tokens that survived merging for
        the just-ingested frame. ``num_kept`` is between 0 and the original
        token count.
    kept_indices:
        ``[num_kept]`` long tensor mapping each kept token back to its
        original spatial index in the frame. Useful for downstream attention
        bookkeeping.
    is_anchor:
        Whether the just-ingested frame was treated as an anchor frame.
        Anchor frames are never merged.
    num_input:
        Original number of tokens in the frame before merging.
    """

    kept_tokens: Tensor
    kept_indices: Tensor
    is_anchor: bool
    num_input: int

    @property
    def num_kept(self) -> int:
        return int(self.kept_tokens.shape[0])

    @property
    def reduction_ratio(self) -> float:
        if self.num_input == 0:
            return 0.0
        return 1.0 - (self.num_kept / self.num_input)


class CausalSlidingTTM:
    """Stateful, causal version of DyCoke's Temporal Token Merging.

    Usage
    -----
    >>> ttm = CausalSlidingTTM(TTMConfig(window_size=4, similarity_threshold=0.9))
    >>> for frame_tokens in stream:                  # [num_tokens, hidden_dim]
    ...     result = ttm.ingest(frame_tokens)
    ...     downstream(result.kept_tokens)
    """

    def __init__(self, config: TTMConfig) -> None:
        self.config = config
        # Buffer of (frame_idx, full_frame_tokens) for the past window-1 frames.
        # We always keep the *full* frame in the buffer (not the post-merge
        # version) so that future similarity comparisons stay calibrated to the
        # original encoder output. Memory cost is O(window_size * num_tokens).
        self._buffer: Deque[Tensor] = deque(maxlen=max(1, config.window_size - 1))
        self._frame_idx: int = 0
        # Running stats for diagnostics / unit tests.
        self._total_input_tokens: int = 0
        self._total_kept_tokens: int = 0

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        """Clear all buffered frames and counters. Call between videos."""
        self._buffer.clear()
        self._frame_idx = 0
        self._total_input_tokens = 0
        self._total_kept_tokens = 0

    @property
    def frame_idx(self) -> int:
        return self._frame_idx

    @property
    def total_reduction_ratio(self) -> float:
        if self._total_input_tokens == 0:
            return 0.0
        return 1.0 - (self._total_kept_tokens / self._total_input_tokens)

    # ----------------------------------------------------------------- ingest

    def ingest(self, frame_tokens: Tensor) -> TTMResult:
        """Process one new frame's tokens and return the surviving tokens.

        Parameters
        ----------
        frame_tokens:
            ``[num_tokens, hidden_dim]`` float tensor. The same number of
            tokens is expected on every call (positions are aligned across
            frames).
        """
        if frame_tokens.dim() != 2:
            raise ValueError(
                f"frame_tokens must be 2D [num_tokens, hidden_dim], "
                f"got shape {tuple(frame_tokens.shape)}"
            )

        num_tokens = frame_tokens.shape[0]
        is_anchor = (self._frame_idx % self.config.anchor_every) == 0
        # Anchor frames are kept verbatim. We also short-circuit when there is
        # no buffered history yet (the very first frame of the stream).
        if is_anchor or len(self._buffer) == 0:
            kept_idx = torch.arange(num_tokens, dtype=torch.long)
            self._push_to_buffer(frame_tokens)
            self._frame_idx += 1
            self._total_input_tokens += num_tokens
            self._total_kept_tokens += num_tokens
            return TTMResult(
                kept_tokens=frame_tokens,
                kept_indices=kept_idx,
                is_anchor=True,
                num_input=num_tokens,
            )

        # Compute per-position cosine similarity against every buffered frame.
        # We take the *max* similarity across the buffer: a token is redundant
        # if at least one recent frame already explains it.
        max_sim = self._max_position_similarity(frame_tokens)  # [num_tokens]

        keep_mask = max_sim < self.config.similarity_threshold
        # Always keep at least one token to avoid pathological empty outputs.
        if not keep_mask.any():
            # Keep the single token with the lowest similarity (i.e. the most
            # informative novel one).
            argmin = int(torch.argmin(max_sim).item())
            keep_mask = torch.zeros_like(keep_mask)
            keep_mask[argmin] = True

        kept_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
        kept = frame_tokens.index_select(0, kept_idx)

        self._push_to_buffer(frame_tokens)
        self._frame_idx += 1
        self._total_input_tokens += num_tokens
        self._total_kept_tokens += int(kept.shape[0])

        return TTMResult(
            kept_tokens=kept,
            kept_indices=kept_idx,
            is_anchor=False,
            num_input=num_tokens,
        )

    # --------------------------------------------------------------- internals

    def _push_to_buffer(self, frame_tokens: Tensor) -> None:
        # Detach + clone so the buffer is independent of any autograd graph
        # held by the caller. Tests rely on this.
        self._buffer.append(frame_tokens.detach().clone())

    def _max_position_similarity(self, frame_tokens: Tensor) -> Tensor:
        """For each spatial position, return the max cosine similarity to any
        buffered frame's token at the same position.

        Returns
        -------
        ``[num_tokens]`` float tensor in [-1, 1].
        """
        cur = torch.nn.functional.normalize(frame_tokens, dim=-1, eps=1e-8)
        max_sim: Optional[Tensor] = None
        for past in self._buffer:
            if past.shape != frame_tokens.shape:
                # Spatial layout changed mid-stream. We treat this as a hard
                # reset of similarity (no merging possible against this frame).
                continue
            past_n = torch.nn.functional.normalize(past, dim=-1, eps=1e-8)
            sim = (cur * past_n).sum(dim=-1)  # [num_tokens]
            if max_sim is None:
                max_sim = sim
            else:
                max_sim = torch.maximum(max_sim, sim)
        if max_sim is None:
            return torch.full((frame_tokens.shape[0],), -1.0)
        return max_sim
