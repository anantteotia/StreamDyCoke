"""Streaming inference loop with anytime answering.

This module wires together :class:`CausalSlidingTTM` and
:class:`BoundedDPCache` into a single object, :class:`StreamDyCoke`, that the
end-to-end runner (or any unit test) can drive frame-by-frame.

The class is deliberately model-agnostic: it does **not** know about Llama,
LLaVA, or any specific transformer. The runner is expected to inject a
``model_callable`` that, given the current active KV state and a query,
produces an answer. This separation lets us:

1. Develop and test the streaming logic on synthetic tensors with no GPU.
2. Plug in a real Video LLM later (LLaVA-OneVision-7B is the planned target)
   without touching the streaming logic.

The active KV state is represented as a plain Python list of ``CacheEntry``
objects. The streaming loop tracks an integer "active capacity" — when more
than that many tokens are in the active list, the lowest-attention ones get
demoted to the DP cache. When attention shifts back, the DP cache is asked to
restore its top-K entries into the active list. This is a faithful, minimal
implementation of DyCoke's two-cache dynamic, adapted for streaming.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch
from torch import Tensor

from streamdycoke.config import StreamDyCokeConfig
from streamdycoke.dp_cache import BoundedDPCache, CacheEntry
from streamdycoke.ttm import CausalSlidingTTM, TTMResult


# A "model callable" takes the active token list and an optional question
# string and returns an answer string. Tests use a deterministic stub; the
# real integration will wrap a HuggingFace generate() call.
ModelCallable = Callable[[List[CacheEntry], Optional[str]], str]


@dataclass
class StreamingState:
    """Public, read-only view of the streaming pipeline's current state."""

    frames_seen: int
    tokens_in_active: int
    tokens_in_dp_cache: int
    total_input_tokens: int
    total_kept_after_ttm: int
    answers: List[str] = field(default_factory=list)


class StreamDyCoke:
    """End-to-end streaming wrapper around TTM + DP cache.

    Parameters
    ----------
    config:
        Top-level :class:`StreamDyCokeConfig`.
    active_capacity:
        Maximum number of tokens kept in the *active* KV cache. Tokens beyond
        this budget are demoted into the DP cache. The DP cache itself is
        bounded separately by ``config.dp_cache.capacity``.
    model_callable:
        Optional function that produces an answer string given the current
        active token list and a question. If ``None``, anytime-answering is
        disabled (useful for unit tests that only exercise the cache logic).
    """

    def __init__(
        self,
        config: StreamDyCokeConfig,
        active_capacity: int,
        model_callable: Optional[ModelCallable] = None,
    ) -> None:
        if active_capacity < 1:
            raise ValueError("active_capacity must be >= 1")
        self.config = config
        self.active_capacity = active_capacity
        self.model_callable = model_callable

        self.ttm = CausalSlidingTTM(config.ttm)
        self.dp_cache = BoundedDPCache(config.dp_cache)

        self._active: List[CacheEntry] = []
        self._frames_seen: int = 0
        self._answers: List[str] = []
        # Used to assign deterministic per-token base_attention scores when no
        # real attention signal is available (tests).
        self._synthetic_attention_seed: int = 0

    # --------------------------------------------------------------- queries

    def state(self) -> StreamingState:
        return StreamingState(
            frames_seen=self._frames_seen,
            tokens_in_active=len(self._active),
            tokens_in_dp_cache=len(self.dp_cache),
            total_input_tokens=self.ttm._total_input_tokens,
            total_kept_after_ttm=self.ttm._total_kept_tokens,
            answers=list(self._answers),
        )

    @property
    def answers(self) -> List[str]:
        return list(self._answers)

    # ------------------------------------------------------------ ingestion

    def ingest_frame(
        self,
        frame_tokens: Tensor,
        attention_scores: Optional[Tensor] = None,
        question: Optional[str] = None,
    ) -> StreamingState:
        """Push one frame through the pipeline.

        Parameters
        ----------
        frame_tokens:
            ``[num_tokens, hidden_dim]`` tensor of visual tokens for this
            frame, straight from the visual encoder.
        attention_scores:
            Optional ``[num_tokens]`` tensor of per-token importance scores.
            If absent, a synthetic deterministic score is used so the cache
            still has something to rank by. The real runner will pass actual
            shallow-layer attention scores (DyCoke uses layer 3).
        question:
            Optional question string. If set and the current frame is an
            answer-trigger frame (every ``answer_every_k_frames``), the
            ``model_callable`` is invoked and its answer is appended to the
            running ``answers`` list.
        """
        ttm_result = self.ttm.ingest(frame_tokens)
        self._frames_seen += 1

        # Build CacheEntry-like objects for the surviving tokens. Each token
        # gets a synthetic per-layer-per-head KV slab if no real one is
        # provided; this keeps the streaming logic exercisable in tests.
        entries = self._wrap_tokens_as_entries(
            ttm_result, attention_scores=attention_scores
        )

        # Append to the active KV state.
        self._active.extend(entries)

        # Bump the DP cache clock once per ingested frame so decay/LRR have
        # something to work with.
        self.dp_cache.advance_time(steps=1)

        # If we're over the active capacity, demote the lowest-attention
        # tokens to the DP cache. This is the "spatial pruning" half of
        # DyCoke, but bounded.
        self._rebalance()

        # Anytime answering hook.
        if (
            question is not None
            and self.model_callable is not None
            and (self._frames_seen % self.config.answer_every_k_frames == 0)
        ):
            answer = self.model_callable(list(self._active), question)
            self._answers.append(answer)

        return self.state()

    # ------------------------------------------------------------ rebalance

    def _rebalance(self) -> None:
        """Move overflow tokens from the active list into the DP cache.

        We pick the **lowest** base_attention tokens to demote. The DP cache
        will then choose its own victim if it's already full.
        """
        overflow = len(self._active) - self.active_capacity
        if overflow <= 0:
            return

        # Sort indices by attention ascending and pop the lowest `overflow`.
        ranked = sorted(
            range(len(self._active)),
            key=lambda i: self._active[i].base_attention,
        )
        to_demote = sorted(ranked[:overflow], reverse=True)
        for idx in to_demote:
            entry = self._active.pop(idx)
            self.dp_cache.insert(
                key=entry.key,
                value=entry.value,
                base_attention=entry.base_attention,
                metadata=entry.metadata,
            )

    def refresh_from_dp_cache(self, k: int) -> int:
        """Restore the top-k DP cache entries back into the active list.

        Returns the number of entries actually restored. The streaming runner
        is expected to call this periodically (or when it detects an attention
        shift). The unit tests call it directly.
        """
        restored = self.dp_cache.restore_top_k(k)
        self._active.extend(restored)
        # Active list might now be over capacity again — rebalance.
        self._rebalance()
        return len(restored)

    def reset(self) -> None:
        self.ttm.reset()
        self.dp_cache.clear()
        self._active.clear()
        self._frames_seen = 0
        self._answers.clear()
        self._synthetic_attention_seed = 0

    # --------------------------------------------------------------- helpers

    def _wrap_tokens_as_entries(
        self,
        ttm_result: TTMResult,
        attention_scores: Optional[Tensor],
    ) -> List[CacheEntry]:
        """Turn the TTM-survived tokens into CacheEntry objects.

        Each entry stores the token vector itself in both ``key`` and
        ``value`` slots. When the real Video LLM is plugged in, the runner
        will replace this with the actual KV slabs from the attention layer.
        """
        entries: List[CacheEntry] = []
        kept = ttm_result.kept_tokens
        n = kept.shape[0]
        if attention_scores is None:
            # Deterministic synthetic scores so tests are reproducible.
            attn = torch.linspace(0.1, 1.0, steps=max(n, 1))
            self._synthetic_attention_seed += 1
        else:
            if attention_scores.shape[0] != ttm_result.num_input:
                raise ValueError(
                    "attention_scores must have one entry per *input* token"
                )
            attn = attention_scores.index_select(0, ttm_result.kept_indices)

        clock = self.dp_cache._clock  # OK to read; the cache is ours.
        for i in range(n):
            tok = kept[i]
            entries.append(
                CacheEntry(
                    token_id=-1,  # not in DP cache yet
                    key=tok.detach().clone(),
                    value=tok.detach().clone(),
                    base_attention=float(attn[i].item()),
                    inserted_at=clock,
                    last_access=clock,
                    metadata={"frame_idx": ttm_result.num_input and self._frames_seen - 1},
                )
            )
        return entries
