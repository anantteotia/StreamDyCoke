"""Bounded Dynamic Pruning (DP) cache.

DyCoke's original DP cache stores **every** pruned KV entry so that tokens
can be restored later when attention shifts back to them. That's fine for an
offline 32-frame video but unbounded in the streaming setting: the cache
grows linearly with stream length, eventually swamping memory.

This module implements a **bounded** DP cache with three pluggable eviction
policies:

* ``FIFO``  — drop the oldest insertion.
* ``LRR``   — drop the least-recently-restored entry. The intuition is that
              if a token has been sitting in DP cache for a long time without
              ever being pulled back into the active KV cache, the model
              probably won't need it again.
* ``DECAY`` — attention-aware exponential decay. Each entry has a priority
              equal to the attention score it earned at insertion time,
              decayed by ``exp(-lambda * dt)`` since its last access (insert
              or restore). Lowest-priority entry is evicted.

The cache stores opaque ``key`` and ``value`` tensors plus per-entry
metadata. It's intentionally not coupled to any specific transformer
implementation: the streaming loop is responsible for slicing the right
``(K, V)`` slabs out of an attention layer and handing them in.

All operations are O(N) in cache size for the DECAY policy and O(1) amortized
for FIFO/LRR. For the cache sizes we care about (a few thousand entries) this
is fast enough on CPU; if it ever becomes a bottleneck the priority queue can
be swapped in transparently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import math

import torch
from torch import Tensor

from streamdycoke.config import DPCacheConfig, EvictionPolicy


@dataclass
class CacheEntry:
    """One stored token in the DP cache.

    Attributes
    ----------
    token_id:
        Globally unique id assigned by the cache at insertion time. The
        streaming loop uses this to refer to entries on later restores.
    key, value:
        Per-token KV slabs. Shapes are caller-defined; we just store them.
        Typically ``[num_layers, num_heads, head_dim]`` for a single token.
    base_attention:
        Attention score (or any importance signal) at insertion time. Used by
        the DECAY eviction policy.
    inserted_at:
        Logical time (e.g. decoding step) at which this token was first
        moved into the DP cache.
    last_access:
        Logical time of the most recent access (insertion or restore). Used
        by both LRR and DECAY policies.
    """

    token_id: int
    key: Tensor
    value: Tensor
    base_attention: float
    inserted_at: int
    last_access: int
    metadata: Dict[str, object] = field(default_factory=dict)


class BoundedDPCache:
    """Bounded version of DyCoke's Dynamic Pruning cache.

    The cache supports three operations:

    * ``insert(...)``      — move a token from the active KV cache into the DP
                             cache. Triggers eviction if at capacity.
    * ``restore(token_id)`` — pull an entry back out for re-insertion into the
                              active KV cache. Updates ``last_access``.
    * ``advance_time()``    — bump the logical clock used by decay/LRR. Called
                              once per decoding step by the streaming loop.

    The cache is intentionally **not** a torch ``nn.Module``; it holds plain
    tensors so it can be tested without any model state and serialized cheaply.
    """

    def __init__(self, config: DPCacheConfig) -> None:
        self.config = config
        self._entries: Dict[int, CacheEntry] = {}
        self._next_id: int = 0
        self._clock: int = 0
        # Stats
        self._num_inserts: int = 0
        self._num_restores: int = 0
        self._num_evictions: int = 0

    # ---------------------------------------------------------------- queries

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def is_full(self) -> bool:
        return len(self._entries) >= self.config.capacity

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "num_inserts": self._num_inserts,
            "num_restores": self._num_restores,
            "num_evictions": self._num_evictions,
            "size": len(self._entries),
            "clock": self._clock,
        }

    def contains(self, token_id: int) -> bool:
        return token_id in self._entries

    # ----------------------------------------------------------------- writes

    def advance_time(self, steps: int = 1) -> None:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        self._clock += steps

    def insert(
        self,
        key: Tensor,
        value: Tensor,
        base_attention: float,
        metadata: Optional[Dict[str, object]] = None,
    ) -> int:
        """Insert one token into the cache.

        Returns the assigned ``token_id`` so the caller can later restore it.
        Triggers a single eviction if the cache is at capacity.
        """
        if self.is_full:
            self._evict_one()

        token_id = self._next_id
        self._next_id += 1
        self._entries[token_id] = CacheEntry(
            token_id=token_id,
            key=key,
            value=value,
            base_attention=float(base_attention),
            inserted_at=self._clock,
            last_access=self._clock,
            metadata=dict(metadata or {}),
        )
        self._num_inserts += 1
        return token_id

    def restore(self, token_id: int) -> CacheEntry:
        """Pull an entry out of the DP cache.

        After restore, the entry is **removed** from the DP cache (the streaming
        loop is expected to put it back into the active KV cache). This is
        consistent with DyCoke's semantics.
        """
        if token_id not in self._entries:
            raise KeyError(f"token_id {token_id} not in DP cache")
        entry = self._entries.pop(token_id)
        entry.last_access = self._clock
        self._num_restores += 1
        return entry

    def restore_top_k(self, k: int) -> List[CacheEntry]:
        """Restore the k highest-priority entries (by current decay score).

        Useful for the streaming loop's "every M steps, refresh the active KV
        cache" pass. Order is highest-priority first.
        """
        if k <= 0 or len(self._entries) == 0:
            return []
        scored = sorted(
            self._entries.values(),
            key=lambda e: self._priority(e),
            reverse=True,
        )[:k]
        out = []
        for e in scored:
            out.append(self.restore(e.token_id))
        return out

    # --------------------------------------------------------------- eviction

    def _evict_one(self) -> None:
        if not self._entries:
            return
        policy = self.config.eviction_policy
        if policy == EvictionPolicy.FIFO:
            victim = min(self._entries.values(), key=lambda e: e.inserted_at)
        elif policy == EvictionPolicy.LRR:
            victim = min(self._entries.values(), key=lambda e: e.last_access)
        elif policy == EvictionPolicy.DECAY:
            victim = min(self._entries.values(), key=lambda e: self._priority(e))
        else:
            raise ValueError(f"unknown eviction policy: {policy}")
        del self._entries[victim.token_id]
        self._num_evictions += 1

    def _priority(self, entry: CacheEntry) -> float:
        """Decayed priority used by the DECAY policy.

        priority = base_attention * exp(-lambda * (clock - last_access))
        """
        dt = max(0, self._clock - entry.last_access)
        return entry.base_attention * math.exp(-self.config.decay_lambda * dt)

    # ------------------------------------------------------------------- misc

    def clear(self) -> None:
        self._entries.clear()
        self._clock = 0
        self._next_id = 0
        self._num_inserts = 0
        self._num_restores = 0
        self._num_evictions = 0

    def snapshot(self) -> List[Tuple[int, float, int, int]]:
        """Lightweight snapshot for tests / debugging.

        Returns a list of ``(token_id, base_attention, inserted_at, last_access)``
        tuples sorted by token_id.
        """
        return sorted(
            (e.token_id, e.base_attention, e.inserted_at, e.last_access)
            for e in self._entries.values()
        )
