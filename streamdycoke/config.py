"""Configuration dataclasses for StreamDyCoke.

Keeping configs as plain dataclasses (not pydantic / hydra) so the algorithm
modules stay framework-agnostic and easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EvictionPolicy(str, Enum):
    """Eviction policy for the bounded Dynamic Pruning cache.

    FIFO   - drop the oldest token (by insertion time).
    LRR    - drop the least-recently-restored token. A token that has not been
             pulled back into the active KV cache for a long time is the first
             to go.
    DECAY  - attention-aware exponential decay. Each token's priority is the
             attention score it earned at insertion time, decayed by
             ``exp(-lambda * dt)`` since the last access (insertion or restore).
             Lowest priority is evicted.
    """

    FIFO = "fifo"
    LRR = "lrr"
    DECAY = "decay"


@dataclass
class TTMConfig:
    """Causal sliding-window Temporal Token Merging.

    Attributes
    ----------
    window_size:
        Number of frames in the causal window. The current frame plus
        ``window_size - 1`` past frames are considered.
    similarity_threshold:
        Cosine similarity above which a token is considered redundant with the
        best-matching past token at the same spatial position and is therefore
        merged away.
    anchor_every:
        Every Nth frame is preserved as a temporal anchor regardless of
        similarity. ``1`` means every frame is an anchor (no merging across
        frame boundaries), ``window_size`` means only the first frame in each
        window is anchored. DyCoke uses 4 with window_size=4.
    """

    window_size: int = 4
    similarity_threshold: float = 0.9
    anchor_every: int = 4

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [0, 1]")
        if self.anchor_every < 1:
            raise ValueError("anchor_every must be >= 1")


@dataclass
class DPCacheConfig:
    """Bounded Dynamic Pruning cache."""

    capacity: int = 2048
    eviction_policy: EvictionPolicy = EvictionPolicy.DECAY
    decay_lambda: float = 0.05  # only used by DECAY policy

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must be >= 1")
        if self.decay_lambda < 0:
            raise ValueError("decay_lambda must be >= 0")


@dataclass
class StreamDyCokeConfig:
    """Top-level config bundling TTM and DP cache settings."""

    ttm: TTMConfig = None  # type: ignore[assignment]
    dp_cache: DPCacheConfig = None  # type: ignore[assignment]
    answer_every_k_frames: int = 8

    def __post_init__(self) -> None:
        if self.ttm is None:
            self.ttm = TTMConfig()
        if self.dp_cache is None:
            self.dp_cache = DPCacheConfig()
        if self.answer_every_k_frames < 1:
            raise ValueError("answer_every_k_frames must be >= 1")
