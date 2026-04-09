"""Small utilities used by the streaming pipeline and the tests."""

from __future__ import annotations

from typing import Iterable, List

import torch
from torch import Tensor


def make_synthetic_frame(
    num_tokens: int,
    hidden_dim: int,
    *,
    seed: int = 0,
    base: Tensor | None = None,
    drift: float = 0.0,
) -> Tensor:
    """Generate a deterministic synthetic frame for tests.

    Parameters
    ----------
    num_tokens, hidden_dim:
        Output shape ``[num_tokens, hidden_dim]``.
    seed:
        RNG seed. Same seed -> same tensor.
    base:
        Optional base frame. If given, the output is ``base + drift * noise``,
        which is useful for simulating slowly changing video where consecutive
        frames are highly similar.
    drift:
        Magnitude of the perturbation when ``base`` is provided.
    """
    g = torch.Generator().manual_seed(seed)
    if base is None:
        return torch.randn(num_tokens, hidden_dim, generator=g)
    if base.shape != (num_tokens, hidden_dim):
        raise ValueError("base shape must match (num_tokens, hidden_dim)")
    noise = torch.randn(num_tokens, hidden_dim, generator=g)
    return base + drift * noise


def make_synthetic_stream(
    num_frames: int,
    num_tokens: int,
    hidden_dim: int,
    *,
    drift: float = 0.05,
    seed: int = 0,
) -> List[Tensor]:
    """Generate a list of ``num_frames`` slowly drifting frames.

    Designed so that consecutive frames are highly cosine-similar (so TTM
    actually does merging) without being literally identical.
    """
    base = make_synthetic_frame(num_tokens, hidden_dim, seed=seed)
    out: List[Tensor] = [base]
    for t in range(1, num_frames):
        out.append(make_synthetic_frame(num_tokens, hidden_dim, seed=seed + t, base=base, drift=drift))
    return out


def cosine_sim_matrix(a: Tensor, b: Tensor) -> Tensor:
    """Pairwise cosine similarity between rows of ``a`` and ``b``.

    Returns ``[a.shape[0], b.shape[0]]``.
    """
    a_n = torch.nn.functional.normalize(a, dim=-1, eps=1e-8)
    b_n = torch.nn.functional.normalize(b, dim=-1, eps=1e-8)
    return a_n @ b_n.T
