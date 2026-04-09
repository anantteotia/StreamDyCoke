"""Unit tests for the bounded DP cache."""

from __future__ import annotations

import math

import torch

from streamdycoke.config import DPCacheConfig, EvictionPolicy
from streamdycoke.dp_cache import BoundedDPCache


def _kv():
    return torch.zeros(2), torch.zeros(2)


def test_insert_and_restore_roundtrip():
    cache = BoundedDPCache(DPCacheConfig(capacity=4))
    k, v = _kv()
    tid = cache.insert(k, v, base_attention=0.5)
    assert len(cache) == 1
    e = cache.restore(tid)
    assert e.token_id == tid
    assert len(cache) == 0
    assert cache.stats["num_inserts"] == 1
    assert cache.stats["num_restores"] == 1


def test_fifo_evicts_oldest():
    cache = BoundedDPCache(DPCacheConfig(capacity=3, eviction_policy=EvictionPolicy.FIFO))
    ids = []
    for i in range(3):
        cache.advance_time()
        ids.append(cache.insert(*_kv(), base_attention=float(i)))
    cache.advance_time()
    cache.insert(*_kv(), base_attention=99.0)  # triggers eviction
    assert not cache.contains(ids[0])
    assert cache.contains(ids[1])
    assert cache.contains(ids[2])
    assert cache.stats["num_evictions"] == 1


def test_lrr_evicts_least_recently_restored():
    cache = BoundedDPCache(DPCacheConfig(capacity=3, eviction_policy=EvictionPolicy.LRR))
    ids = [cache.insert(*_kv(), base_attention=1.0) for _ in range(3)]
    # Touch id[2] by restoring then re-inserting with the same logical time
    cache.advance_time(5)
    cache.restore(ids[2])
    new_id = cache.insert(*_kv(), base_attention=1.0)
    # Now insert one more — should evict the least-recently-touched of the
    # remaining originals (ids[0] or ids[1]).
    cache.advance_time(1)
    cache.insert(*_kv(), base_attention=1.0)
    # ids[2] is gone (we restored it), new_id should still be there.
    assert cache.contains(new_id)
    # At least one of the original two should be evicted.
    survivors = [i for i in ids[:2] if cache.contains(i)]
    assert len(survivors) == 1


def test_decay_evicts_lowest_priority():
    cache = BoundedDPCache(
        DPCacheConfig(capacity=2, eviction_policy=EvictionPolicy.DECAY, decay_lambda=1.0)
    )
    high = cache.insert(*_kv(), base_attention=10.0)
    low = cache.insert(*_kv(), base_attention=0.1)
    # Trigger eviction.
    cache.insert(*_kv(), base_attention=5.0)
    # The low-attention entry should have been evicted, even though it's the
    # most recent of the original two.
    assert cache.contains(high)
    assert not cache.contains(low)


def test_decay_priority_respects_time():
    cache = BoundedDPCache(
        DPCacheConfig(capacity=10, eviction_policy=EvictionPolicy.DECAY, decay_lambda=0.5)
    )
    a = cache.insert(*_kv(), base_attention=1.0)
    cache.advance_time(10)
    b = cache.insert(*_kv(), base_attention=1.0)
    # Snapshot priorities via internal helper for the test.
    pri_a = cache._priority(cache._entries[a])
    pri_b = cache._priority(cache._entries[b])
    assert pri_b > pri_a  # newer entry has higher priority
    expected_a = 1.0 * math.exp(-0.5 * 10)
    assert math.isclose(pri_a, expected_a, rel_tol=1e-6)


def test_restore_top_k_returns_highest_priority_first():
    cache = BoundedDPCache(
        DPCacheConfig(capacity=10, eviction_policy=EvictionPolicy.DECAY, decay_lambda=0.0)
    )
    # No decay -> priority == base_attention
    cache.insert(*_kv(), base_attention=0.1)
    cache.insert(*_kv(), base_attention=0.9)
    cache.insert(*_kv(), base_attention=0.5)
    top = cache.restore_top_k(2)
    assert [round(e.base_attention, 2) for e in top] == [0.9, 0.5]
    assert len(cache) == 1


def test_restore_unknown_id_raises():
    cache = BoundedDPCache(DPCacheConfig(capacity=4))
    try:
        cache.restore(999)
    except KeyError:
        return
    raise AssertionError("expected KeyError")


def test_clear_resets_state():
    cache = BoundedDPCache(DPCacheConfig(capacity=4))
    cache.insert(*_kv(), base_attention=1.0)
    cache.clear()
    assert len(cache) == 0
    assert cache.stats["num_inserts"] == 0
    assert cache.stats["clock"] == 0
