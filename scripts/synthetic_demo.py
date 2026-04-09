"""End-to-end synthetic demo of the StreamDyCoke pipeline.

This script does not need a GPU or any model weights. It feeds a slowly
drifting synthetic stream into the streaming loop and prints the per-frame
state, so you can see TTM merging tokens, the active cache filling up, the
DP cache catching the overflow, and the refresh hook pulling things back.

Run from the project root:

    python -m scripts.synthetic_demo
"""

from __future__ import annotations

from typing import List, Optional

from streamdycoke import (
    DPCacheConfig,
    EvictionPolicy,
    StreamDyCoke,
    StreamDyCokeConfig,
    TTMConfig,
)
from streamdycoke.dp_cache import CacheEntry
from streamdycoke.utils import make_synthetic_stream


def fake_model(active: List[CacheEntry], q: Optional[str]) -> str:
    return f"[answer over {len(active)} tokens] q={q!r}"


def main() -> None:
    cfg = StreamDyCokeConfig(
        ttm=TTMConfig(window_size=4, similarity_threshold=0.92, anchor_every=4),
        dp_cache=DPCacheConfig(
            capacity=64, eviction_policy=EvictionPolicy.DECAY, decay_lambda=0.05
        ),
        answer_every_k_frames=4,
    )
    sd = StreamDyCoke(cfg, active_capacity=24, model_callable=fake_model)

    stream = make_synthetic_stream(
        num_frames=16, num_tokens=32, hidden_dim=16, drift=0.05
    )

    header = f"{'frame':>5} | {'in_active':>9} | {'in_dp':>5} | {'kept_total':>10} | {'in_total':>8}"
    print(header)
    print("-" * len(header))
    for t, frame in enumerate(stream):
        state = sd.ingest_frame(frame, question="what just happened?")
        print(
            f"{t:>5} | {state.tokens_in_active:>9} | {state.tokens_in_dp_cache:>5} | "
            f"{state.total_kept_after_ttm:>10} | {state.total_input_tokens:>8}"
        )

    print()
    print(f"TTM total reduction ratio: {sd.ttm.total_reduction_ratio:.3f}")
    print(f"DP cache stats: {sd.dp_cache.stats}")
    print(f"Anytime answers produced: {len(sd.answers)}")
    for i, a in enumerate(sd.answers):
        print(f"  [{i}] {a}")

    # Demonstrate refresh
    print()
    print("Refreshing top-4 from DP cache back into active...")
    n = sd.refresh_from_dp_cache(k=4)
    print(f"Restored {n} entries. New state: {sd.state()}")


if __name__ == "__main__":
    main()
