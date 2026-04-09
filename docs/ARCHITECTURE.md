# StreamDyCoke Architecture

This doc describes how the modules in `streamdycoke/` fit together. It is meant to be a stable reference as integration with a real Video LLM begins.

## Two-cache model

DyCoke (CVPR 2025) achieves its speedup by maintaining two views of the visual KV cache:

1. **Active KV cache** — what the transformer actually attends over at the current decoding step. Kept small.
2. **Dynamic Pruning (DP) cache** — a "bench" for tokens that the model is not currently looking at, but might want back later. Tokens move between the two as attention shifts.

StreamDyCoke keeps this exact two-cache model but makes both halves work in a streaming setting:

```
        new frame
           |
           v
   +---------------+        merged tokens     +---------------+
   |  CausalTTM    |  ----------------------> | active KV     |
   |  (window w)   |                          | (cap = A)     |
   +---------------+                          +-------+-------+
                                                      |
                                  overflow demotion   |
                                                      v
                                              +---------------+
                                              | bounded DP    |
                                              | cache (cap M) |
                                              +-------+-------+
                                                      |
                                       refresh top-k  |
                                                      v
                                              back into active
```

Per-frame, the streaming loop does:

1. **TTM** the new frame against the past `w-1` frames (causal). Drop tokens whose maximum cosine similarity to any past frame's same-position token exceeds `tau`. First frame in each window of `anchor_every` is always preserved.
2. **Append** the survivors to the active KV cache.
3. **Rebalance**: if active cache is over its budget `A`, demote the lowest-attention tokens into the bounded DP cache. The DP cache evicts on its own if full.
4. **(Optionally) refresh**: pull the top-k highest-priority entries from the DP cache back into the active list. The streaming runner can call this whenever it suspects an attention shift, or on a fixed schedule.
5. **Anytime answer**: every `answer_every_k_frames`, invoke the model callback with the current active list and the user's question.

## Module map

| File | Responsibility | Has tensors? | Has model? |
|---|---|---|---|
| `config.py` | dataclass configs, eviction-policy enum | no | no |
| `ttm.py` | causal sliding-window temporal token merging | yes | no |
| `dp_cache.py` | bounded DP cache + 3 eviction policies | yes (opaque KV slabs) | no |
| `streaming.py` | wires TTM + DP cache, exposes `ingest_frame` and `refresh_from_dp_cache` | yes | optional callback |
| `utils.py` | synthetic frames/streams for tests | yes | no |

The `streamdycoke/` package has **no dependency on transformers, accelerate, or any HuggingFace model**. That's deliberate: the streaming logic is testable in isolation, and the integration with LLaVA-OneVision (or any other Video LLM) lives entirely in `scripts/`.

## Eviction policies

Implemented in `dp_cache.py`. All three are O(N) per eviction over the cache size N (which is typically a few thousand entries; small enough that linear is fine).

* **FIFO** — `min(entries, key=inserted_at)`. Simplest baseline. Likely to drop tokens that the model could still want.
* **LRR** (least-recently-restored) — `min(entries, key=last_access)`. Better for workloads where the same tokens get pulled back repeatedly. Adapts to the model's restore pattern.
* **DECAY** — `priority = base_attention * exp(-lambda * (clock - last_access))`. Combines an importance signal at insertion time with a recency decay. The streaming loop bumps the clock once per frame. Two knobs (`base_attention`, `decay_lambda`) give us a richer ablation surface than FIFO/LRR.

The hypothesis (H2 in the proposal) is that DECAY beats FIFO and LRR on streaming Ego4D-QA at the same memory budget.

## What's deliberately *not* in the package

* **Visual encoder**. The streaming loop assumes you hand it a `[num_tokens, hidden_dim]` tensor per frame. Whatever produces those tokens — CLIP, SigLIP, raw patch embeddings — is the runner's problem.
* **Real KV slabs**. Each `CacheEntry` carries a `key` and `value` tensor of caller-defined shape. In tests these are just the token vector itself; in the real integration they will be the per-layer-per-head slabs sliced out of the model's attention layers.
* **Generation**. There is no decoding loop in `streamdycoke/`. The model callback is expected to wrap a real `model.generate(...)` call; the streaming loop only decides *when* to invoke it and *which tokens* to expose.

## Testing strategy

`tests/` exercises the package end-to-end on synthetic tensors:

* `test_ttm.py` — anchor behaviour, redundancy detection, reset semantics, shape validation.
* `test_dp_cache.py` — insert/restore round-trip, FIFO/LRR/DECAY eviction correctness, decay-priority math, top-k restoration ordering.
* `test_streaming.py` — frame ingestion, active-capacity overflow, refresh, anytime-answer scheduling, external attention scores.

All tests run on CPU in well under a second and require no model download. They are the primary regression net while the algorithm is under development; once a real Video LLM is wired in, a thinner end-to-end smoke test will be added under `scripts/` rather than `tests/` so the unit tests stay fast and offline.

## Open design questions

These are intentionally left unresolved in v0 — they're hypotheses to test, not bugs to fix.

1. **Should refresh be event-driven or time-driven?** Right now `refresh_from_dp_cache` is exposed for the runner to call manually. A natural next step is to detect attention shifts (e.g. when the top-k attended tokens change by more than some Jaccard distance) and trigger a refresh automatically. Worth comparing against a simple "every 4 frames" schedule.
2. **Should `base_attention` be set at insertion time only, or updated on each access?** DyCoke's original method effectively updates it on every decoding step. We currently freeze it at insertion to keep the cache O(1) per access; if accuracy suffers we'll revisit.
3. **How aggressively should TTM merge near scene cuts?** Right now we use a single global threshold `tau`. A scene-aware adaptive threshold (drop merging when frame-to-frame similarity falls off a cliff) is one of the proposal's named extensions.
