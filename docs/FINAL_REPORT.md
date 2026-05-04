# Final Report: StreamDyCoke (Streaming Dynamic Token Compression for Video LLMs)

**Course:** ITCS 6010 / 8010, Spring 2026  
**Institution:** University of North Carolina at Charlotte  
**Author:** Anant Teotia  
**Project repository:** [https://github.com/anantteotia/StreamDyCoke](https://github.com/anantteotia/StreamDyCoke)  
**Date:** May 4, 2026

## Abstract

Video large language models (Video LLMs) are expensive because visual tokens create long key/value (KV) sequences. DyCoke (CVPR 2025) compresses visual tokens for faster offline video question answering, but its temporal token merging (TTM) window is not causal, and its dynamic pruning (DP) storage can grow without a hard cap. This project implements **StreamDyCoke**, a streaming first design that (1) replaces DyCoke’s symmetric TTM with a **causal sliding window** TTM, (2) adds a **bounded DP cache** with three eviction policies, and (3) exposes a **frame by frame** streaming loop with an optional **anytime answering** schedule. The core system is implemented as small, testable PyTorch modules with no HuggingFace dependency, plus optional GPU scripts for a LLaVA OneVision baseline smoke test. On synthetic streaming workloads, an attention aware **DECAY** eviction policy retains higher attention tokens in the DP cache for longer than FIFO and LRR at the same capacity, which is the expected prerequisite for later real model validation.

## 1. Introduction

Real time and assistive applications (wearables, robots, live monitoring) need models that can update continuously as new frames arrive. Most published Video LLM speedups target **offline** clips where the full video is available up front. DyCoke is a strong baseline for token reduction, but its non causal TTM and unbounded DP behavior are mismatched with long running streams. The goal of this project is to keep DyCoke’s useful idea (a small active cache plus a secondary “bench” cache for tokens that may return later) while making the method **causal**, **memory bounded**, and **streamable**.

## 2. Background and related work

**DyCoke** reduces compute by merging redundant tokens over time and by moving low priority tokens out of the active attention context into a DP cache, then restoring them when needed. The published method is aligned with offline evaluation where future frames can be used to decide merges. This project does not re derive DyCoke in full. Instead, it uses DyCoke as the conceptual baseline and implements a compatible, streaming adapted variant in code, with unit tests to lock behavior.

**Streaming constraints.** A practical streaming system must (a) use only past and current frames for temporal redundancy decisions, (b) keep memory near constant as stream length grows, and (c) support partial outputs on a schedule (for example, an update every *k* frames) without requiring a full prefill of a long clip.

## 3. Method

### 3.1 Causal sliding window temporal token merging (TTM)

DyCoke’s original TTM can be viewed as comparing a new frame to nearby frames in time. In StreamDyCoke, TTM is implemented as a **causal** sliding window: each new frame is compared only to the most recent *w-1* buffered frames. Tokens at a fixed spatial index are treated as candidates for merging when their cosine similarity to a best matching past token at the same index exceeds a threshold. Anchor frames are preserved on a schedule so the stream cannot collapse into overly aggressive merging during stable scenes.

### 3.2 Bounded dynamic pruning cache (DP cache)

DyCoke’s DP mechanism can be interpreted as a second memory pool for tokens that are not currently in the active attending set. In a long stream, storing every demoted token is not acceptable. StreamDyCoke therefore uses a **fixed capacity** DP cache. When the cache is full, an eviction policy chooses a victim entry.

Three policies are implemented:

1. **FIFO** evicts the oldest insertion by logical time.  
2. **LRR** evicts the entry that was least recently restored back into the active set.  
3. **DECAY** maintains an importance priority based on attention at insertion time with exponential decay since last access, then evicts the lowest priority entry.

The hypothesis tested in synthetic experiments is that DECAY should retain more useful tokens than FIFO and LRR under non uniform attention, because it combines importance with recency.

### 3.3 Streaming loop and anytime answering

The streaming object composes TTM and the bounded DP cache. Each frame, the pipeline:

1. runs causal TTM to drop temporally redundant tokens,  
2. appends surviving tokens into an active token list with an active capacity budget,  
3. demotes lowest priority tokens into the DP cache when active capacity overflows,  
4. optionally restores top *k* entries from the DP cache (for example on a schedule),  
5. optionally triggers an answer callback every *k* frames if a model callable is provided.

This separation keeps streaming logic independent from any particular transformer implementation. Integration with a real Video LLM is expected to replace synthetic token tensors with real KV tensors and replace heuristic attention with shallow layer attention scores.

## 4. Implementation

The implementation lives in the open GitHub repository **StreamDyCoke**. Core modules include:

- `streamdycoke/ttm.py` causal sliding window TTM  
- `streamdycoke/dp_cache.py` bounded DP cache and eviction policies  
- `streamdycoke/streaming.py` `StreamDyCoke` streaming loop and optional anytime answering  
- `streamdycoke/benchmark.py` and `streamdycoke/viz.py` synthetic benchmarking and plotting helpers  
- `tests/` pytest regression tests intended to run on CPU quickly  

GPU oriented scripts are included under `scripts/` for baseline reproduction work (for example `scripts/run_baseline.py`), but large downloads and experiment outputs are treated as local artifacts and are not committed (see `.gitignore`).

### 4.1 Integration boundary (what is complete versus staged)

What is complete in this repository is the **algorithmic streaming core** with tests and CPU runnable demos. What remains staged for future work is **full Hooking into HuggingFace decoding** to extract real attention scores per token and to splice real KV tensors into the active and DP structures during generation. The repository includes `scripts/streamdycoke_integration_demo.py` as a bridge demo that runs end to end on synthetic embeddings with fake attention to validate the wiring pattern without a GPU stack.

## 5. Experiments

### 5.1 Synthetic eviction policy ablation

Because the course artifact emphasizes measurable behavior before full model integration, the primary controlled experiment is a synthetic streaming workload with non uniform attention scores. The repository README summarizes one representative configuration (32 frames, DP capacity 64, active capacity 24, refresh top 6 every 4 frames, averaged over multiple seeds). In that setting, DECAY increases the mean attention of tokens retained in the DP cache compared to FIFO and LRR, and increases mean retention age, while temporal merge reduction stays consistent across policies as expected (the merge policy is separate from DP eviction).

These synthetic results are best interpreted as **mechanism validation**: the bounded cache behaves differently under DECAY, and the streaming loop can enforce budgets while preserving a pool of retrievable tokens.

### 5.2 GPU baseline smoke test (optional hardware path)

For grounding in a real Video LLM stack, the repo provides an optional script to load LLaVA OneVision and run a short synthetic frame smoke test, reporting basic timing and memory observations. This path depends on CUDA, large downloads, and local setup, so it is treated as an environment dependent milestone rather than a CI requirement.

### 5.3 Planned evaluation on real streaming video QA

The next step for a full research grade evaluation is to run streaming style evaluation on a dataset such as Ego4D QA style benchmarks with real attention traces. That milestone requires stable hook extraction and a defined protocol for partial answers across time. This project identifies the protocol gap explicitly rather than claiming finished dataset numbers without the integration.

## 6. Discussion

**Strengths.** The design intentionally separates (i) causal redundancy removal, (ii) bounded secondary storage, and (iii) scheduling of answers. That separation matches how streaming systems are engineered in practice and makes testing possible without a GPU cluster.

**Limitations.** Synthetic attention is not semantically meaningful. Until real attention signals are wired in, DP eviction results are about whether the machinery acts correctly under stress, not whether downstream QA accuracy improves.

**Risks in real integration.** Real KV tensors are per layer and per head; attention aggregation must match DyCoke’s shallow layer heuristic consistently; refresh scheduling may need event driven triggers when attention shifts abruptly (scene cuts).

## 7. Conclusion

StreamDyCoke reframes DyCoke’s token compression ideas for streaming video settings by enforcing causality in TTM, bounding DP memory with explicit eviction policies, and exposing a streaming inference loop with anytime answering hooks. The repository provides reproducible CPU tests, synthetic experiments that differentiate eviction policies, and optional GPU scripts for baseline smoke testing. The remaining work is primarily integration and dataset evaluation with real attention and real KV tensors, using the tested streaming core as the stable foundation.

## References

1. Tao, K., Qin, C., You, H., Sui, Y., Wang, H. “DyCoke: Dynamic Compression of Tokens for Fast Video Large Language Models.” CVPR 2025. Available at `https://arxiv.org/abs/2411.15024`.

2. LLaVA OneVision model family (HuggingFace model hub entry used for optional baseline experiments). Project page and checkpoints via HuggingFace `llava-hf` organization (see repository scripts for the exact model identifier used).
