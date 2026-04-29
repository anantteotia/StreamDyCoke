# StreamDyCoke — status + numbers (Apr 22, 2026)

**Student**: Anant Teotia (UNC Charlotte)  
**Course**: ITCS 6010/8010 — Efficient AI Computing  
**Project direction**: Streaming / real-time video (frame-by-frame) efficient inference for Video LLMs

## 1) One-paragraph overview (what/why)
Video LLMs are expensive because video inputs create **thousands of visual tokens** and the KV cache grows quickly with frames. Prior work **DyCoke (CVPR 2025)** compresses/prunes tokens effectively for **offline** videos, but assumes the full video is available up front. **StreamDyCoke** adapts this idea to **streaming** settings (live camera / arriving frames) by making token compression and cache management **causal** and **bounded**, enabling **anytime answering** without re-prefill.

## 2) What’s implemented in this repo (already working)
- **Causal sliding-window Temporal Token Merging (TTM)**: merges new-frame tokens only against past frames (no lookahead).
- **Bounded Dynamic-Pruning (DP) cache** with eviction policies:
  - FIFO
  - LRR (least-recently-restored)
  - DECAY (attention-aware with exponential recency decay)
- **Streaming loop / anytime answering** scaffolding (`streamdycoke/streaming.py`).
- **Synthetic benchmark + plots** for policy ablations.
- **Unit tests**: `pytest -q` passes (**21 tests** on CPU in \< 10s locally).

## 3) Numbers you can show today (already run)

### A) Real Video LLM baseline (frame-count scaling)
Collected using **LLaVA-OneVision-7B** `llava-hf/llava-onevision-qwen2-7b-ov-hf` with **int4-nf4** quantization on **RTX 5080 (17.09 GB VRAM)**.

Source: `experiments/baseline/scaling_table.txt` and `experiments/baseline/scaling_results.json`.

| Frames | Visual tokens | Prefill latency (s) | Generate latency (s) | Peak VRAM (GB) |
|---:|---:|---:|---:|---:|
| 2  | 393  | 0.710 | 2.454 | 6.18 |
| 4  | 785  | 0.293 | 2.465 | 6.42 |
| 8  | 1569 | 0.623 | 2.396 | 7.32 |
| 16 | 3137 | 1.635 | 3.879 | 10.06 |
| 32 | 6273 | 21.709 | 27.618 | 19.28 |

**Takeaway**: latency and memory rise sharply with frames; at 32 frames the run hits ~**19.3 GB peak** (over a 17.1 GB card), motivating aggressive streaming-friendly compression/budgeting.

Also captured an 8-frame baseline run summary:
- Source: `experiments/baseline/run_baseline_summary.json`
- Model load time: **13.93 s**
- VRAM after load: **5.82 GB**
- Peak VRAM: **6.72 GB**
- Generation latency: **3.11 s** (max_new_tokens=64)

### B) Synthetic streaming benchmark (policy ablation)
Ablation compares DP-cache eviction policies under the same streaming workload:
- 32-frame stream, 64 tokens/frame, active capacity 24, DP capacity 64
- mean over **3 seeds**

Source: `experiments/synthetic/summary.json` (plots: `experiments/synthetic/policy_summary.png`, `cache_occupancy.png`).

| Policy | DP mean attention (higher is better) | DP mean age (frames) | TTM reduction |
|---|---:|---:|---:|
| FIFO | 0.500 | 2.646 | 0.738 |
| LRR | 0.500 | 2.646 | 0.738 |
| **DECAY** | **0.833** | **5.250** | 0.738 |

**Takeaway**: with the same cache sizes and token traffic, **DECAY retains substantially higher-attention tokens** and keeps them around longer—evidence that eviction policy matters for streaming.

## 4) What’s missing / next milestones (to make it a full course project)
- Integrate streaming token budget + DP-cache refresh into a **real Video LLM** (LLaVA-OneVision in `scripts/` is the target).
- Evaluate on a streaming video QA dataset (planned: **Ego4D-QA** or similar).
- Report accuracy-vs-latency-vs-memory tradeoffs under fixed budgets and ablate:
  - window size, merge threshold, DP capacity, active capacity, refresh schedule
  - eviction policy (FIFO vs LRR vs DECAY)

## 5) Where to find everything in the repo
- **Code**: `streamdycoke/`
- **Tests**: `tests/`
- **Runners**: `scripts/`
- **Artifacts / numbers**: `experiments/` (includes baseline scaling + synthetic plots)
- **Design doc**: `docs/ARCHITECTURE.md`

