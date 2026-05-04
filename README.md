# StreamDyCoke

[![ci](https://github.com/anantteotia/StreamDyCoke/actions/workflows/ci.yml/badge.svg)](https://github.com/anantteotia/StreamDyCoke/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Dynamic Token Compression for Streaming and Real-Time Video Large Language Models**

StreamDyCoke is a streaming-friendly extension of [DyCoke (CVPR 2025)](https://arxiv.org/abs/2411.15024), the dynamic token-pruning method for Video LLMs. DyCoke gives strong speed/memory wins on offline video benchmarks but assumes the entire video is available before inference begins. StreamDyCoke removes that assumption so the same family of techniques can be used in **live, frame-by-frame** settings: assistive vision, robot perception, AR glasses, surveillance, tele-operation, and live commentary.

## What's different from DyCoke

| Component | DyCoke (offline) | StreamDyCoke (this repo) |
|---|---|---|
| Temporal token merging | Symmetric 4-frame window, needs future frames | **Causal sliding-window TTM** merges only against past frames |
| Dynamic Pruning (DP) cache | Unbounded, grows with video length | **Bounded DP cache** with three eviction policies (FIFO / LRR / attention-decay) |
| Answering | One answer per video, after full prefill | **Anytime answering** gives partial answers every *k* frames with no re-prefill |

## Repository layout

```
streamdycoke/         core algorithm modules (PyTorch)
  ttm.py              causal sliding-window temporal token merging
  dp_cache.py         bounded dynamic-pruning cache + eviction policies
  streaming.py        streaming inference loop / anytime answering
  benchmark.py        sweep harness with policy-sensitive metrics
  viz.py              matplotlib plot helpers
  config.py           dataclass configs
  utils.py            shared helpers
tests/                pytest unit tests (21 passing, <1s on CPU)
scripts/              runners
  synthetic_demo.py            end-to-end demo on synthetic stream
  run_synthetic_benchmark.py   eviction-policy ablation + plots
  reproduce_dycoke.py          (placeholder) needs GPU + Video LLM
  benchmark_streaming.py       (placeholder) needs GPU + Ego4D
docs/ARCHITECTURE.md  design doc
experiments/          benchmark output (gitignored)
```

## Status

This is an active course project (Spring 2026, ITCS 6010/8010, UNC Charlotte). The core algorithm modules are pure-PyTorch and CPU-runnable so they can be developed and tested without a Video LLM. End-to-end integration with LLaVA-OneVision lives in `scripts/` and is staged separately.

| Milestone | Status |
|---|---|
| Causal sliding-window TTM | done |
| Bounded DP cache (3 eviction policies) | done |
| Streaming inference loop | done |
| Synthetic benchmark + ablation infrastructure | done |
| Unit tests (21 passing on CPU, sub-second) | done |
| DyCoke baseline reproduction (LLaVA-OneVision-7B) | done (scripts + measured scaling artifacts locally; outputs gitignored) |
| Streaming evaluation on Ego4D-QA | planned |
| Course final report (written summary of approach + experiments + limitations) | done (`docs/FINAL_REPORT.md`) |
| Ablations on real attention (Video LLM hooks + dataset numbers) | planned |

## Preliminary results (synthetic streams)

The bounded DP cache supports three eviction policies. On a 32-frame synthetic stream with non-uniform per-token attention scores (capacity 64, active capacity 24, refresh top-6 every 4 frames, mean over 3 seeds):

| Policy | DP mean attention | DP mean age (frames) | TTM reduction |
|---|---:|---:|---:|
| FIFO  | 0.50 | 2.65 | 0.74 |
| LRR   | 0.50 | 2.65 | 0.74 |
| **DECAY** | **0.83** | **5.25** | 0.74 |

The attention-aware **DECAY** policy retains 66% higher-attention tokens in the DP cache pool than FIFO/LRR while keeping them around twice as long. Insert/evict counts are workload-driven and identical across policies; the *quality* of what each policy retains is what differs. Real-data validation against a Video LLM is the next milestone.

## Quickstart (algorithm-only, no GPU)

```bash
git clone https://github.com/anantteotia/StreamDyCoke.git
cd StreamDyCoke
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
pytest -q
```

## Citation of the paper this builds on

> Tao, K., Qin, C., You, H., Sui, Y., Wang, H. *DyCoke: Dynamic Compression of Tokens for Fast Video Large Language Models.* CVPR 2025.

## License

MIT. See [LICENSE](LICENSE).

## Author

Anant Teotia ([@anantteotia](https://github.com/anantteotia)), University of North Carolina at Charlotte.
