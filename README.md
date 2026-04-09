# StreamDyCoke

**Dynamic Token Compression for Streaming and Real-Time Video Large Language Models**

StreamDyCoke is a streaming-friendly extension of [DyCoke (CVPR 2025)](https://arxiv.org/abs/2411.15024), the dynamic token-pruning method for Video LLMs. DyCoke gives strong speed/memory wins on offline video benchmarks but assumes the entire video is available before inference begins. StreamDyCoke removes that assumption so the same family of techniques can be used in **live, frame-by-frame** settings: assistive vision, robot perception, AR glasses, surveillance, tele-operation, and live commentary.

## What's different from DyCoke

| Component | DyCoke (offline) | StreamDyCoke (this repo) |
|---|---|---|
| Temporal token merging | Symmetric 4-frame window, needs future frames | **Causal sliding-window TTM** — merges only against past frames |
| Dynamic Pruning (DP) cache | Unbounded, grows with video length | **Bounded DP cache** with three eviction policies (FIFO / LRR / attention-decay) |
| Answering | One answer per video, after full prefill | **Anytime answering** — partial answers every *k* frames with no re-prefill |

## Repository layout

```
streamdycoke/         core algorithm modules (PyTorch)
  ttm.py              causal sliding-window temporal token merging
  dp_cache.py         bounded dynamic-pruning cache + eviction policies
  streaming.py        streaming inference loop / anytime answering
  config.py           dataclass configs
  utils.py            shared helpers
tests/                pytest unit tests (run without a GPU or LLM)
scripts/              end-to-end runners (require GPU + Video LLM)
docs/                 architecture notes and design decisions
experiments/          configs and result logs
```

## Status

This is an active course project (Spring 2026, ITCS 6010/8010, UNC Charlotte). The core algorithm modules are pure-PyTorch and CPU-runnable so they can be developed and tested without a Video LLM. End-to-end integration with LLaVA-OneVision lives in `scripts/` and is staged separately.

| Milestone | Status |
|---|---|
| Causal sliding-window TTM | in progress |
| Bounded DP cache (3 eviction policies) | in progress |
| Streaming inference loop | in progress |
| Unit tests on synthetic tensors | in progress |
| DyCoke baseline reproduction (LLaVA-OneVision-7B) | planned |
| Streaming evaluation on Ego4D-QA | planned |
| Ablations and final report | planned |

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

Anant Teotia ([@anantteotia](https://github.com/anantteotia)) — University of North Carolina at Charlotte.
