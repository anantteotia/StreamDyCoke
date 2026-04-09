"""Streaming benchmark for StreamDyCoke on Ego4D-QA / OnlineVideo-Bench.

PLACEHOLDER. Same disclaimer as ``reproduce_dycoke.py``: this script lays out
the intended flow so the project structure is complete, but the real
implementation requires a GPU and a Video LLM.

Planned flow
------------
1. Load LLaVA-OneVision-7B.
2. Wrap its visual encoder + language decoder in a thin adapter that hands
   per-frame token tensors to ``StreamDyCoke.ingest_frame``.
3. Iterate over Ego4D-QA / OnlineVideo-Bench videos in **streaming order**:
   one frame at a time, with timestamped questions arriving mid-stream.
4. For each question, record:
     - whether the model answered correctly
     - time-to-answer from the moment the question arrived
     - peak GPU memory at the time of the answer
     - active cache size and DP cache size
5. Sweep over (window_size, similarity_threshold, dp_capacity, eviction_policy,
   active_capacity) to produce the ablation tables for the report.
6. Dump everything into experiments/streamdycoke/*.jsonl.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "Streaming benchmark is not implemented yet. "
        "Awaiting GPU access and Video LLM integration."
    )


if __name__ == "__main__":
    main()
