"""Reproduce DyCoke baseline numbers on LLaVA-OneVision-7B.

PLACEHOLDER. This script requires a GPU and the LLaVA-OneVision-7B weights.
It's checked in so the project layout is complete; the actual implementation
will land once the GPU is available and we've finished verifying the streaming
algorithm on synthetic data.

Planned flow
------------
1. Load LLaVA-OneVision-7B via HuggingFace transformers.
2. Load MVBench / VideoMME / ActivityNet-QA / EgoSchema via lmms-eval or the
   official DyCoke evaluation harness.
3. Run the official offline DyCoke configuration (window=4, K=0.7, layer=3)
   and record:
     - per-benchmark accuracy
     - peak GPU memory
     - wallclock per example
     - per-token decoding latency
4. Dump results into experiments/dycoke_baseline/<benchmark>.json so the
   StreamDyCoke runs can diff against them later.

This script intentionally raises NotImplementedError today so CI / users
don't think it works yet.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "DyCoke baseline reproduction is not implemented yet. "
        "Awaiting GPU access and LLaVA-OneVision-7B weights."
    )


if __name__ == "__main__":
    main()
