"""
Pre-download LLaVA-OneVision-7B weights to HuggingFace cache.

Run this once before the meeting so the model is ready:
    python scripts/download_model.py

Downloads ~5 GB (int4 shards) and saves to ~/.cache/huggingface/hub/
Progress bar is shown. Safe to interrupt and re-run (resumes).
"""

from huggingface_hub import snapshot_download
import time

MODEL_ID = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

print(f"Downloading {MODEL_ID}")
print("Expected size: ~14 GB fp16 weights (cached once, reused forever)")
print("Press Ctrl+C to pause — it will resume where it left off.\n")

t0 = time.time()
path = snapshot_download(
    repo_id=MODEL_ID,
    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
)
elapsed = time.time() - t0
print(f"\nDone in {elapsed/60:.1f} min — saved to:\n  {path}")
