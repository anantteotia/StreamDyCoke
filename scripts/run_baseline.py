"""
Run LLaVA-OneVision-7B baseline on a small set of test videos.

This is the script the professor asked for: "run the model and give numbers."

Usage:
    python scripts/run_baseline.py

What it does:
    1. Loads LLaVA-OneVision-7B at int4 (fits on RTX 5080, 16 GB VRAM)
    2. Runs a short smoke test with synthetic frames to confirm the pipeline works
    3. Reports token counts, memory usage, and latency

Model will be auto-downloaded from HuggingFace on first run (~5 GB int4).
To use full fp16 (needs ~14 GB VRAM, tight on 16 GB): set QUANTIZE=False below.
"""

import time
import torch
from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
QUANTIZE  = True          # int4 via bitsandbytes — fits on 16 GB VRAM
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
N_FRAMES  = 8             # number of synthetic frames to test with

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"Device : {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print(f"\nLoading {MODEL_ID} ({'int4' if QUANTIZE else 'fp16'}) ...")
t0 = time.time()

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
) if QUANTIZE else None

model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_cfg,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

load_time = time.time() - t0
print(f"Loaded  in {load_time:.1f}s")
if DEVICE == "cuda":
    print(f"VRAM used after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ── Synthetic frames ──────────────────────────────────────────────────────────
# RGB frames, 336x336 (LLaVA-OV default patch size).
# We generate a slow-drift sequence so temporal redundancy is realistic.
rng = np.random.default_rng(42)
base = rng.integers(0, 256, (336, 336, 3), dtype=np.uint8)
frames = []
for i in range(N_FRAMES):
    noise = rng.integers(-8, 8, base.shape, dtype=np.int16)
    frame = np.clip(base.astype(np.int16) + noise * i, 0, 255).astype(np.uint8)
    frames.append(Image.fromarray(frame))

print(f"\nSynthetic stream: {N_FRAMES} frames, 336×336 RGB, slow drift")

# ── Build prompt ──────────────────────────────────────────────────────────────
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video"},
            {"type": "text", "text": "Describe what is happening in this video in one sentence."},
        ],
    }
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# ── Tokenise ──────────────────────────────────────────────────────────────────
inputs = processor(
    text=prompt,
    videos=[frames],
    return_tensors="pt",
    padding=True,
).to(DEVICE, torch.float16)

cfg = model.config
video_tok = getattr(cfg, "video_token_index", None) or getattr(cfg, "video_token_id", None)
image_tok = getattr(cfg, "image_token_index", None) or getattr(cfg, "image_token_id", None)
ids = inputs["input_ids"]
n_video = (ids == video_tok).sum().item() if video_tok is not None else 0
n_image = (ids == image_tok).sum().item() if image_tok is not None else 0
n_total_tokens = ids.shape[-1]
print(f"Visual tokens (video placeholder) : {n_video}")
print(f"Visual tokens (image placeholder) : {n_image}")
print(f"Total input tokens                : {n_total_tokens}")
# Count pixel_values shape — gives true visual-feature count going into the LLM
if "pixel_values_videos" in inputs:
    pv = inputs["pixel_values_videos"]
    print(f"pixel_values_videos shape         : {tuple(pv.shape)}")
elif "pixel_values" in inputs:
    pv = inputs["pixel_values"]
    print(f"pixel_values shape                : {tuple(pv.shape)}")

# ── Inference ─────────────────────────────────────────────────────────────────
if DEVICE == "cuda":
    torch.cuda.reset_peak_memory_stats()

t1 = time.time()
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
latency = time.time() - t1

decoded = processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

print(f"\nGeneration latency : {latency:.2f}s")
if DEVICE == "cuda":
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"Peak VRAM          : {peak_mem:.2f} GB")
print(f"\nModel output:\n  {decoded.strip()}")

print("\n[OK] Baseline pipeline working. Ready to wire StreamDyCoke on top.")

# Write machine-readable summary for the report
import json, datetime
summary = {
    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    "model": MODEL_ID,
    "quantization": "int4-nf4" if QUANTIZE else "fp16",
    "device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
    "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory/1e9, 2) if DEVICE == "cuda" else None,
    "load_time_s": round(load_time, 2),
    "vram_after_load_gb": round(torch.cuda.memory_allocated()/1e9, 2) if DEVICE == "cuda" else None,
    "peak_vram_gb": round(torch.cuda.max_memory_allocated()/1e9, 2) if DEVICE == "cuda" else None,
    "n_frames": N_FRAMES,
    "video_placeholder_tokens": n_video,
    "image_placeholder_tokens": n_image,
    "total_input_tokens": n_total_tokens,
    "generation_latency_s": round(latency, 2),
    "max_new_tokens": 64,
    "model_output": decoded.strip(),
}
import os
os.makedirs("experiments/baseline", exist_ok=True)
with open("experiments/baseline/run_baseline_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("Wrote experiments/baseline/run_baseline_summary.json")
