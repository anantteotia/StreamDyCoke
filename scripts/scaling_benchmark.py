"""
Scaling benchmark for LLaVA-OneVision-7B (int4) on RTX 5080.

Measures VRAM, prefill latency, and generation latency as the number of
input frames grows. These are the REAL numbers to report to the professor.

Output:
    experiments/baseline/scaling_results.json
    experiments/baseline/scaling_table.txt
"""
import gc
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaOnevisionForConditionalGeneration,
)

MODEL_ID = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
FRAME_COUNTS = [2, 4, 8, 16, 32]
PROMPT_TEXT = "Describe what is happening in this video in one sentence."

# ── Load once ────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_ID} (int4) ...")
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
t0 = time.time()
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_cfg,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
load_time = time.time() - t0
vram_after_load = torch.cuda.memory_allocated() / 1e9
print(f"  load_time={load_time:.1f}s vram_after_load={vram_after_load:.2f}GB")

# ── Frame generator ──────────────────────────────────────────────────────────
def make_frames(n, seed=42):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (384, 384, 3), dtype=np.uint8)
    frames = []
    for i in range(n):
        noise = rng.integers(-8, 8, base.shape, dtype=np.int16)
        frame = np.clip(base.astype(np.int16) + noise * i, 0, 255).astype(np.uint8)
        frames.append(Image.fromarray(frame))
    return frames


conversation = [{
    "role": "user",
    "content": [
        {"type": "video"},
        {"type": "text", "text": PROMPT_TEXT},
    ],
}]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# ── Sweep ────────────────────────────────────────────────────────────────────
results = []
for n in FRAME_COUNTS:
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    frames = make_frames(n)
    inputs = processor(
        text=prompt, videos=[frames], return_tensors="pt", padding=True,
    ).to("cuda", torch.float16)

    total_input = inputs["input_ids"].shape[-1]
    video_tok = getattr(model.config, "video_token_index", None) or getattr(
        model.config, "video_token_id", None
    )
    visual_tok_count = (inputs["input_ids"] == video_tok).sum().item()

    # Prefill only (no generation)
    torch.cuda.synchronize()
    t1 = time.time()
    with torch.inference_mode():
        out_pre = model(**inputs)
    torch.cuda.synchronize()
    prefill_s = time.time() - t1
    prefill_peak = torch.cuda.max_memory_allocated() / 1e9

    # Generation
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t2 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    torch.cuda.synchronize()
    gen_s = time.time() - t2
    gen_peak = torch.cuda.max_memory_allocated() / 1e9

    decoded = processor.decode(
        out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    ).strip()

    row = {
        "n_frames": n,
        "visual_tokens": visual_tok_count,
        "total_input_tokens": total_input,
        "prefill_latency_s": round(prefill_s, 3),
        "prefill_peak_vram_gb": round(prefill_peak, 2),
        "generate_latency_s": round(gen_s, 3),
        "generate_peak_vram_gb": round(gen_peak, 2),
        "output": decoded,
    }
    results.append(row)
    print(
        f"  n_frames={n:3d}  visual_tokens={visual_tok_count:5d}  "
        f"prefill={prefill_s:.2f}s  gen(32tok)={gen_s:.2f}s  "
        f"peak_vram={gen_peak:.2f}GB"
    )
    del inputs, out, out_pre
    torch.cuda.empty_cache()
    gc.collect()

# ── Write ────────────────────────────────────────────────────────────────────
os.makedirs("experiments/baseline", exist_ok=True)
out = {
    "model": MODEL_ID,
    "quantization": "int4-nf4",
    "gpu": torch.cuda.get_device_name(0),
    "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
    "model_load_time_s": round(load_time, 2),
    "vram_after_load_gb": round(vram_after_load, 2),
    "runs": results,
}
with open("experiments/baseline/scaling_results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

# Text table
with open("experiments/baseline/scaling_table.txt", "w", encoding="utf-8") as f:
    f.write("LLaVA-OneVision-7B (int4) on RTX 5080 - frame-count scaling\n")
    f.write("=" * 78 + "\n")
    f.write(f"{'frames':>7} {'vis_tok':>8} {'prefill_s':>11} {'gen_s':>8} "
            f"{'peak_vram_GB':>13}\n")
    f.write("-" * 78 + "\n")
    for r in results:
        f.write(
            f"{r['n_frames']:>7d} {r['visual_tokens']:>8d} "
            f"{r['prefill_latency_s']:>11.3f} {r['generate_latency_s']:>8.3f} "
            f"{r['generate_peak_vram_gb']:>13.2f}\n"
        )
print("\nWrote experiments/baseline/scaling_results.json")
print("Wrote experiments/baseline/scaling_table.txt")
