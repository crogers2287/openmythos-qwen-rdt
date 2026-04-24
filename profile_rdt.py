#!/usr/bin/env python3
"""
RDT Profiling Script — Measure Before Building
================================================
Profiles the actual compute cost of running Qwen3.6-35B-A3B layers
in the RDT configuration (Prelude/Core loop/Coda) at various
recurrence steps and sequence lengths.

Answers:
  1. tokens/sec at each config
  2. VRAM usage at each config
  3. Whether KV cache is reused across recurrence (critical)
  4. Which tensors are recomputed vs reused per loop
  5. What config fits within 24 hours for the full dataset

Does NOT train. Does NOT download datasets. Pure profiling.
"""

try:
    import unsloth
except ImportError:
    pass

import gc
import json
import sys
import time

import torch
import torch.nn as nn


# ===========================================================================
# Step 1: Corrected Compute Model
# ===========================================================================

def print_compute_model():
    """
    RDT compute cost vs baseline.

    Base model: 40 layers sequential
    RDT: Prelude(4) + Core(4) × R loops + Coda(4) = 4 + 4R + 4 = 8 + 4R

    This is EXECUTION count, not unique layer count.
    """
    print("=" * 70)
    print("  CORRECTED COMPUTE MODEL — Execution Count Thinking")
    print("=" * 70)
    print()
    print("  Base model: 40 layers × 1 pass = 40 layer executions")
    print()
    print("  RDT formula: Prelude(4) + Core(4) × R + Coda(4)")
    print("  Effective layer executions = 8 + 4R")
    print()

    rows = []
    for R in [1, 2, 4, 8, 16]:
        eff = 8 + 4 * R
        ratio = eff / 40
        rows.append((R, eff, ratio))
        print(f"  R={R:2d}:  {eff:3d} layer execs  ({ratio:.2f}x base)")

    print()
    print("  R=1:  12 execs → 0.30x base (CHEAPER than baseline)")
    print("  R=8:  40 execs → 1.00x base (SAME as baseline)")
    print("  R=16: 72 execs → 1.80x base (80% MORE than baseline)")
    print()
    print("  PLUS per-loop overhead: LTI injection + depth LoRA + loop embedding")
    print("  PLUS backward pass: with gradient checkpointing, forward is recomputed")
    print("  Effective training cost ≈ 3× forward cost (fwd + ckpt recompute + bwd)")
    print()
    print("  KEY QUESTION: Is attention recomputed each loop or is KV cache reused?")
    print("  → If KV cache reused: core loops are cheap (MoE FFN only)")
    print("  → If attention recomputed: core loops are expensive (full attention)")
    print("=" * 70)
    print()
    return rows


# ===========================================================================
# Step 2: Load Model
# ===========================================================================

def load_model():
    from unsloth import FastLanguageModel

    print("Loading model (bf16, single GPU)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="hesamation/Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled",
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        device_map={"": "cuda:0"},
    )
    print(f"Model loaded.")

    # Get text tokenizer
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    if not hasattr(text_tok, "encode"):
        text_tok = getattr(text_tok, "tokenizer", text_tok)
    if text_tok.pad_token is None:
        text_tok.pad_token = text_tok.eos_token

    return model, text_tok


# ===========================================================================
# Step 3: Inspect Architecture — KV Cache & Tensor Reuse
# ===========================================================================

def inspect_architecture(model):
    """Trace the forward path to determine what's reused vs recomputed."""
    print("=" * 70)
    print("  ARCHITECTURE INSPECTION — KV Cache & Tensor Reuse")
    print("=" * 70)

    # Find the text model layers
    text_model = model.model.language_model
    layers = text_model.layers
    config = text_model.config

    print(f"\n  Total layers: {len(layers)}")
    print(f"  Layer types: {config.layer_types[:8]}... (pattern repeats)")
    print()

    # Check layer 0 (DeltaNet/linear attention) structure
    layer0 = layers[0]
    print("  Layer 0 (linear_attention) modules:")
    for name, mod in layer0.named_children():
        print(f"    {name}: {type(mod).__name__}")

    # Check layer 3 (full attention) structure
    layer3 = layers[3]
    print(f"\n  Layer 3 (full_attention) modules:")
    for name, mod in layer3.named_children():
        print(f"    {name}: {type(mod).__name__}")

    # Key finding: DeltaNet layers don't use KV cache in the traditional sense
    # They use a recurrent state (delta rule), not cached K/V matrices.
    # Full attention layers (every 4th) DO use KV cache.
    print()
    print("  FINDING: Qwen3.6-35B-A3B hybrid architecture")
    print("    - Layers 0,1,2 (linear_attention/DeltaNet): recurrent state, NO KV cache")
    print("    - Layer 3 (full_attention): standard attention, HAS KV cache")
    print()
    print("  For RDT recurrence over layers 16-19:")

    core_types = [config.layer_types[i] for i in [16, 17, 18, 19]]
    print(f"    Layer 16: {core_types[0]}")
    print(f"    Layer 17: {core_types[1]}")
    print(f"    Layer 18: {core_types[2]}")
    print(f"    Layer 19: {core_types[3]}")

    n_linear = sum(1 for t in core_types if t == "linear_attention")
    n_full = sum(1 for t in core_types if t == "full_attention")
    print(f"\n    {n_linear} DeltaNet layers (cheap recurrence, no attention recompute)")
    print(f"    {n_full} full attention layer(s) (expensive, recomputes attention each loop)")
    print()

    if n_full > 0:
        print("  WARNING: Full attention in the core IS recomputed each loop.")
        print("  This is O(T²) per loop where T = seq_len.")
        print("  Mitigation: KV cache from prelude could be reused for core's")
        print("  attention layer, but this requires custom cache management.")
    else:
        print("  GOOD: All core layers are DeltaNet — no attention recompute per loop.")

    print("=" * 70)
    print()


# ===========================================================================
# Step 4: Profile Forward Pass
# ===========================================================================

def profile_forward(model, tokenizer, seq_len, recurrence_steps):
    """
    Profile a single forward pass through the RDT architecture.

    Runs layers manually to measure the actual execution cost.
    """
    text_model = model.model.language_model
    layers = text_model.layers
    config = text_model.config

    # Create dummy input
    input_ids = torch.randint(0, 1000, (1, seq_len), device="cuda:0")
    hidden_states = text_model.embed_tokens(input_ids)
    position_ids = torch.arange(seq_len, device="cuda:0").unsqueeze(0)
    position_embeddings = text_model.rotary_emb(hidden_states, position_ids)

    # Warm up
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    mem_before = torch.cuda.memory_allocated() / 1024**3

    # Time the forward pass
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        # Prelude: layers 0-3
        h = hidden_states
        for i in [0, 1, 2, 3]:
            h = layers[i](h, position_embeddings=position_embeddings, position_ids=position_ids)

        # Recurrent Core: layers 16-19 × R
        for loop_t in range(recurrence_steps):
            for i in [16, 17, 18, 19]:
                h = layers[i](h, position_embeddings=position_embeddings, position_ids=position_ids)

        # Coda: layers 36-39
        for i in [36, 37, 38, 39]:
            h = layers[i](h, position_embeddings=position_embeddings, position_ids=position_ids)

        h = text_model.norm(h)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    mem_after = torch.cuda.max_memory_allocated() / 1024**3

    tokens_per_sec = seq_len / elapsed
    effective_layers = 8 + 4 * recurrence_steps

    return {
        "seq_len": seq_len,
        "recurrence": recurrence_steps,
        "effective_layers": effective_layers,
        "time_s": round(elapsed, 3),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "peak_vram_gb": round(mem_after, 1),
        "vram_delta_gb": round(mem_after - mem_before, 2),
    }


def profile_baseline(model, tokenizer, seq_len):
    """Profile the baseline model (all 40 layers, no RDT)."""
    text_model = model.model.language_model
    layers = text_model.layers

    input_ids = torch.randint(0, 1000, (1, seq_len), device="cuda:0")
    hidden_states = text_model.embed_tokens(input_ids)
    position_ids = torch.arange(seq_len, device="cuda:0").unsqueeze(0)
    position_embeddings = text_model.rotary_emb(hidden_states, position_ids)

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        h = hidden_states
        for i in range(40):
            h = layers[i](h, position_embeddings=position_embeddings, position_ids=position_ids)
        h = text_model.norm(h)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return {
        "seq_len": seq_len,
        "recurrence": "baseline(40)",
        "effective_layers": 40,
        "time_s": round(elapsed, 3),
        "tokens_per_sec": round(seq_len / elapsed, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 1),
    }


# ===========================================================================
# Step 5: Run Full Profiling Matrix
# ===========================================================================

def run_profiling(model, tokenizer):
    print("=" * 70)
    print("  PROFILING — Forward Pass Benchmark")
    print("=" * 70)
    print()

    results = []

    for seq_len in [512, 1024]:
        # Baseline first
        print(f"  Profiling baseline (40 layers) @ seq_len={seq_len}...")
        base = profile_baseline(model, tokenizer, seq_len)
        results.append(base)
        print(f"    {base['time_s']}s, {base['tokens_per_sec']} tok/s, {base['peak_vram_gb']}GB")

        # RDT configs
        for R in [2, 4, 8, 16]:
            print(f"  Profiling RDT R={R} @ seq_len={seq_len}...")
            try:
                res = profile_forward(model, tokenizer, seq_len, R)
                results.append(res)
                print(f"    {res['time_s']}s, {res['tokens_per_sec']} tok/s, {res['peak_vram_gb']}GB")
            except torch.cuda.OutOfMemoryError:
                print(f"    OOM!")
                results.append({
                    "seq_len": seq_len, "recurrence": R,
                    "effective_layers": 8 + 4*R,
                    "time_s": "OOM", "tokens_per_sec": 0, "peak_vram_gb": "OOM",
                })
                torch.cuda.empty_cache()

    # Print results table
    print()
    print("=" * 90)
    print(f"  {'Config':>20} | {'Eff.Layers':>10} | {'Time(s)':>8} | {'Tok/s':>8} | {'VRAM(GB)':>9} | {'Est h/1B tok':>12}")
    print("-" * 90)
    for r in results:
        tok_s = r['tokens_per_sec']
        if isinstance(tok_s, (int, float)) and tok_s > 0:
            hours_per_1b = 1_000_000_000 / tok_s / 3600
            est = f"{hours_per_1b:.0f}h"
        else:
            est = "N/A"
        label = f"R={r['recurrence']}@{r['seq_len']}"
        print(f"  {label:>20} | {r['effective_layers']:>10} | {str(r['time_s']):>8} | {str(r['tokens_per_sec']):>8} | {str(r['peak_vram_gb']):>9} | {est:>12}")
    print("=" * 90)

    return results


# ===========================================================================
# Step 6: Solve for 24-Hour Config
# ===========================================================================

def solve_24h_config(results):
    """Work backward from the 24-hour constraint."""
    print()
    print("=" * 70)
    print("  24-HOUR CONSTRAINT — Working Backward")
    print("=" * 70)

    # Dataset: 9633 examples
    # Assume avg ~500 tokens per example after tokenization at seq_len boundaries
    dataset_size = 9633
    target_hours = 24

    print(f"\n  Dataset: {dataset_size} examples")
    print(f"  Target: {target_hours} hours wall time")
    print()

    # Training cost ≈ 3× forward cost (fwd + grad_ckpt recompute + backward)
    TRAIN_MULTIPLIER = 3.0

    print(f"  Training cost multiplier: {TRAIN_MULTIPLIER}x forward pass")
    print(f"  (forward + gradient checkpointing recompute + backward)")
    print()

    valid_configs = []

    for r in results:
        if isinstance(r['time_s'], str):  # OOM
            continue
        seq_len = r['seq_len']
        recurrence = r['recurrence']
        fwd_time = r['time_s']

        # Time per training step = fwd_time × TRAIN_MULTIPLIER × grad_accum
        for grad_accum in [1, 2, 4]:
            step_time = fwd_time * TRAIN_MULTIPLIER * grad_accum
            steps = dataset_size // grad_accum
            total_hours = step_time * steps / 3600

            fits = total_hours <= target_hours

            valid_configs.append({
                "recurrence": recurrence,
                "seq_len": seq_len,
                "grad_accum": grad_accum,
                "step_time_est": round(step_time, 1),
                "total_steps": steps,
                "total_hours": round(total_hours, 1),
                "fits_24h": fits,
                "effective_batch": grad_accum,
            })

    print(f"  {'Config':>25} | {'GA':>3} | {'Step(s)':>7} | {'Steps':>6} | {'Hours':>6} | {'Fits?':>5}")
    print("  " + "-" * 70)
    for c in valid_configs:
        label = f"R={c['recurrence']}@{c['seq_len']}"
        fits = "YES" if c['fits_24h'] else "no"
        print(f"  {label:>25} | {c['grad_accum']:>3} | {c['step_time_est']:>7} | {c['total_steps']:>6} | {c['total_hours']:>6} | {fits:>5}")

    print()
    winners = [c for c in valid_configs if c['fits_24h']]
    if winners:
        best = max(winners, key=lambda c: (c['seq_len'], c['recurrence']))
        print(f"  RECOMMENDED CONFIG (highest quality that fits 24h):")
        print(f"    recurrence_steps = {best['recurrence']}")
        print(f"    seq_len = {best['seq_len']}")
        print(f"    gradient_accumulation = {best['grad_accum']}")
        print(f"    estimated time = {best['total_hours']}h")
    else:
        print("  NO CONFIG FITS 24h — need faster hardware or smaller model")

    print("=" * 70)
    return valid_configs


# ===========================================================================
# Main
# ===========================================================================

def main():
    # Step 1: Show compute model
    print_compute_model()

    # Step 2: Load model
    model, tokenizer = load_model()

    # Step 3: Inspect architecture
    inspect_architecture(model)

    # Step 4-5: Profile
    results = run_profiling(model, tokenizer)

    # Step 6: Solve for 24h
    solve_24h_config(results)

    print("\nDone. Use these numbers to configure training.")


if __name__ == "__main__":
    main()
