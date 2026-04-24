#!/usr/bin/env python3
"""
RDT Training — Non-Unsloth, Profile-First
==========================================
True Recurrent-Depth Transformer training using plain Transformers + PEFT.
No Unsloth. No compiled kernels. No accelerate hooks fighting the RDT loop.

The RDT loop calls decoder layers directly in a controlled forward pass.
This works because plain HF layers are simple nn.Module with no wrapper magic.

Architecture:
    Prelude (layers 0-3)   → run once, capture input signal e
    Core    (layers 16-19) → loop R times with LTI injection
    Coda    (layers 36-39) → run once, produce output

Control switches:
    --max_recurrence_steps R    (default: 8)
    --seq_len N                 (default: 1024)
    --target_hours H            (default: 24)
    --dry_run_profile           profile only, no training
    --confirm_runtime           required to start training

Usage:
    python train_rdt.py --dry_run_profile          # profile and show ETA
    python train_rdt.py --confirm_runtime           # train after profile check
"""

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rdt")


# ===========================================================================
# Config
# ===========================================================================

BASE_MODEL = "hesamation/Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled"
DATASET = "Roman1111111/claude-opus-4.6-10000x"
PRELUDE_LAYERS = [0, 1, 2, 3]
CORE_LAYERS = [16, 17, 18, 19]
CODA_LAYERS = [36, 37, 38, 39]
LTI_DIM = 2048
DEPTH_LORA_RANK = 32
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj", "shared_expert_gate"]
MoE_TEMPERATURE = 2.0
OUTPUT_DIR = "/data/share311/rdt/output"


# ===========================================================================
# Model Loading — Plain Transformers, No Unsloth
# ===========================================================================

def load_model(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    log.info(f"Loading {BASE_MODEL} via transformers (bf16, no Unsloth)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info(f"Model loaded. Type: {type(model).__name__}")
    return model, tokenizer


# ===========================================================================
# Locate Model Internals
# ===========================================================================

def find_text_model(model):
    """Navigate to the text model and its layers."""
    # Qwen3.5 MoE CausalLM: model.model.layers
    # Qwen3.5 MoE ConditionalGen (Unsloth): model.model.language_model.layers
    for path in [
        "model",                        # Qwen3_5MoeForCausalLM via transformers
        "model.language_model",         # Qwen3_5MoeForConditionalGeneration via Unsloth
        "model.model.language_model",
        "model.model",
    ]:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "layers") and hasattr(obj, "embed_tokens"):
                log.info(f"Text model found at: {path}")
                return obj
        except AttributeError:
            continue
    raise RuntimeError("Cannot locate text model with layers + embed_tokens")


# ===========================================================================
# RDT Forward Pass
# ===========================================================================

class RDTForward(nn.Module):
    """
    RDT forward pass module. Registered as a submodule on the text model
    so its parameters (LTI, depth_lora) are included in the optimizer.

    Does NOT wrap the model. Called explicitly in the training loop.
    """

    def __init__(self, text_model, args):
        super().__init__()
        from open_mythos import LTIInjection, LoRAAdapter, loop_index_embedding

        self.text_model = text_model
        self.layers = text_model.layers
        self.R = args.max_recurrence_steps
        self.loop_index_embedding = loop_index_embedding

        # Trainable RDT components
        self.lti = LTIInjection(dim=LTI_DIM).float().to("cuda:0")
        self.depth_lora = LoRAAdapter(
            dim=LTI_DIM, rank=DEPTH_LORA_RANK, max_loops=args.max_recurrence_steps,
        ).to(device="cuda:0", dtype=torch.bfloat16)

        log.info(f"RDT components: LTI(dim={LTI_DIM}), DepthLoRA(rank={DEPTH_LORA_RANK})")

    def forward(self, hidden_states, position_ids=None):
        """
        Run Prelude → Core×R → Coda on hidden_states (after embedding).
        Returns final hidden_states (before norm + lm_head).
        """
        # Position embeddings (RoPE)
        position_embeddings = self.text_model.rotary_emb(hidden_states, position_ids)
        config = self.text_model.config

        def run_layer(i, h):
            return self.layers[i](
                h, position_embeddings=position_embeddings, position_ids=position_ids,
            )

        # Prelude
        for i in PRELUDE_LAYERS:
            hidden_states = run_layer(i, hidden_states)

        # Capture input signal for LTI injection
        e = hidden_states.clone()
        h = hidden_states

        # Core × R
        for loop_t in range(self.R):
            h = self.loop_index_embedding(h, loop_t, 64)

            transformer_out = h
            for i in CORE_LAYERS:
                transformer_out = run_layer(i, transformer_out)

            # LTI stable update (fp32 for numerical stability)
            h_new = self.lti(h.float(), e.float(), transformer_out.float())
            h = h_new.to(torch.bfloat16)

            # Depth LoRA
            h = h + self.depth_lora(h, loop_t)

        # Coda
        hidden_states = h
        for i in CODA_LAYERS:
            hidden_states = run_layer(i, hidden_states)

        return hidden_states


# ===========================================================================
# MoE Temperature Override
# ===========================================================================

def override_moe_temperature(model, text_model, temperature):
    """Patch MoE router softmax temperature."""
    import torch.nn.functional as F
    patched = 0

    for name, module in model.named_modules():
        if name.endswith(".mlp.gate") and hasattr(module, "top_k") and hasattr(module, "weight"):
            def make_fwd(mod, temp):
                def fwd(hidden_states):
                    h = hidden_states.reshape(-1, mod.hidden_dim)
                    logits = F.linear(h, mod.weight) / temp
                    probs = F.softmax(logits, dtype=torch.float, dim=-1)
                    top_val, top_idx = torch.topk(probs, mod.top_k, dim=-1)
                    top_val = (top_val / top_val.sum(dim=-1, keepdim=True)).to(probs.dtype)
                    return probs, top_val, top_idx
                return fwd
            module.forward = make_fwd(module, temperature)
            patched += 1

    log.info(f"MoE temperature {temperature} applied to {patched} routers")


# ===========================================================================
# LoRA Setup (plain PEFT, no Unsloth)
# ===========================================================================

def setup_lora(model, args):
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        target_modules=LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ===========================================================================
# Profiling
# ===========================================================================

def profile(model, text_model, rdt, tokenizer, args):
    """Measure actual forward pass time and compute ETA."""
    log.info("=" * 70)
    log.info("  PROFILING — Measuring Real Forward Pass")
    log.info("=" * 70)

    seq_len = args.seq_len
    R = args.max_recurrence_steps
    effective_layers = 8 + 4 * R

    log.info(f"  Config: R={R}, seq_len={seq_len}")
    log.info(f"  Effective layer executions: {effective_layers} (baseline=40)")

    # Dummy input
    input_ids = torch.randint(0, 1000, (1, seq_len), device="cuda:0")
    position_ids = torch.arange(seq_len, device="cuda:0").unsqueeze(0)

    # Warm-up
    with torch.no_grad():
        h = text_model.embed_tokens(input_ids)
        _ = rdt(h, position_ids)
    torch.cuda.synchronize()

    # Measure forward (inference)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for trial in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            h = text_model.embed_tokens(input_ids)
            out = rdt(h, position_ids)
            out = text_model.norm(out)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    fwd_time = min(times)
    vram_gb = torch.cuda.max_memory_allocated() / 1024**3

    # Measure forward+backward (training)
    gc.collect()
    torch.cuda.empty_cache()

    h = text_model.embed_tokens(input_ids)
    h.requires_grad_(True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = rdt(h, position_ids)
    out = text_model.norm(out)
    loss = out.sum()  # dummy loss
    loss.backward()
    torch.cuda.synchronize()
    train_step_time = time.perf_counter() - t0

    peak_vram_train = torch.cuda.max_memory_allocated() / 1024**3

    # ETA
    dataset_size = 9633
    grad_accum = max(1, args.grad_accum)
    steps = dataset_size // grad_accum
    total_hours = train_step_time * grad_accum * steps / 3600

    log.info(f"\n  RESULTS:")
    log.info(f"    Forward (inference):  {fwd_time:.3f}s  ({seq_len/fwd_time:.0f} tok/s)")
    log.info(f"    Forward+Backward:    {train_step_time:.3f}s")
    log.info(f"    VRAM (inference):    {vram_gb:.1f} GB")
    log.info(f"    VRAM (training):     {peak_vram_train:.1f} GB")
    log.info(f"    Dataset:             {dataset_size} examples")
    log.info(f"    Grad accumulation:   {grad_accum}")
    log.info(f"    Total steps:         {steps}")
    log.info(f"    Estimated runtime:   {total_hours:.1f} hours")
    fits = total_hours <= args.target_hours
    log.info(f"    Fits {args.target_hours}h target:    {'YES' if fits else 'NO'}")
    log.info("=" * 70)

    return {
        "fwd_time": fwd_time,
        "train_step_time": train_step_time,
        "vram_inference_gb": vram_gb,
        "vram_training_gb": peak_vram_train,
        "total_hours": total_hours,
        "fits_target": fits,
    }


# ===========================================================================
# Dataset
# ===========================================================================

def load_dataset_formatted(tokenizer, args):
    from datasets import load_dataset

    log.info(f"Loading dataset: {DATASET}")
    dataset = load_dataset(DATASET, split="train")
    log.info(f"Dataset: {len(dataset)} examples")

    def format_example(example):
        messages = example.get("messages", example.get("conversations", []))
        if isinstance(messages, list) and len(messages) > 0:
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                )
            except Exception:
                parts = [f"<|im_start|>{m.get('role','user')}\n{m.get('content','')}<|im_end|>"
                         for m in messages]
                text = "\n".join(parts)
        else:
            text = example.get("text", str(example))
        return {"text": text}

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    def tokenize(example):
        return tokenizer(example["text"], truncation=True,
                         max_length=args.seq_len, padding="max_length")

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    log.info(f"Dataset tokenized: {len(dataset)} examples")
    return dataset


# ===========================================================================
# Training Loop
# ===========================================================================

def train(model, text_model, rdt, tokenizer, profile_result, args):
    log.info("=" * 70)
    log.info("  TRAINING — RDT Recurrent Distillation")
    log.info("=" * 70)

    dataset = load_dataset_formatted(tokenizer, args)

    # Freeze base model, only train LoRA + RDT components
    trainable = [p for p in model.parameters() if p.requires_grad]
    rdt_params = list(rdt.lti.parameters()) + list(rdt.depth_lora.parameters())
    all_trainable = trainable + rdt_params

    total_t = sum(p.numel() for p in all_trainable)
    total_all = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in rdt_params)
    log.info(f"Trainable: {total_t:,} / {total_all:,} ({100*total_t/total_all:.4f}%)")

    optimizer = torch.optim.AdamW(all_trainable, lr=2e-4, weight_decay=0.01)

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)

    model.train()
    rdt.train()

    log.info(f"Starting: R={args.max_recurrence_steps}, seq_len={args.seq_len}, "
             f"steps={len(dataloader)}, ETA={profile_result['total_hours']:.1f}h")

    for step, batch in enumerate(dataloader):
        t0 = time.perf_counter()

        input_ids = batch["input_ids"].to("cuda:0")
        labels = input_ids.clone()
        position_ids = torch.arange(input_ids.shape[1], device="cuda:0").unsqueeze(0)

        # Forward: embed → RDT → norm → lm_head
        h = text_model.embed_tokens(input_ids)
        h = rdt(h, position_ids)
        h = text_model.norm(h)
        logits = model.lm_head(h)

        # Loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), ignore_index=-100,
        )

        loss.backward()

        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad()

        elapsed = time.perf_counter() - t0
        eta_h = elapsed * (len(dataloader) - step - 1) / 3600

        if (step + 1) % 1 == 0:
            rho = rdt.lti.get_A().max().item()
            log.info(
                f"Step {step+1}/{len(dataloader)} "
                f"loss={loss.item():.4f} "
                f"rho(A)={rho:.4f} "
                f"step={elapsed:.1f}s "
                f"ETA={eta_h:.1f}h"
            )

        if (step + 1) % 500 == 0:
            save_dir = Path(OUTPUT_DIR) / f"checkpoint-{step+1}"
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "lti": rdt.lti.state_dict(),
                "depth_lora": rdt.depth_lora.state_dict(),
            }, save_dir / "rdt_components.pt")
            model.save_pretrained(save_dir / "lora")
            log.info(f"Checkpoint saved: {save_dir}")

    log.info("Training complete.")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="RDT Training — Profile-First")
    parser.add_argument("--max_recurrence_steps", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--target_hours", type=float, default=24.0)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--dry_run_profile", action="store_true",
                        help="Profile only, no training")
    parser.add_argument("--confirm_runtime", action="store_true",
                        help="Required to start training")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("  RDT Recurrent Distillation — Profile-First Pipeline")
    log.info(f"  R={args.max_recurrence_steps}, seq_len={args.seq_len}, "
             f"target={args.target_hours}h")
    log.info("=" * 70)

    # Load model (plain transformers, no Unsloth)
    model, tokenizer = load_model(args)
    text_model = find_text_model(model)

    # Build RDT forward module
    rdt = RDTForward(text_model, args)

    # Override MoE temperature
    override_moe_temperature(model, text_model, MoE_TEMPERATURE)

    # Apply LoRA
    model = setup_lora(model, args)

    # Re-find text model after PEFT wrapping
    text_model = find_text_model(model)

    # Profile
    log.info("\nProfiling...")
    prof = profile(model, text_model, rdt, tokenizer, args)

    if args.dry_run_profile:
        log.info("\nDry run complete. No training started.")
        log.info(f"VERDICT: RDT under plain Transformers {'is viable' if prof['fits_target'] else 'exceeds target'}.")
        log.info(f"RDT requires non-Unsloth training: YES (confirmed by 11 failed Unsloth attempts).")
        sys.exit(0)

    if not args.confirm_runtime:
        log.error("\nTraining requires --confirm_runtime flag.")
        log.error(f"Estimated runtime: {prof['total_hours']:.1f}h "
                  f"({'fits' if prof['fits_target'] else 'EXCEEDS'} {args.target_hours}h target)")
        sys.exit(1)

    if not prof['fits_target']:
        log.warning(f"\nWARNING: Estimated {prof['total_hours']:.1f}h exceeds "
                    f"{args.target_hours}h target. Proceeding anyway (--confirm_runtime).")

    train(model, text_model, rdt, tokenizer, prof, args)


if __name__ == "__main__":
    main()
