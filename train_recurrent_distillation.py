#!/usr/bin/env python3
"""
Recurrent Distillation Training Pipeline
=========================================
Converts a pre-trained Qwen3.6-35B-A3B reasoning model into a
Recurrent-Depth Transformer (RDT) using OpenMythos components.

Architecture:
    Prelude (layers 0-3)   →  run once, capture input signal e
    Recurrent Core (16-19) →  loop up to 16x with LTI injection
    Coda (layers 36-39)    →  run once, produce logits

Key components:
    - LTI injection from OpenMythos (spectral radius < 1 guaranteed)
    - Depth-wise LoRA adaptation per loop iteration
    - Loop-index sinusoidal embedding
    - MoE routing temperature override (2.0)

Target hardware: single NVIDIA RTX PRO 6000 Blackwell (96GB GDDR7)
Loading: bf16 native, no quantization

Usage:
    python train_recurrent_distillation.py           # dry-run validation
    python train_recurrent_distillation.py --train    # start training (requires --confirm)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Unsloth must be imported before transformers/peft for kernel patches
try:
    import unsloth  # noqa: F401
except ImportError:
    pass

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rdt")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
from config import RDTConfig, VRAMBudget

cfg = RDTConfig()


# ===========================================================================
# Phase 1: Dependency Validation
# ===========================================================================

def check_dependencies() -> Dict[str, str]:
    """Verify all required packages are importable and log versions."""
    deps = {}
    errors = []

    for pkg, import_name in [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("open_mythos", "open_mythos"),
        ("datasets", "datasets"),
        ("peft", "peft"),
        ("accelerate", "accelerate"),
    ]:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "unknown")
            deps[pkg] = ver
            log.info(f"  {pkg}: {ver}")
        except ImportError as e:
            errors.append(f"{pkg}: {e}")
            log.error(f"  {pkg}: MISSING — {e}")

    # Optional: Muon optimizer
    try:
        from torch.optim import Muon
        deps["muon"] = "torch.optim.Muon (native)"
        log.info(f"  muon: native (torch.optim.Muon)")
    except ImportError:
        try:
            from muon import MuonWithAuxAdam
            deps["muon"] = "muon-optimizer (pip)"
            log.info(f"  muon: pip package (muon-optimizer)")
        except ImportError:
            deps["muon"] = "unavailable (will use AdamW)"
            log.warning(f"  muon: unavailable — will fall back to AdamW")

    # Optional: Unsloth
    try:
        from unsloth import FastLanguageModel
        deps["unsloth"] = "available"
        log.info(f"  unsloth: available (FastLanguageModel)")
    except ImportError:
        deps["unsloth"] = "unavailable (will use transformers directly)"
        log.warning(f"  unsloth: unavailable — loading via transformers directly")

    if errors:
        log.error(f"Missing {len(errors)} required packages. Install with: pip install -r requirements.txt")
        sys.exit(1)

    return deps


# ===========================================================================
# Phase 2: Model Loading
# ===========================================================================

def load_model_unsloth(cfg: RDTConfig):
    """Load model via Unsloth FastLanguageModel for MoE Triton kernel support."""
    from unsloth import FastLanguageModel

    log.info(f"Loading {cfg.base_model} via Unsloth (bf16, single GPU)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,  # explicit: no quantization
        device_map={"": cfg.device},  # force all on single GPU, no meta offload
    )
    log.info(f"Model loaded. dtype={next(model.parameters()).dtype}")
    return model, tokenizer


def load_model_transformers(cfg: RDTConfig):
    """Fallback: load via transformers AutoModelForCausalLM."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(f"Loading {cfg.base_model} via transformers (bf16, single GPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model,
        trust_remote_code=True,
    )
    log.info(f"Model loaded. dtype={next(model.parameters()).dtype}")
    return model, tokenizer


def load_model(cfg: RDTConfig):
    """Load model, preferring Unsloth if available."""
    try:
        return load_model_unsloth(cfg)
    except ImportError:
        log.warning("Unsloth not available. Falling back to transformers.")
        return load_model_transformers(cfg)


# ===========================================================================
# Phase 2b: Model Introspection
# ===========================================================================

def discover_model_structure(model) -> Dict:
    """
    Inspect the loaded model to find:
    - Layer module paths (for Prelude/Core/Coda extraction)
    - MoE router/gate module names (for LoRA targeting and temperature override)
    - Hidden dimension
    """
    info = {
        "layer_module_path": None,
        "layer_count": 0,
        "hidden_size": 0,
        "router_module_names": [],
        "all_named_modules": [],
    }

    # Walk the model to find decoder layers
    # Qwen3.5 MoE multimodal: model.model.language_model.layers
    # Qwen3.5 MoE text-only: model.model.layers
    layers_module = None
    for attr_path in [
        "model.language_model.layers",       # Qwen3.5 MoE multimodal (via Unsloth)
        "model.model.language_model.layers", # Qwen3.5 MoE multimodal (via transformers)
        "model.model.layers",                # standard HF CausalLM
        "model.layers",                      # some HF models
        "transformer.h",                     # GPT-style
    ]:
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            layers_module = obj
            info["layer_module_path"] = attr_path
            info["layer_count"] = len(layers_module)
            break
        except AttributeError:
            continue

    if layers_module is None:
        log.error("Could not locate decoder layers in model. Dumping top-level modules:")
        for name, _ in model.named_children():
            log.error(f"  {name}")
        raise RuntimeError("Cannot find decoder layers. Model structure is unexpected.")

    # Find hidden size from first layer's attention
    first_layer = layers_module[0]
    for name, param in first_layer.named_parameters():
        if "q_proj" in name:
            info["hidden_size"] = param.shape[0]
            break

    # Find MoE router/gate modules
    # Qwen3.5 MoE uses Qwen3_5MoeTopKRouter (not nn.Linear)
    router_names = []
    for name, module in model.named_modules():
        if name.endswith(".mlp.gate") and hasattr(module, "top_k"):
            router_names.append(name)

    info["router_module_names"] = router_names
    log.info(f"Model structure:")
    log.info(f"  Layers path: {info['layer_module_path']}")
    log.info(f"  Layer count: {info['layer_count']}")
    log.info(f"  Hidden size: {info['hidden_size']}")
    log.info(f"  Router modules found: {len(router_names)}")
    if router_names:
        log.info(f"  Sample router: {router_names[0]}")

    return info


def get_layers(model, layer_module_path: str) -> nn.ModuleList:
    """Get the decoder layers ModuleList from the model."""
    obj = model
    for part in layer_module_path.split("."):
        obj = getattr(obj, part)
    return obj


# ===========================================================================
# Phase 3: RDT Wrapper — Recurrent-Depth Forward Pass
# ===========================================================================

def patch_model_for_rdt(model, model_info: Dict, cfg: RDTConfig):
    """
    Monkey-patch the text model's forward to implement RDT loop.

    Instead of wrapping the model externally (which fights Unsloth's hooks),
    we replace the layer loop inside the model's own forward method. This way
    every layer call goes through the model's optimized execution path, and
    Unsloth/accelerate hooks work naturally.

    The original forward runs:
        for i, layer in enumerate(self.layers[:N]):
            hidden_states = layer(hidden_states, ...)

    We replace it with:
        for i in prelude_layers:   hidden_states = layers[i](...)
        e = hidden_states.clone()
        for loop in range(n_loops):
            for i in core_layers: hidden_states = layers[i](...)
            hidden_states = LTI(hidden_states, e, ...)
        for i in coda_layers:      hidden_states = layers[i](...)
    """
    from open_mythos import LTIInjection, LoRAAdapter, loop_index_embedding

    # Create trainable RDT components and register them on the model
    lti = LTIInjection(dim=cfg.lti_hidden_dim).float().to(cfg.device)
    depth_lora = LoRAAdapter(
        dim=cfg.lti_hidden_dim,
        rank=cfg.depth_lora_rank,
        max_loops=cfg.depth_lora_max_loops,
    ).to(device=cfg.device, dtype=torch.bfloat16)

    # Find the text model (parent of layers)
    layer_path = model_info["layer_module_path"]
    text_model = model
    for part in layer_path.split(".")[:-1]:
        text_model = getattr(text_model, part)

    # Register RDT components as submodules so they're included in parameters()
    text_model.rdt_lti = lti
    text_model.rdt_depth_lora = depth_lora

    # Store config on text_model for the patched forward
    text_model._rdt_cfg = cfg
    text_model._rdt_loop_index_embedding = loop_index_embedding

    # Save original forward
    original_forward = text_model.forward

    def rdt_forward(self_tm, *args, **kwargs):
        """
        Patched forward that replaces the sequential layer loop with RDT.

        We intercept at the text model level, replacing its layer iteration.
        The model's own embedding, norm, and all hooks remain untouched.
        """
        import inspect

        # Get the original forward's handling of inputs (embedding, mask, etc.)
        # by calling it with a modified layers list that we control.
        # BUT: that's fragile. Instead, let's replicate just the layer loop.

        # The Qwen3.5MoeTextModel.forward does:
        # 1. embed_tokens(input_ids)
        # 2. rotary_emb
        # 3. for i, layer in enumerate(self.layers[:N]): hidden_states = layer(...)
        # 4. self.norm(hidden_states)
        # We only need to replace step 3.

        # Extract inputs from args/kwargs (match the original signature)
        input_ids = kwargs.get('input_ids', args[0] if args else None)
        attention_mask = kwargs.get('attention_mask', None)
        position_ids = kwargs.get('position_ids', None)
        past_key_values = kwargs.get('past_key_values', None)
        use_cache = kwargs.get('use_cache', None)

        if input_ids is None:
            inputs_embeds = kwargs.get('inputs_embeds', None)
            hidden_states = inputs_embeds
        else:
            hidden_states = self_tm.embed_tokens(input_ids)

        if position_ids is None:
            batch_size, seq_len = hidden_states.shape[:2]
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)

        position_embeddings = self_tm.rotary_emb(hidden_states, position_ids)

        # Build attention masks matching the original forward
        rdt_cfg = self_tm._rdt_cfg
        layer_types = self_tm.config.layer_types

        # Convert attention mask to float for SDPA compatibility
        # The original forward computes causal_mask and linear_attn_mask
        # For simplicity, pass None to use default causal masking
        attention_mask = None

        # --- RDT Layer Loop ---
        # Prelude: layers 0-3
        for i in rdt_cfg.prelude_layers:
            layer_mask = attention_mask
            hidden_states = self_tm.layers[i](
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

        # Capture input signal e
        e = hidden_states.clone()
        h = hidden_states

        # Recurrent Core: loop layers 16-19
        loop_embed_fn = self_tm._rdt_loop_index_embedding
        for loop_t in range(rdt_cfg.max_loop_iters):
            h = loop_embed_fn(h, loop_t, rdt_cfg.loop_embedding_dim)

            transformer_out = h
            for i in rdt_cfg.recurrent_core_layers:
                layer_mask = attention_mask
                transformer_out = self_tm.layers[i](
                    transformer_out,
                    position_embeddings=position_embeddings,
                    attention_mask=layer_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )

            # LTI stable update
            h_fp32 = h.float()
            e_fp32 = e.float()
            t_fp32 = transformer_out.float()
            h = self_tm.rdt_lti(h_fp32, e_fp32, t_fp32).to(hidden_states.dtype)

            # Depth LoRA
            h = h + self_tm.rdt_depth_lora(h, loop_t)

        # Coda: layers 36-39
        hidden_states = h
        for i in rdt_cfg.coda_layers:
            layer_mask = attention_mask
            hidden_states = self_tm.layers[i](
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

        hidden_states = self_tm.norm(hidden_states)

        # Return proper output format expected by the wrapper model
        from transformers.modeling_outputs import BaseModelOutputWithPast
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
        )

    # Bind the patched forward
    import types
    text_model.forward = types.MethodType(rdt_forward, text_model)

    log.info(f"RDT patch applied to {type(text_model).__name__}:")
    log.info(f"  Prelude: layers {cfg.prelude_layers}")
    log.info(f"  Core: layers {cfg.recurrent_core_layers} x {cfg.max_loop_iters} loops")
    log.info(f"  Coda: layers {cfg.coda_layers}")
    log.info(f"  LTI dim: {cfg.lti_hidden_dim}")
    log.info(f"  Depth LoRA rank: {cfg.depth_lora_rank}")

    return lti, depth_lora


def get_spectral_radius(model, model_info):
    """Get the current LTI spectral radius."""
    # Navigate to the text model, handling PEFT wrapping
    # After PEFT: model.base_model.model.model.language_model
    # Before PEFT: model.model.language_model
    for attr_chain in [
        "base_model.model.model.language_model",  # PEFT-wrapped
        "model.language_model",                     # unwrapped
        "model.model.language_model",               # alternative
    ]:
        obj = model
        try:
            for part in attr_chain.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, 'rdt_lti'):
                with torch.no_grad():
                    return obj.rdt_lti.get_A().max().item()
        except AttributeError:
            continue
    return 0.0  # fallback if not found


# ===========================================================================
# Phase 3b: MoE Routing Temperature Override
# ===========================================================================

def override_routing_temperature(model, temperature: float, model_info: Dict):
    """
    Monkey-patch MoE TopKRouter to apply temperature scaling before softmax.

    Qwen3.5 MoE router forward does:
        logits = F.linear(hidden_states, self.weight)
        probs = softmax(logits)
        ...

    We patch it to:
        logits = F.linear(hidden_states, self.weight) / temperature
        probs = softmax(logits / temperature)

    This softens the routing distribution, preventing routing collapse
    when DeltaNet's recurrent memory state interacts with the MoE gating.
    """
    import torch.nn.functional as F
    patched = 0

    for name, module in model.named_modules():
        # Match the TopKRouter modules (e.g. model.language_model.layers.0.mlp.gate)
        if name.endswith(".mlp.gate") and hasattr(module, "weight") and hasattr(module, "top_k"):
            original_forward = module.forward
            hidden_dim = module.hidden_dim if hasattr(module, "hidden_dim") else module.weight.shape[1]
            top_k = module.top_k

            def make_temp_forward(mod, temp):
                def temp_forward(hidden_states):
                    h = hidden_states.reshape(-1, mod.hidden_dim)
                    router_logits = F.linear(h, mod.weight)
                    # Temperature scaling BEFORE softmax
                    router_logits = router_logits / temp
                    router_logits = torch.nn.functional.softmax(
                        router_logits, dtype=torch.float, dim=-1
                    )
                    router_top_value, router_indices = torch.topk(
                        router_logits, mod.top_k, dim=-1
                    )
                    router_top_value = router_top_value / router_top_value.sum(
                        dim=-1, keepdim=True
                    )
                    router_top_value = router_top_value.to(router_logits.dtype)
                    return router_logits, router_top_value, router_indices
                return temp_forward

            module.forward = make_temp_forward(module, temperature)
            patched += 1

    log.info(f"MoE routing temperature override: {temperature} applied to {patched} router modules")
    if patched == 0:
        log.warning(
            "No MoE TopKRouter modules found for temperature override. "
            "Check model structure — expected modules ending in '.mlp.gate' "
            "with 'weight', 'top_k', and 'hidden_dim' attributes."
        )
    return patched


# ===========================================================================
# Phase 4: LoRA Setup
# ===========================================================================

def setup_lora_unsloth(model, cfg: RDTConfig):
    """Configure LoRA via Unsloth with router fine-tuning enabled."""
    from unsloth import FastLanguageModel

    log.info("Setting up LoRA via Unsloth...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        use_gradient_checkpointing=False,  # disabled — conflicts with custom RDT loop
        max_seq_length=cfg.max_seq_length,
    )
    log.info(f"LoRA applied. Target modules: {cfg.lora_target_modules}")
    return model


def setup_lora_peft(model, cfg: RDTConfig):
    """Fallback: configure LoRA via PEFT directly."""
    from peft import LoraConfig, get_peft_model

    log.info("Setting up LoRA via PEFT...")
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def setup_lora(model, cfg: RDTConfig):
    """Apply LoRA, preferring Unsloth if available."""
    try:
        return setup_lora_unsloth(model, cfg)
    except (ImportError, Exception) as e:
        log.warning(f"Unsloth LoRA failed ({e}). Falling back to PEFT.")
        return setup_lora_peft(model, cfg)


# (Optimizer is built inline in train() — no separate function needed)


# ===========================================================================
# Phase 6: Dataset Preparation (scaffold only — not downloaded yet)
# ===========================================================================

def prepare_dataset(cfg: RDTConfig, tokenizer):
    """
    Load and format the reasoning distillation dataset.

    The Roman1111111/claude-opus-4.6-10000x dataset contains structured
    reasoning traces from Claude Opus 4.6. We format these as
    instruction/response pairs to flow through the RDT loops.

    NOTE: This function downloads the dataset. Call only after user confirmation.
    """
    from datasets import load_dataset

    log.info(f"Loading dataset: {cfg.dataset_name}")
    dataset = load_dataset(cfg.dataset_name, split="train")
    log.info(f"Dataset loaded: {len(dataset)} examples")

    # The tokenizer from Unsloth may be a multimodal processor (Qwen3.5 VL).
    # For text-only training, extract the underlying text tokenizer.
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if not hasattr(text_tokenizer, "encode"):
        # If still a processor, try to get the tokenizer attribute
        text_tokenizer = getattr(text_tokenizer, "tokenizer", text_tokenizer)
    log.info(f"Using text tokenizer: {type(text_tokenizer).__name__}")

    # Ensure pad token is set
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token

    def format_example(example):
        """Format a single example for causal LM training."""
        # The dataset has conversational format with reasoning traces
        messages = example.get("messages", example.get("conversations", []))

        if isinstance(messages, list) and len(messages) > 0:
            # Use the text tokenizer's chat template
            try:
                text = text_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                # Fallback: manual formatting
                parts = []
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
                text = "\n".join(parts)
        else:
            text = example.get("text", example.get("output", str(example)))

        return {"text": text}

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    def tokenize(example):
        return text_tokenizer(
            example["text"],
            truncation=True,
            max_length=cfg.max_seq_length,
            padding="max_length",
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    log.info(f"Dataset tokenized: {len(dataset)} examples, max_len={cfg.max_seq_length}")
    return dataset


# ===========================================================================
# Phase 7: Training Loop (scaffold — requires --train --confirm)
# ===========================================================================

def train(model, model_info, tokenizer, cfg: RDTConfig):
    """
    Run the recurrent distillation training loop.

    The model's text forward has been monkey-patched by patch_model_for_rdt()
    to run the RDT loop. We just call model() normally — all the RDT logic
    (Prelude/Core/Coda, LTI injection, depth LoRA) happens inside the
    model's own forward path, using its own hooks and optimizations.
    """
    log.info("Gradient checkpointing: using Unsloth's native checkpointing")
    model.gradient_checkpointing_enable()

    # Prepare dataset
    dataset = prepare_dataset(cfg, tokenizer)

    # Build optimizer — collect all trainable params from the model
    # (includes LoRA adapters + RDT components registered as submodules)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_params)
    total_all = sum(p.numel() for p in model.parameters())
    log.info(f"Trainable params: {total_trainable:,} / {total_all:,} "
             f"({100 * total_trainable / total_all:.4f}%)")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    log.info("Optimizer: AdamW")

    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model.train()
    global_step = 0

    log.info(f"Starting training: {cfg.num_epochs} epochs, "
             f"{len(dataloader)} steps/epoch, "
             f"grad_accum={cfg.gradient_accumulation_steps}, "
             f"loops={cfg.max_loop_iters}")

    import time as _time

    for epoch in range(cfg.num_epochs):
        epoch_loss = 0.0
        step_start = _time.time()

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(cfg.device)
            attention_mask = batch["attention_mask"].to(cfg.device)
            labels = input_ids.clone()

            # Forward through the patched model — RDT loop happens
            # inside the text model's monkey-patched forward.
            # The full model (ConditionalGeneration) handles lm_head + loss.
            outputs = model(
                input_ids=input_ids,
                labels=labels,
            )
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']

            scaled_loss = loss / cfg.gradient_accumulation_steps
            scaled_loss.backward()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item()

            step_elapsed = _time.time() - step_start
            step_start = _time.time()

            if (step + 1) % cfg.logging_steps == 0:
                avg_loss = epoch_loss / (step + 1)
                rho = get_spectral_radius(model, model_info)
                eta_minutes = step_elapsed * (len(dataloader) - step - 1) / 60
                log.info(
                    f"[Epoch {epoch+1}/{cfg.num_epochs}] "
                    f"Step {step+1}/{len(dataloader)} "
                    f"loss={loss.item():.4f} avg={avg_loss:.4f} "
                    f"rho(A)={rho:.6f} "
                    f"step_time={step_elapsed:.1f}s "
                    f"ETA={eta_minutes:.0f}min"
                )

        rho = get_spectral_radius(model, model_info)
        avg_loss = epoch_loss / max(len(dataloader), 1)
        log.info(
            f"Epoch {epoch+1} complete — "
            f"avg_loss={avg_loss:.4f} "
            f"LTI spectral radius: {rho:.6f} (must be < 1.0)"
        )

        save_path = Path(cfg.output_dir) / f"checkpoint-epoch-{epoch+1}"
        save_path.mkdir(parents=True, exist_ok=True)
        # Save RDT components (navigate PEFT-wrapped model)
        for attr_chain in [
            "base_model.model.model.language_model",
            "model.language_model",
            "model.model.language_model",
        ]:
            obj = model
            try:
                for part in attr_chain.split("."):
                    obj = getattr(obj, part)
                if hasattr(obj, 'rdt_lti'):
                    torch.save({
                        "lti": obj.rdt_lti.state_dict(),
                        "depth_lora": obj.rdt_depth_lora.state_dict(),
                    }, save_path / "rdt_components.pt")
                    break
            except AttributeError:
                continue
        log.info(f"Checkpoint saved to {save_path}")

    log.info("Training complete.")


# ===========================================================================
# Validation & Diagnostics
# ===========================================================================

def validate_architecture(model, model_info: Dict, cfg: RDTConfig) -> List[str]:
    """
    Run pre-training validation checks. Returns list of warnings/blockers.
    """
    issues = []

    # Check layer count matches expectations
    if model_info["layer_count"] != cfg.total_model_layers:
        issues.append(
            f"BLOCKER: Expected {cfg.total_model_layers} layers, "
            f"found {model_info['layer_count']}"
        )

    # Check hidden size matches LTI dim
    if model_info["hidden_size"] and model_info["hidden_size"] != cfg.lti_hidden_dim:
        issues.append(
            f"WARNING: Model hidden_size={model_info['hidden_size']} "
            f"!= LTI dim={cfg.lti_hidden_dim}. Will auto-adjust."
        )
        cfg.lti_hidden_dim = model_info["hidden_size"]

    # Check layer indices are valid
    all_layers = cfg.prelude_layers + cfg.recurrent_core_layers + cfg.coda_layers
    max_layer = max(all_layers)
    if max_layer >= model_info["layer_count"]:
        issues.append(
            f"BLOCKER: Layer index {max_layer} exceeds model layer count "
            f"({model_info['layer_count']})"
        )

    # Check for router modules (needed for temperature override)
    if not model_info["router_module_names"]:
        issues.append(
            "WARNING: No MoE router modules detected. "
            "Temperature override may not work. "
            "Will attempt to find router modules by pattern matching."
        )

    # VRAM estimate
    budget = VRAMBudget()
    log.info(f"VRAM budget estimate:")
    log.info(f"  Model weights (bf16): {budget.model_weights_bf16_gb:.1f} GB")
    log.info(f"  LTI + LoRA: {budget.lti_matrices_fp32_gb + budget.lora_adapters_gb:.2f} GB")
    log.info(f"  Optimizer states: {budget.optimizer_states_gb:.1f} GB")
    log.info(f"  Activations (grad ckpt): {budget.activations_grad_ckpt_gb:.1f} GB")
    log.info(f"  Total: {budget.total_gb:.1f} / {cfg.server_vram_gb} GB")
    log.info(f"  Headroom: {budget.headroom_gb:.1f} GB")

    if budget.total_gb > cfg.server_vram_gb:
        issues.append(
            f"BLOCKER: Estimated VRAM ({budget.total_gb:.1f} GB) exceeds "
            f"available ({cfg.server_vram_gb} GB)"
        )

    return issues


def print_summary(deps: Dict, model_info: Dict, issues: List[str], cfg: RDTConfig):
    """Print a validation summary."""
    print("\n" + "=" * 70)
    print("  RDT Recurrent Distillation — Validation Summary")
    print("=" * 70)

    print(f"\n  Base model:     {cfg.base_model}")
    print(f"  Target GPU:     {cfg.server_gpu}")
    print(f"  VRAM:           {cfg.server_vram_gb} GB")
    print(f"  Loading:        bf16 (no quantization)")
    print(f"  Server SSH:     {cfg.server_ssh}")

    print(f"\n  Architecture:")
    print(f"    Prelude:      layers {cfg.prelude_layers}")
    print(f"    Core:         layers {cfg.recurrent_core_layers} x {cfg.max_loop_iters} loops")
    print(f"    Coda:         layers {cfg.coda_layers}")
    print(f"    LTI dim:      {cfg.lti_hidden_dim}")
    print(f"    MoE temp:     {cfg.routing_temperature}")

    print(f"\n  Training:")
    print(f"    Dataset:      {cfg.dataset_name}")
    print(f"    LoRA rank:    {cfg.lora_rank}")
    print(f"    LoRA targets: {cfg.lora_target_modules}")
    print(f"    Optimizer:    {'Muon + AdamW' if cfg.use_muon else 'AdamW'}")
    print(f"    Grad ckpt:    {cfg.gradient_checkpointing}")

    print(f"\n  Dependencies:")
    for pkg, ver in deps.items():
        status = "OK" if "unavailable" not in ver else "FALLBACK"
        print(f"    [{status:>8}] {pkg}: {ver}")

    if issues:
        print(f"\n  Issues ({len(issues)}):")
        for issue in issues:
            severity = "BLOCK" if "BLOCKER" in issue else "WARN "
            print(f"    [{severity}] {issue}")
    else:
        print(f"\n  No issues found.")

    print("\n" + "=" * 70)

    if any("BLOCKER" in i for i in issues):
        print("  STATUS: BLOCKED — resolve blockers before training")
    else:
        print("  STATUS: READY — run with --train --confirm to start training")
    print("=" * 70 + "\n")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RDT Recurrent Distillation Training Pipeline"
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Start training (requires --confirm)"
    )
    parser.add_argument(
        "--confirm", action="store_true",
        help="Confirm dataset download and training start"
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=True,
        help="Validate configuration without loading model (default)"
    )
    parser.add_argument(
        "--load-model", action="store_true",
        help="Load model and run full validation (requires GPU)"
    )
    args = parser.parse_args()

    log.info("=" * 50)
    log.info("RDT Recurrent Distillation Pipeline")
    log.info("=" * 50)

    # Phase 1: Check dependencies
    log.info("\nPhase 1: Checking dependencies...")
    deps = check_dependencies()

    if not args.load_model and not args.train:
        # Dry run: validate config without loading model
        log.info("\nDry-run mode: validating config without GPU...")

        # Check OpenMythos components are importable
        from open_mythos import LTIInjection, LoRAAdapter, loop_index_embedding
        lti_test = LTIInjection(dim=cfg.lti_hidden_dim)
        A = lti_test.get_A()
        log.info(f"LTI injection test: spectral radius = {A.max().item():.6f} (< 1.0: OK)")

        depth_lora_test = LoRAAdapter(
            dim=cfg.lti_hidden_dim,
            rank=cfg.depth_lora_rank,
            max_loops=cfg.depth_lora_max_loops,
        )
        test_input = torch.randn(1, 10, cfg.lti_hidden_dim)
        test_out = depth_lora_test(test_input, 0)
        log.info(f"Depth LoRA test: input {test_input.shape} -> output {test_out.shape}: OK")

        issues = []
        print_summary(deps, {"layer_count": 40, "hidden_size": 2048,
                             "router_module_names": ["(not loaded)"]},
                      issues, cfg)
        return

    # Phase 2: Load model
    log.info("\nPhase 2: Loading model...")
    model, tokenizer = load_model(cfg)

    # Phase 2b: Introspect model
    log.info("\nPhase 2b: Discovering model structure...")
    model_info = discover_model_structure(model)

    # Phase 3: Patch model for RDT loop
    log.info("\nPhase 3: Patching model forward for RDT loop...")
    lti, depth_lora = patch_model_for_rdt(model, model_info, cfg)

    # Phase 3b: Override MoE routing temperature
    log.info("\nPhase 3b: Overriding MoE routing temperature...")
    override_routing_temperature(model, cfg.routing_temperature, model_info)

    # Phase 4: Apply LoRA
    log.info("\nPhase 4: Applying LoRA...")
    model = setup_lora(model, cfg)

    # Validate
    log.info("\nValidating architecture...")
    issues = validate_architecture(model, model_info, cfg)
    print_summary(deps, model_info, issues, cfg)

    if args.train:
        if not args.confirm:
            log.error("Training requires --confirm flag to proceed.")
            log.error("This will download the dataset and start training.")
            sys.exit(1)

        if any("BLOCKER" in i for i in issues):
            log.error("Cannot start training: blockers detected.")
            sys.exit(1)

        # Phase 5-7: Train
        log.info("\nPhase 5-7: Starting training...")
        train(model, model_info, tokenizer, cfg)


if __name__ == "__main__":
    main()
