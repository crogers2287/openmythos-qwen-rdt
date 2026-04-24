"""
Hyperparameter and architecture configuration for RDT recurrent distillation.

All magic numbers live here. Change layer ranges, loop count, routing
temperature, and training hyperparameters in one place.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RDTConfig:
    """Recurrent-Depth Transformer configuration."""

    # --- Model source ---
    base_model: str = "hesamation/Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled"
    base_model_original: str = "Qwen/Qwen3.6-35B-A3B"

    # --- Layer mapping (into OpenMythos Prelude / Core / Coda) ---
    prelude_layers: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    recurrent_core_layers: List[int] = field(default_factory=lambda: [16, 17, 18, 19])
    coda_layers: List[int] = field(default_factory=lambda: [36, 37, 38, 39])
    total_model_layers: int = 40

    # --- Recurrence ---
    max_loop_iters: int = 4  # start with 4, extrapolate to 16 at inference
    loop_embedding_dim: int = 64  # channels for sinusoidal loop-index signal

    # --- LTI injection (Parcae stability) ---
    lti_hidden_dim: int = 2048  # must match Qwen hidden_size

    # --- MoE routing ablation ---
    routing_temperature: float = 2.0

    # --- LoRA ---
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # LoRA target modules for Qwen3.5 MoE layers
    # Note: "gate" (Qwen3_5MoeTopKRouter) is NOT nn.Linear, so PEFT can't
    # target it. Router training is handled by temperature override instead.
    # DeltaNet layers use in_proj_*/out_proj, attention layers use q/k/v/o_proj.
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # full attention layers
        "out_proj",                                # DeltaNet layers
        "shared_expert_gate",                      # shared expert gating (nn.Linear)
    ])

    # --- OpenMythos depth-wise LoRA (separate from PEFT LoRA) ---
    depth_lora_rank: int = 32
    depth_lora_max_loops: int = 16

    # --- Training ---
    dataset_name: str = "Roman1111111/claude-opus-4.6-10000x"
    max_seq_length: int = 1024  # keep short to fit in VRAM with RDT loop
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 1
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    bf16: bool = True
    gradient_checkpointing: bool = True

    # --- Optimizer ---
    use_muon: bool = True  # fallback to AdamW if unavailable
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5

    # --- Hardware ---
    device: str = "cuda:0"  # single RTX 6000 Blackwell 96GB

    # --- Output ---
    output_dir: str = "./output"
    save_steps: int = 500
    logging_steps: int = 1  # log every step so we can see progress immediately

    # --- Server (Packet.ai) ---
    server_ssh: str = "ssh -p 30355 ubuntu@50.35.188.53"
    server_gpu: str = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    server_vram_gb: int = 96


@dataclass
class VRAMBudget:
    """Estimated VRAM breakdown for the RTX 6000 Blackwell 96GB."""

    model_weights_bf16_gb: float = 70.0  # 35B params * 2 bytes
    lti_matrices_fp32_gb: float = 0.01
    lora_adapters_gb: float = 0.05
    optimizer_states_gb: float = 0.2  # only trainable params
    activations_grad_ckpt_gb: float = 6.0  # estimate with gradient checkpointing
    total_gb: float = 76.26
    headroom_gb: float = 19.74  # 96 - 76.26
