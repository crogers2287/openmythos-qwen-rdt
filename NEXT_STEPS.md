# RDT Recurrent Distillation — Assumptions, Blockers, and Next Steps

## What This Is

A training pipeline that converts a pre-trained Qwen3.6-35B-A3B reasoning model
into a Recurrent-Depth Transformer (RDT) using OpenMythos components.

Instead of running all 40 transformer layers sequentially, the RDT architecture
runs 12 selected layers in a Prelude/Recurrent Core/Coda pattern, where the
Core loops up to 16 times with LTI-stabilized state injection.

## Architecture Decisions Made

1. **Unsloth + custom forward pass** (not pure OpenMythos model build).
   Unsloth loads the Qwen model with MoE Triton kernels for speed. The RDT
   loop is implemented as a custom forward-pass wrapper that uses OpenMythos
   components (LTIInjection, LoRAAdapter, loop_index_embedding) standalone.

2. **Layer mapping**: Prelude (0-3), Core (16-19 looped 16x), Coda (36-39).
   Layers 4-15 and 20-35 are skipped entirely. This is intentional in the
   RDT design — loops replace depth.

3. **MoE router fine-tuning enabled** with temperature override (2.0).
   Unsloth discourages router training by default. The temperature override
   is the safety mechanism against routing collapse.

4. **bf16 loading** on single RTX 6000 Blackwell (96GB GDDR7). No quantization.

## Hardware

- **GPU**: NVIDIA RTX PRO 6000 Blackwell Server Edition (96GB GDDR7)
- **Provider**: Packet.ai
- **SSH**: `ssh -p 30355 ubuntu@50.35.188.53`
- **VRAM budget**: ~74-78 GB used / 96 GB available (~18-22 GB headroom)

## Verified Facts (from research)

- OpenMythos v0.5.0 is real and complete. LTIInjection guarantees spectral
  radius < 1 via log-space parameterization. API verified by reading source.
- Qwen3.6-35B-A3B has 40 layers, 256 MoE experts, 8+1 per token, hidden_size=2048.
  Hybrid 3:1 DeltaNet/Attention pattern confirmed from config.
- Architecture class is `Qwen3_5MoeForConditionalGeneration` (multimodal wrapper).
  The text model is inside `model.model`.
- hesamation/Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled exists on HF.
  Fine-tuned with LoRA rank 32 on ~14,200 samples. Apache-2.0.
- Roman1111111/claude-opus-4.6-10000x exists. 9,633 rows, 27.2M tokens. MIT.
- Muon optimizer available natively in PyTorch 2.11+ (torch.optim.Muon).
  Works for 2D weight matrices. AdamW for 1D params and LTI matrices.
- Unsloth supports Qwen3.6-35B-A3B MoE. Uses FastLanguageModel.
  4-bit QLoRA NOT recommended for MoE. bf16 LoRA is the correct path.

## Known Blockers / Risks

### VRAM is tight but feasible
35B params in bf16 = ~70GB. With gradient checkpointing and only ~50M trainable
params, optimizer states are small. 16-loop BPTT with gradient checkpointing
trades compute for memory. Should fit in 96GB with ~18-22GB headroom, but first
training step will be the real test.

### RDT forward pass integration is partially scaffolded
The `RecurrentDepthWrapper.forward_rdt()` method handles the Prelude/Core/Coda
loop correctly. However, the full training loop integration (connecting the
wrapper's output back through the model's LM head for loss computation) has a
TODO. This requires GPU testing to verify the exact model structure path:
embedding → RDT wrapper → LM head.

### MoE router module discovery
The temperature override and LoRA targeting depend on finding the right module
names for the MoE gate/router. The code discovers these dynamically via
`discover_model_structure()`. If the naming convention doesn't match "gate"
or "router", the override will log a warning and skip. GPU testing needed.

### Layer output format
Qwen layers may return tuples, dataclasses, or other structures. The wrapper
handles tuples and plain tensors. If the actual model uses a different return
format, `forward_rdt()` will need adjustment. GPU testing needed.

### Unsloth on server
Unsloth needs to be installed on the Packet.ai server. The script falls back
to transformers + PEFT if Unsloth is unavailable, but you lose the MoE Triton
kernel speedup.

## What Needs User Confirmation

1. **Dataset download**: The script will download Roman1111111/claude-opus-4.6-10000x
   from Hugging Face when you run with `--train --confirm`. Not done yet.

2. **Model download**: The script will download the ~70GB model weights from HF
   when you run with `--load-model` or `--train`. Not done yet.

3. **Server setup**: Install dependencies on the Packet.ai server:
   ```bash
   ssh -p 30355 ubuntu@50.35.188.53
   pip install -r requirements.txt
   pip install unsloth  # for MoE Triton kernels
   ```

## Next Steps (in order)

1. **Copy repo to server** and install dependencies including Unsloth
2. **Run `--load-model`** to test model loading and structure discovery on GPU
3. **Verify RDT forward pass** with a single batch (fix any layer output format issues)
4. **Run `--train --confirm`** to start the recurrent distillation training
5. **Monitor**: LTI spectral radius should stay < 1.0; loss should decrease; no OOM

## Files

| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependencies |
| `config.py` | All hyperparameters and architecture config |
| `train_recurrent_distillation.py` | Main pipeline script |
| `notes/user_prompt.txt` | Original task specification |
| `notes/followup_updated_prompt.txt` | Updated scope (A100 → RTX 6000) |
| `notes/context7_summary.txt` | Library research notes |
| `notes/server_connection_details_from_image.txt` | Packet.ai server details |
