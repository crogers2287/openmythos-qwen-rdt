#!/usr/bin/env python3
"""
Recurrent Distillation — Phase 1: SFT Training
================================================
Train LoRA adapters on structured reasoning traces using Unsloth's
optimized pipeline. No custom forward pass — let Unsloth do what it
does best.

The RDT architecture (Prelude/Core/Coda loop with LTI injection)
is applied at inference time in a separate script, using the LoRA
weights trained here.

Usage:
    python train_sft.py              # start training
    python train_sft.py --dry-run    # validate config only
"""

# Unsloth must be imported first
try:
    import unsloth
except ImportError:
    pass

import argparse
import logging
import sys
import time
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sft")


# ===========================================================================
# Config
# ===========================================================================

BASE_MODEL = "hesamation/Qwen3.6-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled"
DATASET = "Roman1111111/claude-opus-4.6-10000x"
MAX_SEQ_LENGTH = 1024
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "out_proj", "shared_expert_gate",
]
OUTPUT_DIR = "./output"
NUM_EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM = 4
LR = 2e-4


# ===========================================================================
# Model Loading
# ===========================================================================

def load_model():
    from unsloth import FastLanguageModel

    log.info(f"Loading {BASE_MODEL} via Unsloth (bf16)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        device_map={"": "cuda:0"},
    )
    log.info(f"Model loaded.")

    log.info("Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        target_modules=LORA_TARGETS,
        bias="none",
        use_gradient_checkpointing="unsloth",
        max_seq_length=MAX_SEQ_LENGTH,
    )
    log.info(f"LoRA applied. Targets: {LORA_TARGETS}")

    return model, tokenizer


# ===========================================================================
# Dataset
# ===========================================================================

def load_and_format_dataset(tokenizer):
    from datasets import load_dataset

    log.info(f"Loading dataset: {DATASET}")
    dataset = load_dataset(DATASET, split="train")
    log.info(f"Dataset: {len(dataset)} examples")

    # Get the underlying text tokenizer (not the multimodal processor)
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    if not hasattr(text_tok, "encode"):
        text_tok = getattr(text_tok, "tokenizer", text_tok)
    if text_tok.pad_token is None:
        text_tok.pad_token = text_tok.eos_token

    def format_example(example):
        messages = example.get("messages", example.get("conversations", []))
        if isinstance(messages, list) and len(messages) > 0:
            try:
                text = text_tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            except Exception:
                parts = []
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
                text = "\n".join(parts)
        else:
            text = example.get("text", str(example))
        return {"text": text}

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    log.info(f"Dataset formatted: {len(dataset)} examples")
    return dataset, text_tok


# ===========================================================================
# Training
# ===========================================================================

def train(model, tokenizer, dataset, text_tok):
    from trl import SFTTrainer, SFTConfig

    log.info(f"Starting SFT training: {NUM_EPOCHS} epochs, "
             f"batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM}, "
             f"lr={LR}, max_seq={MAX_SEQ_LENGTH}")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=1,
        save_steps=500,
        save_total_limit=2,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=text_tok,
        train_dataset=dataset,
        args=training_args,
    )

    log.info("Training...")
    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start

    log.info(f"Training complete in {elapsed/3600:.1f} hours")
    log.info(f"Final loss: {result.training_loss:.4f}")

    # Save
    model.save_pretrained(f"{OUTPUT_DIR}/final")
    text_tok.save_pretrained(f"{OUTPUT_DIR}/final")
    log.info(f"Model saved to {OUTPUT_DIR}/final")

    return result


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    log.info("=" * 50)
    log.info("RDT Recurrent Distillation — SFT Phase")
    log.info("=" * 50)

    model, tokenizer = load_model()
    dataset, text_tok = load_and_format_dataset(tokenizer)

    if args.dry_run:
        log.info("Dry run complete. Model loaded, dataset ready.")
        log.info(f"  {len(dataset)} examples, {MAX_SEQ_LENGTH} max seq len")
        return

    train(model, tokenizer, dataset, text_tok)


if __name__ == "__main__":
    main()
