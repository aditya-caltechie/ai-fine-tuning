#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""

This file is intentionally **not meant to be run locally** in this repo.
It exists to explain the *conceptual steps* of LoRA / QLoRA training in a clean,
plain-Python format (no notebook cells, no Colab `!` magics).

High-level steps:
1) Choose a base model and a dataset of instruction-style prompts.
2) Configure quantization (QLoRA = 4-bit base model + LoRA adapters).
3) Load tokenizer + quantized base model (works on a T4 for lite settings).
4) Configure LoRA target modules and training hyperparameters.
5) Train using TRL's SFTTrainer and push adapter weights to Hugging Face.

Notes on hardware:
- A **T4** can handle the **LITE** configuration (small dataset, small batch).
- An **A100** is recommended for larger datasets / larger LoRA ranks / longer runs.

Credentials:
- Hugging Face: set `HF_TOKEN` (to download base model + push trained adapters)
"""

from __future__ import annotations

import os
from datetime import datetime


# ---------------------------------------------------------------------------
# Global configuration (kept as constants for readability)
# ---------------------------------------------------------------------------

# Base model: small enough for QLoRA on a T4 (lite settings).
BASE_MODEL = "meta-llama/Llama-3.2-3B"
PROJECT_NAME = "price"

# Dataset: instruction-style data with item descriptions and "Price is $" target.
# In the original course material, these live on Hugging Face under `ed-donner/...`.
DATA_USER = "ed-donner"
LITE_MODE = True
DATASET_NAME = f"{DATA_USER}/items_prompts_lite" if LITE_MODE else f"{DATA_USER}/items_prompts_full"

# Run naming: timestamped so each training run is distinct on the Hub.
RUN_NAME = f"{datetime.now():%Y-%m-%d_%H.%M.%S}" + ("-lite" if LITE_MODE else "")
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"

# Where to push the trained adapter weights.
HF_USER = "adityakrverma"  # replace with your HF username if you actually run this
HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_RUN_NAME}"

# Training knobs:
EPOCHS = 1 if LITE_MODE else 3
BATCH_SIZE = 32 if LITE_MODE else 256
MAX_SEQUENCE_LENGTH = 128
GRADIENT_ACCUMULATION_STEPS = 1

# QLoRA / LoRA knobs:
# - QLoRA: quantize base model to 4-bit (bitsandbytes), then train small LoRA adapters.
QUANT_4_BIT = True
LORA_R = 32 if LITE_MODE else 256
LORA_ALPHA = LORA_R * 2
LORA_DROPOUT = 0.1

# LoRA target modules (where adapters are injected).
# LITE targets attention only; larger runs often include MLP blocks too.
ATTENTION_LAYERS = ["q_proj", "v_proj", "k_proj", "o_proj"]
MLP_LAYERS = ["gate_proj", "up_proj", "down_proj"]
TARGET_MODULES = ATTENTION_LAYERS if LITE_MODE else ATTENTION_LAYERS + MLP_LAYERS

# Optimizer / scheduler:
LEARNING_RATE = 1e-4
WARMUP_RATIO = 0.01
LR_SCHEDULER_TYPE = "cosine"
WEIGHT_DECAY = 0.001
OPTIMIZER = "paged_adamw_32bit"

# Logging / checkpoints:
VAL_SIZE = 500 if LITE_MODE else 1000
LOG_STEPS = 5 if LITE_MODE else 10
SAVE_STEPS = 100 if LITE_MODE else 200


def login_to_huggingface() -> None:
    """Authenticate so we can pull/push models from the Hugging Face Hub."""
    from huggingface_hub import login

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Missing HF_TOKEN. Export it to authenticate with Hugging Face.")

    # add_to_git_credential=True is convenient in Colab; harmless elsewhere.
    login(hf_token, add_to_git_credential=True)


def pick_precision() -> dict:
    """
    Decide whether we can use bf16.

    - A100 (Ampere/SM80+) supports bf16 well.
    - T4 (SM75) does NOT have native bf16; you typically use fp16.
    """

    import torch

    capability = torch.cuda.get_device_capability()
    use_bf16 = capability[0] >= 8
    return {"use_bf16": use_bf16}


def load_dataset_splits(dataset_name: str, val_size: int):
    """Load train/val/test splits from Hugging Face Datasets."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_name)
    train = dataset["train"]
    val = dataset["val"].select(range(val_size))
    test = dataset["test"]
    return train, val, test


def load_tokenizer_and_quantized_model(base_model: str, quant_4bit: bool, use_bf16: bool):
    """
    Load:
    - Tokenizer
    - Base model in 4-bit or 8-bit (bitsandbytes) with device_map="auto"

    This is the heart of QLoRA: the base model weights are kept quantized/frozen,
    and we train only small LoRA adapter matrices.
    """

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if quant_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto",
    )
    base.generation_config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, base


def build_lora_config(lora_r: int, lora_alpha: int, lora_dropout: float, target_modules: list[str]):
    """Define where and how LoRA adapters will be injected."""
    from peft import LoraConfig

    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )


def build_training_config(use_bf16: bool):
    """
    TRL's SFTConfig describes training hyperparameters and hub/checkpointing behavior.

    Key idea:
    - We *fine-tune* using supervised fine-tuning (SFT) on instruction-style prompts.
    - We save and push the resulting adapter weights to the Hub periodically.
    """
    from trl import SFTConfig

    return SFTConfig(
        output_dir=PROJECT_RUN_NAME,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        optim=OPTIMIZER,
        save_steps=SAVE_STEPS,
        save_total_limit=10,
        logging_steps=LOG_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fp16=not use_bf16,
        bf16=use_bf16,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=WARMUP_RATIO,
        group_by_length=True,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        run_name=RUN_NAME,
        max_length=MAX_SEQUENCE_LENGTH,
        save_strategy="steps",
        hub_strategy="every_save",
        push_to_hub=True,
        hub_model_id=HUB_MODEL_NAME,
        hub_private_repo=True,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
    )


def train_model(model, train_dataset, eval_dataset, lora_cfg, sft_cfg):
    """
    Run supervised fine-tuning (SFT) with LoRA adapters.

    Under the hood:
    - The base model remains quantized and mostly frozen.
    - The trainer inserts LoRA adapters and trains only those parameters.
    """
    from trl import SFTTrainer

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_cfg,
        args=sft_cfg,
    )

    # Kick off training. On Colab, this can run for minutes to hours.
    trainer.train()
    return trainer


def push_to_hub(trainer, project_run_name: str) -> None:
    """Push the trained (adapter) weights to Hugging Face Hub."""
    trainer.model.push_to_hub(project_run_name, private=True)
    print(f"Saved to the hub: {project_run_name}")


def main() -> None:
    # Login steps are shown for completeness. In a true "reference-only" read,
    # you can treat them as optional prerequisites.
    login_to_huggingface()

    # Decide whether we can use bf16 for training or not
    precision = pick_precision()
    use_bf16 = precision["use_bf16"]

    # 1) Load dataset
    train, val, _test = load_dataset_splits(DATASET_NAME, VAL_SIZE)

    # 2) Load tokenizer + quantized base model (QLoRA)
    _tokenizer, base_model = load_tokenizer_and_quantized_model(
        BASE_MODEL, QUANT_4_BIT, use_bf16
    )

    # 3) Define LoRA adapters (what to train)
    lora_cfg = build_lora_config(
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
    )

    # 4) Define training hyperparameters and hub behavior
    sft_cfg = build_training_config(use_bf16=use_bf16)

    # 5) Train adapters
    trainer = train_model(
        model=base_model,  # quantized base model
        train_dataset=train,  # training dataset
        eval_dataset=val,  # validation dataset
        lora_cfg=lora_cfg,  # LoRA adapters to train
        sft_cfg=sft_cfg,  # training hyperparameters
    )

    # 6) Push trained adapters to Hugging Face
    push_to_hub(trainer, PROJECT_RUN_NAME)


if __name__ == "__main__":
    main()

