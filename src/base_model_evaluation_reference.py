#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Reference script: Evaluate a BASE model (no LoRA) on the pricing task.

This is a cleaned-up, notebook-free version of "Week 7 Day 2: base model evaluation".
It is intentionally **not meant to be run locally** in this repo; it exists to
explain the steps and concepts in plain Python.

Goal
----
Load a *base* LLM (quantized) and check how well it can predict prices *without*
task-specific fine-tuning.

Why this matters
----------------
- The base model is general-purpose; it often **doesn't follow the exact output
  format** we want (e.g., it might produce extra words, or not output a clean number).
- Even if it outputs a number, accuracy is usually worse than a LoRA fine-tuned model.
- This motivates LoRA/QLoRA fine-tuning (see `lora_training_reference.py`).

Hardware notes (T4 vs A100)
---------------------------
- On a **T4**, bf16 isn't natively supported; use fp16 compute dtype.
- On an **A100**, bf16 is typically preferred.
"""

from __future__ import annotations

import os
import re


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_MODEL = "meta-llama/Llama-3.2-3B"
PROJECT_NAME = "price"

# Dataset contains items with:
# - item["prompt"]: the full prompt ending with "Price is $"
# - item["completion"]: the ground-truth price as a string (e.g. "219.0")
DATA_USER = "ed-donner"
LITE_MODE = True
DATASET_NAME = f"{DATA_USER}/items_prompts_lite" if LITE_MODE else f"{DATA_USER}/items_prompts_full"

# Hyper-parameters
QUANT_4_BIT = True

# Inference settings (small: we only need the price)
MAX_NEW_TOKENS = 8


def parse_first_number(text: str) -> float | None:
    """
    Pull the first numeric value out of model output.

    Base models often produce things like:
      "349.99. What is the price"
    so we regex the first number instead of expecting a perfect format.
    """
    cleaned = text.replace(",", "")
    match = re.search(r"[-+]?\d*\.\d+|\d+", cleaned)
    return float(match.group()) if match else None


def main() -> None:
    # 1) Log in to Hugging Face (so we can download the base model)
    #
    # In Colab you can also do:
    #   from google.colab import userdata
    #   hf_token = userdata.get("HF_TOKEN")
    from huggingface_hub import login

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Missing HF_TOKEN. Export it to authenticate with Hugging Face.")
    login(hf_token, add_to_git_credential=True)

    # 2) Load our data
    from datasets import load_dataset

    dataset = load_dataset(DATASET_NAME)
    train = dataset["train"]
    val = dataset["val"]
    test = dataset["test"]

    # 3) Pick the right quantization (4-bit keeps memory low)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # T4 (SM75) => fp16 compute; A100 (SM80+) => bf16 compute
    major_cc, _minor_cc = torch.cuda.get_device_capability()
    compute_dtype = torch.bfloat16 if major_cc >= 8 else torch.float16

    if QUANT_4_BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
        )
    else:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=compute_dtype,
        )

    # 4) Load the tokenizer and the BASE model (no LoRA adapters)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
    )
    base_model.generation_config.pad_token_id = tokenizer.pad_token_id

    print(f"Memory footprint: {base_model.get_memory_footprint() / 1e9:.1f} GB")

    # 5) Predict with the BASE model (simple helper)
    def model_predict(item) -> str:
        inputs = tokenizer(item["prompt"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            output_ids = base_model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, prompt_len:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Try one example
    pred = model_predict(test[0])
    print("Model raw output:", repr(pred))
    print("Parsed number:", parse_first_number(pred), "| Ground truth:", test[0]["completion"])

    # Takeaway:
    # This works, but it's usually not accurate/consistent enough.
    # LoRA/QLoRA fine-tuning is used to specialize the model for this pricing task.


if __name__ == "__main__":
    main()

