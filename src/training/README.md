# Training vs Fine-tuning (what’s the difference?)

This folder is **reference-only**. In this repo, the production runtime lives in `src/inference/`, while “training/fine-tuning” material is kept for learning and documentation.

## Definitions (plain English)

- **Training a model (from scratch / pretraining)**  
  You start with random (or mostly untrained) weights and teach the model general language (or multimodal) behavior from a *very large* dataset.  
  This is how foundation models are created.

- **Fine-tuning a model**  
  You start with an already-trained base model and adapt it to your task/domain by continuing training on a *much smaller* dataset.

- **LoRA / QLoRA fine-tuning (common in practice)**  
  A parameter-efficient fine-tuning method: instead of updating all model weights, you train small “adapter” matrices (LoRA).  
  With **QLoRA**, the base model is quantized (e.g., 4-bit) to reduce memory, while training the LoRA adapters.

## What changes when you “train” vs “fine-tune”?

- **Goal**
  - **Training**: learn broad capabilities from general data.
  - **Fine-tuning**: specialize an existing model for a narrower behavior (task, style, domain).

- **Data**
  - **Training**: typically billions+ tokens, diverse sources.
  - **Fine-tuning**: often thousands to millions of tokens; curated and task-focused.

- **Compute / cost**
  - **Training**: extremely expensive (large clusters, long runs).
  - **Fine-tuning**: much cheaper; LoRA/QLoRA can be done with a single GPU (depending on model size).

- **Risk**
  - **Training**: highest risk (you can end up with a weak model if data/compute aren’t sufficient).
  - **Fine-tuning**: lower risk; main failure modes are overfitting, reduced generality, or “forgetting” behaviors.

- **Output artifacts**
  - **Training**: a full new base checkpoint.
  - **Fine-tuning**: either a new full checkpoint **or** (LoRA) a small adapter you load on top of the base model.

## When to use which (rules of thumb)

### Prefer fine-tuning when…

- You already have a strong base model that “mostly works”.
- You need **domain adaptation** (e.g., your product catalog, pricing style, internal terminology).
- You need **consistent formatting** or structured outputs.
- You have **limited compute** and a modest dataset.

In this repo’s context: if you want a better price predictor for your product descriptions, **fine-tuning (LoRA/QLoRA)** is the typical choice.

### Consider training from scratch only when…

- You need a new foundation model because existing base models can’t meet requirements.
- You have access to **massive datasets + substantial compute**.
- You need full control over training data, tokenizer, architecture, and licensing constraints.

For most application teams (and for this repo), **training from scratch is not the intended path**.

## Quick decision checklist

If any of these are true, choose **fine-tuning**:

- “The base model is close; I just need it to behave more like my examples.”
- “I want better accuracy on my domain-specific distribution.”
- “I want a small artifact to ship (LoRA adapter) rather than retraining the whole model.”

Choose **training from scratch** only if:

- “No available base model can be adapted to my needs.”
- “I have enough data/compute to pretrain effectively.”

## Where to look next in this repo

- **LoRA/QLoRA walkthrough (reference)**: `src/fine_tuning/notebooks/lora_training_reference.py`
- **Serving / inference (Modal)**: `src/inference/pricing_service.py` (loads base model + LoRA adapter at container start)

