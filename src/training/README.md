# Training vs Fine-tuning (in the context of *your repos*)

This folder is **reference-only**. In this repo, production runtime lives in `src/inference/` (Modal service + CLI), while “training/fine-tuning” material is here mainly to explain concepts and workflows.

## The short version

- **Training from scratch**: you build a model (architecture) and learn weights from random initialization on your dataset.
- **Fine-tuning**: you start from a strong pre-trained model (LLM) and adapt it to your task with additional training.
- **LoRA / QLoRA**: a common, practical fine-tuning style for LLMs where you train small adapter weights (often with the base model quantized).

## How your two repos map to these ideas

### Example of “training from scratch” (your `ai-deep-learning` repo)

In your other repo (`aditya-caltechie/ai-deep-learning`), you’re doing classic training-from-scratch:

- **You define the model architecture** (a residual MLP) and initialize it with random weights.
- **You run a training loop** (optimize weights end-to-end) on a Hugging Face dataset like `ed-donner/items_lite` / `ed-donner/items_full`.
- **You save a local weights file** (a `.pth` checkpoint) and then evaluate.

Repo link: `https://github.com/aditya-caltechie/ai-deep-learning`

### Example of “fine-tuning” (this `ai-fine-tuning` repo)

This repo is the opposite approach:

- We start with a **pre-trained LLM** (example used throughout: `meta-llama/Llama-3.2-3B`).
- We adapt it for the pricing task using **QLoRA / LoRA adapters** (small trainable weights).
- At inference time, we load **base model + adapter** together.

Relevant places in *this* repo:

- **Colab notebooks (hands-on fine-tuning)**: `src/fine_tuning/notebooks/`
  - `1_basemodel_evaluation.ipynb` (baseline eval)
  - `2_fine-tuning_via_QLORA.ipynb` (train adapters)
  - `3_testing_fine-tuned-model.ipynb` (evaluate fine-tuned)
- **Plain-Python walkthrough (reference-only)**: `src/fine_tuning/notebooks/lora_training_reference.py`
- **Serving (loads base + adapter)**: `src/inference/pricing_service.py`
- **CLI entrypoint (deploy/price/agent/logs)**: `src/inference.py`

## What’s actually different (mechanically)

### Starting point

- **Training from scratch**: random weights + your architecture.
- **Fine-tuning**: pre-trained weights + small task dataset.

### What weights get updated

- **Training from scratch**: typically *all* parameters are trainable.
- **Fine-tuning (full)**: often all parameters are trainable (but with a small learning rate).
- **Fine-tuning (LoRA/QLoRA)**: mostly **only adapter weights** are trained; base model stays frozen (and may be quantized).

### Artifacts you produce

- **Training from scratch** (often): a local checkpoint like `model.pth`.
- **LoRA/QLoRA fine-tuning** (often): a small **adapter** you can push to Hugging Face; inference loads adapter on top of the base model.

## When to use which (practical guidance)

### Use training from scratch when…

- No pre-trained model fits your constraints (data modality, licensing, architecture).
- You have **lots of data** and **enough compute** to train a good model.
- You want full control over learned representations (and accept higher engineering effort).

### Use fine-tuning when…

- A base model already “mostly works”, but you want it to match your domain/task better.
- You have **limited compute/time** and a smaller curated dataset.
- You want to ship a small artifact (LoRA adapter) and keep the base model unchanged.

## Where to go next (repo links)

- **Fine-tuning notebooks (Colab workflow)**: `src/fine_tuning/README.md`
- **Fine-tuning notebooks (files)**: `src/fine_tuning/notebooks/`
- **Inference + deploy CLI**: `src/inference.py`
- **Modal service code**: `src/inference/pricing_service.py`

