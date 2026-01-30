## Pricing service overview

`src/pricing_service.py` defines a **Modal-deployed** pricing service (Modal app name: `pricer-service`).

### What it does (high level)

- **Step 1 — Infrastructure**: Creates a Modal app + container image with ML deps and a persistent HF cache volume.
- **Step 2 — Load models once per container** (`@modal.enter()`):
  - Loads the **tokenizer** (needed to convert prompt text ↔ tokens).
  - Loads the **base model** (4-bit quantized).
  - Loads the **LoRA/PEFT adapter weights** and applies them on top of the base model (this is the fine-tuning).
- **Step 3 — Inference** (`Pricer.price(description)`):
  - Builds a prompt
  - Tokenizes it
  - Generates a few tokens for the price
  - Parses the numeric value and returns a `float`

### How to run

From repo root:

```bash
# Deploy the service to Modal
uv run python src/main.py deploy

# Call the service (auto-preprocesses input using Groq first)
uv run python src/main.py price "iphone 10"

# Or call via the agent wrapper (also auto-preprocesses first)
uv run python src/main.py agent "iphone 10"

# View remote container logs (this is where `print()` from the service shows up)
uv run python src/main.py logs
```

### Required secrets / environment

- **Modal secret**: `huggingface-secret` with key `HF_TOKEN` (so Modal can download models from Hugging Face)
- **Local `.env`**: `GROQ_API_KEY=...` (used by `src/preprocessor.py` before calling Modal)

