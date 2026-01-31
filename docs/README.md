## Pricing service overview

`src/inference/pricing_service.py` defines a **Modal-deployed** pricing service (Modal app name: `pricer-service`).

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
uv run python src/inference.py deploy

# Call the service (auto-preprocesses input using Groq first)
uv run python src/inference.py price "iphone 10"

# View remote container logs (this is where `print()` from the service shows up)
uv run modal app logs pricer-service --timestamps
```

### FYI: why `price` can differ between runs (raw text inputs)

The `price` command ultimately calls the **same deployed Modal service method**:

- **Service**: Modal app `pricer-service`
- **Class/method**: `Pricer.price(...)` (defined in `src/inference/pricing_service.py`)

So why might you see different answers (e.g. `299` vs `350`) for the “same” raw query like `"iphone 10"`?

- `price` runs `preprocess_if_needed()` first (in `src/inference.py`).
- For raw inputs (like `"iphone 10"`) that *aren’t already structured*, `preprocess_if_needed()` calls `Preprocessor().preprocess(...)` (in `src/inference/preprocessor.py`).
- The preprocessor uses **LiteLLM** to call a **Groq-hosted LLM** (default model: `groq/openai/gpt-oss-20b`) to rewrite your raw text into a structured format (Title/Category/Brand/Description/Details).
- If the preprocessing output changes between runs, the **final prompt sent to the fine-tuned model changes**, so the predicted price can change too.

Important detail:

- **The fine-tuned model inference is seeded** (`set_seed(42)` in `src/inference/pricing_service.py`). That means **given the same preprocessed text**, pricing should be stable. Variability usually comes from **the preprocessing step**.

#### Determinism guidance (recommended use cases)

- **Most deterministic option**: pass already-structured input so preprocessing is skipped.
- **If you want repeated runs of the same raw text to match**: the preprocessor now supports low-randomness settings and a simple on-disk cache so the same input reuses the same structured output.

Example “structured input” (preprocessor skips when it sees enough structured fields):

```bash
uv run python src/inference.py price $'Title: iPhone X\nCategory: Electronics\nBrand: Apple\nDescription: Smartphone\nDetails: 64GB'
```

### Required secrets / environment

- **Modal secret**: `huggingface-secret` with key `HF_TOKEN` (so Modal can download models from Hugging Face)
- **Local `.env`**: `GROQ_API_KEY=...` (used by `src/inference/preprocessor.py` before calling Modal)

