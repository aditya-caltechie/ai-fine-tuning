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

### FYI: why `price` and `agent` can differ (and how they differ)

Both commands ultimately call the **same deployed Modal service method**:

- **Service**: Modal app `pricer-service`
- **Class/method**: `Pricer.price(...)` (defined in `src/pricing_service.py`)

So why might you see different answers (e.g. `299` vs `350`) for the “same” query like `"iphone 10"`?

- **Both commands run `preprocess_if_needed()` first** (in `src/main.py`).
- For raw inputs (like `"iphone 10"`) that *aren’t already structured*, `preprocess_if_needed()` calls `Preprocessor().preprocess(...)` (in `src/preprocessor.py`).
- The preprocessor uses **LiteLLM** to call a **Groq-hosted LLM** (default model: `groq/openai/gpt-oss-20b`) to rewrite your raw text into a structured format (Title/Category/Brand/Description/Details).
- If the preprocessing output changes between runs, the **final prompt sent to the fine-tuned model changes**, so the predicted price can change too.

Important detail:

- **The fine-tuned model inference is seeded** (`set_seed(42)` in `src/pricing_service.py`). That means **given the same preprocessed text**, pricing should be stable. Variability usually comes from **the preprocessing step**.

#### How `price` differs from `agent`

- **`price`**: directly instantiates the deployed Modal class and calls `Pricer.price.remote(processed_text)`.
- **`agent`**: instantiates `SpecialistAgent` (in `src/specialist_agent.py`), which is a thin wrapper that *also* calls `Pricer.price.remote(processed_text)` and adds logging.

In other words: **`agent` is a wrapper around the same remote call**; it doesn’t use a different pricing model.

#### Determinism guidance (recommended use cases)

- **Most deterministic option**: pass already-structured input so preprocessing is skipped.
- **If you want repeated runs of the same raw text to match**: the preprocessor now supports low-randomness settings and a simple on-disk cache so the same input reuses the same structured output.

Example “structured input” (preprocessor skips when it sees enough structured fields):

```bash
uv run python src/main.py price $'Title: iPhone X\nCategory: Electronics\nBrand: Apple\nDescription: Smartphone\nDetails: 64GB'
uv run python src/main.py agent $'Title: iPhone X\nCategory: Electronics\nBrand: Apple\nDescription: Smartphone\nDetails: 64GB'
```

### Required secrets / environment

- **Modal secret**: `huggingface-secret` with key `HF_TOKEN` (so Modal can download models from Hugging Face)
- **Local `.env`**: `GROQ_API_KEY=...` (used by `src/preprocessor.py` before calling Modal)

