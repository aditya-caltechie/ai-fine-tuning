## AGENTS — Project overview (ai-fine-tuning)

This document is a **contributor-oriented map** of this repo: where things live, what each module does, and the common run commands.

---

### Repository layout

```
ai-fine-tuning/
├── README.md
├── docs/
│   ├── README.md                     # Deeper notes about the Modal pricing service
│   ├── inference.md                  # Inference workflow + diagrams
│   └── fine_tuning.md                # Fine-tuning notes + diagrams
├── pyproject.toml                    # Dependencies (managed by uv)
├── uv.lock
├── src/
│   ├── inference.py                  # CLI entrypoint (deploy / price)
│   ├── inference/
│   │   ├── pricing_service.py        # Modal app + Pricer.price(...) (inference/serving)
│   │   ├── preprocessor.py           # LLM-based input structuring (Groq via LiteLLM)
│   │   └── logger.py                 # Minimal logger used by the CLI
│   ├── fine_tuning/
│   │   └── notebooks/                # QLoRA/LoRA notebooks + reference scripts
│   └── training/                     # Reference-only training notes
└── tests/
    ├── __init__.py                   # Makes `tests/` a package (enables unittest discovery)
    ├── unit/
    │   ├── __init__.py
    │   ├── test_inference_cli.py     # CLI + preprocess unit tests
    │   └── test_compare_prices_parsing.py
    └── integration/
        ├── __init__.py
        ├── compare_prices.py         # Script: compare HF dataset vs CLI `price` output
        └── test_compare_prices_integration.py  # Skipped unless RUN_INTEGRATION_TESTS=1
```

Notes:

- This repo runs as **scripts under `src/`**, not as a published Python package.
- `src/training/` and `src/fine_tuning/` are primarily **reference / learning** material.

---

### Key components (where to start reading)

- **Entrypoint (CLI)**: `src/inference.py`
  - `deploy`: deploys the Modal app defined in `src/inference/pricing_service.py`
  - `price`: calls the deployed `Pricer.price(...)` directly (after preprocessing if needed)

- **Inference/service (Modal)**: `src/inference/pricing_service.py`
  - Loads tokenizer + quantized base model + LoRA adapter weights on container start
  - Exposes `Pricer.price(description) -> float` as a Modal method (inference)

- **Preprocessing (important for “surprising” outputs)**: `src/inference/preprocessor.py`
  - Converts raw text (e.g. `"iphone 10"`) into a structured “Title/Category/Brand/…” block
  - Uses LiteLLM (default: Groq model `groq/openai/gpt-oss-20b`)
  - Uses best-effort deterministic settings + a small cache to reduce run-to-run variance

---

### Running locally (common commands)

Run from repo root (recommended via `uv`):

```bash
# Install deps
uv sync

# Deploy the Modal service (must do once, or after changing service code)
uv run python src/inference.py deploy

# Call the service (auto-preprocesses raw text first)
uv run python src/inference.py price "iphone 10"

# Stream remote container logs (print() from the Modal container)
uv run modal app logs pricer-service --timestamps

# Run unit tests (CI runs this)
uv run python -m unittest discover -s tests -p "test_*.py"

# Compare dataset ground-truth vs model output (defaults to 5 test rows)
uv run python tests/integration/compare_prices.py

# Run integration tests (skipped by default; requires network + deployed Modal app)
RUN_INTEGRATION_TESTS=1 uv run python -m unittest discover -s tests -p "test_*.py"
```

---

### Configuration & environment

Create a `.env` in the repo root (or export env vars). Key settings:

- **Modal**
  - You must be authenticated with Modal (`modal token set ...`)
  - The service expects a Modal secret named **`huggingface-secret`** containing `HF_TOKEN`

- **Groq / preprocessing**
  - `GROQ_API_KEY`: used by `src/inference/preprocessor.py`
  - `PRICER_PREPROCESSOR_MODEL`: override default preprocessor model (defaults to `groq/openai/gpt-oss-20b`)

- **Preprocessor determinism (best-effort)**
  - `PRICER_PREPROCESSOR_TEMPERATURE` (default `0`)
  - `PRICER_PREPROCESSOR_TOP_P` (default `1`)
  - `PRICER_PREPROCESSOR_SEED` (default `42`)
  - `PRICER_PREPROCESSOR_CACHE` (default enabled)
  - Cache file: `.cache/pricer_preprocess_cache.json`

---

### Common pitfalls

- **Forgetting to deploy**: `price`/`agent` require the Modal app to be deployed first.
- **Different answers for the same raw query**:
  - Both `price` and `agent` call the same remote method, but raw input goes through an LLM preprocessor first.
  - Prefer passing already-structured input if you want maximum stability.
- **Missing secrets/keys**:
  - `huggingface-secret` (Modal) must contain `HF_TOKEN`
  - `.env` needs `GROQ_API_KEY` for preprocessing

