## AGENTS — Project overview (ai-fine-tuning)

This document is a **contributor-oriented map** of this repo: where things live, what each module does, and the common run commands.

---

### Repository layout

```
ai-fine-tuning/
├── README.md
├── docs/
│   └── README.md                     # Deeper notes about the Modal pricing service
├── pyproject.toml                    # Dependencies (managed by uv)
├── uv.lock
└── src/
    ├── main.py                       # CLI entrypoint (deploy / price / agent / logs)
    ├── inference/
    │   ├── pricing_service.py        # Modal app + Pricer.price(...) (inference/serving)
    │   ├── preprocessor.py           # LLM-based input structuring (Groq via LiteLLM)
    │   ├── specialist_agent.py       # Thin wrapper around Pricer.price.remote(...)
    │   └── agent.py                  # Minimal logging base class
    ├── evaluation/
    │   └── base_model_evaluation.py  # Reference-only: baseline eval (no LoRA)
    └── training/
        └── lora_training_reference.py # Reference-only: LoRA/QLoRA training walkthrough
```

Notes:

- This repo runs as **scripts under `src/`**, not as a published Python package.
- `src/training/` and `src/evaluation/` are **reference-only** (kept for learning / documentation).

---

### Key components (where to start reading)

- **Entrypoint (CLI)**: `src/main.py`
  - `deploy`: deploys the Modal app defined in `src/inference/pricing_service.py`
  - `price`: calls the deployed `Pricer.price(...)` directly
  - `agent`: calls the same remote method via `SpecialistAgent` (adds logging)
  - `logs`: streams Modal container logs

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
uv run python src/main.py deploy

# Call the service (auto-preprocesses raw text first)
uv run python src/main.py price "iphone 10"

# Same remote call, via the agent wrapper (adds logs)
uv run python src/main.py agent "iphone 10"

# Stream remote container logs (print() from the Modal container)
uv run python src/main.py logs
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

