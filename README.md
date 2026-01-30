# ai-fine-tuning

## Run Week 8 Day 1 (`src/inference.py`)

Prereqs:
- `uv` installed
- Modal set up (`uv run modal token set ...`)
- Modal Secret created for Hugging Face token (usually named **`huggingface-secret`** with key `HF_TOKEN`)
- `.env` contains your Groq key:
  - `GROQ_API_KEY=...`

Service files:
- **Modal service**: `src/inference/pricing_service.py` (deployed app name: `pricer-service`)
- **Client runner**: `src/inference.py`
- **Inference helpers**: `src/inference/` (agent wrapper + preprocessor)
- **Reference-only training**: `src/training/`
- **Reference-only evaluation**: `src/evaluation/`

Commands (run from repo root):

```bash
# Step 1 (MUST): Deploy the Modal app (do this once, or whenever you change the service code)
uv run python src/inference.py deploy

# Step 2 (choose one): Call the service directly OR use the agent (agent calls the same service)
uv run python src/inference.py price "raw text here"
uv run python src/inference.py agent "raw text here"

# Optional: watch the remote Modal logs (this is where container print() output shows up)
uv run python src/inference.py logs
```

Optional: set a different default preprocess model:
- Add to `.env`: `PRICER_PREPROCESSOR_MODEL=groq/openai/gpt-oss-20b` (or another LiteLLM-supported model)

Notes:
- `price` and `agent` both call the same deployed Modal method (`Pricer.price`). If you pass raw text (like `"iphone 10"`), both commands run an LLM-based preprocessor first, which can affect the final price.
- To make repeated runs stable for the same raw input, the preprocessor uses best-effort deterministic settings and a small on-disk cache at `.cache/pricer_preprocess_cache.json`.
  - You can control this with env vars like `PRICER_PREPROCESSOR_TEMPERATURE`, `PRICER_PREPROCESSOR_SEED`, `PRICER_PREPROCESSOR_CACHE`.

## Repo structure (quick guide)

- **`src/inference/`**: all runtime inference/service code (Modal service, agent wrapper, preprocessing)
- **`src/inference.py`**: CLI entrypoint; kept at `src/` so commands stay the same
- **`src/training/`**: reference-only scripts explaining LoRA/QLoRA training (not intended to run locally on Mac)
- **`src/evaluation/`**: reference-only scripts showing baseline evaluation (base model, no LoRA)

## Convert a Colab notebook to a commented Python script (optional)

Because Google Drive/Colab links often require sign-in, the simplest workflow is:

- In Colab: **File → Download → Download `.ipynb`**
- Put the downloaded notebook into this repo (example: `my_notebook.ipynb`)
- Convert it using your preferred tool (for example, `jupytext`) or a simple notebook-to-script exporter.