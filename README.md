# ai-fine-tuning

This repo shows an end-to-end LLM fine-tuning + serving workflow for a product pricing task: run QLoRA/LoRA fine-tuning notebooks in src/fine_tuning/, then load the adapter for inference.
It deploys a Modal pricing service in src/inference/ and provides a CLI (src/inference.py) to deploy, query prices, use an agent wrapper, and stream logs.

## Architecture (high level)

```mermaid
flowchart LR
  %% ---------------------------
  %% Training (reference-only)
  %% ---------------------------
  subgraph Train["Fine-tuning (reference-only)"]
    TCODE["src/fine_tuning<br/>notebooks + lora_training_reference.py"]
    TGPU["GPU runtime<br/>Colab or similar"]
    TGPU --> TCODE
    TCODE --> ADAPT["LoRA adapter weights<br/>(PEFT)"]
  end

  %% ---------------------------
  %% Hugging Face (model + data)
  %% ---------------------------
  subgraph HF["Hugging Face Hub"]
    BASE["Base model<br/>meta-llama/Llama-3.2-3B"]
    DATA["Dataset<br/>ed-donner/items_prompts_lite or full"]
    ADAPT_HUB["Published adapters<br/>HF_USER/price-RUN_NAME"]
  end

  %% ---------------------------
  %% Inference + serving
  %% ---------------------------
  subgraph Local["Local machine"]
    CLI["CLI<br/>src/inference.py<br/>deploy | price | agent | logs"]
    PRE["Optional preprocessing<br/>src/inference/preprocessor.py<br/>LiteLLM completion()"]
    ENV_GROQ[".env<br/>GROQ_API_KEY"]
    ENV_GROQ --> PRE
  end

  subgraph LLM["External LLM provider (preprocess)"]
    GROQ["Groq (default)<br/>PRICER_PREPROCESSOR_MODEL"]
  end

  subgraph Modal["Modal cloud"]
    DEPLOY["Deploy + logs<br/>modal deploy / app logs"]
    RPC["RPC routing<br/>Pricer.price.remote(...)"]
    CONTAINER["GPU container<br/>App: pricer-service"]
    SECRET["Modal Secret<br/>huggingface-secret (HF_TOKEN)"]
    CACHE["Modal Volume<br/>HF cache (/cache)"]
    SERVICE["Service code<br/>src/inference/pricing_service.py"]
    MODEL["Runtime model<br/>base + LoRA adapters"]

    DEPLOY --> CONTAINER
    RPC --> CONTAINER
    SECRET --> CONTAINER
    CACHE --> CONTAINER
    CONTAINER --> SERVICE
    SERVICE --> MODEL
  end

  %% ---------------------------
  %% Relationships (data/control)
  %% ---------------------------
  BASE --> TCODE
  DATA --> TCODE
  ADAPT --> ADAPT_HUB

  CLI -->|"deploy, logs"| DEPLOY
  CLI -->|"price, agent"| PRE
  PRE --> GROQ

  CLI -->|"Modal SDK"| RPC
  PRE -->|"structured input"| RPC

  BASE -->|"download on cold start"| CONTAINER
  ADAPT_HUB -->|"download on cold start"| CONTAINER
```

## Run (`src/inference.py`)

Prereqs:
- `uv` installed
- `uv sync`
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