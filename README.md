# ai-fine-tuning

## Run Week 8 Day 1 (`src/main.py`)

Prereqs:
- `uv` installed
- Modal set up (`uv run modal token set ...`)
- Modal Secret created for Hugging Face token (usually named **`huggingface-secret`** with key `HF_TOKEN`)
- `.env` contains your Groq key:
  - `GROQ_API_KEY=...`

Service files:
- **Modal service**: `src/pricing_service.py` (deployed app name: `pricer-service`)
- **Client runner**: `src/main.py`

Commands (run from repo root):

```bash
# Step 1 (MUST): Deploy the Modal app (do this once, or whenever you change the service code)
uv run python src/main.py deploy

# Step 2 (choose one): Call the service directly OR use the agent (agent calls the same service)
uv run python src/main.py price "raw text here"
uv run python src/main.py agent "raw text here"

# Optional: watch the remote Modal logs (this is where container print() output shows up)
uv run python src/main.py logs
```

Optional: set a different default preprocess model:
- Add to `.env`: `PRICER_PREPROCESSOR_MODEL=groq/openai/gpt-oss-20b` (or another LiteLLM-supported model)

## Convert a Colab notebook to a commented Python script

Because Google Drive/Colab links often require sign-in, the simplest workflow is:

- In Colab: **File → Download → Download `.ipynb`**
- Put the downloaded notebook into this repo (example: `my_notebook.ipynb`)
- Run:

```bash
python convert_ipynb_to_py.py my_notebook.ipynb -o my_notebook.py
```

What you get in the output `.py`:
- Markdown cells → comment blocks
- Code cells → normal Python code
- Clear `# Cell N (type)` separators between cells