# Inference workflow (`src/inference.py` + `src/inference/`)

This doc is a walkthrough of the **end-to-end inference path** in this repo:

- **CLI entrypoint**: `src/inference.py`
- **Modal service**: `src/inference/pricing_service.py` (Modal app name: `pricer-service`)
- **Preprocessing** (optional): `src/inference/preprocessor.py` (LiteLLM → default Groq model)
- **Agent wrapper** (optional): `src/inference/specialist_agent.py`

## Overall architecture (components + where they run)

```mermaid
flowchart LR
  subgraph Local[Local machine]
    CLI["src/inference.py<br/>CLI: deploy | price | agent | logs"]
    PRE["src/inference/preprocessor.py<br/>Preprocessor (LiteLLM)"]
    AG["src/inference/specialist_agent.py<br/>SpecialistAgent (logging wrapper)"]
  end

  subgraph LLM[External LLM provider]
    GROQ["Groq (default)<br/>Model: PRICER_PREPROCESSOR_MODEL<br/>Auth: GROQ_API_KEY"]
  end

  subgraph Modal[Modal cloud]
    CP["Modal control plane<br/>(deploy, logs, RPC routing)"]
    CONTAINER["GPU container<br/>(deployed app: pricer-service)"]

    PS["src/inference/pricing_service.py<br/>Pricer.setup() + Pricer.price()"]
    TOK[Tokenizer]
    BASE["Base model<br/>(4-bit quantized)"]
    LORA["LoRA/PEFT<br/>adapter weights"]

    VOL["Modal Volume<br/>HF cache (/cache)"]
    SEC["Modal Secret<br/>huggingface-secret (HF_TOKEN)"]

    CP --> CONTAINER
    VOL --> CONTAINER
    SEC --> CONTAINER

    CONTAINER --> PS
    PS --> TOK
    PS --> BASE
    PS --> LORA
  end

  subgraph HF[Hugging Face Hub]
    HUB["Model artifacts<br/>BASE_MODEL + FINETUNED_MODEL adapters"]
  end

  CLI -->|"price/agent raw text"| PRE
  PRE -->|"completion()"| GROQ
  CLI -->|"deploy/logs"| CP
  CLI -->|"Modal SDK RPC<br/>Pricer.price.remote(...)"| CP
  AG -->|"Modal SDK RPC<br/>Pricer.price.remote(...)"| CP
  CONTAINER -->|"download on cold start<br/>(then cached)"| HUB
```

## Big picture (what happens when you run a command)

```mermaid
flowchart TD
  U[User runs: uv run python src/inference.py <command> ...] --> M[main() in src/inference.py]
  M -->|deploy| D[cmd_deploy(): modal deploy -m inference.pricing_service]
  M -->|logs| L[cmd_logs(): modal app logs pricer-service --timestamps]
  M -->|price <text>| P[cmd_price(text)]
  M -->|agent <text>| A[cmd_agent(text)]

  P --> PRE[preprocess_if_needed(text)]
  A --> PRE

  PRE -->|already structured| ST[use text as-is]
  PRE -->|raw text| PR[Preprocessor().preprocess(text)]
  PR --> ST

  ST --> RPC1[Modal RPC: Pricer.price.remote(structured_text)]
  RPC1 --> SVC[Modal container: Pricer.price(description)]
  SVC --> OUT[float price printed locally]

  A --> RPC2[SpecialistAgent.price(): Pricer.price.remote(structured_text)]
  RPC2 --> SVC
```

## CLI command dispatch (the surface API)

`src/inference.py` provides four commands via `main()`:

- **`deploy`**: deploy the Modal service module `inference.pricing_service`
- **`price "<text>"`**: preprocess (if needed) and call the deployed service directly
- **`agent "<text>"`**: preprocess (if needed) and call the deployed service via `SpecialistAgent` (adds logs)
- **`logs`**: stream the Modal app logs for `pricer-service` (where container `print()` output goes)

## Walkthrough: `deploy`

When you run:

```bash
uv run python src/inference.py deploy
```

`cmd_deploy()` executes:

- `uv run modal deploy -m inference.pricing_service`
- working directory: `src/` (so `-m inference.pricing_service` resolves correctly)

### What gets deployed

In `src/inference/pricing_service.py`:

- A Modal app is defined as `modal.App("pricer-service")`
- A container image is built with ML deps (Transformers, bitsandbytes, PEFT, etc.)
- A `Pricer` class is registered with `@app.cls(...)`, including GPU selection and a Hugging Face cache volume

### What happens on container startup (cold start)

On container start, Modal calls `Pricer.setup()` because it is decorated with `@modal.enter()`.
That method loads **once per container**:

- **Tokenizer**: `AutoTokenizer.from_pretrained(BASE_MODEL)`
- **Base model** (4-bit quantized): `AutoModelForCausalLM.from_pretrained(..., quantization_config=..., device_map="auto")`
- **LoRA adapter weights** applied to base model: `PeftModel.from_pretrained(self.base_model, FINETUNED_MODEL, revision=REVISION)`

This setup is why inference can be fast after the container is warm: the heavy model loads are not repeated per request.

## Walkthrough: `price "<text>"`

When you run:

```bash
uv run python src/inference.py price "iphone 10"
```

the flow is:

1) `cmd_price(text)` calls `preprocess_if_needed(text)`  
2) The processed text is sent to the deployed Modal class method via:  
   `Pricer = modal.Cls.from_name("pricer-service", "Pricer")` then `pricer.price.remote(processed)`
3) The returned `float` is printed locally

### The preprocessing gate (`preprocess_if_needed`)

`preprocess_if_needed(text)` checks whether the input already looks like the structured format your fine-tuned model expects.

It counts lines starting with these prefixes (case-insensitive):

- `Title:`
- `Category:`
- `Brand:`
- `Description:`
- `Details:`

If it sees **3 or more** of those fields, it treats the input as already structured and skips preprocessing.

Otherwise, it calls `Preprocessor().preprocess(text)` (LiteLLM) to rewrite raw text into the structured format.

### Why raw text can yield different answers

The deployed model inference is seeded (`set_seed(42)` in `Pricer.price`), so **given the same structured text**, the output should be stable.

If you pass raw text, the **preprocessor is an LLM call**, and changes in its output will change the prompt sent to the fine-tuned model, which can change the predicted price.

## Walkthrough: `agent "<text>"`

When you run:

```bash
uv run python src/inference.py agent "iphone 10"
```

the flow is:

1) `cmd_agent(text)` calls `preprocess_if_needed(text)` (same logic as `price`)
2) It creates `SpecialistAgent()` and calls `agent.price(processed)`
3) `SpecialistAgent.price()` calls the same deployed Modal method: `Pricer.price.remote(description)`

The difference vs `price` is **only** that the agent wrapper adds lightweight logging (via `src/inference/agent.py`).

## Walkthrough: `logs`

When you run:

```bash
uv run python src/inference.py logs
```

`cmd_logs()` executes:

- `uv run modal app logs pricer-service --timestamps`

This is the best way to see **container-side `print()` output**, such as:

- `prompt: ...`
- `Querying the fine-tuned model`

Those `print()` calls happen inside `Pricer.price()` on the Modal container.

## What the Modal service does per request (`Pricer.price`)

Once `cmd_price` / `SpecialistAgent` reaches the deployed service, `Pricer.price(description)`:

```mermaid
sequenceDiagram
  participant Client as Local CLI (src/inference.py)
  participant Modal as Modal RPC
  participant Pricer as Modal container (Pricer.price)
  participant Model as LoRA-applied Llama

  Client->>Modal: Pricer.price.remote(structured_text)
  Modal->>Pricer: invoke price(description)
  Pricer->>Pricer: set_seed(42)
  Pricer->>Pricer: build prompt = QUESTION + description + "Price is $"
  Pricer->>Pricer: tokenize prompt (tokenizer.encode → CUDA)
  Pricer->>Model: generate(max_new_tokens=5)
  Model-->>Pricer: token IDs
  Pricer->>Pricer: decode + parse numeric value via regex
  Pricer-->>Modal: float
  Modal-->>Client: float
```

Key implementation details:

- **Prompt format**:
  - `QUESTION = "What does this cost to the nearest dollar?"`
  - `PREFIX = "Price is $"`
  - prompt is `QUESTION\n\n{description}\n\n{PREFIX}`
- **Generation budget**: `max_new_tokens=5` (small, to emit a short price)
- **Parsing**: splits on `Price is $`, removes commas, extracts first number with a regex, returns `0` if none found

## Practical tips (stable runs)

- **Most deterministic**: pass already-structured input so preprocessing is skipped:

```bash
uv run python src/inference.py price $'Title: iPhone X\nCategory: Electronics\nBrand: Apple\nDescription: Smartphone\nDetails: 64GB'
```

- **If preprocessing is required**:
  - configure the preprocessor model via `PRICER_PREPROCESSOR_MODEL`
  - ensure `GROQ_API_KEY` is set (LiteLLM → Groq default in this repo)

