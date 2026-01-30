"""
Modal pricing service (deployed app: `pricer-service`).

High-level flow:
1) Define a Modal App + container image with the required ML dependencies.
2) On container start (`@modal.enter()`), load:
   - Base Llama model (quantized to 4-bit via bitsandbytes)
   - Fine-tuned LoRA adapter weights (PEFT) and apply them on top of the base model
   - Tokenizer (needed to convert prompt text -> token IDs, and tokens -> text)
3) Expose a remote method `Pricer.price(description)` used by:
   - `src/main.py price "..."`
   - `src/specialist_agent.py` (agent wrapper)

Notes:
- `print()` statements here run on the Modal container. View them with:
  `uv run python src/main.py logs`
"""

import modal
from modal import Volume, Image

# ---------------------------------------------------------------------------
# STEP 1) Define infrastructure (app + image + secrets + GPU + caching)
# ---------------------------------------------------------------------------

app = modal.App("pricer-service")
image = Image.debian_slim().pip_install(
    "huggingface", "torch", "transformers", "bitsandbytes", "accelerate", "peft"
)

# Modal secret that contains your Hugging Face token so the container can download models.
# Depending on your Modal configuration, you may need to replace "huggingface-secret" with "hf-secret".
secrets = [modal.Secret.from_name("huggingface-secret")]

GPU = "T4"
BASE_MODEL = "meta-llama/Llama-3.2-3B"
PROJECT_NAME = "price"
HF_USER = "ed-donner"  # your HF name here! Or use mine if you just want to reproduce my results.
RUN_NAME = "2025-11-28_18.47.07"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
REVISION = "b19c8bfea3b6ff62237fbb0a8da9779fc12cefbd"
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"
CACHE_DIR = "/cache"

# Change this to 1 if you want Modal to be always running, otherwise it will go cold after 2 mins
MIN_CONTAINERS = 0

PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"

# Persist Hugging Face downloads between container restarts (speeds up cold starts)
hf_cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)


@app.cls(
    image=image.env({"HF_HUB_CACHE": CACHE_DIR}),
    secrets=secrets,
    gpu=GPU,
    timeout=1800,
    min_containers=MIN_CONTAINERS,
    volumes={CACHE_DIR: hf_cache_volume},
)
class Pricer:
    @modal.enter()
    def setup(self):
        # -------------------------------------------------------------------
        # STEP 2) Load model + tokenizer once per container
        # -------------------------------------------------------------------
        #
        # Why tokenizer?
        # LLMs don't read strings directly. Tokenizers convert text -> token IDs
        # and decode token IDs -> text.
        #
        # Why load base + adapter weights?
        # The fine-tuned model here is stored as LoRA/PEFT adapter weights.
        # We load the base model, then apply the adapter weights on top.
        # -------------------------------------------------------------------
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        # STEP 2a) Quantization config (4-bit) to reduce GPU memory usage.
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        # STEP 2b) Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # STEP 2c) Load base model (quantized) onto the GPU
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quant_config,
            device_map="auto",
        )

        # STEP 2d) Load LoRA adapter weights and apply them to the base model
        self.fine_tuned_model = PeftModel.from_pretrained(
            self.base_model,
            FINETUNED_MODEL,
            revision=REVISION,
        )

    @modal.method()
    def price(self, description: str) -> float:
        # -------------------------------------------------------------------
        # STEP 3) Inference
        # - Build prompt
        # - Tokenize prompt
        # - Generate a few new tokens (the price)
        # - Decode and parse out the numeric value
        # -------------------------------------------------------------------
        import re
        import torch
        from transformers import set_seed

        set_seed(42)
        prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"

        # These run on the Modal container. If you want to see them locally:
        # `uv run python src/main.py logs`
        print("prompt:", prompt, flush=True)
        print("Querying the fine-tuned model", flush=True)

        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.fine_tuned_model.generate(inputs, max_new_tokens=5)
        result = self.tokenizer.decode(outputs[0])
        contents = result.split("Price is $")[1]
        contents = contents.replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
        return float(match.group()) if match else 0
