#!/usr/bin/env python3
# Generated from: Copy of NEW Week 7 Day 5 - Testing our Fine-tuned model
# NOTE: This file was auto-generated; edit the notebook if you want to keep changes in sync.

#-------------------------------------------------------------------------------
# Cell 1 (markdown)
#-------------------------------------------------------------------------------
# # The Price Is Right
#
# ### And now, to evaluate our fine-tuned open source model
#

#-------------------------------------------------------------------------------
# Cell 2 (code)
#-------------------------------------------------------------------------------
!pip install -q --upgrade bitsandbytes trl
!wget -q https://raw.githubusercontent.com/ed-donner/llm_engineering/main/week7/util.py -O util.py

#-------------------------------------------------------------------------------
# Cell 3 (code)
#-------------------------------------------------------------------------------
# imports

import os
import re
import math
from tqdm import tqdm
from google.colab import userdata
from huggingface_hub import login
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from datasets import load_dataset, Dataset, DatasetDict
from datetime import datetime
from peft import PeftModel
from util import evaluate

#-------------------------------------------------------------------------------
# Cell 4 (code)
#-------------------------------------------------------------------------------
# Constants

BASE_MODEL = "meta-llama/Llama-3.2-3B"
PROJECT_NAME = "price"
HF_USER = "ed-donner" # your HF name here!

LITE_MODE = False

DATA_USER = "ed-donner"
DATASET_NAME = f"{DATA_USER}/items_prompts_lite" if LITE_MODE else f"{DATA_USER}/items_prompts_full"

if LITE_MODE:
  RUN_NAME = "2025-11-30_15.10.55-lite"
  REVISION = None
else:
  RUN_NAME = "2025-11-28_18.47.07"
  REVISION = "b19c8bfea3b6ff62237fbb0a8da9779fc12cefbd"

PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_RUN_NAME}"


# Hyper-parameters - QLoRA

QUANT_4_BIT = True
capability = torch.cuda.get_device_capability()
use_bf16 = capability[0] >= 8

#-------------------------------------------------------------------------------
# Cell 5 (markdown)
#-------------------------------------------------------------------------------
# ### Log in to HuggingFace
#
# If you don't already have a HuggingFace account, visit https://huggingface.co to sign up and create a token.
#
# Then select the Secrets for this Notebook by clicking on the key icon in the left, and add a new secret called `HF_TOKEN` with the value as your token.

#-------------------------------------------------------------------------------
# Cell 6 (code)
#-------------------------------------------------------------------------------
# Log in to HuggingFace

hf_token = userdata.get('HF_TOKEN')
login(hf_token, add_to_git_credential=True)

#-------------------------------------------------------------------------------
# Cell 7 (code)
#-------------------------------------------------------------------------------
dataset = load_dataset(DATASET_NAME)
test = dataset['test']

#-------------------------------------------------------------------------------
# Cell 8 (code)
#-------------------------------------------------------------------------------
test[0]

#-------------------------------------------------------------------------------
# Cell 9 (markdown)
#-------------------------------------------------------------------------------
# ## Now load the Tokenizer and Model

#-------------------------------------------------------------------------------
# Cell 10 (code)
#-------------------------------------------------------------------------------
# pick the right quantization

if QUANT_4_BIT:
  quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    bnb_4bit_quant_type="nf4"
  )
else:
  quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
  )

#-------------------------------------------------------------------------------
# Cell 11 (code)
#-------------------------------------------------------------------------------
# Load the Tokenizer and the Model

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
)
base_model.generation_config.pad_token_id = tokenizer.pad_token_id

# Load the fine-tuned model with PEFT
if REVISION:
  fine_tuned_model = PeftModel.from_pretrained(base_model, HUB_MODEL_NAME, revision=REVISION)
else:
  fine_tuned_model = PeftModel.from_pretrained(base_model, HUB_MODEL_NAME)


print(f"Memory footprint: {fine_tuned_model.get_memory_footprint() / 1e6:.1f} MB")

#-------------------------------------------------------------------------------
# Cell 12 (code)
#-------------------------------------------------------------------------------
fine_tuned_model

#-------------------------------------------------------------------------------
# Cell 13 (markdown)
#-------------------------------------------------------------------------------
# # THE MOMENT OF TRUTH!
#
# ## Use the model in inference mode
#
# Trying to beat "human" level of performance at $87.62
#
# Or possibly close to gpt-4.1-nano at $62.51
#
# ## Caveat
#
# Keep in mind that prices of goods vary considerably; the model can't predict things like sale prices that it doesn't have any information about.

#-------------------------------------------------------------------------------
# Cell 14 (code)
#-------------------------------------------------------------------------------
def model_predict(item):
    inputs = tokenizer(item["prompt"],return_tensors="pt").to("cuda")
    with torch.no_grad():
        output_ids = fine_tuned_model.generate(**inputs, max_new_tokens=8)
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, prompt_len:]
    return tokenizer.decode(generated_ids)

#-------------------------------------------------------------------------------
# Cell 15 (code)
#-------------------------------------------------------------------------------
set_seed(42)
evaluate(model_predict, test)

#-------------------------------------------------------------------------------
# Cell 16 (code)
#-------------------------------------------------------------------------------
# (empty code cell)
