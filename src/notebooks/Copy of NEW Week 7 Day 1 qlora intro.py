#!/usr/bin/env python3
# Generated from: Copy of NEW Week 7 Day 1 qlora intro.ipynb
# NOTE: This file was auto-generated; edit the notebook if you want to keep changes in sync.

#-------------------------------------------------------------------------------
# Cell 1 (markdown)
#-------------------------------------------------------------------------------
# ## Predict Product Prices
#
# ### Week 7 Day 1
#
# An introduction to LoRA and QLoRA
#

#-------------------------------------------------------------------------------
# Cell 2 (markdown)
#-------------------------------------------------------------------------------
# ## A reminder of important pro-tip for using Colab:
#
#
#
# **Pro-tip:**
#
# In the middle of running a Colab, you might get an error like this:
#
# > Runtime error: CUDA is required but not available for bitsandbytes. Please consider installing [...]
#
# This is a super-misleading error message! Please don't try changing versions of packages...
#
# This actually happens because Google has switched out your Colab runtime, perhaps because Google Colab was too busy. The solution is:
#
# 1. Runtime menu >> Disconnect and delete runtime
# 2. Reload the colab from fresh and Edit menu >> Clear All Outputs
# 3. Connect to a new T4 using the button at the top right
# 4. Select "View resources" from the menu on the top right to confirm you have a GPU
# 5. Rerun the cells in the colab, from the top down, starting with the pip installs
#
# And all should work great - otherwise, ask me!

#-------------------------------------------------------------------------------
# Cell 3 (code)
#-------------------------------------------------------------------------------
#!pip install -q --upgrade bitsandbytes

# --- Outputs (from notebook) ---
# [{"output_type": "stream", "name": "stdout", "text": ["\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m59.1/59.1 MB\u001b[0m \u001b[31m10.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n", "\u001b[?25h"]}]

#-------------------------------------------------------------------------------
# Cell 4 (code)
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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, set_seed
from peft import LoraConfig, PeftModel
from datetime import datetime

#-------------------------------------------------------------------------------
# Cell 5 (code)
#-------------------------------------------------------------------------------
# Constants

BASE_MODEL = "meta-llama/Llama-3.2-3B"
PROJECT_NAME = "price"
RUN_NAME =  f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"

LITE_MODE = False

DATA_USER = "ed-donner"
DATASET_NAME = f"{DATA_USER}/items_prompts_lite" if LITE_MODE else f"{DATA_USER}/items_prompts_full"


FINETUNED_MODEL = f"ed-donner/price-2025-11-30_15.10.55-lite"

#-------------------------------------------------------------------------------
# Cell 6 (markdown)
#-------------------------------------------------------------------------------
# ### Log in to HuggingFace
#
# If you don't already have a HuggingFace account, visit https://huggingface.co to sign up and create a token.
#
# Then select the Secrets for this Notebook by clicking on the key icon in the left, and add a new secret called `HF_TOKEN` with the value as your token.

#-------------------------------------------------------------------------------
# Cell 7 (code)
#-------------------------------------------------------------------------------
# Log in to HuggingFace

hf_token = userdata.get('HF_TOKEN')
login(hf_token, add_to_git_credential=True)

#-------------------------------------------------------------------------------
# Cell 8 (markdown)
#-------------------------------------------------------------------------------
# ## Trying out different Quantization
#

#-------------------------------------------------------------------------------
# Cell 9 (code)
#-------------------------------------------------------------------------------
# Load the Base Model without quantization

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")

#-------------------------------------------------------------------------------
# Cell 10 (code)
#-------------------------------------------------------------------------------
print(f"Memory footprint: {base_model.get_memory_footprint() / 1e9:,.1f} GB")

#-------------------------------------------------------------------------------
# Cell 11 (code)
#-------------------------------------------------------------------------------
base_model

#-------------------------------------------------------------------------------
# Cell 12 (markdown)
#-------------------------------------------------------------------------------
# ## Restart your session!
#
# In order to load the next model and clear out the cache of the last model, you'll now need to go to Runtime >> Restart session and run the initial cells (installs and imports and HuggingFace login) again.
#
# This is to clean out the GPU.

#-------------------------------------------------------------------------------
# Cell 13 (code)
#-------------------------------------------------------------------------------
# Load the Base Model using 8 bit

quant_config = BitsAndBytesConfig(load_in_8bit=True)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
)

#-------------------------------------------------------------------------------
# Cell 14 (markdown)
#-------------------------------------------------------------------------------
# (empty markdown cell)

#-------------------------------------------------------------------------------
# Cell 15 (code)
#-------------------------------------------------------------------------------
print(f"Memory footprint: {base_model.get_memory_footprint() / 1e9:,.1f} GB")

#-------------------------------------------------------------------------------
# Cell 16 (code)
#-------------------------------------------------------------------------------
base_model

#-------------------------------------------------------------------------------
# Cell 17 (markdown)
#-------------------------------------------------------------------------------
# ## Restart your session!
#
# In order to load the next model and clear out the cache of the last model, you'll now need to go to Runtime >> Restart session and run the initial cells (imports and HuggingFace login) again.
#
# This is to clean out the GPU.

#-------------------------------------------------------------------------------
# Cell 18 (code)
#-------------------------------------------------------------------------------
# Load the Tokenizer and the Base Model using 4 bit

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
)

#-------------------------------------------------------------------------------
# Cell 19 (code)
#-------------------------------------------------------------------------------
print(f"Memory footprint: {base_model.get_memory_footprint() / 1e9:,.2f} GB")

#-------------------------------------------------------------------------------
# Cell 20 (code)
#-------------------------------------------------------------------------------
base_model

#-------------------------------------------------------------------------------
# Cell 21 (code)
#-------------------------------------------------------------------------------
fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)

#-------------------------------------------------------------------------------
# Cell 22 (code)
#-------------------------------------------------------------------------------
print(f"Memory footprint: {fine_tuned_model.get_memory_footprint() / 1e9:,.2f} GB")

#-------------------------------------------------------------------------------
# Cell 23 (code)
#-------------------------------------------------------------------------------
fine_tuned_model

#-------------------------------------------------------------------------------
# Cell 24 (markdown)
#-------------------------------------------------------------------------------
# # Understanding LoRA weights and dimensions
#
# ## More common QLoRA setup that we'll use for the Lite Training `LITE_MODE=True`
#
# r = 32, which is a common value for r
#
# Target modules are the Attention layers only

#-------------------------------------------------------------------------------
# Cell 25 (code)
#-------------------------------------------------------------------------------
# Each of the Target Modules has 2 LoRA Adaptor matrices, called lora_A and lora_B
# These are designed so that weights can be adapted by adding alpha * lora_A * lora_B
# Let's count the number of weights using their dimensions:

r = 32

# See the matrix dimensions above

# Attention layers
lora_q_proj = 3072 * r + 3072 * r
lora_k_proj = 3072 * r + 1024 * r
lora_v_proj = 3072 * r + 1024 * r
lora_o_proj = 3072 * r + 3072 * r

# Each layer comes to
lora_layer = lora_q_proj + lora_k_proj + lora_v_proj + lora_o_proj

# There are 28 layers
params = lora_layer * 28

# So the total size in MB is
size = (params * 4) / 1_000_000

print(f"Total number of params: {params:,} and size {size:,.1f}MB")

#-------------------------------------------------------------------------------
# Cell 26 (markdown)
#-------------------------------------------------------------------------------
# ## And now, for training with the Full Dataset `LITE_MODE=False`
#
# This is a pretty extreme LoRA setup. We have so much training data that I'm trying to train a lot!
#
# We use 256 dimensions in the LoRA matrices and we have LoRA matrices for the attention layers and the MLP layers:

#-------------------------------------------------------------------------------
# Cell 27 (code)
#-------------------------------------------------------------------------------
# Each of the Target Modules has 2 LoRA Adaptor matrices, called lora_A and lora_B
# These are designed so that weights can be adapted by adding alpha * lora_A * lora_B
# Let's count the number of weights using their dimensions:

r = 256

# See the matrix dimensions above

# Attention layers
lora_q_proj = 3072 * r + 3072 * r
lora_k_proj = 3072 * r + 1024 * r
lora_v_proj = 3072 * r + 1024 * r
lora_o_proj = 3072 * r + 3072 * r

# MLP layers
lora_gate_proj = 3072 * r + 8192 * r
lora_up_proj = 3072 * r + 8192 * r
lora_down_proj = 3072 * r + 8192 * r

# Each layer comes to
lora_layer = lora_q_proj + lora_k_proj + lora_v_proj + lora_o_proj + lora_gate_proj + lora_up_proj + lora_down_proj

# There are 28 layers
params = lora_layer * 28

# So the total size in MB is
size = (params * 4) / 1_000_000

print(f"Total number of params: {params:,} and size {size:,.1f}MB")

#-------------------------------------------------------------------------------
# Cell 28 (code)
#-------------------------------------------------------------------------------
# (empty code cell)
