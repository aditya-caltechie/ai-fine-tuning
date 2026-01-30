#!/usr/bin/env python3
# Generated from: Copy of NEW Week 7 Day 3-4 TRAINING.ipynb
# NOTE: This file was auto-generated; edit the notebook if you want to keep changes in sync.

#-------------------------------------------------------------------------------
# Cell 1 (markdown)
#-------------------------------------------------------------------------------
# # The Price Is Right - Week 7
#
# ## Day 3 and Day 4: Training!
#
# This is what it's all been about!
#
# If you are using LITE_MODE=True, then please run this on a free T4 box.
#
# If you are using LITE_MODE-False, then please use a paid A100 with high memory.

#-------------------------------------------------------------------------------
# Cell 2 (code)
#-------------------------------------------------------------------------------
!pip install -q --upgrade bitsandbytes==0.48.2 trl==0.25.1
!wget -q https://raw.githubusercontent.com/ed-donner/llm_engineering/main/week7/util.py -O util.py

#-------------------------------------------------------------------------------
# Cell 3 (code)
#-------------------------------------------------------------------------------
import os
import re
import math
from tqdm import tqdm
from google.colab import userdata
from huggingface_hub import login
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed, BitsAndBytesConfig
from datasets import load_dataset, Dataset, DatasetDict
import wandb
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datetime import datetime
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------
# Cell 4 (code)
#-------------------------------------------------------------------------------
# Constants

BASE_MODEL = "meta-llama/Llama-3.2-3B"
PROJECT_NAME = "price"
HF_USER = "ed-donner" # your HF name here!

LITE_MODE = True

DATA_USER = "ed-donner"
DATASET_NAME = f"{DATA_USER}/items_prompts_lite" if LITE_MODE else f"{DATA_USER}/items_prompts_full"

RUN_NAME =  f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
if LITE_MODE:
  RUN_NAME += "-lite"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_RUN_NAME}"

# Hyper-parameters - overall

EPOCHS = 1 if LITE_MODE else 3
BATCH_SIZE = 32 if LITE_MODE else 256
MAX_SEQUENCE_LENGTH = 128
GRADIENT_ACCUMULATION_STEPS = 1

# Hyper-parameters - QLoRA

QUANT_4_BIT = True
LORA_R = 32 if LITE_MODE else 256
LORA_ALPHA = LORA_R * 2
ATTENTION_LAYERS = ["q_proj", "v_proj", "k_proj", "o_proj"]
MLP_LAYERS = ["gate_proj", "up_proj", "down_proj"]
TARGET_MODULES = ATTENTION_LAYERS if LITE_MODE else ATTENTION_LAYERS + MLP_LAYERS
LORA_DROPOUT = 0.1

# Hyper-parameters - training

LEARNING_RATE = 1e-4
WARMUP_RATIO = 0.01
LR_SCHEDULER_TYPE = 'cosine'
WEIGHT_DECAY = 0.001
OPTIMIZER = "paged_adamw_32bit"

capability = torch.cuda.get_device_capability()
use_bf16 = capability[0] >= 8

# Tracking

VAL_SIZE = 500 if LITE_MODE else 1000
LOG_STEPS = 5 if LITE_MODE else 10
SAVE_STEPS = 100 if LITE_MODE else 200
LOG_TO_WANDB = True

#-------------------------------------------------------------------------------
# Cell 5 (code)
#-------------------------------------------------------------------------------
# A100 GPU supports this; T4 does not natively

use_bf16

#-------------------------------------------------------------------------------
# Cell 6 (markdown)
#-------------------------------------------------------------------------------
# # More on Optimizers
#
# https://huggingface.co/docs/transformers/main/en/perf_train_gpu_one#optimizers
#
# The most common is Adam or AdamW (Adam with Weight Decay).  
# Adam achieves good convergence by storing the rolling average of the previous gradients; however, it adds an additional memory footprint of the order of the number of model parameters.

#-------------------------------------------------------------------------------
# Cell 7 (markdown)
#-------------------------------------------------------------------------------
# ### Log in to HuggingFace and Weights & Biases
#
# If you don't already have a HuggingFace account, visit https://huggingface.co to sign up and create a token.
#
# Then select the Secrets for this Notebook by clicking on the key icon in the left, and add a new secret called `HF_TOKEN` with the value as your token.
#
# Repeat this for weightsandbiases at https://wandb.ai and add a secret called `WANDB_API_KEY`

#-------------------------------------------------------------------------------
# Cell 8 (code)
#-------------------------------------------------------------------------------
# Log in to HuggingFace

hf_token = userdata.get('HF_TOKEN')
login(hf_token, add_to_git_credential=True)

#-------------------------------------------------------------------------------
# Cell 9 (markdown)
#-------------------------------------------------------------------------------
# (empty markdown cell)

#-------------------------------------------------------------------------------
# Cell 10 (code)
#-------------------------------------------------------------------------------
# Log in to Weights & Biases
wandb_api_key = userdata.get('WANDB_API_KEY')
os.environ["WANDB_API_KEY"] = wandb_api_key
wandb.login()

# Configure Weights & Biases to record against our project
os.environ["WANDB_PROJECT"] = PROJECT_NAME
os.environ["WANDB_LOG_MODEL"] = "false"
os.environ["WANDB_WATCH"] = "false"

#-------------------------------------------------------------------------------
# Cell 11 (code)
#-------------------------------------------------------------------------------
dataset = load_dataset(DATASET_NAME)
train = dataset['train']
val = dataset['val'].select(range(VAL_SIZE))
test = dataset['test']

#-------------------------------------------------------------------------------
# Cell 12 (code)
#-------------------------------------------------------------------------------
# if you wish to reduce the training dataset to 10,000 points instead, then uncomment this line:

# train = train.select(range(10000))

#-------------------------------------------------------------------------------
# Cell 13 (code)
#-------------------------------------------------------------------------------
if LOG_TO_WANDB:
  wandb.init(project=PROJECT_NAME, name=RUN_NAME)

#-------------------------------------------------------------------------------
# Cell 14 (markdown)
#-------------------------------------------------------------------------------
# ## Now load the Tokenizer and Model
#
# The model is "quantized" - we are reducing the precision to 4 bits.

#-------------------------------------------------------------------------------
# Cell 15 (code)
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
# Cell 16 (code)
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

print(f"Memory footprint: {base_model.get_memory_footprint() / 1e6:.1f} MB")

#-------------------------------------------------------------------------------
# Cell 17 (markdown)
#-------------------------------------------------------------------------------
# # AND NOW
#
# ## We set up the configuration for Training
#
# We need to create 2 objects:
#
# A LoraConfig object with our hyperparameters for LoRA
#
# An SFTConfig with our overall Training parameters

#-------------------------------------------------------------------------------
# Cell 18 (code)
#-------------------------------------------------------------------------------
# LoRA Parameters

lora_parameters = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=TARGET_MODULES,
)

#-------------------------------------------------------------------------------
# Cell 19 (code)
#-------------------------------------------------------------------------------
# Training parameters

train_parameters = SFTConfig(
    output_dir=PROJECT_RUN_NAME,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    optim=OPTIMIZER,
    save_steps=SAVE_STEPS,
    save_total_limit=10,
    logging_steps=LOG_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.001,
    fp16=not use_bf16,
    bf16=use_bf16,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=WARMUP_RATIO,
    group_by_length=True,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    report_to="wandb" if LOG_TO_WANDB else None,
    run_name=RUN_NAME,
    max_length=MAX_SEQUENCE_LENGTH,
    save_strategy="steps",
    hub_strategy="every_save",
    push_to_hub=True,
    hub_model_id=HUB_MODEL_NAME,
    hub_private_repo=True,
    eval_strategy="steps",
    eval_steps=SAVE_STEPS
)

#-------------------------------------------------------------------------------
# Cell 20 (markdown)
#-------------------------------------------------------------------------------
# # AND NOW - create the trainer

#-------------------------------------------------------------------------------
# Cell 21 (code)
#-------------------------------------------------------------------------------
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=train,
    eval_dataset=val,
    peft_config=lora_parameters,
    args=train_parameters
)

#-------------------------------------------------------------------------------
# Cell 22 (markdown)
#-------------------------------------------------------------------------------
# ## In the next cell, we kick off fine-tuning!
#
# This will run for some time, uploading to the hub every SAVE_STEPS steps.
#
# After some time, Google might stop your colab. For people on free plans, it can happen whenever Google is low on resources. For anyone on paid plans, they can give you up to 24 hours, but there's no guarantee.
#
# If your server is stopped, you can follow my colab here to resume from your last save:
#
# https://colab.research.google.com/drive/1qGTDVIas_Vwoby4UVi2vwsU0tHXy8OMO#scrollTo=R_O04fKxMMT-
#
# I've saved this colab with my final run in the output so you can see the example. The trick is that I needed to set `is_trainable=True` when loading the fine_tuned model.
#
# ### Anyway, with that in mind, let's kick this off!

#-------------------------------------------------------------------------------
# Cell 23 (code)
#-------------------------------------------------------------------------------
# Fine-tune!
fine_tuning.train()

# Push our fine-tuned model to Hugging Face
fine_tuning.model.push_to_hub(PROJECT_RUN_NAME, private=True)
print(f"Saved to the hub: {PROJECT_RUN_NAME}")

#-------------------------------------------------------------------------------
# Cell 24 (code)
#-------------------------------------------------------------------------------
if LOG_TO_WANDB:
  wandb.finish()
