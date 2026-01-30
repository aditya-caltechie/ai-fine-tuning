#!/usr/bin/env python3
# Generated from: Copy of NEW Week 7 Day 2 base model evaluation.ipynb
# NOTE: This file was auto-generated; edit the notebook if you want to keep changes in sync.

#-------------------------------------------------------------------------------
# Cell 1 (markdown)
#-------------------------------------------------------------------------------
# # The Price is Right
#
# ### Week 7 Day 2
#
# Selecting our model and evaluating the base model against the task.
#
# Keep in mind: our base model has 8 billion params, quantized down to 4 bits
# Compared with GPT-5 at TRILLIONS of params!

#-------------------------------------------------------------------------------
# Cell 2 (code)
#-------------------------------------------------------------------------------
!pip install -q --upgrade bitsandbytes

# --- Outputs (from notebook) ---
# [{"output_type": "stream", "name": "stdout", "text": ["\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m59.4/59.4 MB\u001b[0m \u001b[31m14.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n", "\u001b[?25h"]}]

#-------------------------------------------------------------------------------
# Cell 3 (code)
#-------------------------------------------------------------------------------
!wget -q https://raw.githubusercontent.com/ed-donner/llm_engineering/main/week7/util.py -O util.py

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
from datasets import load_dataset, Dataset, DatasetDict
from datetime import datetime
import matplotlib.pyplot as plt
from util import evaluate

#-------------------------------------------------------------------------------
# Cell 5 (code)
#-------------------------------------------------------------------------------
# Constants

BASE_MODEL = "meta-llama/Llama-3.2-3B"
PROJECT_NAME = "price"

LITE_MODE = True

DATA_USER = "ed-donner"
DATASET_NAME = f"{DATA_USER}/items_prompts_lite" if LITE_MODE else f"{DATA_USER}/items_prompts_full"

# Hyper-parameters

QUANT_4_BIT = True

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
# # Load our data
#
# We uploaded it to Hugging Face, so it's easy to retrieve it now

#-------------------------------------------------------------------------------
# Cell 9 (code)
#-------------------------------------------------------------------------------
dataset = load_dataset(DATASET_NAME)
train = dataset['train']
val = dataset['val']
test = dataset['test']

# --- Outputs (from notebook) ---
# [{"output_type": "display_data", "data": {"text/plain": ["README.md:   0%|          | 0.00/509 [00:00<?, ?B/s]"], "application/vnd.jupyter.widget-view+json": {"version_major": 2, "version_minor": 0, "model_id": "9ee15c9f70564e0d929af20e5e431fc0"}}, "metadata": {}}, {"output_type": "display_data", "data": {"text/plain": ["data/train-00000-of-00001.parquet:   0%|          | 0.00/4.31M [00:00<?, ?B/s]"], "application/vnd.jupyter.widget-view+json": {"version_major": 2, "version_minor": 0, "model_id": "753775eb70804c538188b2861ddc78f4"}}, "metadata": {}}, {"output_type": "display_data", "data": {"text/plain": ["data/val-00000-of-00001.parquet:   0%|          | 0.00/216k [00:00<?, ?B/s]"], "application/vnd.jupyter.widget-view+json": {"version_major": 2, "version_minor": 0, "model_id": "ce2a08b8458c4ea2a5f7ca630a61c529"}}, "metadata": {}}, {"output_type": "display_data", "data": {"text/plain": ["data/test-00000-of-00001.parquet:   0%|          | 0.00/218k [00:00<?, ?B/s]"], "application/vnd.jupyter.widget-view+json": {"version_major": 2, "version_minor": 0, "model_id": "617f3efaed134a6d917891b05f874720"}}, "metadata": {}}, {"output_type": "display_data", "data": {"text/plain": ["Generating train split:   0%|          | 0/20000 [00:00<?, ? examples/s]"], "application/vnd.jupyter.widget-view+json": {"version_major": 2, "version_minor": 0, "model_id": "10d19324e0e04f3bbff696004de1661c"}}, "metadata": {}}, {"output_type": "display_data", "data": {"text/plain": ["Generating val split:   0%|          | 0/1000 [00:00<?, ? examples/s]"], "application/vnd.jupyter.widget-view+json": {"version_major": 2, "version_minor": 0, "model_id": "73b5523fa41f4ecf9f70a57434ec3faa"}}, "metadata": {}}, {"output_type": "display_data", "data": {"text/plain": ["Generating test split:   0%|          | 0/1000 [00:00<?, ? examples/s]"], "application/vnd.jupyter.widget-view+json": {"version_major": 2, "version_minor": 0, "model_id": "0f8569f0b8a1432a9713fd9021d38c22"}}, "metadata": {}}]

#-------------------------------------------------------------------------------
# Cell 10 (code)
#-------------------------------------------------------------------------------
train[0]

# --- Outputs (from notebook) ---
# [{"output_type": "execute_result", "data": {"text/plain": ["{'prompt': 'What does this cost to the nearest dollar?\\n\\nTitle: Schlage F59 & 613 Andover Interior Knob (Deadbolt Included)  \\nCategory: Home Hardware  \\nBrand: Schlage  \\nDescription: A single‑piece oil‑rubbed bronze knob that mounts to a deadbolt for secure, easy interior door use.  \\nDetails: Designed for a 4\" minimum center‑to‑center door prep, it offers a lifetime mechanical and finish warranty and comes ready for quick installation.\\n\\nPrice is $',\n", " 'completion': '64.00'}"]}, "metadata": {}, "execution_count": 9}]

#-------------------------------------------------------------------------------
# Cell 11 (markdown)
#-------------------------------------------------------------------------------
# # Prepare our Base Llama Model for evaluation
#
# Load our base model with 4 bit quantization and try out 1 example

#-------------------------------------------------------------------------------
# Cell 12 (code)
#-------------------------------------------------------------------------------
## pick the right quantization

if QUANT_4_BIT:
  quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
  )
else:
  quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16
  )

#-------------------------------------------------------------------------------
# Cell 13 (code)
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

print(f"Memory footprint: {base_model.get_memory_footprint() / 1e9:.1f} GB")

# --- Outputs (from notebook) ---
# [{"output_type": "display_data", "data": {"text/plain": ["tokenizer_config.json:   0%|          | 0.00/50.5k [00:00<?, ?B/s]"], "application/vnd.jupyter.widget-view+json": {"version_major": 2, "version_minor": 0, "model_id": "63246cb0c8394b0b974eb7b156390154"}}, "metadata": {}}, {"output_type": "display_data", "data": {"text/plain": ["tokenizer.json:   0%|          | 0.00/9.09M [00:00<?, ?B/s]"], "application/vnd.jupyter.widget-view+json": {"version_major": 2, "version_minor": 0, "model_id": "83e40719cc4f44f0854013bbc0e1e575"}}, "metadata": {}}, {"output_type": "display_data", "data": {"text/plain": ["special_tokens_map.json:   0%|          | 0.00/301 [00:00<?, ?B/s]"], "application/vnd.jupyter.widget-view+json": {"version_major": 2, "version_minor": 0, "model_id": "9a4722c19f2648a3bcb9489fb47ddbba"}}, "metadata": {}}, {"output_type": "display_data", "data": {"text/plain": ["config.json:   0%|          | 0.00/844 [00:00<?, ?B/s]"], "application/vnd.jupyter.widget-view+json": {"version_major": 2, "version_minor": 0, "model_id": "edba4c3fd5de4054881363369b1416ff"}}, "metadata": {}}, {"output_type": "display_data", "data": {"text/plain": ["model.safetensors.index.json:   0%|          | 0.00/20.9k [00:00<?, ?B/s]"], "application/vnd.jupyter.widget-view+json": {"version_major": 2, "version_minor": 0, "model_id": "da0b080bd41449e3a65bf422df1a20a0"}}, "metadata": {}}, {"output_type": "display_data", "data": {"text/plain": ["Fetching 2 files:   0%|          | 0/2 [00:00<?, ?it/s]"], "application/vnd.jupyter.widget-view+json": {"version_major": 2, "version_minor": 0, "model_id": "7f28fa9c4282482f93cca5632b1155a8"}}, "metadata": {}}, {"output_type": "display_data", "data": {"text/plain": ["model-00002-of-00002.safetensors:   0%|          | 0.00/1.46G [00:00<?, ?B/s]"], "application/vnd.jupyter.widget-view+json": {"version_major": 2, "version_minor": 0, "model_id": "430dab005fc8474896e5503238118072"}}, "metadata": {}}, {"output_type": "display_data", "data": {"text/plain": ["model-00001-of-00002.safetensors:   0%|          | 0.00/4.97G [00:00<?, ?B/s]"], "application/vnd.jupyter.widget-view+json": {"version_major": 2, "version_minor": 0, "model_id": "6e0d8ba88ef545fd9cc9fea6eb6d3c7f"}}, "metadata": {}}, {"output_type": "display_data", "data": {"text/plain": ["Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]"], "application/vnd.jupyter.widget-view+json": {"version_major": 2, "version_minor": 0, "model_id": "2064a8819ac040409c4997d12ca5c131"}}, "metadata": {}}, {"output_type": "display_data", "data": {"text/plain": ["generation_config.json:   0%|          | 0.00/185 [00:00<?, ?B/s]"], "application/vnd.jupyter.widget-view+json": {"version_major": 2, "version_minor": 0, "model_id": "48895729906d4eedaa7f87db64f16643"}}, "metadata": {}}, {"output_type": "stream", "name": "stdout", "text": ["Memory footprint: 2.2 GB\n"]}]

#-------------------------------------------------------------------------------
# Cell 14 (code)
#-------------------------------------------------------------------------------
def model_predict(item):
    inputs = tokenizer(item["prompt"],return_tensors="pt").to("cuda")
    with torch.no_grad():
        output_ids = base_model.generate(**inputs, max_new_tokens=8)
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, prompt_len:]
    return tokenizer.decode(generated_ids)

#-------------------------------------------------------------------------------
# Cell 15 (code)
#-------------------------------------------------------------------------------
test[0]

# --- Outputs (from notebook) ---
# [{"output_type": "execute_result", "data": {"text/plain": ["{'prompt': 'What does this cost to the nearest dollar?\\n\\nTitle: Excess V2 Distortion/Modulation Pedal  \\nCategory: Music Pedals  \\nBrand: Old Blood Noise  \\nDescription: A versatile pedal offering distortion and three modulation modes—delay, chorus, and harmonized fifths—with full control over signal routing and expression.  \\nDetails: Features include separate gain, tone, and volume controls; time, depth, and volume per modulation; order switching, soft‑touch bypass, and expression jack for dynamic control.\\n\\nPrice is $',\n", " 'completion': '219.0'}"]}, "metadata": {}, "execution_count": 13}]

#-------------------------------------------------------------------------------
# Cell 16 (code)
#-------------------------------------------------------------------------------
model_predict(test[0])

# --- Outputs (from notebook) ---
# [{"output_type": "execute_result", "data": {"text/plain": ["'349.99. What is the price'"], "application/vnd.google.colaboratory.intrinsic+json": {"type": "string"}}, "metadata": {}, "execution_count": 14}]

#-------------------------------------------------------------------------------
# Cell 17 (markdown)
#-------------------------------------------------------------------------------
# # Evaluation!
#
# Trying out our base Llama 3.2 model against the Test dataset

#-------------------------------------------------------------------------------
# Cell 18 (code)
#-------------------------------------------------------------------------------
evaluate(model_predict, test)

# --- Outputs (from notebook) ---
# [{"output_type": "display_data", "data": {"text/html": ["<html>\n", "<head><meta charset=\"utf-8\" /></head>\n", "<body>\n", "    <div>            <script src=\"https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG\"></script><script type=\"text/javascript\">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: \"STIX-Web\"}});}</script>                <script type=\"text/javascript\">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>\n", "        <script charset=\"utf-8\" src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>                <div id=\"9d6da6f6-43cf-417c-adde-4bcf99ff35a5\" class=\"plotly-graph-div\" style=\"height:300px; width:800px;\"></div>            <script type=\"text/javascript\">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"9d6da6f6-43cf-417c-adde-4bcf99ff35a5\")) {                    Plotly.newPlot(                        \"9d6da6f6-43cf-417c-adde-4bcf99ff35a5\",                        [{\"fill\":\"toself\",\"fillcolor\":\"rgba(128,128,128,0.2)\",\"hoverinfo\":\"skip\",\"line\":{\"color\":\"rgba(255,255,255,0)\"},\"name\":\"95% CI\",\"showlegend\":false,\"x\":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,200,199,198,197,196,195,194,193,192,191,190,189,188,187,186,185,184,183,182,181,180,179,178,177,176,175,174,173,172,171,170,169,168,167,166,165,164,163,162,161,160,159,158,157,156,155,154,153,152,151,150,149,148,147,146,145,144,143,142,141,140,139,138,137,136,135,134,133,132,131,130,129,128,127,126,125,124,123,122,121,120,119,118,117,116,115,114,113,112,111,110,109,108,107,106,105,104,103,102,101,100,99,98,97,96,95,94,93,92,91,90,89,88,87,86,85,84,83,82,81,80,79,78,77,76,75,74,73,72,71,70,69,68,67,66,65,64,63,62,61,60,59,58,57,56,55,54,53,52,51,50,49,48,47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1],\"y\":[60.99000000000001,78.89639932989789,76.50430430322979,79.82837749056361,75.19962282882058,121.81378245674574,126.4184171134668,141.9459807142033,131.1607497072982,144.19549951980974,242.38555430373782,283.13060299576097,264.66817671173203,250.29995636925162,241.71165145065632,228.66214712308502,217.59107213448416,207.58977836742747,204.60236715043996,199.3879964401374,195.29103686518332,187.82320872741627,182.82061477767658,178.4416338220935,183.94128750297602,187.358968285636,181.32664855245457,175.667723911718,171.0670168415874,167.8505271397167,165.06966645158437,161.37665827627916,157.17639712704468,154.94782096929438,154.3597367496571,155.26789222228234,152.51776296373305,149.86396752262365,147.95139973238267,145.13876373371016,146.31275166251456,144.29947045844574,141.85892526547983,143.56137473948067,145.31202901952133,142.78842939616928,140.195146193582,137.72814592599084,135.4116827145926,134.7434535077403,132.81234453690556,132.59462269982882,214.0605972039556,210.57778015263932,209.41554661046877,207.35338443946102,203.9846422529049,200.9739808096788,198.98164808798916,195.83731698155447,193.27759068195775,190.68177061311337,188.65229125664985,187.4238088003975,192.79230163498198,191.54018758299216,192.34770103998923,193.21007615734797,191.38014658122972,189.062011659703,186.776

#-------------------------------------------------------------------------------
# Cell 19 (code)
#-------------------------------------------------------------------------------
# (empty code cell)
