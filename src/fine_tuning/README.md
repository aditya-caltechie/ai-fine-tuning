# Fine-tuning on Google Colab (T4) — how to run these notebooks

This folder contains **fine-tuning notebooks** (and a plain-Python reference script) for the “Price is Right” project.

If you want to run them, the intended environment is **Google Colab** (GPU). Start here: [Google Colaboratory](https://colab.google/).

Open these notebook on Google Collab and choose T4 envionment, which would run these notebooks on T4 GPU. It would load base modal and LORA weights there. Later you can run inference on those fine-tuned model ( base model + LORA weights)

---

## What’s in here?

Notebooks live under `src/fine_tuning/notebooks/`:

- **`1_basemodel_evaluation.ipynb`**: installs training deps, loads a quantized base model, and evaluates how the **base model** performs on the pricing task (before fine-tuning).
- **`2_fine-tuning_via_QLORA.ipynb`**: runs **QLoRA supervised fine-tuning (SFT)** and pushes the resulting **LoRA adapter** to Hugging Face.
- **`3_testing_fine-tuned-model.ipynb`**: loads the base model + your pushed adapter and evaluates the **fine-tuned** model on the same task.

Also:

- **`notebooks/lora_training_reference.py`**: a **reference-only** plain Python walkthrough of the same LoRA/QLoRA steps (no Colab `!` commands). It’s meant for understanding, not for running locally.

---

## Option A (recommended): Open from GitHub in Colab

If these notebooks are pushed to GitHub, you can open them directly in Colab:

1. In a browser, open Colab: [Google Colaboratory](https://colab.google/).
2. **File → Open notebook → GitHub**
3. Paste your repo URL, then pick one of:
   - `src/fine_tuning/notebooks/1_basemodel_evaluation.ipynb`
   - `src/fine_tuning/notebooks/2_fine-tuning_via_QLORA.ipynb`
   - `src/fine_tuning/notebooks/3_testing_fine-tuned-model.ipynb`

Tip: Colab also supports URLs of the form:

`https://colab.research.google.com/github/<USER>/<REPO>/blob/main/src/fine_tuning/notebooks/<NOTEBOOK>.ipynb`

---

## Option B: Upload the notebook(s) to Colab

If you’re running locally and haven’t pushed to GitHub:

1. Go to [Google Colaboratory](https://colab.google/).
2. **File → Upload notebook**
3. Upload one of the `.ipynb` files from `src/fine_tuning/notebooks/`.

---

## Connect a T4 GPU (Colab)

1. In Colab: **Runtime → Change runtime type**
2. Set:
   - **Hardware accelerator**: `GPU`
   - **GPU type**: `T4` (on free tier, you often get T4 automatically; the menu may vary)
3. Click **Save**

Verify the GPU:

```python
!nvidia-smi
```

and (optional):

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

## Required secrets (Hugging Face, and optionally W&B)

These notebooks download a base model + dataset from Hugging Face and (during training) push your adapter back to the Hub.

### Hugging Face token (required)

1. Create a token at Hugging Face (Settings → Access Tokens).
2. In Colab, click the **key icon** (Secrets) on the left sidebar.
3. Add a secret named **`HF_TOKEN`** with your token value.

Notes:
- Some base models (e.g. Llama-family) require you to accept model terms on Hugging Face before download will work.

### Weights & Biases (optional; only used in training notebook)

`2_fine-tuning_via_QLORA.ipynb` can log training to W&B.

1. Create a W&B API key.
2. Add a Colab secret named **`WANDB_API_KEY`**.

---

## Installing dependencies (why `pip` matters)

These notebooks use Colab-style cells like:

- `!pip install ...`
- `!wget ...`

That’s why they’re meant to run on Colab. If you ever see:

`pip: command not found`

it usually means you’re **not** running inside a normal Colab/Python environment (or the kernel is misconfigured). A common workaround is:

```python
!python -m pip install -U pip
!python -m pip install -q --upgrade bitsandbytes trl
```

---

## What to expect in each notebook

### `1_basemodel_evaluation.ipynb` (baseline)

- **Goal**: see how the base model performs on the pricing task before training.
- **What it does**:
  - Installs quantization/training helpers (e.g., bitsandbytes)
  - Pulls a helper script `util.py` via `wget` (used for evaluation)
  - Loads the dataset (lite/full) from Hugging Face
  - Loads a quantized base model (4-bit) and runs evaluation
- **Output**: printed evaluation metrics / plots showing baseline performance.

### `2_fine-tuning_via_QLORA.ipynb` (train adapters)

- **Goal**: fine-tune using **QLoRA** on GPU.
- **Hardware guidance (from the notebook)**:
  - `LITE_MODE=True` → intended for a **free T4** (smaller dataset/settings)
  - `LITE_MODE=False` → intended for a higher-memory GPU like an **A100**
- **What it does**:
  - Logs into Hugging Face (and optionally W&B)
  - Loads dataset splits
  - Loads the base model quantized (4-bit) and configures LoRA target modules
  - Runs TRL’s SFT training
  - **Pushes adapter checkpoints to Hugging Face** every `SAVE_STEPS`
- **Output**:
  - Training logs (loss curves; optionally W&B dashboards)
  - A Hugging Face Hub repo containing your adapter checkpoints (private by default in the notebook)

Important: Colab sessions can end unexpectedly. Because the notebook pushes to the Hub during training, you can often resume by re-opening the notebook and pointing at the last saved revision/checkpoint.

### `3_testing_fine-tuned-model.ipynb` (evaluate tuned model)

- **Goal**: compare fine-tuned vs baseline performance.
- **What it does**:
  - Downloads the base model + your fine-tuned adapter from the Hub
  - Loads them together (base + LoRA adapter)
  - Runs the same evaluation as the baseline notebook
- **Output**: evaluation results showing the improvement you got from fine-tuning.

---

## Where this connects to “serving” in this repo

After you fine-tune and publish adapter weights, the inference side (`src/inference/`) can load:

- the **base model**, plus
- your **LoRA adapter**

to serve predictions (in this repo: via Modal in `src/inference/pricing_service.py`).

