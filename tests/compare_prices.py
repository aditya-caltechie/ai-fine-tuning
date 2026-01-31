#!/usr/bin/env python3
"""
Compare dataset prices vs deployed agent inference.

This script:
1) Loads N items from the Hugging Face dataset used in training
2) Extracts each item's Title and ground-truth price from the row text
3) Calls the deployed Modal model via the repo CLI:
     uv run python src/inference.py agent "<input>"
4) Prints a small report (per-item + summary stats)

  uv run python src/inference.py agent "<Title>"

and prints per-item comparisons plus simple error stats.
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_CLI = REPO_ROOT / "src" / "inference.py"

DEFAULT_DATASET = os.getenv("PRICER_DATASET_NAME", "ed-donner/items_prompts_lite")

TITLE_RE = re.compile(r"(?im)^\s*title:\s*(.+?)\s*$")
PRICE_RE = re.compile(r"(?i)price\s+is\s*\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)")


def run_price(title: str) -> float:
    cmd = ["uv", "run", "python", str(INFERENCE_CLI), "price", title]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Price CLI failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")

    # The CLI is expected to print a float on success.
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(
            "Price CLI returned 0 but printed nothing.\n\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    out = lines[-1]
    try:
        return float(out)
    except ValueError as e:
        raise RuntimeError(f"Could not parse float from CLI output: {out!r}\n\nFull stdout:\n{proc.stdout}") from e


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare dataset prices vs agent inference.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset name (default: %(default)s)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--n", type=int, default=5, help="Number of examples to evaluate (default: %(default)s)")
    args = parser.parse_args()

    try:
        from datasets import load_dataset  # type: ignore
    except ModuleNotFoundError:
        print("Missing dependency: datasets. Run `uv sync` first.", file=sys.stderr)
        return 2

    print(f"Loading dataset: {args.dataset} split={args.split!r}")
    ds = load_dataset(args.dataset)[args.split]
    n = min(args.n, len(ds))
    print(f"Evaluating {n} rows\n")

    abs_errors: list[float] = []
    rows_evaluated = 0

    for i in range(n):
        row = ds[i]
        # Best-effort: dataset rows can have multiple text fields. Join all string-like content.
        parts: list[str] = []
        for v in row.values():
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, (int, float)):
                parts.append(str(v))
            elif isinstance(v, (list, tuple)):
                parts.extend([x for x in v if isinstance(x, str)])
        blob = "\n".join(parts)

        m_title = TITLE_RE.search(blob)
        m_price = PRICE_RE.search(blob)
        if not m_title or not m_price:
            print(f"[{i:02d}] Skipping: could not parse title/price from row keys={list(row.keys())}")
            continue

        title = m_title.group(1).strip()
        true_price = float(m_price.group(1).replace(",", ""))

        # Call the deployed Modal model via the repo CLI:
        pred = run_price(title)
        err = abs(pred - true_price)
        abs_errors.append(err)
        rows_evaluated += 1

        print(
            f"[{i:02d}] title={title!r}\n"
            f"     true=${true_price:,.2f}  pred=${pred:,.2f}  abs_err=${err:,.2f}\n"
        )

    if rows_evaluated == 0:
        print("No rows evaluated (could not parse any examples).", file=sys.stderr)
        return 1

    mean_abs = sum(abs_errors) / len(abs_errors)
    med_abs = statistics.median(abs_errors)
    p90_abs = sorted(abs_errors)[max(0, int(0.9 * len(abs_errors)) - 1)]

    print("Summary")
    print(f"- rows_evaluated: {rows_evaluated}")
    print(f"- mean_abs_error: {mean_abs:,.2f}")
    print(f"- median_abs_error: {med_abs:,.2f}")
    print(f"- p90_abs_error: {p90_abs:,.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Example output:
# ===============================

# Adityas-Laptop:ai-fine-tuning averma$ uv run python src/compare_dataset_agent_prices.py --n 5 --split test
# Loading dataset: ed-donner/items_prompts_lite split='test'
# Evaluating 5 rows

# [00] title='Excess V2 Distortion/Modulation Pedal'
#      true=$219.00  pred=$189.00  abs_err=$30.00

# [01] title='Telpo Headlight Assembly for 2015‑2017 Toyota Camry'
#      true=$115.99  pred=$110.00  abs_err=$5.99

# [02] title='NewPower99 6000\u202fmAh Battery Replacement Kit for Samsung Galaxy Tab S3 9.7'
#      true=$54.95  pred=$40.00  abs_err=$14.95

# [03] title='EXMAX NG-10X Fresnel Lens Focusing Adapter'
#      true=$69.98  pred=$25.00  abs_err=$44.98

# [04] title='Raven Pro Document Scanner Stand'
#      true=$29.85  pred=$110.00  abs_err=$80.15

# Summary
# - rows_evaluated: 5
# - mean_abs_error: 35.21
# - median_abs_error: 30.00
# - p90_abs_error: 44.98
# Adityas-Laptop:ai-fine-tuning averma$ 