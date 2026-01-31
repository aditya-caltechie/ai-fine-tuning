#!/usr/bin/env python3
"""
Simple runner for the pricing project.

Run from repo root:

Step 1 (MUST): deploy the Modal service
  uv run python src/inference.py deploy

Step 2: call the service directly 
  uv run python src/inference.py price "raw text here"
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    load_dotenv = None


SRC_DIR = Path(__file__).resolve().parent
SERVICE_MODULE = "inference.pricing_service"  # file: src/inference/pricing_service.py
APP_NAME = "pricer-service"  # modal.App("pricer-service")

USAGE = """\
Usage (run from repo root):

  Step 1 (MUST):
    uv run python src/inference.py deploy

  Step 2:
    uv run python src/inference.py price "<raw text>"
"""


def preprocess_if_needed(text: str) -> str | None:
    """
    Ensure input is in the structured format the fine-tuned model expects.
    - If already structured, return as-is.
    - Else run Preprocessor() (Groq default) and return structured text.
    """

    prefixes = ("title:", "category:", "brand:", "description:", "details:")
    hits = sum(
        1 for line in text.splitlines() if any(line.strip().lower().startswith(p) for p in prefixes)
    )
    if hits >= 3:
        return text

    from inference.preprocessor import Preprocessor

    try:
        return Preprocessor().preprocess(text)
    except Exception as e:
        print(f"Preprocess failed: {e}")
        return None


def cmd_deploy() -> int:
    cmd = ["uv", "run", "modal", "deploy", "-m", SERVICE_MODULE]
    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(SRC_DIR))


def cmd_price(text: str) -> int:
    import modal
    from inference.logger import Logger

    logger = Logger()
    logger.log("Querying the fine-tuned model")
    processed = preprocess_if_needed(text)
    if processed is None:
        return 1

    try:
        Pricer = modal.Cls.from_name(APP_NAME, "Pricer")
        pricer = Pricer()
        # Deployed-app logs are best viewed via `logs`, but enable_output can still show some SDK output.
        with modal.enable_output():
            result = pricer.price.remote(processed)
            logger.log("Result: " + str(result))
    except Exception as e:
        print(f"Pricing failed: {e}")
        return 1
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO)

    if load_dotenv is not None:
        load_dotenv(override=True)

    # Allow `import inference.preprocessor`, etc. when running `python src/inference.py ...`
    src_dir_str = str(SRC_DIR)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)

    argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(USAGE)
        return 0

    cmd = argv[0]

    if cmd == "deploy":
        return cmd_deploy()
    if cmd == "price":
        if len(argv) < 2:
            print(USAGE)
            return 2
        return cmd_price(argv[1])

    print(f"Unknown command: {cmd}\n")
    print(USAGE)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

