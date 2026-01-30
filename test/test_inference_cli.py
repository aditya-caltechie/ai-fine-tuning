import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
INFERENCE_CLI_PATH = SRC_DIR / "inference.py"


def load_inference_cli():
    """
    Load `src/inference.py` under a safe module name.

    This repo has BOTH:
    - `src/inference.py` (CLI module)
    - `src/inference/` (package)

    Loading by file path avoids import-name ambiguity.
    """

    spec = importlib.util.spec_from_file_location("inference_cli", INFERENCE_CLI_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load inference CLI module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPreprocessIfNeeded(unittest.TestCase):
    def setUp(self):
        self.cli = load_inference_cli()

    def test_returns_original_for_structured_input(self):
        text = "\n".join(
            [
                "Title: iPhone X",
                "Category: Electronics",
                "Brand: Apple",
                "Description: Smartphone",
                "Details: 64GB",
            ]
        )
        out = self.cli.preprocess_if_needed(text)
        self.assertEqual(out, text)

    def test_calls_preprocessor_for_raw_input(self):
        # Ensure the src folder is importable so `from inference.preprocessor ...` works.
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))

        # Avoid importing the real `inference.preprocessor` (it depends on litellm).
        calls: list[str] = []

        class FakePreprocessor:
            def preprocess(self, text: str) -> str:
                calls.append(text)
                return "Title: X\nCategory: Y\nBrand: Z\nDescription: D\nDetails: E"

        fake_mod = ModuleType("inference.preprocessor")
        fake_mod.Preprocessor = FakePreprocessor  # type: ignore[attr-defined]

        old = sys.modules.get("inference.preprocessor")
        sys.modules["inference.preprocessor"] = fake_mod
        try:
            out = self.cli.preprocess_if_needed("iphone 10")
            self.assertIsInstance(out, str)
            self.assertIn("Title:", out)
            self.assertEqual(calls, ["iphone 10"])
        finally:
            if old is None:
                sys.modules.pop("inference.preprocessor", None)
            else:
                sys.modules["inference.preprocessor"] = old

    def test_returns_none_if_preprocessor_raises(self):
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))

        class FakePreprocessor:
            def preprocess(self, text: str) -> str:
                raise RuntimeError("boom")

        fake_mod = ModuleType("inference.preprocessor")
        fake_mod.Preprocessor = FakePreprocessor  # type: ignore[attr-defined]

        old = sys.modules.get("inference.preprocessor")
        sys.modules["inference.preprocessor"] = fake_mod
        try:
            with patch("builtins.print") as p:
                out = self.cli.preprocess_if_needed("raw text")
                self.assertIsNone(out)
                # The CLI is expected to log failures to stdout.
                self.assertTrue(p.called)
        finally:
            if old is None:
                sys.modules.pop("inference.preprocessor", None)
            else:
                sys.modules["inference.preprocessor"] = old


class TestCliIntegration(unittest.TestCase):
    def test_help_exits_zero(self):
        proc = subprocess.run(
            [sys.executable, str(INFERENCE_CLI_PATH), "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)

    def test_unknown_command_exits_2(self):
        proc = subprocess.run(
            [sys.executable, str(INFERENCE_CLI_PATH), "definitely-not-a-command"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 2, msg=proc.stdout + proc.stderr)

