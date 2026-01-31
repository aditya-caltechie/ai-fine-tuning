import os
import subprocess
import sys
import unittest
from pathlib import Path


class TestComparePricesIntegration(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("RUN_INTEGRATION_TESTS") == "1",
        "Set RUN_INTEGRATION_TESTS=1 to run integration tests (network + Modal required).",
    )
    def test_compare_prices_script_runs(self):
        repo_root = Path(__file__).resolve().parents[2]
        script = repo_root / "tests" / "integration" / "compare_prices.py"

        proc = subprocess.run(
            [sys.executable, str(script), "--n", "5", "--split", "test"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)

