import os
import unittest


class TestCompareDatasetAgentPrices(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("RUN_DATASET_AGENT_COMPARE") == "1",
        "Set RUN_DATASET_AGENT_COMPARE=1 to run this integration test (network + Modal required).",
    )
    def test_compare_20_items_runs(self):
        """
        Integration smoke test:
        - loads 20 rows from HF dataset
        - calls deployed Modal service via `src/inference.py agent`

        Skipped by default so CI stays offline-safe.
        """
        import subprocess
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "src" / "compare_dataset_agent_prices.py"

        proc = subprocess.run(
            [
                sys.executable,
                str(script),
                "--n",
                "20",
                "--split",
                "test",
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)

