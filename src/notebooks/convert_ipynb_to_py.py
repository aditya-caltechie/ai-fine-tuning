#!/usr/bin/env python3
"""
Convert a Jupyter/Colab .ipynb notebook into a single .py script with comments.

What it does:
- Markdown cells become commented blocks.
- Code cells become executable Python code blocks.
- Cell boundaries are preserved with clear headers.

Usage:
  python convert_ipynb_to_py.py path/to/notebook.ipynb -o output.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _as_lines(source: Any) -> list[str]:
    """Normalize notebook cell `source` to a list of lines (without trailing newlines)."""
    if source is None:
        return []
    if isinstance(source, str):
        # Keep line endings stable across platforms; splitlines() drops trailing newline.
        return source.splitlines()
    if isinstance(source, list):
        # Jupyter typically stores a list of strings (often with trailing '\n').
        out: list[str] = []
        for chunk in source:
            if chunk is None:
                continue
            out.extend(str(chunk).splitlines())
        return out
    # Fallback: stringify unknown shapes.
    return str(source).splitlines()


def _write_markdown_as_comments(lines: list[str]) -> list[str]:
    """Render markdown cell content as Python comments."""
    if not lines:
        return ["# (empty markdown cell)"]
    return [f"# {line}" if line.strip() else "#" for line in lines]


def convert_ipynb_to_py(ipynb_path: Path) -> list[str]:
    nb = json.loads(ipynb_path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])

    out: list[str] = []
    out.append("#!/usr/bin/env python3")
    out.append(f"# Generated from: {ipynb_path.name}")
    out.append("# NOTE: This file was auto-generated; edit the notebook if you want to keep changes in sync.")
    out.append("")

    for idx, cell in enumerate(cells, start=1):
        cell_type = cell.get("cell_type", "unknown")
        source_lines = _as_lines(cell.get("source"))

        out.append("#" + "-" * 79)
        out.append(f"# Cell {idx} ({cell_type})")
        out.append("#" + "-" * 79)

        if cell_type == "markdown":
            out.extend(_write_markdown_as_comments(source_lines))
            out.append("")
            continue

        if cell_type == "code":
            if not source_lines:
                out.append("# (empty code cell)")
            else:
                # Write code as-is.
                out.extend(source_lines)

            # Preserve cell outputs as comments if present (optional, best-effort).
            outputs = cell.get("outputs")
            if outputs:
                out.append("")
                out.append("# --- Outputs (from notebook) ---")
                try:
                    # Keep it concise; dump shapes, not huge binary blobs.
                    out.append("# " + json.dumps(outputs, ensure_ascii=False)[:4000].replace("\n", "\\n"))
                except Exception:
                    out.append("# (outputs could not be serialized)")
            out.append("")
            continue

        # Unknown cell types (raw, etc.)
        out.append(f"# (unhandled cell_type={cell_type!r})")
        out.extend(_write_markdown_as_comments(source_lines))
        out.append("")

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert .ipynb to .py with comments.")
    parser.add_argument("ipynb", type=Path, help="Path to notebook file (.ipynb or JSON notebook)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .py path")
    args = parser.parse_args()

    ipynb_path: Path = args.ipynb
    if not ipynb_path.exists():
        raise SystemExit(f"Notebook not found: {ipynb_path}")

    # Some Colab exports (or user-renamed files) can be valid notebook JSON without the `.ipynb` suffix.
    # Validate by inspecting the JSON structure rather than relying on the extension.
    try:
        nb_probe = json.loads(ipynb_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Input is not valid JSON: {e}") from e

    if not isinstance(nb_probe, dict) or "cells" not in nb_probe or "nbformat" not in nb_probe:
        raise SystemExit("Input does not look like a Jupyter notebook (missing 'nbformat'/'cells').")

    out_path = args.output or ipynb_path.with_suffix(".py")
    lines = convert_ipynb_to_py(ipynb_path)
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

