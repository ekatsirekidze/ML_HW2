"""Validate every model_experiment_*.ipynb in `notebooks/`.

Checks:
  1. The file is valid JSON.
  2. Every code cell is syntactically valid Python.

Run from repo root:
    python tools/validate_notebooks.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def main() -> int:
    failures = 0
    for path in sorted(NB_DIR.glob("model_experiment_*.ipynb")):
        try:
            nb = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[FAIL] {path.name}: invalid JSON: {exc}")
            failures += 1
            continue

        n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
        cell_failures = []
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            try:
                ast.parse(src)
            except SyntaxError as exc:
                cell_failures.append((i, exc))

        if cell_failures:
            print(f"[FAIL] {path.name}: {len(cell_failures)} cells with syntax errors")
            for i, exc in cell_failures:
                print(f"        cell {i}: line {exc.lineno}, {exc.msg}")
            failures += 1
        else:
            print(f"[OK]   {path.name}: {len(nb['cells'])} cells ({n_code} code), all parse")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
