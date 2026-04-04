#!/usr/bin/env python3
"""Execute a notebook's code cells without launching a Jupyter kernel.

This is useful in restricted environments where `jupyter nbconvert --execute`
cannot start a kernel because local socket creation is blocked.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run notebook code cells sequentially in a plain Python process."
    )
    parser.add_argument("notebook", type=Path, help="Path to the .ipynb file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    notebook_path = args.notebook.resolve()
    if not notebook_path.exists():
        print(f"Notebook not found: {notebook_path}", file=sys.stderr)
        return 1

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    with notebook_path.open() as f:
        notebook = json.load(f)

    global_ns = {"__name__": "__main__"}

    for idx, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        print(f"\n===== RUN CELL {idx} =====", flush=True)
        try:
            exec(compile(source, f"{notebook_path.name}:cell_{idx}", "exec"), global_ns)
            if "plt" in global_ns:
                global_ns["plt"].close("all")
        except Exception as exc:
            print(f"ERROR IN CELL {idx}: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
            return 1

    print("\nNOTEBOOK CELLS COMPLETED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
