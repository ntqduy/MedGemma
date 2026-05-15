#!/usr/bin/env python3
"""Med3DVLM-style captioning evaluation entrypoint for MedGemma."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.medgemma_eval import main as evaluate_main  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not any(arg == "--config" or arg.startswith("--config=") for arg in args):
        args = ["--config", str(PROJECT_ROOT / "config" / "CAP_task.yaml"), *args]
    if not any(arg == "--task" or arg.startswith("--task=") for arg in args):
        args = ["--task", "cap", *args]
    return evaluate_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
