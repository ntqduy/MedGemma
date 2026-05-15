#!/usr/bin/env python3
"""Backward-compatible evaluation shim.

The MedGemma evaluator now lives under `src/eval` to match the Med3DVLM-style
project layout. This file keeps older commands and `train.py` imports working.
"""

from __future__ import annotations

from src.eval.medgemma_eval import *  # noqa: F401,F403
from src.eval.medgemma_eval import main


if __name__ == "__main__":
    raise SystemExit(main())
