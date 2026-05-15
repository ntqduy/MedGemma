"""Thin model helper exports for MedGemma.

MedGemma uses Hugging Face image-text classes instead of a local custom
architecture, so the actual loader remains shared with the evaluator. This
module gives the project a Med3DVLM-like `src/model` import surface.
"""

from __future__ import annotations

from src.eval.medgemma_eval import ModelBundle, collect_model_stats, generate_prediction, load_model_bundle

__all__ = [
    "ModelBundle",
    "collect_model_stats",
    "generate_prediction",
    "load_model_bundle",
]
