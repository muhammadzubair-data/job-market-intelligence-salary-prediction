"""Backward-compatible API (v1). New code should import from the submodules.

Kept so existing scripts that used ``jobscope.core`` keep working.
"""
from __future__ import annotations

import pandas as pd

from .config import CATEGORICAL_FEATURES as CATEGORIES  # noqa: F401
from .config import REQUIRED_COLUMNS as REQUIRED  # noqa: F401
from .modeling import predict_salary, train_and_compare  # noqa: F401
from .nlp import extract_skills as _extract_canonical
from .nlp import skill_frequency as _skill_frequency
from .validation import validate  # noqa: F401


def extract_skills(text: str) -> list[str]:
    """v1 behaviour: lowercase canonical skill names, sorted."""
    return sorted(s.lower() for s in _extract_canonical(text))


def skill_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """v1 behaviour: columns ``skill`` and ``postings``."""
    return _skill_frequency(df)[["skill", "postings"]]


def train(data: pd.DataFrame):
    """v1 behaviour: returns (model, metrics dict) using the selected best model."""
    result = train_and_compare(data)
    metrics = {
        "best_model": result.best_model_name,
        "mae_usd": round(result.holdout_metrics["mae_usd"], 2),
        "rmse_usd": round(result.holdout_metrics["rmse_usd"], 2),
        "r2": round(result.holdout_metrics["r2"], 3),
        "train_rows": result.train_rows,
        "test_rows": result.test_rows,
    }
    return result.model, metrics
