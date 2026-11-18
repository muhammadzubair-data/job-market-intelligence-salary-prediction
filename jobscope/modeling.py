"""Salary model training, comparison, evaluation, explainability and persistence.

Evaluation protocol (designed to avoid leakage and optimistic scores):

1. Exact duplicate rows are removed during validation.
2. A single 80/20 random holdout split is made **once**, before any model
   fitting. The holdout is never used for model selection.
3. Every preprocessing step (imputation, encoding, scaling, skill vocabulary)
   lives inside a scikit-learn ``Pipeline`` so it is learned from training
   folds only.
4. Candidate models are compared with 5-fold cross-validation on the training
   portion; the lowest CV MAE wins.
5. All candidates are then scored once on the untouched holdout for reporting.
6. Prediction intervals use split-conformal calibration on out-of-fold
   residuals from the training portion; their empirical coverage is checked
   on the holdout.
7. The deployed estimator is the selected pipeline refit on all valid rows.

The free-text ``description`` is *not* a model feature, so text that might
echo the salary cannot leak into the model.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import __version__
from .config import (
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    MIN_TRAIN_ROWS,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    SKILLS_FEATURE,
    TARGET,
    TEST_SIZE,
)
from .nlp import extract_skills


class SkillBinarizer(BaseEstimator, TransformerMixin):
    """Turn a free-text skills column into multi-hot canonical skill flags.

    The vocabulary is learned during ``fit`` (training data only) and limited to
    skills seen at least ``min_count`` times.
    """

    def __init__(self, min_count: int = 5):
        self.min_count = min_count

    @staticmethod
    def _column(X) -> pd.Series:
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0].astype(str)
        return pd.Series(np.asarray(X).ravel()).astype(str)

    def fit(self, X, y=None):
        counts: dict[str, int] = {}
        for skills in self._column(X).map(extract_skills):
            for s in skills:
                counts[s] = counts.get(s, 0) + 1
        self.vocabulary_ = sorted(s for s, c in counts.items() if c >= self.min_count)
        return self

    def transform(self, X):
        parsed = self._column(X).map(lambda t: set(extract_skills(t)))
        out = np.zeros((len(parsed), len(self.vocabulary_)), dtype=float)
        for i, skills in enumerate(parsed):
            for j, s in enumerate(self.vocabulary_):
                if s in skills:
                    out[i, j] = 1.0
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"skill__{s}" for s in self.vocabulary_], dtype=object)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=3)),
        ]), CATEGORICAL_FEATURES),
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC_FEATURES),
        ("skills", SkillBinarizer(), [SKILLS_FEATURE]),
    ])


def candidate_models() -> dict[str, Any]:
    return {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, random_state=RANDOM_STATE
        ),
    }


def build_pipeline(regressor) -> Pipeline:
    """Preprocessing + regressor trained on log(salary) for multiplicative effects."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)),
    ])


def _metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae_usd": float(mean_absolute_error(y_true, y_pred)),
        "rmse_usd": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


@dataclass
class TrainingResult:
    best_model_name: str
    model: Pipeline                 # selected pipeline refit on ALL valid rows
    holdout_model: Pipeline         # selected pipeline fit on the training split only
    cv_results: pd.DataFrame
    holdout_results: pd.DataFrame
    holdout_metrics: dict[str, float]
    interval_log_halfwidth: float
    interval_level: float
    holdout_interval_coverage: float
    permutation_importance: pd.DataFrame
    error_by_role: pd.DataFrame
    error_by_country: pd.DataFrame
    holdout_predictions: pd.DataFrame
    train_rows: int
    test_rows: int
    meta: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "best_model": self.best_model_name,
            **{k: round(v, 4) for k, v in self.holdout_metrics.items()},
            "interval_level": self.interval_level,
            "holdout_interval_coverage": round(self.holdout_interval_coverage, 4),
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            **self.meta,
        }


def _group_errors(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    g = frame.assign(abs_error=(frame.actual - frame.predicted).abs()).groupby(col)
    out = g.agg(rows=("abs_error", "size"), mae_usd=("abs_error", "mean"), median_salary_usd=("actual", "median"))
    out["mae_pct_of_median"] = out.mae_usd / out.median_salary_usd
    return out.reset_index().sort_values("mae_usd", ascending=False)


def train_and_compare(
    data: pd.DataFrame,
    models: dict[str, Any] | None = None,
    cv_folds: int = CV_FOLDS,
    interval_level: float = 0.8,
    importance_repeats: int = 5,
) -> TrainingResult:
    """Run the full comparison / evaluation protocol described in the module docstring."""
    if len(data) < MIN_TRAIN_ROWS:
        raise ValueError(f"At least {MIN_TRAIN_ROWS} validated rows are needed to train and evaluate a model (got {len(data)}).")
    missing = [c for c in MODEL_FEATURES + [TARGET] if c not in data.columns]
    if missing:
        raise ValueError("Data is missing model columns: " + ", ".join(missing) + ". Run validate() first.")

    models = models or candidate_models()
    X = data[MODEL_FEATURES]
    y = data[TARGET].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    folds = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    cv_rows, holdout_rows, fitted = [], [], {}
    scoring = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}
    baseline = DummyRegressor(strategy="median").fit(X_train, y_train)
    holdout_rows.append({"model": "Baseline (median)", **_metrics(y_test, baseline.predict(X_test))})

    for name, reg in models.items():
        pipe = build_pipeline(clone(reg))
        cv = cross_validate(pipe, X_train, y_train, cv=folds, scoring=scoring, n_jobs=1)
        cv_rows.append({
            "model": name,
            "cv_mae_usd": -cv["test_mae"].mean(),
            "cv_mae_std": cv["test_mae"].std(),
            "cv_rmse_usd": -cv["test_rmse"].mean(),
            "cv_r2": cv["test_r2"].mean(),
            "cv_r2_std": cv["test_r2"].std(),
            "fit_seconds": cv["fit_time"].mean(),
        })
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        holdout_rows.append({"model": name, **_metrics(y_test, pipe.predict(X_test))})

    cv_df = pd.DataFrame(cv_rows).sort_values("cv_mae_usd").reset_index(drop=True)
    holdout_df = pd.DataFrame(holdout_rows)
    best = cv_df.iloc[0]["model"]
    best_pipe = fitted[best]
    test_pred = best_pipe.predict(X_test)
    holdout_metrics = _metrics(y_test, test_pred)

    # Split-conformal interval on the log scale using out-of-fold residuals.
    oof = cross_val_predict(build_pipeline(clone(models[best])), X_train, y_train, cv=folds)
    resid = np.abs(np.log1p(y_train.to_numpy()) - np.log1p(np.clip(oof, 0, None)))
    n = len(resid)
    q_level = min(1.0, math.ceil((n + 1) * interval_level) / n)
    halfwidth = float(np.quantile(resid, q_level, method="higher"))
    low = np.expm1(np.log1p(test_pred) - halfwidth)
    high = np.expm1(np.log1p(test_pred) + halfwidth)
    coverage = float(((y_test.to_numpy() >= low) & (y_test.to_numpy() <= high)).mean())

    imp = permutation_importance(
        best_pipe, X_test, y_test, scoring="neg_mean_absolute_error",
        n_repeats=importance_repeats, random_state=RANDOM_STATE, n_jobs=1,
    )
    importance = pd.DataFrame({
        "feature": X_test.columns,
        "mae_increase_usd": imp.importances_mean,
        "std": imp.importances_std,
    }).sort_values("mae_increase_usd", ascending=False).reset_index(drop=True)

    preds = X_test.assign(actual=y_test.to_numpy(), predicted=test_pred, low=low, high=high)
    final = build_pipeline(clone(models[best])).fit(X, y)

    return TrainingResult(
        best_model_name=best,
        model=final,
        holdout_model=best_pipe,
        cv_results=cv_df,
        holdout_results=holdout_df,
        holdout_metrics=holdout_metrics,
        interval_log_halfwidth=halfwidth,
        interval_level=interval_level,
        holdout_interval_coverage=coverage,
        permutation_importance=importance,
        error_by_role=_group_errors(preds, "job_title"),
        error_by_country=_group_errors(preds, "country"),
        holdout_predictions=preds.reset_index(drop=True),
        train_rows=len(X_train),
        test_rows=len(X_test),
        meta={"cv_folds": cv_folds, "random_state": RANDOM_STATE, "target_transform": "log1p", "package_version": __version__},
    )


def make_input(title, country, years, employment, remote, skills: str) -> pd.DataFrame:
    skills = skills or ""
    return pd.DataFrame([{
        "job_title": title,
        "country": country,
        "employment_type": employment,
        "remote_type": remote,
        "experience_years": float(years),
        "skill_count": len([s for s in skills.split(",") if s.strip()]),
        "skills": skills,
    }])[MODEL_FEATURES]


def predict_salary(model, title, country, years, employment, remote, skills: str) -> float:
    return max(0.0, float(model.predict(make_input(title, country, years, employment, remote, skills))[0]))


def predict_with_interval(result: TrainingResult, title, country, years, employment, remote, skills: str):
    point = predict_salary(result.model, title, country, years, employment, remote, skills)
    low = float(np.expm1(np.log1p(point) - result.interval_log_halfwidth))
    high = float(np.expm1(np.log1p(point) + result.interval_log_halfwidth))
    return point, max(0.0, low), high


def save_artifacts(result: TrainingResult, output: Path, data_type: str) -> None:
    """Persist the deployable model plus human-readable evaluation outputs."""
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": result.model, "interval_log_halfwidth": result.interval_log_halfwidth,
                 "interval_level": result.interval_level, "best_model": result.best_model_name},
                output / "salary_model.joblib")
    (output / "metrics.json").write_text(json.dumps({**result.summary(), "data_type": data_type}, indent=2))
    result.cv_results.round(4).to_csv(output / "cv_comparison.csv", index=False)
    result.holdout_results.round(4).to_csv(output / "holdout_comparison.csv", index=False)
    result.permutation_importance.round(4).to_csv(output / "permutation_importance.csv", index=False)


def load_model(path: Path) -> dict:
    """Load a model bundle. Only load files you created yourself: joblib can execute code."""
    return joblib.load(path)
