"""Schema validation, cleaning and a data-quality report."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import (
    CATEGORICAL_FEATURES,
    MAX_EXPERIENCE_YEARS,
    MAX_SALARY_USD,
    REQUIRED_COLUMNS,
)


@dataclass
class QualityReport:
    input_rows: int = 0
    missing_target: int = 0
    out_of_range: int = 0
    duplicates: int = 0
    output_rows: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def removed(self) -> int:
        return self.input_rows - self.output_rows

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame([
            ("Input rows", self.input_rows),
            ("Missing / non-numeric salary or experience", self.missing_target),
            ("Out-of-range salary or experience", self.out_of_range),
            ("Exact duplicate rows", self.duplicates),
            ("Valid rows used", self.output_rows),
        ], columns=["Check", "Rows"])


def _count_skills(text: str) -> int:
    return len([v for v in str(text).split(",") if v.strip()])


def validate(df: pd.DataFrame, return_report: bool = False):
    """Validate and clean a job-postings frame.

    * required columns must be present (case-insensitive)
    * text columns are stripped and missing values become ``"Unknown"``
    * salary must be numeric, > 0 and <= 1,000,000 annual USD
    * experience must be numeric and within 0-60 years
    * exact duplicate rows are removed (prevents train/test leakage of copies)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("validate() expects a pandas DataFrame")
    data = df.copy()
    data.columns = [str(c).strip().lower() for c in data.columns]
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    report = QualityReport(input_rows=len(data))
    for col in CATEGORICAL_FEATURES + ["skills", "description"]:
        data[col] = data[col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    for col in ("experience_years", "salary_usd"):
        data[col] = pd.to_numeric(data[col], errors="coerce")

    before = len(data)
    data = data.dropna(subset=["salary_usd", "experience_years"])
    report.missing_target = before - len(data)

    before = len(data)
    in_range = (
        (data.salary_usd > 0)
        & (data.salary_usd <= MAX_SALARY_USD)
        & (data.experience_years >= 0)
        & (data.experience_years <= MAX_EXPERIENCE_YEARS)
    )
    data = data[in_range]
    report.out_of_range = before - len(data)

    before = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    report.duplicates = before - len(data)

    if data.empty:
        raise ValueError("No valid rows remain after validation")

    data["skill_count"] = data["skills"].apply(_count_skills)
    report.output_rows = len(data)
    if "source" in data.columns and data["source"].astype(str).str.contains("SYNTHETIC", case=False).all():
        report.notes.append("All rows are labelled as synthetic demonstration data.")
    return (data, report) if return_report else data


def is_synthetic(df: pd.DataFrame) -> bool:
    """True when every row is flagged as synthetic in the ``source`` column."""
    return "source" in df.columns and df["source"].astype(str).str.contains("SYNTHETIC", case=False).all()
