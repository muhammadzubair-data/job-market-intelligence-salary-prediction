"""Backward-compatibility tests for the original v1 API (jobscope.core)."""
import pandas as pd
import pytest

from jobscope.core import extract_skills, predict_salary, skill_frequency, train, validate
from jobscope.data import generate_demo


def test_demo_is_reproducible():
    assert generate_demo(10).equals(generate_demo(10))


def test_validation_filters_invalid():
    df = generate_demo(4)
    df.loc[0, "salary_usd"] = -5
    assert len(validate(df)) == 3


def test_missing_column():
    with pytest.raises(ValueError, match="Missing required"):
        validate(pd.DataFrame({"salary_usd": [1]}))


def test_skill_extraction_v1_lowercase():
    assert extract_skills("Python, SQL and Power BI") == ["power bi", "python", "sql"]


def test_training_and_prediction_v1():
    df = validate(generate_demo(120))
    model, metrics = train(df)
    assert metrics["test_rows"] == 24
    assert {"mae_usd", "rmse_usd", "r2", "best_model"} <= set(metrics)
    assert predict_salary(model, "Data Analyst", "Canada", 3, "Full-time", "Remote", "Python, SQL") > 0


def test_skill_frequency_v1():
    counts = skill_frequency(validate(generate_demo(10)))
    assert list(counts.columns) == ["skill", "postings"]
    assert not counts.empty
