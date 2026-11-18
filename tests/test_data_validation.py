import pandas as pd
import pytest

from jobscope.config import DEMO_CSV
from jobscope.data import generate_demo, load_csv, load_demo
from jobscope.validation import is_synthetic, validate


def test_bundled_demo_csv_matches_generator():
    """The committed CSV must be exactly what the seeded generator produces."""
    pd.testing.assert_frame_equal(pd.read_csv(DEMO_CSV), generate_demo(), check_dtype=False)


def test_load_demo_is_synthetic():
    assert is_synthetic(validate(load_demo()))


def test_load_csv_from_bytes_normalises_headers():
    raw = generate_demo(5).rename(columns={"job_title": " Job_Title "}).to_csv(index=False).encode()
    df = load_csv(raw)
    assert "job_title" in df.columns


def test_load_csv_empty():
    with pytest.raises(ValueError, match="empty"):
        load_csv(b"")


def test_load_csv_row_cap():
    raw = generate_demo(20).to_csv(index=False).encode()
    with pytest.raises(ValueError, match="more than 10"):
        load_csv(raw, max_rows=10)


def test_validation_report_counts():
    df = generate_demo(10)
    df["salary_usd"] = df["salary_usd"].astype(object)
    df.loc[0, "salary_usd"] = "not a number"
    df.loc[1, "experience_years"] = 99
    df.loc[2, "salary_usd"] = 5_000_000
    df = pd.concat([df, df.iloc[[3]]], ignore_index=True)  # exact duplicate
    clean, report = validate(df, return_report=True)
    assert report.input_rows == 11
    assert report.missing_target == 1
    assert report.out_of_range == 2
    assert report.duplicates == 1
    assert report.output_rows == len(clean) == 7
    assert report.removed == 4


def test_validation_case_insensitive_columns_and_blank_text():
    df = generate_demo(3)
    df.columns = [c.upper() for c in df.columns]
    df.loc[0, "COUNTRY"] = "   "
    clean = validate(df)
    assert clean.loc[0, "country"] == "Unknown"
    assert "skill_count" in clean.columns


def test_validation_all_invalid_raises():
    df = generate_demo(3)
    df["salary_usd"] = -1
    with pytest.raises(ValueError, match="No valid rows"):
        validate(df)


def test_validate_rejects_non_dataframe():
    with pytest.raises(TypeError):
        validate([1, 2, 3])
