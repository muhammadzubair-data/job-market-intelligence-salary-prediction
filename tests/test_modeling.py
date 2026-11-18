import numpy as np
import pandas as pd
import pytest

from jobscope.config import MODEL_FEATURES, RANDOM_STATE, TEST_SIZE
from jobscope.data import generate_demo
from jobscope.modeling import (
    SkillBinarizer,
    load_model,
    predict_salary,
    predict_with_interval,
    save_artifacts,
    train_and_compare,
)
from jobscope.validation import validate


def test_all_candidates_compared(trained):
    assert set(trained.cv_results.model) == {"Ridge Regression", "Random Forest", "Gradient Boosting", "HistGradientBoosting"}
    assert "Baseline (median)" in set(trained.holdout_results.model)
    # selected model is the CV winner, not the holdout winner
    assert trained.best_model_name == trained.cv_results.sort_values("cv_mae_usd").iloc[0].model


def test_beats_naive_baseline(trained):
    base = trained.holdout_results.set_index("model").loc["Baseline (median)"]
    assert trained.holdout_metrics["mae_usd"] < base.mae_usd
    assert 0 < trained.holdout_metrics["r2"] <= 1


def test_split_sizes_and_disjoint(trained, demo_df):
    assert trained.test_rows == int(np.ceil(len(demo_df) * TEST_SIZE))
    assert trained.train_rows + trained.test_rows == len(demo_df)


def test_interval_and_coverage(trained):
    point, low, high = predict_with_interval(trained, "Data Scientist", "Germany", 5, "Full-time", "Hybrid", "Python, SQL")
    assert low < point < high
    assert 0.5 <= trained.holdout_interval_coverage <= 1.0


def test_permutation_importance_covers_inputs(trained):
    assert set(trained.permutation_importance.feature) == set(MODEL_FEATURES)


def test_unseen_category_does_not_crash(trained):
    assert predict_salary(trained.model, "Quantum Wizard", "Atlantis", 3, "Freelance", "Remote", "Rust") > 0


def test_more_experience_not_cheaper_on_average(trained):
    lo = np.mean([predict_salary(trained.model, r, "Canada", 1, "Full-time", "Hybrid", "Python") for r in ["Data Analyst", "ML Engineer"]])
    hi = np.mean([predict_salary(trained.model, r, "Canada", 12, "Full-time", "Hybrid", "Python") for r in ["Data Analyst", "ML Engineer"]])
    assert hi > lo


def test_skill_binarizer_learns_vocab_from_fit_data_only():
    train = pd.DataFrame({"skills": ["Python, SQL"] * 6})
    test = pd.DataFrame({"skills": ["Spark, Python"]})
    sb = SkillBinarizer(min_count=5).fit(train)
    assert sb.vocabulary_ == ["Python", "SQL"]
    assert sb.transform(test).tolist() == [[1.0, 0.0]]
    assert list(sb.get_feature_names_out()) == ["skill__Python", "skill__SQL"]


def test_description_is_not_a_feature():
    assert "description" not in MODEL_FEATURES and "salary_usd" not in MODEL_FEATURES


def test_too_few_rows():
    with pytest.raises(ValueError, match="At least"):
        train_and_compare(validate(generate_demo(20)))


def test_reproducible_metrics():
    df = validate(generate_demo(150))
    a = train_and_compare(df, importance_repeats=1).holdout_metrics
    b = train_and_compare(df, importance_repeats=1).holdout_metrics
    assert a == b
    assert RANDOM_STATE == 42


def test_artifacts_roundtrip(trained, tmp_path):
    save_artifacts(trained, tmp_path, "synthetic_demo")
    for name in ["salary_model.joblib", "metrics.json", "cv_comparison.csv", "holdout_comparison.csv", "permutation_importance.csv"]:
        assert (tmp_path / name).exists()
    bundle = load_model(tmp_path / "salary_model.joblib")
    args = ("Data Analyst", "India", 2, "Contract", "Remote", "Excel, SQL")
    assert predict_salary(bundle["model"], *args) == pytest.approx(predict_salary(trained.model, *args))
