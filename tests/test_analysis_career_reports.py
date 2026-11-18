import numpy as np
import pandas as pd

from jobscope import analysis, career, reports


def test_kpis(demo_df):
    k = analysis.kpis(demo_df)
    assert k["postings"] == len(demo_df)
    assert k["p25_salary"] <= k["median_salary"] <= k["p75_salary"]


def test_statistical_summary_detects_built_in_role_effect(demo_df):
    stats = analysis.statistical_summary(demo_df).set_index("factor")
    # the synthetic generator multiplies salary by role and country, so these must be detected
    assert stats.loc["job_title", "p_value"] < 0.001
    assert stats.loc["country", "p_value"] < 0.001
    assert stats.loc["experience_years", "statistic"] > 0


def test_benjamini_hochberg():
    q = analysis.benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.5]))
    assert np.allclose(q, [0.04, 0.0533333, 0.0533333, 0.5])
    assert (q >= [0.01, 0.04, 0.03, 0.5]).all()


def test_skill_salary_association(demo_df):
    out = analysis.skill_salary_association(demo_df)
    assert {"skill", "role_adjusted_premium", "p_value", "q_value"} <= set(out.columns)


def test_skill_gap_and_recommendations(demo_df, trained):
    gap = career.skill_gap(demo_df, "Data Scientist", "Python, SQL")
    assert {"Have", "Gap"} <= set(gap.status)
    assert set(gap[gap.status == "Have"].skill) == {"Python", "SQL"}
    candidates = gap[gap.status == "Gap"].skill.tolist()[:3]
    wi = career.skill_what_if(trained.model, demo_df, "Data Scientist", "Canada", 3, "Full-time", "Remote", "Python, SQL", candidates)
    assert len(wi) == len(candidates)
    recs = career.recommendations(gap, wi)
    assert recs and all(isinstance(r, str) for r in recs)


def test_experience_curve(trained):
    curve = career.experience_curve(trained.model, "Data Analyst", "Canada", "Full-time", "Remote", "Python", 5)
    assert len(curve) == 6


def test_csv_formula_injection_neutralised():
    df = pd.DataFrame({"a": ["=HYPERLINK(\"x\")", "+1", "@SUM(A1)", "-2", "safe"], "n": [1, 2, 3, 4, 5]})
    text = reports.to_csv_bytes(df).decode()
    assert "'=HYPERLINK" in text and "'+1" in text and "'@SUM" in text and "'-2" in text and "\nsafe," in text


def test_csv_lists_are_joined(demo_df):
    text = reports.to_csv_bytes(demo_df.head(2)).decode()
    assert "[" not in text.splitlines()[1]


def test_html_report(demo_df, trained):
    data = reports.build_html_report(
        data_mode="Synthetic demo", kpis=analysis.kpis(demo_df), quality=pd.DataFrame({"Check": ["x"], "Rows": [1]}),
        by_role=analysis.salary_by(demo_df, "job_title"), by_country=analysis.salary_by(demo_df, "country"),
        skills=pd.DataFrame({"skill": ["<script>"], "postings": [1]}), stats_table=analysis.statistical_summary(demo_df),
        model_summary=trained.summary(), cv_table=trained.cv_results, holdout_table=trained.holdout_results,
        importance=trained.permutation_importance,
    ).decode()
    assert data.startswith("<!DOCTYPE html>")
    assert "Synthetic demonstration data" in data
    assert "<script>" not in data and "&lt;script&gt;" in data
