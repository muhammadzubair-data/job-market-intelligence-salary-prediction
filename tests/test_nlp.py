import pandas as pd

from jobscope import nlp


def test_synonyms_map_to_canonical():
    found = nlp.extract_skills("PostgreSQL, PySpark on Databricks, scikit-learn and k8s")
    assert found == ["Kubernetes", "Machine Learning", "SQL", "Spark"]


def test_word_boundaries():
    # "gitlab" -> Git is intended; "digital" / "restaurant" must not match Git or R
    assert nlp.extract_skills("digital restaurant") == []
    assert "Git" in nlp.extract_skills("CI on GitLab")


def test_negation_is_skipped():
    assert "Tableau" not in nlp.extract_skills("Python needed. No Tableau required.")
    assert "Spark" not in nlp.extract_skills("Spark is not required")
    assert nlp.extract_skills("Python needed. No Tableau required.") == ["Python"]


def test_negation_does_not_hide_positive_mention():
    assert "Tableau" in nlp.extract_skills("Tableau and Power BI. Spark not required.")


def test_empty_text():
    assert nlp.extract_skills("") == []
    assert nlp.extract_skills(None) == []


def test_skill_frequency_shares(demo_df):
    freq = nlp.skill_frequency(demo_df)
    assert {"skill", "category", "postings", "share", "median_salary_usd"} <= set(freq.columns)
    assert freq.share.between(0, 1).all()
    assert freq.postings.is_monotonic_decreasing


def test_skill_pairs_and_matrix():
    df = nlp.add_extracted_skills(pd.DataFrame({
        "skills": ["Python, SQL", "Python, SQL", "Excel"],
        "description": ["", "", ""],
        "salary_usd": [1, 2, 3],
    }))
    pairs = nlp.skill_pairs(df)
    assert pairs.iloc[0][["skill_a", "skill_b", "postings"]].tolist() == ["Python", "SQL", 2]
    assert pairs.iloc[0].lift == 1.5
    mat = nlp.cooccurrence_matrix(df, ["Python", "SQL", "Excel"])
    assert mat.loc["Python", "SQL"] == 2 and mat.loc["Excel", "Excel"] == 1
