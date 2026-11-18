"""Descriptive analytics and statistical tests.

All tests describe the *supplied sample*. With synthetic demo data they only
recover the generator's own formula and must not be read as market findings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def kpis(df: pd.DataFrame) -> dict[str, float]:
    return {
        "postings": int(len(df)),
        "median_salary": float(df.salary_usd.median()),
        "p25_salary": float(df.salary_usd.quantile(0.25)),
        "p75_salary": float(df.salary_usd.quantile(0.75)),
        "countries": int(df.country.nunique()),
        "roles": int(df.job_title.nunique()),
        "remote_share": float((df.remote_type.str.lower() == "remote").mean()),
        "median_experience": float(df.experience_years.median()),
    }


def salary_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = (
        df.groupby(col)
        .salary_usd.agg(postings="size", median="median", p25=lambda s: s.quantile(0.25), p75=lambda s: s.quantile(0.75), mean="mean")
        .reset_index()
        .sort_values("median", ascending=False)
    )
    out["share"] = out.postings / out.postings.sum()
    return out


def experience_band(years: pd.Series) -> pd.Series:
    return pd.cut(
        years,
        bins=[-0.1, 2, 5, 9, 60],
        labels=["Entry (0-2y)", "Mid (3-5y)", "Senior (6-9y)", "Lead (10y+)"],
    )


def kruskal_test(df: pd.DataFrame, col: str, min_group: int = 5) -> dict:
    """Kruskal-Wallis H test: do salary distributions differ across groups?

    Effect size: epsilon-squared (0 = none, 1 = all variance between groups).
    """
    groups = [g.salary_usd.to_numpy() for _, g in df.groupby(col) if len(g) >= min_group]
    if len(groups) < 2:
        return {"test": "Kruskal-Wallis", "factor": col, "groups": len(groups), "statistic": np.nan, "p_value": np.nan, "effect_size": np.nan}
    h, p = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    eps2 = (h - len(groups) + 1) / (n - len(groups)) if n > len(groups) else np.nan
    return {"test": "Kruskal-Wallis", "factor": col, "groups": len(groups), "statistic": float(h), "p_value": float(p), "effect_size": float(max(eps2, 0))}


def spearman_test(df: pd.DataFrame, col: str = "experience_years") -> dict:
    if df[col].nunique() < 2 or len(df) < 5:
        return {"test": "Spearman", "factor": col, "groups": np.nan, "statistic": np.nan, "p_value": np.nan, "effect_size": np.nan}
    rho, p = stats.spearmanr(df[col], df.salary_usd)
    return {"test": "Spearman rank correlation", "factor": col, "groups": np.nan, "statistic": float(rho), "p_value": float(p), "effect_size": float(rho)}


def statistical_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        kruskal_test(df, "job_title"),
        kruskal_test(df, "country"),
        kruskal_test(df, "remote_type"),
        kruskal_test(df, "employment_type"),
        spearman_test(df, "experience_years"),
        spearman_test(df, "skill_count"),
    ]
    out = pd.DataFrame(rows)
    out["significant_at_0.05"] = out.p_value < 0.05
    return out


def skill_salary_association(df: pd.DataFrame, min_postings: int = 15) -> pd.DataFrame:
    """Median salary of postings with vs without each skill, within-role adjusted.

    Adjustment: salary is divided by the median of its job title first, so a
    skill common in high-paying roles is not automatically flagged as "valuable".
    This is an association, not a causal effect.
    """
    if "extracted_skills" not in df.columns:
        raise ValueError("Run nlp.add_extracted_skills first")
    rel = df.salary_usd / df.groupby("job_title").salary_usd.transform("median")
    skills = sorted({s for row in df.extracted_skills for s in row})
    rows = []
    for s in skills:
        has = df.extracted_skills.map(lambda xs, s=s: s in xs)
        if has.sum() < min_postings or (~has).sum() < min_postings:
            continue
        _, p = stats.mannwhitneyu(rel[has], rel[~has], alternative="two-sided")
        rows.append({
            "skill": s,
            "postings_with": int(has.sum()),
            "role_adjusted_premium": float(rel[has].median() / rel[~has].median() - 1),
            "p_value": float(p),
        })
    out = pd.DataFrame(rows, columns=["skill", "postings_with", "role_adjusted_premium", "p_value"])
    out["q_value"] = benjamini_hochberg(out.p_value.to_numpy()) if len(out) else []
    return out.sort_values("role_adjusted_premium", ascending=False).reset_index(drop=True)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """False-discovery-rate adjusted p-values (q-values) for many simultaneous tests."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out
