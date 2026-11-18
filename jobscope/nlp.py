"""Transparent rule-based NLP for skill extraction.

A controlled vocabulary maps canonical skill names to synonyms/aliases.
Matching is word-boundary aware and skips simple negations such as
"no Tableau required" or "Spark not required". It is intentionally
explainable rather than clever: every match can be traced to a pattern.
"""
from __future__ import annotations

import itertools
import re
from collections import Counter
from functools import cache

import pandas as pd

# canonical skill -> (category, aliases)
SKILL_TAXONOMY: dict[str, tuple[str, list[str]]] = {
    "Python": ("Programming", ["python", "py3"]),
    "SQL": ("Data & Databases", ["sql", "postgresql", "postgres", "mysql", "t-sql", "tsql"]),
    "R": ("Programming", ["r programming", "rstudio", "r language"]),
    "Excel": ("BI & Visualization", ["excel", "spreadsheets", "vba"]),
    "Power BI": ("BI & Visualization", ["power bi", "powerbi", "dax"]),
    "Tableau": ("BI & Visualization", ["tableau"]),
    "Looker": ("BI & Visualization", ["looker"]),
    "Pandas": ("Programming", ["pandas"]),
    "NumPy": ("Programming", ["numpy"]),
    "Statistics": ("Analytics & ML", ["statistics", "statistical", "hypothesis testing", "a/b testing", "ab testing"]),
    "Machine Learning": ("Analytics & ML", ["machine learning", "ml models", "scikit-learn", "sklearn"]),
    "Deep Learning": ("Analytics & ML", ["deep learning", "neural network", "neural networks"]),
    "NLP": ("Analytics & ML", ["nlp", "natural language processing", "text mining"]),
    "PyTorch": ("Analytics & ML", ["pytorch", "torch"]),
    "TensorFlow": ("Analytics & ML", ["tensorflow", "keras"]),
    "LLMs": ("Analytics & ML", ["llm", "llms", "large language model", "large language models", "generative ai", "genai"]),
    "Spark": ("Data Engineering", ["spark", "pyspark", "databricks"]),
    "Airflow": ("Data Engineering", ["airflow"]),
    "dbt": ("Data Engineering", ["dbt"]),
    "Snowflake": ("Data Engineering", ["snowflake"]),
    "AWS": ("Cloud & MLOps", ["aws", "amazon web services", "sagemaker", "s3"]),
    "Azure": ("Cloud & MLOps", ["azure"]),
    "GCP": ("Cloud & MLOps", ["gcp", "google cloud", "bigquery"]),
    "Docker": ("Cloud & MLOps", ["docker", "containers", "containerization"]),
    "Kubernetes": ("Cloud & MLOps", ["kubernetes", "k8s"]),
    "MLOps": ("Cloud & MLOps", ["mlops", "mlflow", "model deployment"]),
    "Git": ("Engineering Practice", ["git", "github", "gitlab", "version control"]),
    "Streamlit": ("BI & Visualization", ["streamlit"]),
}

_NEGATION_BEFORE = r"(?:no|not|without|never)\s+(?:\w+\s+){0,2}"
_NEGATION_AFTER = r"\s+(?:is\s+|are\s+)?(?:not\s+(?:required|needed|necessary)|optional\s+not)"


def _normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9+#/\-\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@cache
def _patterns() -> list[tuple[str, re.Pattern, re.Pattern, re.Pattern]]:
    compiled = []
    for skill, (_, aliases) in SKILL_TAXONOMY.items():
        alt = "|".join(sorted((re.escape(a) for a in aliases), key=len, reverse=True))
        core = rf"(?<![a-z0-9])(?:{alt})(?![a-z0-9])"
        compiled.append((
            skill,
            re.compile(core),
            re.compile(_NEGATION_BEFORE + rf"(?:{alt})(?![a-z0-9])"),
            re.compile(rf"(?<![a-z0-9])(?:{alt})" + _NEGATION_AFTER),
        ))
    return compiled


def extract_skills(text: str) -> list[str]:
    """Return sorted canonical skills mentioned (and not negated) in ``text``."""
    normalized = _normalize(text)
    if not normalized:
        return []
    found = set()
    for skill, pattern, neg_before, neg_after in _patterns():
        hits = len(pattern.findall(normalized))
        if not hits:
            continue
        negated = len(neg_before.findall(normalized)) + len(neg_after.findall(normalized))
        if hits > negated:
            found.add(skill)
    return sorted(found)


def skill_category(skill: str) -> str:
    return SKILL_TAXONOMY.get(skill, ("Other", []))[0]


def add_extracted_skills(df: pd.DataFrame) -> pd.DataFrame:
    """Add an ``extracted_skills`` list column from skills + description text."""
    out = df.copy()
    combined = out["skills"].astype(str) + " . " + out["description"].astype(str)
    out["extracted_skills"] = combined.map(extract_skills)
    return out


def skill_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Postings mentioning each skill, share of postings and median salary."""
    data = df if "extracted_skills" in df.columns else add_extracted_skills(df)
    total = max(len(data), 1)
    exploded = data[["extracted_skills", "salary_usd"]].explode("extracted_skills").dropna()
    if exploded.empty:
        return pd.DataFrame(columns=["skill", "category", "postings", "share", "median_salary_usd"])
    out = (
        exploded.groupby("extracted_skills")
        .agg(postings=("salary_usd", "size"), median_salary_usd=("salary_usd", "median"))
        .reset_index()
        .rename(columns={"extracted_skills": "skill"})
    )
    out["share"] = out["postings"] / total
    out["category"] = out["skill"].map(skill_category)
    out = out.sort_values(["postings", "skill"], ascending=[False, True]).reset_index(drop=True)
    return out[["skill", "category", "postings", "share", "median_salary_usd"]]


def skill_pairs(df: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """Most frequent co-occurring skill pairs with lift (observed / expected)."""
    data = df if "extracted_skills" in df.columns else add_extracted_skills(df)
    n = max(len(data), 1)
    single = Counter()
    pairs = Counter()
    for skills in data["extracted_skills"]:
        s = sorted(set(skills))
        single.update(s)
        pairs.update(itertools.combinations(s, 2))
    rows = []
    for (a, b), count in pairs.most_common(top):
        expected = single[a] * single[b] / n
        rows.append({
            "skill_a": a,
            "skill_b": b,
            "postings": count,
            "lift": round(count / expected, 2) if expected else None,
        })
    return pd.DataFrame(rows, columns=["skill_a", "skill_b", "postings", "lift"])


def cooccurrence_matrix(df: pd.DataFrame, skills: list[str]) -> pd.DataFrame:
    data = df if "extracted_skills" in df.columns else add_extracted_skills(df)
    mat = pd.DataFrame(0, index=skills, columns=skills, dtype=int)
    for found in data["extracted_skills"]:
        present = [s for s in found if s in mat.index]
        for a in present:
            for b in present:
                mat.loc[a, b] += 1
    return mat


def parse_user_skills(text: str) -> list[str]:
    """Map a free-text comma-separated skill list onto the canonical taxonomy."""
    return extract_skills(text)
