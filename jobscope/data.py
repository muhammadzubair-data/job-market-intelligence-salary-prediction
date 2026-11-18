"""Data ingestion: deterministic synthetic demo data and safe CSV loading.

The demo generator produces ENTIRELY SYNTHETIC records. Its salary formula is
hand-written, so any pattern found in it is an artefact of that formula and
says nothing about the real job market.
"""
from __future__ import annotations

import io
import random
from pathlib import Path
from typing import IO, Union

import pandas as pd

from .config import DEMO_CSV, MAX_UPLOAD_ROWS

TITLES = {
    "Data Analyst": 70000,
    "Data Scientist": 110000,
    "ML Engineer": 135000,
    "AI Engineer": 145000,
    "Analytics Engineer": 115000,
    "BI Developer": 85000,
}
COUNTRIES = {
    "United States": 1.0,
    "United Kingdom": 0.75,
    "Canada": 0.80,
    "Germany": 0.78,
    "India": 0.32,
    "Pakistan": 0.22,
    "Australia": 0.88,
}
SKILLS = [
    "Python", "SQL", "Excel", "Power BI", "Tableau", "AWS", "Docker",
    "Machine Learning", "Statistics", "NLP", "Spark", "Git", "Streamlit", "Pandas",
]


def generate_demo(n: int = 750, seed: int = 42) -> pd.DataFrame:
    """Return ``n`` reproducible synthetic job postings (seeded)."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        title = rng.choice(list(TITLES))
        country = rng.choice(list(COUNTRIES))
        years = rng.randint(0, 15)
        remote = rng.choice(["Remote", "Hybrid", "On-site"])
        employment = rng.choice(["Full-time", "Contract"])
        skills = rng.sample(SKILLS, rng.randint(3, 7))
        salary = (
            (TITLES[title] + years * 3900)
            * COUNTRIES[country]
            * (1.04 if remote == "Remote" else 1)
            * rng.uniform(0.82, 1.18)
        )
        rows.append({
            "job_title": title,
            "country": country,
            "experience_years": years,
            "employment_type": employment,
            "remote_type": remote,
            "skills": ", ".join(skills),
            "description": (
                f"Seeking {title} with {', '.join(skills)}. "
                "Collaborate on analytics and production workflows."
            ),
            "salary_usd": round(salary),
            "source": "SYNTHETIC DEMO",
            "posting_id": f"demo-{i + 1:04d}",
        })
    return pd.DataFrame(rows)


def load_demo() -> pd.DataFrame:
    """Load the bundled demo CSV, regenerating it if the file is missing."""
    if Path(DEMO_CSV).exists():
        return pd.read_csv(DEMO_CSV)
    return generate_demo()


def load_csv(source: Union[str, Path, IO[bytes], bytes], max_rows: int = MAX_UPLOAD_ROWS) -> pd.DataFrame:
    """Read a CSV from a path, bytes or file-like object with a row cap.

    Raises ``ValueError`` with a user-friendly message for unreadable input.
    """
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    try:
        df = pd.read_csv(source, nrows=max_rows + 1, encoding_errors="replace")
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The CSV file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"The CSV file could not be parsed: {exc}") from exc
    except UnicodeDecodeError as exc:  # pragma: no cover - encoding_errors guards this
        raise ValueError("The CSV file is not valid UTF-8 text.") from exc
    if len(df) > max_rows:
        raise ValueError(f"The CSV has more than {max_rows:,} rows. Please upload a smaller sample.")
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df
