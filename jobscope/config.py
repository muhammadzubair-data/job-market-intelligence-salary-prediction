"""Project-wide constants and paths."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DEMO_CSV = DATA_DIR / "demo_jobs.csv"
ARTIFACTS_DIR = ROOT / "artifacts"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
MIN_TRAIN_ROWS = 60

# Upload guard rails (keep Streamlit Community Cloud memory usage reasonable)
MAX_UPLOAD_ROWS = 100_000
MAX_SALARY_USD = 1_000_000
MAX_EXPERIENCE_YEARS = 60

REQUIRED_COLUMNS = [
    "job_title",
    "country",
    "experience_years",
    "employment_type",
    "remote_type",
    "skills",
    "description",
    "salary_usd",
]
OPTIONAL_COLUMNS = ["source", "posting_id"]
CATEGORICAL_FEATURES = ["job_title", "country", "employment_type", "remote_type"]
NUMERIC_FEATURES = ["experience_years", "skill_count"]
SKILLS_FEATURE = "skills"
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES + [SKILLS_FEATURE]
TARGET = "salary_usd"

BRAND = "ZubairLabs"
WEBSITE = "https://zubairlabs.com"
GITHUB_URL = "https://github.com/muhammadzubair-data/job-market-intelligence-salary-prediction"

# Country name -> ISO-3166 alpha-3 (used for map rendering; unknown names are listed, not dropped silently)
COUNTRY_ISO3 = {
    "argentina": "ARG", "australia": "AUS", "austria": "AUT", "bangladesh": "BGD", "belgium": "BEL",
    "brazil": "BRA", "canada": "CAN", "chile": "CHL", "china": "CHN", "colombia": "COL", "czechia": "CZE",
    "czech republic": "CZE", "denmark": "DNK", "egypt": "EGY", "finland": "FIN", "france": "FRA",
    "germany": "DEU", "greece": "GRC", "hong kong": "HKG", "hungary": "HUN", "india": "IND", "indonesia": "IDN",
    "ireland": "IRL", "israel": "ISR", "italy": "ITA", "japan": "JPN", "kenya": "KEN", "malaysia": "MYS",
    "mexico": "MEX", "netherlands": "NLD", "new zealand": "NZL", "nigeria": "NGA", "norway": "NOR",
    "pakistan": "PAK", "philippines": "PHL", "poland": "POL", "portugal": "PRT", "qatar": "QAT",
    "romania": "ROU", "saudi arabia": "SAU", "singapore": "SGP", "south africa": "ZAF", "south korea": "KOR",
    "korea": "KOR", "spain": "ESP", "sweden": "SWE", "switzerland": "CHE", "taiwan": "TWN", "thailand": "THA",
    "turkey": "TUR", "turkiye": "TUR", "ukraine": "UKR", "united arab emirates": "ARE", "uae": "ARE",
    "united kingdom": "GBR", "uk": "GBR", "england": "GBR", "united states": "USA", "usa": "USA", "us": "USA",
    "vietnam": "VNM", "oman": "OMN", "kuwait": "KWT", "bahrain": "BHR", "sri lanka": "LKA", "nepal": "NPL",
}
