import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobscope.data import generate_demo  # noqa: E402
from jobscope.modeling import train_and_compare  # noqa: E402
from jobscope.nlp import add_extracted_skills  # noqa: E402
from jobscope.validation import validate  # noqa: E402


@pytest.fixture(scope="session")
def demo_df():
    return add_extracted_skills(validate(generate_demo(400)))


@pytest.fixture(scope="session")
def trained(demo_df):
    return train_and_compare(demo_df, importance_repeats=2)
