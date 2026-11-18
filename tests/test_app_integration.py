"""Integration tests: dashboard start-up (Streamlit AppTest) and the CLI training script."""
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent


def test_dashboard_starts_without_exceptions():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Demo mode" in w.value for w in at.warning)
    assert any("Executive Overview" in m.value for m in at.markdown)


def test_dashboard_filters_rerun():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)
    at.run()
    at.sidebar.multiselect[1].set_value(["Data Analyst"]).run()  # job roles filter
    assert not at.exception
    assert any("of 750 postings in view" in m.value for m in at.markdown)


def test_train_script_cli(tmp_path):
    from scripts.train import main

    summary = main(["--output", str(tmp_path)])
    assert summary["data_type"] == "synthetic_demo"
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["best_model"] == summary["best_model"]
