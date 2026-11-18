"""End-to-end browser test of the running dashboard (Playwright + Chromium).

Not collected by pytest by default (file name). Run against a live server:
    streamlit run app.py --server.port 8501 &
    python tests/e2e_ui.py http://localhost:8501
Checks every page loads without exceptions, the CSV upload switches to user-data
mode, a salary prediction runs, and CSV + HTML report downloads work.
Optionally writes screenshots: python tests/e2e_ui.py URL --shots assets/screenshots
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PAGES = [("", "overview"), ("market", "market"), ("skills", "skills"), ("predict", "prediction"),
         ("career", "career"), ("explorer", "explorer"), ("model", "model")]
FIXTURE = Path(__file__).parent / "fixtures" / "sample_upload.csv"


def wait_idle(page, extra=2.0):
    page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=60_000)
    time.sleep(1.5)
    for _ in range(400):
        if page.locator("[data-testid='stSpinner']").count() == 0:
            break
        time.sleep(0.5)
    time.sleep(extra)


def check(page, name, failures):
    exc = page.locator("[data-testid='stException']").count()
    errs = [e.inner_text()[:150] for e in page.locator("[data-testid='stAlertContentError']").all()]
    print(f"  {name:<12} exceptions={exc} errors={len(errs)} charts={page.locator('.js-plotly-plot').count()}")
    if exc or errs:
        failures.append((name, exc, errs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", default="http://localhost:8501")
    ap.add_argument("--shots", help="directory for screenshots")
    ap.add_argument("--chrome", default=os.environ.get("CHROME_PATH"), help="path to a Chrome/Chromium binary")
    args = ap.parse_args()
    base = args.url.rstrip("/")
    failures: list = []
    with sync_playwright() as p:
        launch = {"args": ["--no-sandbox"]}
        if args.chrome:
            launch["executable_path"] = args.chrome
        browser = p.chromium.launch(**launch)
        for theme in (["light", "dark"] if args.shots else ["light"]):
            print(f"Pages ({theme} theme):")
            page = browser.new_page(viewport={"width": 1440, "height": 900}, color_scheme=theme, accept_downloads=True)
            for path, name in PAGES:
                page.goto(f"{base}/{path}", wait_until="networkidle")
                wait_idle(page)
                if name == "prediction":
                    page.get_by_role("button", name="Estimate salary").click()
                    wait_idle(page, 3)
                    assert page.get_by_text("prediction interval").count() > 0, "prediction result missing"
                check(page, name, failures)
                if args.shots:
                    Path(args.shots).mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=f"{args.shots}/{name}-{theme}.png")
            page.close()

        print("Upload + downloads:")
        page = browser.new_page(viewport={"width": 1440, "height": 900}, accept_downloads=True)
        page.goto(f"{base}/explorer", wait_until="networkidle")
        wait_idle(page)
        page.get_by_text("Data workspace").click()
        page.locator("input[type='file']").set_input_files(str(FIXTURE))
        wait_idle(page, 4)
        assert page.get_by_text("User data mode").count() > 0, "upload did not switch to user data mode"
        print("  upload       user-data mode banner shown")
        check(page, "upload", failures)
        with page.expect_download() as dl:
            page.get_by_role("button", name="⬇ Filtered data (CSV)").click()
        csv_text = Path(dl.value.path()).read_text()
        assert "'=HYPERLINK" in csv_text, "formula injection not neutralised"
        assert "-100" not in csv_text.split("\n")[2], "invalid row leaked"
        print(f"  csv download {len(csv_text.splitlines()) - 1} rows, formula cells neutralised")
        page.get_by_role("button", name="Build HTML report").click()
        wait_idle(page, 4)
        with page.expect_download() as dl:
            page.get_by_role("button", name="⬇ Download HTML report").click()
        report = Path(dl.value.path()).read_text()
        assert "User-supplied data" in report and "Salary model" in report
        print(f"  html report  {len(report):,} bytes, includes model section")
        for link, name in [("Model Performance", "model(upload)"), ("Job Market Intelligence", "market(upload)")]:
            page.get_by_role("link", name=link).click()  # in-app navigation keeps the uploaded file
            wait_idle(page, 3)
            assert page.get_by_text("User data mode").count() > 0, "uploaded data lost on navigation"
            check(page, name, failures)
        assert page.get_by_text("Atlantis").count() > 0, "unmapped-country note missing"
        browser.close()
    if failures:
        print("FAILED:", failures)
        return 1
    print("E2E OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
