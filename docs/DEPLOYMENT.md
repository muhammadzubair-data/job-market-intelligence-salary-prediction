# Deployment guide

## Local

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud (free)

1. Push the repository to GitHub (public).
2. Sign in at <https://share.streamlit.io> with GitHub and click **Create app → Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `muhammadzubair-data/job-market-intelligence-salary-prediction`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL (optional):** e.g. `jobscope-ai`
4. Under **Advanced settings**, choose Python **3.12**. No secrets are required.
5. Click **Deploy**. Cloud installs `requirements.txt` only (dev tools are kept in `requirements-dev.txt`).

### Notes
- The first visit trains and cross-validates four models (about 10–20 s on the free tier). Results are cached for later visits.
- `.streamlit/config.toml` enables static serving for the bundled map outlines, sets light/dark themes and limits uploads to 25 MB.
- Uploaded CSVs are processed in memory for the session and not stored by the app. The hosting provider's own logging policies apply.
- If the app sleeps after inactivity, visitors can wake it with one click.

## Quality gates

```bash
pip install -r requirements-dev.txt
ruff check .
python -m pytest -q
python scripts/train.py
playwright install chromium && streamlit run app.py &  python tests/e2e_ui.py
```

The same checks run in GitHub Actions on every push to `main`.
