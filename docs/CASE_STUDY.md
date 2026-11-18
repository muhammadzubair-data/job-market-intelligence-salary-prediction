# Case study — JobScope AI

## Problem
Job seekers and hiring teams want clear answers about role demand, in-demand skills and pay. Raw postings make that hard: titles and skill names vary, salaries mix currencies and periods, and most dashboards show numbers without saying how reliable they are.

## Objective
Build a reusable, honest analytics and modelling product that:
1. accepts normalised job-posting data (or runs on clearly labelled synthetic data),
2. explains the market through interactive charts and statistical tests,
3. extracts skills transparently from text,
4. estimates salary with measured accuracy and calibrated uncertainty,
5. turns the analysis into practical career guidance and shareable reports.

## Solution
- **Data layer:** validation with a quality report, upload guard rails, synthetic-data labelling everywhere.
- **NLP:** a 28-skill taxonomy with synonyms and negation handling, plus co-occurrence and lift.
- **Statistics:** Kruskal-Wallis with effect sizes, Spearman correlation, role-adjusted skill associations with Benjamini-Hochberg correction.
- **ML:** four model families compared by cross-validation inside leakage-safe pipelines, a holdout scored against a naive baseline, conformal prediction intervals, permutation importance and per-group error.
- **Product:** a seven-section Streamlit app with light/dark themes, a world map, career gap analysis, and CSV/HTML exports.
- **Engineering:** modular package, 46 automated tests, a browser E2E test, and CI on three Python versions.

## Results on synthetic data (method demonstration only)
The selected Gradient Boosting model reached a holdout MAE of $9,105 (R² 0.924), against $37,960 for a median baseline. The 80% interval covered 82% of unseen rows. Importance analysis correctly found that skills carry no signal in this data, as designed, which shows the pipeline does not invent effects.

## What I would do with real data
Use a licensed postings feed with provenance, validate on time-based splits, normalise currencies with dated FX rates, add SHAP explanations, and monitor drift and per-country error.
