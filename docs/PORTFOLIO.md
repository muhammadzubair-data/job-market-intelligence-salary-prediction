# Portfolio materials — JobScope AI

All claims below are verified features or measured results from the synthetic demo data. Keep the "synthetic data" qualifier wherever metrics appear.

**Links**
- Live app: https://jobscope-ai-zubairlabs.streamlit.app
- Code: https://github.com/muhammadzubair-data/job-market-intelligence-salary-prediction
- Website: https://zubairlabs.com

---

## GitHub

**Description:** JobScope AI: job market intelligence, NLP skills analytics and salary prediction with a Streamlit dashboard (scikit-learn, Plotly). ZubairLabs portfolio.

**Topics:** data-science, machine-learning, nlp, streamlit, salary-prediction, job-market, scikit-learn, plotly, python, data-visualization, career-analytics, portfolio-project

---

## LinkedIn post

I just shipped JobScope AI, a job market intelligence and salary prediction app built end to end in Python.

The question behind it: which roles are in demand, which skills keep showing up, and what does a role pay in a given country at a given level of experience? More importantly, how sure can you be about that number?

What it does:
📊 Validates and cleans job posting data, with a data quality report
🧠 Pulls skills out of job descriptions using a transparent NLP layer (28 skills, synonyms like PostgreSQL → SQL, and it ignores "no Tableau required")
📈 Runs real statistics: Kruskal-Wallis, Spearman, and skill-salary tests corrected for multiple comparisons
🤖 Compares Ridge, Random Forest, Gradient Boosting and HistGradientBoosting with 5-fold cross-validation, then reports the winner on a holdout it never saw
🎯 Gives every salary estimate an 80% prediction interval, and checks how often that interval is actually right on unseen data
🚀 Adds a career page with a skill-gap chart and "what if I learn X" scenarios
📤 Exports filtered CSVs and a full HTML report

On the bundled demo data, the selected model reached a holdout MAE of about $9.1k (R² 0.92), against $38k for a naive baseline, and the 80% interval covered 82% of unseen rows.

One thing I care about: the demo dataset is synthetic, and the app says so on every page. Skills were generated with no effect on pay, and the model correctly found that skills carry almost no signal. A pipeline that doesn't invent patterns is worth more to me than a flashy score.

Built with Python, pandas, scikit-learn, SciPy, Plotly and Streamlit. 46 automated tests plus a browser end-to-end test run in GitHub Actions on Python 3.11–3.13.

Try it: https://jobscope-ai-zubairlabs.streamlit.app
Code: https://github.com/muhammadzubair-data/job-market-intelligence-salary-prediction

#DataScience #MachineLearning #NLP #Python #Streamlit #Analytics #ZubairLabs

---

## Upwork portfolio item

**Title:** JobScope AI: Job Market Analytics, NLP & Salary Prediction Dashboard

**Role:** Data Scientist & Python Developer (end to end: data pipeline, NLP, ML, dashboard, testing, deployment)

**Description:**
An interactive analytics web app that turns raw job-posting data into market insights, skill trends and salary estimates. Users can upload their own CSV or explore the built-in demo data.

What I built:
- Data validation pipeline with a clear data quality report
- Rule-based NLP that extracts skills from job descriptions, including synonyms and simple negation
- Statistical analysis of salary differences by role, country and experience
- Salary model selected from four algorithms with cross-validation, with prediction intervals and feature importance
- Career page with skill-gap analysis and what-if scenarios
- Seven-section Streamlit dashboard with light and dark themes, a world map and downloadable CSV and HTML reports
- 46 automated tests, browser end-to-end tests and GitHub Actions CI; deployed on Streamlit Community Cloud

Result on demo data: holdout MAE about $9.1k vs $38k for a baseline (R² 0.92); 80% prediction interval covered 82% of unseen rows.

**Skills:** Python, Data Science, Machine Learning, Natural Language Processing, scikit-learn, pandas, Statistical Analysis, Data Visualization, Plotly, Streamlit, Dashboard Development, Predictive Modeling

**Deliverables:** Live web app, GitHub repository with documentation, model card, reproducible training script, automated test suite, CSV and HTML report exports

**Links:** live app and GitHub (above)

---

## ZubairLabs website

**Project name:** JobScope AI

**One-liner:** Job market intelligence and salary prediction in one interactive dashboard.

**Summary:**
JobScope AI helps job seekers, recruiters and analysts understand the job market. It shows which roles are in demand, which skills appear most often, how pay changes with location and experience, and what a role is likely to pay, with an honest range rather than a single number. Upload your own job-posting data or explore the demo.

**Key features:**
- Executive overview with salary KPIs and distributions
- Job market view with a world salary map and role × country comparison
- Skills intelligence powered by transparent NLP
- Salary prediction with an 80% prediction interval
- Career skill-gap analysis and what-if scenarios
- Data explorer with CSV and HTML report downloads
- Model performance page with cross-validation, holdout metrics and explainability

**Tech stack:** Python · pandas · NumPy · SciPy · scikit-learn · Plotly · Streamlit · pytest · Playwright · GitHub Actions

**Results (synthetic demo data):** Gradient Boosting selected by 5-fold CV; holdout MAE $9,105, R² 0.924; 82% coverage for an 80% interval.

**Buttons:** Live demo → https://jobscope-ai-zubairlabs.streamlit.app · View code → GitHub link above
