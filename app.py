"""JobScope AI — Global Job Market Intelligence & Salary Prediction (Streamlit app).

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from jobscope import __version__, analysis, career, nlp, reports
from jobscope.config import COUNTRY_ISO3, GITHUB_URL, MIN_TRAIN_ROWS, REQUIRED_COLUMNS, WEBSITE
from jobscope.data import load_csv, load_demo
from jobscope.modeling import TrainingResult, predict_with_interval, train_and_compare
from jobscope.validation import is_synthetic, validate

st.set_page_config(
    page_title="JobScope AI · Job Market Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": f"JobScope AI v{__version__} · A ZubairLabs data science portfolio project · {WEBSITE}"},
)

# --------------------------------------------------------------------------- styling
st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1440px;}
.js-hero {padding: 26px 30px; border-radius: 20px; margin-bottom: 18px; color: #F8FAFC;
  background: radial-gradient(1200px 300px at 0% 0%, rgba(56,189,248,.35), transparent 60%),
              linear-gradient(115deg, #0B2742 0%, #13213D 55%, #1E1B4B 100%);
  border: 1px solid rgba(148,163,184,.25); box-shadow: 0 10px 30px rgba(2,8,23,.18);}
.js-hero .eyebrow {letter-spacing: .18em; font-size: 11px; font-weight: 700; color: #7DD3FC; text-transform: uppercase;}
.js-hero h1 {font-size: 34px; line-height: 1.15; margin: 6px 0 6px 0; font-weight: 800; color: #F8FAFC; padding: 0;}
.js-hero p {margin: 0; color: #CBD5E1; font-size: 15px;}
.js-pill {display:inline-block; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; margin: 10px 6px 0 0;
  background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.18); color: #E2E8F0;}
.js-pill.warn {background: rgba(245,158,11,.18); border-color: rgba(245,158,11,.5); color: #FDE68A;}
.js-pill.ok {background: rgba(16,185,129,.18); border-color: rgba(16,185,129,.5); color: #A7F3D0;}
.kpi-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 4px 0 14px 0;}
.kpi {border-radius: 16px; padding: 14px 16px; border: 1px solid rgba(148,163,184,.28);
  background: linear-gradient(180deg, rgba(148,163,184,.10), rgba(148,163,184,.03));}
.kpi .label {font-size: 12px; font-weight: 600; opacity: .72; text-transform: uppercase; letter-spacing: .06em;}
.kpi .value {font-size: 26px; font-weight: 800; margin-top: 4px; line-height: 1.1;}
.kpi .sub {font-size: 12px; opacity: .65; margin-top: 4px;}
.section-title {font-size: 24px; font-weight: 800; margin: 2px 0 2px 0;}
.section-sub {opacity: .7; margin-bottom: 14px; font-size: 14px;}
.insight {border-left: 4px solid #0EA5E9; padding: 10px 14px; border-radius: 10px; margin-bottom: 8px;
  background: rgba(14,165,233,.08);}
.result-card {border-radius: 18px; padding: 22px 24px; border: 1px solid rgba(14,165,233,.45);
  background: linear-gradient(135deg, rgba(14,165,233,.14), rgba(139,92,246,.10));}
.result-card .big {font-size: 40px; font-weight: 800; line-height: 1.1;}
.result-card .range {font-size: 15px; opacity: .85; margin-top: 6px;}
.brand-side {font-weight: 800; font-size: 18px;} .brand-side span {color: #0EA5E9;}
.footer {opacity: .6; font-size: 12px; margin-top: 30px; text-align: center;}
[data-testid="stMetricValue"] {font-weight: 800;}
div[data-testid="stForm"] {border-radius: 16px;}
</style>
""",
    unsafe_allow_html=True,
)

CHART_HEIGHT = 380


def money(x: float) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"${x:,.0f}"


def kpi_cards(items: list[tuple[str, str, str]]) -> None:
    e = html.escape
    cards = "".join(
        f"<div class='kpi'><div class='label'>{e(label)}</div><div class='value'>{e(value)}</div><div class='sub'>{e(sub)}</div></div>"
        for label, value, sub in items
    )
    st.markdown(f"<div class='kpi-grid'>{cards}</div>", unsafe_allow_html=True)


def insight(text: str) -> None:
    """Callout box; converts **bold** markdown to HTML since the content is raw HTML."""
    safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(text))  # values may come from an uploaded CSV
    st.markdown(f"<div class='insight'>{safe}</div>", unsafe_allow_html=True)


def section(title: str, subtitle: str) -> None:
    st.markdown(f"<div class='section-title'>{title}</div><div class='section-sub'>{subtitle}</div>", unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = CHART_HEIGHT) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="",
                      title_font=dict(size=15), hoverlabel=dict(font_size=12))
    return fig


def chart(fig: go.Figure, height: int = CHART_HEIGHT) -> None:
    # topojsonURL points maps at the bundled static/topojson files instead of the plot.ly CDN.
    st.plotly_chart(style_fig(fig, height), width="stretch",
                    config={"displaylogo": False, "topojsonURL": "app/static/topojson/"})


# --------------------------------------------------------------------------- data + model (cached)
@st.cache_data(show_spinner=False, max_entries=8)
def prepare_data(raw_bytes: bytes | None):
    raw = load_demo() if raw_bytes is None else load_csv(raw_bytes)
    df, report = validate(raw, return_report=True)
    df = nlp.add_extracted_skills(df)
    return df, report.as_frame()


@st.cache_resource(show_spinner=False, max_entries=4)
def get_model(fingerprint: str, _df: pd.DataFrame) -> TrainingResult:
    return train_and_compare(_df)


def fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df.drop(columns=["extracted_skills"]), index=False).values.tobytes()).hexdigest()


# --------------------------------------------------------------------------- sidebar: data workspace
ASSETS = Path(__file__).parent / "assets"
st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "icon.svg"), link=WEBSITE, size="large")

uploaded = None
with st.sidebar.expander("📂 Data workspace", expanded=False):
    uploaded = st.file_uploader("Upload job postings CSV", type=["csv"], help="Max 25 MB / 100,000 rows. Annual salaries in USD.")
    st.caption("Required columns: " + ", ".join(f"`{c}`" for c in REQUIRED_COLUMNS))
    template = load_demo().head(5)
    st.download_button("Download CSV template", reports.to_csv_bytes(template), "jobscope_template.csv", "text/csv", width="stretch")

load_error = None
with st.spinner("Loading and validating data…"):
    try:
        df_all, quality = prepare_data(uploaded.getvalue() if uploaded is not None else None)
    except (ValueError, TypeError) as exc:
        load_error = str(exc)
        df_all, quality = prepare_data(None)

SYNTHETIC = is_synthetic(df_all)
MAX_EXP = max(1, int(df_all.experience_years.max()))
DATA_MODE = "Synthetic demo" if SYNTHETIC else "User-supplied upload"

with st.sidebar:
    st.markdown("**Filters**")
    countries_all = sorted(df_all.country.unique())
    roles_all = sorted(df_all.job_title.unique())
    sel_countries = st.multiselect("Countries", countries_all, placeholder="All countries")
    sel_roles = st.multiselect("Job roles", roles_all, placeholder="All roles")
    sel_remote = st.multiselect("Work arrangement", sorted(df_all.remote_type.unique()), placeholder="All arrangements")
    lo, hi = int(df_all.experience_years.min()), int(df_all.experience_years.max())
    sel_exp = st.slider("Experience (years)", lo, hi, (lo, hi)) if hi > lo else (lo, hi)
    st.caption("🌗 Light/dark theme: ⋮ menu → Settings → Theme")

mask = df_all.experience_years.between(*sel_exp)
if sel_countries:
    mask &= df_all.country.isin(sel_countries)
if sel_roles:
    mask &= df_all.job_title.isin(sel_roles)
if sel_remote:
    mask &= df_all.remote_type.isin(sel_remote)
df = df_all[mask]

# --------------------------------------------------------------------------- hero
mode_pill = ("<span class='js-pill warn'>⚠ Synthetic demo data</span>" if SYNTHETIC
             else "<span class='js-pill ok'>✓ User-supplied data</span>")
st.markdown(
    f"""<div class='js-hero'><div class='eyebrow'>ZubairLabs · Data Science Portfolio</div>
<h1>JobScope AI</h1><p>Global job market intelligence · NLP skills analytics · salary prediction · career insights</p>
{mode_pill}<span class='js-pill'>{len(df):,} of {len(df_all):,} postings in view</span><span class='js-pill'>v{__version__}</span></div>""",
    unsafe_allow_html=True,
)
if load_error:
    st.error(f"Your file could not be used: {load_error} — showing the synthetic demo dataset instead.")
if SYNTHETIC:
    st.warning("**Demo mode.** Every posting and salary here is synthetic, generated by a seeded formula. "
               "Charts, statistics and model scores demonstrate the method only — they are **not** real job-market findings.", icon="⚠️")
else:
    st.info("**User data mode.** Results describe the uploaded sample, not the whole job market. "
            "Check source quality, currency normalisation and coverage before drawing conclusions.", icon="ℹ️")


def require_rows() -> bool:
    if df.empty:
        st.info("No postings match the current filters. Widen the filters in the sidebar.", icon="🔎")
        return False
    return True


def model_or_none() -> TrainingResult | None:
    if len(df_all) < MIN_TRAIN_ROWS:
        st.warning(f"At least {MIN_TRAIN_ROWS} valid rows are needed to train the salary model (dataset has {len(df_all)}).")
        return None
    try:
        with st.spinner("Training and cross-validating 4 models (cached after the first run)…"):
            return get_model(fingerprint(df_all), df_all)
    except Exception as exc:  # surface a readable message rather than a stack trace
        st.error(f"The salary model could not be trained on this dataset: {exc}")
        return None


# --------------------------------------------------------------------------- pages
def page_overview():
    section("Executive Overview", "Headline metrics and salary structure for the postings currently in view.")
    if not require_rows():
        return
    k = analysis.kpis(df)
    kpi_cards([
        ("Job postings", f"{k['postings']:,}", f"{k['roles']} roles · {k['countries']} countries"),
        ("Median salary", money(k["median_salary"]), "annual, USD"),
        ("Interquartile range", f"${k['p25_salary']/1000:,.0f}k–{k['p75_salary']/1000:,.0f}k", "middle 50% of salaries"),
        ("Remote share", f"{k['remote_share']:.0%}", "of postings in view"),
        ("Median experience", f"{k['median_experience']:.0f} yrs", "requested"),
    ])
    by_role = analysis.salary_by(df, "job_title")
    by_country = analysis.salary_by(df, "country")
    skills = nlp.skill_frequency(df)

    c1, c2 = st.columns([1.15, 1])
    with c1:
        fig = px.histogram(df, x="salary_usd", nbins=40, title="Salary distribution",
                           labels={"salary_usd": "Annual salary (USD)"}, color_discrete_sequence=["#0EA5E9"])
        fig.add_vline(x=k["median_salary"], line_dash="dash", annotation_text="median", annotation_position="top")
        fig.update_yaxes(title="Postings")
        chart(fig)
    with c2:
        fig = px.bar(by_role.sort_values("median"), x="median", y="job_title", orientation="h",
                     error_x=by_role.sort_values("median")["p75"] - by_role.sort_values("median")["median"],
                     title="Median salary by role (whisker → 75th percentile)", labels={"median": "Median annual USD", "job_title": ""})
        chart(fig)

    st.markdown("##### Key observations in this dataset")
    top_role, low_role = by_role.iloc[0], by_role.iloc[-1]
    top_c = by_country.iloc[0]
    rho = analysis.spearman_test(df)["statistic"]
    obs = [
        f"Highest median salary role: **{top_role.job_title}** at {money(top_role['median'])} "
        f"vs **{low_role.job_title}** at {money(low_role['median'])}.",
        f"Highest median salary country: **{top_c.country}** ({money(top_c['median'])}, {int(top_c.postings)} postings).",
        (f"Experience and salary have a Spearman correlation of **{rho:.2f}**." if not np.isnan(rho) else "Not enough variation in experience to correlate."),
    ]
    if not skills.empty:
        obs.append(f"Most mentioned skill: **{skills.iloc[0].skill}**, in {skills.iloc[0].share:.0%} of postings.")
    for o in obs:
        insight(o)
    if SYNTHETIC:
        st.caption("These observations restate patterns built into the synthetic generator.")

    c3, c4 = st.columns(2)
    with c3:
        chart(px.box(df, x="job_title", y="salary_usd", color="job_title", points=False,
                     title="Salary spread by role", labels={"salary_usd": "Annual USD", "job_title": ""}).update_layout(showlegend=False))
    with c4:
        vol = df.groupby("country").size().reset_index(name="postings").sort_values("postings", ascending=False)
        chart(px.bar(vol, x="country", y="postings", title="Posting volume by country", labels={"country": ""},
                     color_discrete_sequence=["#8B5CF6"]))
    with st.expander("Data quality report"):
        st.dataframe(quality, hide_index=True, width="stretch")


def page_market():
    section("Job Market Intelligence", "Role demand, experience levels, locations and employment patterns.")
    if not require_rows():
        return
    d = df.assign(experience_band=analysis.experience_band(df.experience_years))
    c1, c2 = st.columns(2)
    with c1:
        demand = d.groupby("job_title").size().reset_index(name="postings").sort_values("postings")
        chart(px.bar(demand, x="postings", y="job_title", orientation="h", title="Role demand (postings)",
                     labels={"job_title": ""}, color_discrete_sequence=["#0EA5E9"]))
    with c2:
        band = d.groupby("experience_band", observed=False).salary_usd.agg(["median", "size"]).reset_index()
        fig = px.bar(band, x="experience_band", y="median", text="size", title="Median salary by experience level",
                     labels={"experience_band": "", "median": "Median annual USD"}, color_discrete_sequence=["#10B981"])
        fig.update_traces(texttemplate="%{text} postings", textposition="outside")
        chart(fig)

    geo = analysis.salary_by(d, "country")
    geo["iso3"] = geo.country.str.strip().str.lower().map(COUNTRY_ISO3)
    fig = px.choropleth(geo.dropna(subset=["iso3"]), locations="iso3", locationmode="ISO-3", color="median", hover_name="country",
                        hover_data={"iso3": False, "postings": True, "p25": ":,.0f", "p75": ":,.0f", "median": ":,.0f"},
                        color_continuous_scale="Blues", title="Median salary by country")
    fig.update_geos(showframe=False, showcoastlines=False, showland=True, landcolor="rgba(148,163,184,0.18)",
                    showcountries=True, countrycolor="rgba(148,163,184,0.35)", projection_type="natural earth",
                    bgcolor="rgba(0,0,0,0)", lataxis_range=[-58, 85])
    fig.update_layout(coloraxis_colorbar=dict(title="USD"))
    chart(fig, 460)
    unmapped = geo[geo.iso3.isna()].country.tolist()
    if unmapped:
        st.caption("Not shown on the map (unrecognised country names): " + ", ".join(unmapped))

    c3, c4 = st.columns([1.3, 1])
    with c3:
        pivot = d.pivot_table(index="job_title", columns="country", values="salary_usd", aggfunc="median")
        fig = px.imshow(pivot.round(-2), text_auto=".2s", aspect="auto", color_continuous_scale="Tealgrn",
                        title="Median salary: role × country", labels=dict(color="USD", x="", y=""))
        chart(fig, 420)
    with c4:
        mix = d.groupby(["remote_type", "employment_type"]).size().reset_index(name="postings")
        chart(px.sunburst(mix, path=["remote_type", "employment_type"], values="postings", color="remote_type",
                          color_discrete_sequence=["#0EA5E9", "#8B5CF6", "#10B981", "#F59E0B"],
                          title="Work arrangement & employment type"), 420)

    st.markdown("##### Salary summary by country")
    st.dataframe(
        geo.rename(columns={"median": "median_usd", "p25": "p25_usd", "p75": "p75_usd", "mean": "mean_usd"}),
        hide_index=True, width="stretch",
        column_config={c: st.column_config.NumberColumn(format="$%,.0f") for c in ["median_usd", "p25_usd", "p75_usd", "mean_usd"]}
        | {"share": st.column_config.ProgressColumn("share", format="percent", min_value=0, max_value=1)},
    )
    with st.expander("Statistical tests (sample-level)"):
        st.caption("Kruskal-Wallis tests whether salary distributions differ between groups; effect size is ε². "
                   "Spearman ρ measures monotonic association. Significance is not causation.")
        st.dataframe(analysis.statistical_summary(d).round(4), hide_index=True, width="stretch")


def page_skills():
    section("Skills Intelligence", "Rule-based NLP extraction across skills lists and job descriptions.")
    if not require_rows():
        return
    freq = nlp.skill_frequency(df)
    if freq.empty:
        st.info("No skills from the controlled vocabulary were found in the postings in view.")
        return
    kpi_cards([
        ("Distinct skills found", f"{len(freq)}", f"of {len(nlp.SKILL_TAXONOMY)} in the taxonomy"),
        ("Avg skills / posting", f"{df.extracted_skills.map(len).mean():.1f}", "after negation filtering"),
        ("Top skill", freq.iloc[0].skill, f"{freq.iloc[0].share:.0%} of postings"),
    ])
    c1, c2 = st.columns([1.2, 1])
    with c1:
        top = freq.head(15).sort_values("postings")
        chart(px.bar(top, x="share", y="skill", color="category", orientation="h", title="Top skills by share of postings",
                     labels={"share": "Share of postings", "skill": ""}).update_xaxes(tickformat=".0%"), 440)
    with c2:
        chart(px.treemap(freq, path=["category", "skill"], values="postings", title="Skill categories"), 440)

    c3, c4 = st.columns([1.1, 1])
    with c3:
        top10 = freq.head(10).skill.tolist()
        mat = nlp.cooccurrence_matrix(df, top10)
        chart(px.imshow(mat, text_auto=True, color_continuous_scale="Purples", title="Skill co-occurrence (top 10)"), 440)
    with c4:
        st.markdown("##### Strongest skill combinations")
        st.caption("Lift > 1: the pair appears together more often than chance would predict.")
        st.dataframe(nlp.skill_pairs(df, 12), hide_index=True, width="stretch", height=380)

    assoc = analysis.skill_salary_association(df)
    if not assoc.empty:
        fig = px.bar(assoc, x="role_adjusted_premium", y="skill", orientation="h",
                     color=np.where(assoc.q_value < 0.05, "Significant (FDR q < 0.05)", "Not significant"),
                     color_discrete_map={"Significant (FDR q < 0.05)": "#10B981", "Not significant": "#94A3B8"},
                     title="Role-adjusted salary association by skill",
                     labels={"role_adjusted_premium": "Median salary difference vs. postings without the skill", "skill": "", "color": ""})
        fig.update_xaxes(tickformat="+.0%")
        chart(fig, 460)
        st.caption("Association only. Salaries are normalised by role median first, so skills common in well-paid roles are not automatically flagged. "
                   "Mann-Whitney U tests with Benjamini-Hochberg correction for testing many skills at once.")

    with st.expander("🔬 Try the skill extractor on your own text"):
        txt = st.text_area("Job description", "We need a Data Scientist with strong Python, PostgreSQL and scikit-learn. "
                           "Experience with PySpark or Databricks is a plus. No Tableau required.", height=110)
        found = nlp.extract_skills(txt)
        st.write("**Extracted skills:** " + (", ".join(found) if found else "none"))
        st.caption("Synonyms map to canonical names (PostgreSQL → SQL, PySpark → Spark). Simple negations such as “No Tableau required” are skipped.")


def page_prediction():
    section("Salary Prediction", "Estimate an annual salary from role, location, experience and skills.")
    res = model_or_none()
    if res is None:
        return
    with st.form("predict"):
        c1, c2, c3 = st.columns(3)
        with c1:
            role = st.selectbox("Role", sorted(df_all.job_title.unique()))
            country = st.selectbox("Country", sorted(df_all.country.unique()))
        with c2:
            years = st.slider("Years of experience", 0, MAX_EXP, min(4, MAX_EXP))
            employment = st.selectbox("Employment type", sorted(df_all.employment_type.unique()))
        with c3:
            remote = st.selectbox("Work arrangement", sorted(df_all.remote_type.unique()))
            skills = st.text_input("Skills (comma separated)", "Python, SQL, Machine Learning")
        submitted = st.form_submit_button("Estimate salary", type="primary", width="stretch")

    if not submitted and "last_pred" not in st.session_state:
        st.info("Fill in the profile and press **Estimate salary**.", icon="🎯")
        return
    if submitted:
        st.session_state.last_pred = dict(role=role, country=country, years=years, employment=employment, remote=remote, skills=skills)
    p = st.session_state.last_pred
    point, low, high = predict_with_interval(res, p["role"], p["country"], p["years"], p["employment"], p["remote"], p["skills"])
    peers = df_all[(df_all.job_title == p["role"]) & (df_all.country == p["country"])]
    left, right = st.columns([1, 1.2])
    with left:
        st.markdown(
            f"""<div class='result-card'><div style='opacity:.75;font-weight:600'>Model estimate · {html.escape(p['role'])} · {html.escape(p['country'])}</div>
<div class='big'>{money(point)}</div><div class='range'>{res.interval_level:.0%} prediction interval: <b>{money(low)} – {money(high)}</b></div>
<div class='range' style='font-size:13px;opacity:.7'>Interval calibrated on out-of-fold residuals; observed holdout coverage {res.holdout_interval_coverage:.0%}.</div></div>""",
            unsafe_allow_html=True,
        )
        st.write("")
        recognised = nlp.extract_skills(p["skills"])
        kpi_cards([
            ("Peer median", money(peers.salary_usd.median()) if len(peers) else "–", f"{len(peers)} postings, same role & country"),
            ("Recognised skills", str(len(recognised)), ", ".join(recognised) or "none matched the taxonomy"),
        ])
    with right:
        curve = career.experience_curve(res.model, p["role"], p["country"], p["employment"], p["remote"], p["skills"], MAX_EXP)
        fig = px.line(curve, x="experience_years", y="estimate_usd", markers=True, title="Estimated salary vs. experience (other inputs fixed)",
                      labels={"experience_years": "Years of experience", "estimate_usd": "Estimate (USD)"})
        fig.add_scatter(x=[p["years"]], y=[point], mode="markers", marker=dict(size=14, color="#EF4444"), name="Your profile")
        if len(peers):
            fig.add_scatter(x=peers.experience_years, y=peers.salary_usd, mode="markers", opacity=0.35, name="Peer postings",
                            marker=dict(color="#94A3B8"))
        chart(fig, 400)
    st.caption(f"Model: **{res.best_model_name}** (selected by 5-fold CV MAE). Illustrative estimate, not salary advice or a job offer benchmark."
               + (" Trained on synthetic data." if SYNTHETIC else ""))


def page_career():
    section("Career Intelligence", "Skill-gap analysis and evidence-based next steps for a target role.")
    res = model_or_none()
    c1, c2 = st.columns([1, 2])
    with c1:
        role = st.selectbox("Target role", sorted(df_all.job_title.unique()), key="career_role")
        country = st.selectbox("Country", sorted(df_all.country.unique()), key="career_country")
        years = st.slider("Your experience (years)", 0, MAX_EXP, min(2, MAX_EXP), key="career_years")
    with c2:
        e1, e2 = st.columns(2)
        employment = e1.selectbox("Employment type", sorted(df_all.employment_type.unique()), key="career_emp")
        remote = e2.selectbox("Work arrangement", sorted(df_all.remote_type.unique()), key="career_remote")
        mine = st.text_area("Your current skills", "Python, Excel, SQL", height=68, key="career_skills")
        st.caption("Free text is fine — synonyms such as *PostgreSQL* or *scikit-learn* are recognised.")
    gap = career.skill_gap(df_all, role, mine)
    if gap.empty:
        st.info("No postings for this role in the data.")
        return
    candidates = gap[gap.status == "Gap"].skill.tolist()
    what_if = career.skill_what_if(res.model, df_all, role, country, years, employment, remote, mine, candidates) if res and candidates else pd.DataFrame()

    have = (gap.status == "Have").sum()
    kpi_cards([
        ("Skills you have", f"{have}", f"of {len(gap)} commonly requested"),
        ("Coverage", f"{have / len(gap):.0%}", f"for {role}"),
        ("Current estimate", money(career.predict_salary(res.model, role, country, years, employment, remote, mine)) if res else "–",
         f"{employment} · {remote}"),
    ])
    l, r = st.columns(2)
    with l:
        fig = px.bar(gap.sort_values("share_of_role_postings"), x="share_of_role_postings", y="skill", color="status", orientation="h",
                     color_discrete_map={"Have": "#10B981", "Gap": "#F59E0B"}, title=f"What {role} postings ask for",
                     labels={"share_of_role_postings": "Share of role postings", "skill": ""})
        fig.update_xaxes(tickformat=".0%")
        chart(fig, 440)
    with r:
        if not what_if.empty:
            fig = px.bar(what_if.sort_values("change_usd"), x="change_usd", y="skill", orientation="h",
                         title="Model what-if: estimate change if you add each skill",
                         labels={"change_usd": "Change in estimate (USD)", "skill": ""}, color_discrete_sequence=["#8B5CF6"])
            chart(fig, 440)
        else:
            st.info("No missing skills to simulate — or the model is unavailable.")
    st.markdown("##### Recommendations")
    for msg in career.recommendations(gap, what_if):
        insight(msg)
    st.caption("Based on posting frequency and model associations in the loaded data. Skills rarely act alone: "
               "projects, domain knowledge and seniority matter too." + (" Synthetic data: illustrative only." if SYNTHETIC else ""))


def page_explorer():
    section("Data Explorer", "Search, inspect and export the validated postings.")
    if not require_rows():
        return
    q = st.text_input("Search titles, skills and descriptions", placeholder="e.g. Spark, Remote, Data Scientist")
    view = df
    if q.strip():
        needle = q.strip().lower()
        hay = (view.job_title + " " + view.skills + " " + view.description + " " + view.country + " " + view.remote_type).str.lower()
        view = view[hay.str.contains(needle, regex=False)]
    cols_default = [c for c in ["posting_id", "job_title", "country", "experience_years", "employment_type", "remote_type", "skills", "salary_usd"] if c in view.columns]
    cols = st.multiselect("Columns", list(view.columns), default=cols_default)
    st.caption(f"{len(view):,} rows")
    st.dataframe(view[cols] if cols else view, hide_index=True, width="stretch", height=460,
                 column_config={"salary_usd": st.column_config.NumberColumn("salary_usd", format="$%,.0f")})

    st.markdown("##### Downloads")
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇ Filtered data (CSV)", reports.to_csv_bytes(view), "jobscope_filtered.csv", "text/csv", width="stretch")
    d2.download_button("⬇ Skill frequency (CSV)", reports.to_csv_bytes(nlp.skill_frequency(view) if len(view) else pd.DataFrame()),
                       "jobscope_skills.csv", "text/csv", width="stretch", disabled=view.empty)
    if d3.button("Build HTML report", width="stretch", disabled=view.empty):
        r = model_or_none()
        st.session_state.report_bytes = reports.build_html_report(
            data_mode=DATA_MODE, kpis=analysis.kpis(view), quality=quality,
            by_role=analysis.salary_by(view, "job_title"), by_country=analysis.salary_by(view, "country"),
            skills=nlp.skill_frequency(view), stats_table=analysis.statistical_summary(view),
            model_summary=r.summary() if r else None, cv_table=r.cv_results if r else None,
            holdout_table=r.holdout_results if r else None, importance=r.permutation_importance if r else None,
        )
    if st.session_state.get("report_bytes"):
        st.download_button("⬇ Download HTML report", st.session_state.report_bytes, "jobscope_report.html", "text/html", type="primary")
    st.caption("CSV exports neutralise spreadsheet formulas (cells starting with =, +, -, @) for safe opening in Excel.")


def page_model():
    section("Model Performance", "How the salary model was selected, how well it does, and what drives it.")
    res = model_or_none()
    if res is None:
        return
    m = res.holdout_metrics
    base = res.holdout_results.set_index("model").loc["Baseline (median)"]
    kpi_cards([
        ("Selected model", res.best_model_name, "lowest 5-fold CV MAE"),
        ("Holdout MAE", money(m["mae_usd"]), f"baseline {money(base.mae_usd)}"),
        ("Holdout RMSE", money(m["rmse_usd"]), f"baseline {money(base.rmse_usd)}"),
        ("Holdout R²", f"{m['r2']:.3f}", f"{res.test_rows} unseen rows"),
        (f"{res.interval_level:.0%} interval coverage", f"{res.holdout_interval_coverage:.0%}", "observed on holdout"),
    ])
    if SYNTHETIC:
        st.warning("These scores measure how well models recover the synthetic generator's formula. They say nothing about real-world salary prediction accuracy.", icon="⚠️")

    c1, c2 = st.columns(2)
    with c1:
        cv = res.cv_results
        fig = px.bar(cv, x="model", y="cv_mae_usd", error_y="cv_mae_std", title="5-fold CV MAE on training split (lower is better)",
                     labels={"cv_mae_usd": "MAE (USD)", "model": ""}, color="model")
        chart(fig.update_layout(showlegend=False))
    with c2:
        hp = res.holdout_predictions
        fig = px.scatter(hp, x="actual", y="predicted", color="job_title", opacity=0.75, title="Holdout: actual vs. predicted",
                         labels={"actual": "Actual (USD)", "predicted": "Predicted (USD)"})
        lim = [float(min(hp.actual.min(), hp.predicted.min())), float(max(hp.actual.max(), hp.predicted.max()))]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(dash="dash", color="#94A3B8"), name="perfect"))
        chart(fig)

    st.markdown("##### Model comparison")
    t1, t2 = st.columns(2)
    with t1:
        st.caption("Cross-validation (training 80%, used for selection)")
        st.dataframe(res.cv_results, hide_index=True, width="stretch", column_config={
            "cv_mae_usd": st.column_config.NumberColumn("CV MAE", format="$%,.0f"), "cv_mae_std": st.column_config.NumberColumn("± std", format="$%,.0f"),
            "cv_rmse_usd": st.column_config.NumberColumn("CV RMSE", format="$%,.0f"), "cv_r2": st.column_config.NumberColumn("CV R²", format="%.3f"),
            "cv_r2_std": st.column_config.NumberColumn("R² std", format="%.3f"), "fit_seconds": st.column_config.NumberColumn("fit s", format="%.2f")})
    with t2:
        st.caption("Holdout (20%, reporting only)")
        st.dataframe(res.holdout_results, hide_index=True, width="stretch", column_config={
            "mae_usd": st.column_config.NumberColumn("MAE", format="$%,.0f"), "rmse_usd": st.column_config.NumberColumn("RMSE", format="$%,.0f"),
            "r2": st.column_config.NumberColumn("R²", format="%.3f")})

    c3, c4 = st.columns(2)
    with c3:
        imp = res.permutation_importance.sort_values("mae_increase_usd")
        chart(px.bar(imp, x="mae_increase_usd", y="feature", error_x="std", orientation="h",
                     title="Permutation importance (MAE increase when shuffled)",
                     labels={"mae_increase_usd": "MAE increase (USD)", "feature": ""}, color_discrete_sequence=["#0EA5E9"]))
    with c4:
        resid = res.holdout_predictions.assign(residual=lambda x: x.actual - x.predicted)
        chart(px.histogram(resid, x="residual", nbins=35, title="Holdout residuals", labels={"residual": "Actual − predicted (USD)"},
                           color_discrete_sequence=["#8B5CF6"]))

    e1, e2 = st.columns(2)
    fmt = {"mae_usd": st.column_config.NumberColumn(format="$%,.0f"), "median_salary_usd": st.column_config.NumberColumn(format="$%,.0f"),
           "mae_pct_of_median": st.column_config.NumberColumn(format="percent")}
    with e1:
        st.caption("Holdout error by role")
        st.dataframe(res.error_by_role, hide_index=True, width="stretch", column_config=fmt)
    with e2:
        st.caption("Holdout error by country")
        st.dataframe(res.error_by_country, hide_index=True, width="stretch", column_config=fmt)

    with st.expander("Methodology, assumptions & limitations", expanded=False):
        st.markdown(f"""
1. **Validation** — required columns, numeric coercion, salary in (0, 1,000,000] annual USD, experience 0–60 years, exact duplicates removed (prevents copies landing on both sides of the split).
2. **Split** — one random 80/20 split (seed 42) *before* any fitting. The holdout is never used for model selection.
3. **Features** — role, country, employment type, work arrangement (one-hot), experience and skill count (scaled), and canonical skills extracted with NLP (multi-hot, vocabulary learned on training folds). Free-text descriptions are **not** model inputs.
4. **Target** — `log1p(salary)`, so effects are multiplicative (a country factor scales every role). Predictions are converted back to USD.
5. **Selection** — Ridge, Random Forest, Gradient Boosting and HistGradientBoosting compared by 5-fold CV MAE on the training split; a median baseline is shown for context.
6. **Uncertainty** — split-conformal interval on the log scale from out-of-fold residuals; coverage checked on the holdout ({res.holdout_interval_coverage:.0%} observed for a {res.interval_level:.0%} target).
7. **Explainability** — permutation importance on the holdout; what-if curves on the prediction and career pages.
8. **Deployment model** — the selected pipeline is refit on all {len(df_all):,} valid rows for predictions.

**Limitations:** random splits can overstate performance for new time periods, employers or regions; salary currencies and periods must be normalised before upload; associations are not causal; small groups have unstable estimates. Avoid protected characteristics as features and audit errors by group before any real use.
""")


pages = [
    st.Page(page_overview, title="Executive Overview", icon=":material/dashboard:", url_path="overview", default=True),
    st.Page(page_market, title="Job Market Intelligence", icon=":material/public:", url_path="market"),
    st.Page(page_skills, title="Skills Intelligence", icon=":material/psychology:", url_path="skills"),
    st.Page(page_prediction, title="Salary Prediction", icon=":material/payments:", url_path="predict"),
    st.Page(page_career, title="Career Intelligence", icon=":material/trending_up:", url_path="career"),
    st.Page(page_explorer, title="Data Explorer", icon=":material/table_view:", url_path="explorer"),
    st.Page(page_model, title="Model Performance", icon=":material/insights:", url_path="model"),
]
st.navigation(pages, position="sidebar").run()

with st.sidebar:
    st.divider()
    st.markdown(f"[🌐 zubairlabs.com]({WEBSITE}) · [GitHub]({GITHUB_URL})")
st.markdown(
    f"<div class='footer'>© 2026 ZubairLabs · JobScope AI v{__version__} · Portfolio demonstration · No paid APIs · "
    f"<a href='{WEBSITE}' target='_blank'>zubairlabs.com</a></div>",
    unsafe_allow_html=True,
)
