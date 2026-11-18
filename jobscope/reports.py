"""Downloadable exports: safe CSV and a self-contained HTML report."""
from __future__ import annotations

import datetime as dt
import html

import pandas as pd

_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def sanitize_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Neutralise spreadsheet formula injection in text cells (CWE-1236)."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].map(
                lambda v: "'" + v if isinstance(v, str) and v.startswith(_FORMULA_PREFIXES) else v
            )
    return out


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    frame = df.copy()
    for col in frame.columns:
        if frame[col].map(lambda v: isinstance(v, (list, set, tuple))).any():
            frame[col] = frame[col].map(lambda v: ", ".join(v) if isinstance(v, (list, set, tuple)) else v)
    return sanitize_for_csv(frame).to_csv(index=False).encode("utf-8")


def _table(df: pd.DataFrame, floatfmt: str = "{:,.2f}") -> str:
    return df.to_html(index=False, border=0, classes="tbl", float_format=lambda x: floatfmt.format(x), escape=True)


def build_html_report(
    *,
    data_mode: str,
    kpis: dict,
    quality: pd.DataFrame,
    by_role: pd.DataFrame,
    by_country: pd.DataFrame,
    skills: pd.DataFrame,
    stats_table: pd.DataFrame | None = None,
    model_summary: dict | None = None,
    cv_table: pd.DataFrame | None = None,
    holdout_table: pd.DataFrame | None = None,
    importance: pd.DataFrame | None = None,
) -> bytes:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC")
    synthetic = data_mode.lower().startswith("synthetic")
    banner = (
        "<div class='warn'><b>Synthetic demonstration data.</b> Every posting and salary in this report was generated "
        "by a seeded formula. Nothing here describes the real job market.</div>"
        if synthetic else
        "<div class='info'><b>User-supplied data.</b> Results describe the uploaded sample only; verify source quality, "
        "currency normalisation and coverage before drawing conclusions.</div>"
    )
    k = kpis
    kpi_html = "".join(
        f"<div class='kpi'><span>{html.escape(label)}</span><b>{html.escape(value)}</b></div>"
        for label, value in [
            ("Postings", f"{k['postings']:,}"),
            ("Median salary", f"${k['median_salary']:,.0f}"),
            ("Interquartile range", f"${k['p25_salary']:,.0f} – ${k['p75_salary']:,.0f}"),
            ("Countries", str(k["countries"])),
            ("Roles", str(k["roles"])),
            ("Remote share", f"{k['remote_share']:.0%}"),
        ]
    )
    sections = [
        "<h2>Data quality</h2>" + _table(quality, "{:,.0f}"),
        "<h2>Salary by role</h2>" + _table(by_role.round(2)),
        "<h2>Salary by country</h2>" + _table(by_country.round(2)),
        "<h2>Skill mentions (rule-based NLP)</h2>" + _table(skills.head(25).round(3)),
    ]
    if stats_table is not None:
        sections.append("<h2>Statistical tests</h2><p class='muted'>Tests describe this sample only. Significance is not causation.</p>"
                        + _table(stats_table.round(4), "{:,.4f}"))
    if model_summary:
        rows = "".join(f"<tr><td>{html.escape(str(a))}</td><td>{html.escape(str(b))}</td></tr>" for a, b in model_summary.items())
        sections.append(f"<h2>Salary model</h2><table class='tbl'><tbody>{rows}</tbody></table>")
    if cv_table is not None:
        sections.append("<h3>Cross-validation comparison (training split)</h3>" + _table(cv_table.round(3)))
    if holdout_table is not None:
        sections.append("<h3>Holdout comparison</h3>" + _table(holdout_table.round(3)))
    if importance is not None:
        sections.append("<h3>Permutation importance (holdout)</h3>" + _table(importance.round(2)))

    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>JobScope AI report</title>
<style>
body{{font-family:Inter,Segoe UI,Arial,sans-serif;max-width:1000px;margin:32px auto;padding:0 20px;color:#0f172a;line-height:1.5}}
h1{{margin:0;font-size:30px}} h2{{margin-top:34px;border-bottom:2px solid #e2e8f0;padding-bottom:6px}}
.brand{{color:#0284c7;font-weight:700;letter-spacing:2px;font-size:12px}} .muted{{color:#64748b;font-size:13px}}
.warn{{background:#fef3c7;border:1px solid #f59e0b;padding:12px 14px;border-radius:10px;margin:18px 0}}
.info{{background:#e0f2fe;border:1px solid #38bdf8;padding:12px 14px;border-radius:10px;margin:18px 0}}
.kpis{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}} .kpi{{border:1px solid #e2e8f0;border-radius:12px;padding:12px}}
.kpi span{{display:block;color:#64748b;font-size:12px}} .kpi b{{font-size:20px}}
.tbl{{border-collapse:collapse;width:100%;font-size:13px}} .tbl th{{background:#f1f5f9;text-align:left}}
.tbl td,.tbl th{{padding:6px 8px;border-bottom:1px solid #e2e8f0}}
</style></head><body>
<div class="brand">ZUBAIRLABS · JOBSCOPE AI</div><h1>Job Market Intelligence Report</h1>
<p class="muted">Generated {stamp} · Data mode: {html.escape(data_mode)}</p>{banner}
<div class="kpis">{kpi_html}</div>{''.join(sections)}
<p class="muted" style="margin-top:40px">Generated by JobScope AI (https://zubairlabs.com). Model estimates are illustrative, not salary advice.</p>
</body></html>"""
    return doc.encode("utf-8")
