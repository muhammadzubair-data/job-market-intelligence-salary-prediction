"""Evidence-based career intelligence: skill gaps and model what-if scenarios.

Recommendations are derived from (a) how often each skill appears in postings
for the target role and (b) how the trained model's estimate changes when a
skill is added. Both are associations in the supplied data, not guarantees.
"""
from __future__ import annotations

import pandas as pd

from .modeling import predict_salary
from .nlp import extract_skills


def role_skill_profile(df: pd.DataFrame, role: str) -> pd.DataFrame:
    sub = df[df.job_title == role]
    if sub.empty:
        return pd.DataFrame(columns=["skill", "share_of_role_postings", "postings"])
    exploded = sub.extracted_skills.explode().dropna()
    counts = exploded.value_counts()
    return pd.DataFrame({
        "skill": counts.index,
        "postings": counts.values,
        "share_of_role_postings": counts.values / len(sub),
    }).reset_index(drop=True)


def skill_gap(df: pd.DataFrame, role: str, user_skills_text: str, min_share: float = 0.15) -> pd.DataFrame:
    """Compare a user's skills with what postings for ``role`` mention."""
    have = set(extract_skills(user_skills_text))
    profile = role_skill_profile(df, role)
    profile["status"] = profile.skill.map(lambda s: "Have" if s in have else "Gap")
    profile = profile[(profile.share_of_role_postings >= min_share) | (profile.status == "Have")]
    return profile.sort_values(["status", "share_of_role_postings"], ascending=[False, False]).reset_index(drop=True)


def skill_what_if(model, df: pd.DataFrame, role, country, years, employment, remote, user_skills_text: str, candidates: list[str]) -> pd.DataFrame:
    """Model-estimated change when each candidate skill is added to the profile."""
    base = predict_salary(model, role, country, years, employment, remote, user_skills_text)
    rows = []
    for skill in candidates:
        new_text = (user_skills_text + ", " + skill) if user_skills_text.strip() else skill
        est = predict_salary(model, role, country, years, employment, remote, new_text)
        rows.append({"skill": skill, "estimate_usd": est, "change_usd": est - base, "change_pct": (est - base) / base if base else 0.0})
    return pd.DataFrame(rows, columns=["skill", "estimate_usd", "change_usd", "change_pct"]).sort_values("change_usd", ascending=False).reset_index(drop=True)


def experience_curve(model, role, country, employment, remote, skills_text: str, max_years: int = 15) -> pd.DataFrame:
    return pd.DataFrame([
        {"experience_years": y, "estimate_usd": predict_salary(model, role, country, y, employment, remote, skills_text)}
        for y in range(0, max_years + 1)
    ])


def recommendations(gap: pd.DataFrame, what_if: pd.DataFrame, top: int = 3, materiality: float = 0.03) -> list[str]:
    """Plain-language recommendations, each tied to the evidence behind it."""
    msgs = []
    gaps = gap[gap.status == "Gap"].head(top)
    impact = what_if.set_index("skill") if not what_if.empty else pd.DataFrame()
    for _, row in gaps.iterrows():
        line = f"**{row.skill}** appears in {row.share_of_role_postings:.0%} of postings for this role"
        if not impact.empty and row.skill in impact.index:
            pct = impact.loc[row.skill, "change_pct"]
            delta = impact.loc[row.skill, "change_usd"]
            if abs(pct) >= materiality:
                line += f"; adding it moves the model estimate by {delta:+,.0f} USD ({pct:+.1%})"
            else:
                line += f"; the model shows no material salary difference from it alone ({pct:+.1%}, within noise)"
        msgs.append(line + ".")
    if not msgs:
        msgs.append("Your listed skills already cover the skills most frequently mentioned for this role in the data.")
    return msgs
