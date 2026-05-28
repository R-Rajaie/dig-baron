"""League Objective Setup Analytics — Dash web application.

Run from project root:
    python src/lolobj/webapp/app.py

Requires:
    pip install dash plotly
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

# ── path setup ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[3]
_SRC  = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dash import Dash, Input, Output, State, callback, dcc, html
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from lolobj.config import PROCESSED_DIR, RiotConfig, load_riot_config
from lolobj.analysis.baseline_report import OUTCOME_ORDER, PROFILE_ORDER, assign_setup_profiles
from lolobj.analysis.state_first import make_state_first_frame, summarize_action_by_state
from lolobj.riot_client import RiotClient, PLATFORM_TO_REGION

# ── data ────────────────────────────────────────────────────────────────────
print("[app] Loading data…")
df = pd.read_parquet(PROCESSED_DIR / "objective_windows.parquet")
df["setup_profile"] = assign_setup_profiles(df)
df_sf = make_state_first_frame(df)
_N = len(df)
_MATCH_IDS = set(df["match_id"].unique())
print(f"[app] {_N:,} rows, {df['match_id'].nunique():,} matches")

RANKS = [r for r in ["low", "mid", "high", "elite"] if r in df["rank_bucket"].unique()]

# ── pre-compute stats ────────────────────────────────────────────────────────
STATS = {
    "matches":     df["match_id"].nunique(),
    "objectives":  df.groupby(["match_id", "objective_instance"]).ngroups,
    "secure_rate": round(df["secured"].mean() * 100, 1),
    "obj_types":   df["objective_type"].nunique(),
}

_vc = df["outcome_label"].value_counts()
outcome_df = pd.DataFrame({
    "outcome": OUTCOME_ORDER,
    "n": [int(_vc.get(o, 0)) for o in OUTCOME_ORDER],
})
outcome_df = outcome_df[outcome_df["n"] > 0].copy()
outcome_df["pct"] = (outcome_df["n"] / _N * 100).round(1)

profile_df = (
    df.groupby("setup_profile")["secured"]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns={"mean": "secure_rate", "count": "n"})
)
profile_df["secure_pct"] = (profile_df["secure_rate"] * 100).round(1)
_pm = {p: i for i, p in enumerate(PROFILE_ORDER)}
profile_df = profile_df.sort_values("setup_profile", key=lambda s: s.map(_pm).fillna(99))

_FEATURE_LABELS = {
    "arrived_first":      "Arrived first",
    "jungler_alive_T_30": "Jungler alive (T-30)",
    "support_alive_T_30": "Support alive (T-30)",
    "jungler_alive_T_60": "Jungler alive (T-60)",
}
feat_rows = []
for col, label in _FEATURE_LABELS.items():
    if col not in df.columns:
        continue
    r1 = round(df[df[col] == 1]["secured"].mean() * 100, 1)
    r0 = round(df[df[col] == 0]["secured"].mean() * 100, 1)
    feat_rows.append({"feature": label, "present": r1, "absent": r0, "diff": round(r1 - r0, 1)})
feat_df = pd.DataFrame(feat_rows).sort_values("diff")

deaths_df = (
    df[df["team_deaths_60s"] <= 4]
    .groupby("team_deaths_60s")["secured"]
    .agg(["mean", "count"])
    .reset_index()
)
deaths_df["secure_pct"] = (deaths_df["mean"] * 100).round(1)
deaths_df.columns = ["deaths", "mean", "n", "secure_pct"]

arrived_tbl = summarize_action_by_state(df_sf, "arrived_first",     min_n=50)
numbers_tbl  = summarize_action_by_state(df_sf, "numbers_adv_T_30",  min_n=50)

rank_rows = []
for rank in RANKS:
    sub = df[df["rank_bucket"] == rank]
    for outcome in OUTCOME_ORDER:
        cnt = (sub["outcome_label"] == outcome).sum()
        if cnt > 0:
            rank_rows.append({"rank": rank, "outcome": outcome,
                               "pct": round(cnt / len(sub) * 100, 2)})
rank_outcome_df = pd.DataFrame(rank_rows)

# ── models ───────────────────────────────────────────────────────────────────
print("[app] Training models…")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.preprocessing import StandardScaler

_SN = [c for c in ["gold_diff_T_60", "team_alive_T_60", "enemy_alive_T_60",
       "team_deaths_90s", "enemy_deaths_60s", "previous_same_obj_team",
       "previous_same_obj_enemy", "objective_number"]
       if c in df_sf.columns and pd.api.types.is_numeric_dtype(df_sf[c])]
_SC = [c for c in ["objective_type", "rank_bucket", "gold_state_T_60",
       "alive_state_T_60", "death_state_pre_obj"] if c in df_sf.columns]
_SA = [c for c in ["arrived_first", "team_grouped_T_30", "numbers_adv_T_30",
       "numbers_down_T_30", "gave_setup_T_30", "free_setup_T_30", "nearby_diff_T_30"]
       if c in df_sf.columns]

_Xn  = df_sf[_SN].fillna(0).astype(float)
_Xc  = pd.get_dummies(df_sf[_SC], drop_first=False).astype(float) if _SC else pd.DataFrame(index=df_sf.index)
_Xa  = df_sf[_SA].fillna(0).astype(float)
_Xs  = pd.concat([_Xn, _Xc], axis=1)
_Xb  = pd.concat([_Xn, _Xc, _Xa], axis=1)
_y   = df_sf["secured"].astype(int).values
_g   = df_sf["match_id"].values

_gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
_tr, _te = next(_gss.split(_Xs, _y, _g))
_ytr, _yte = _y[_tr], _y[_te]

model_rows = []
_saved_models: dict[str, tuple] = {}   # name → (lr, sc, X_full)
for _mname, _X, _nf in [
    ("State (T-60)",  _Xs, len(_Xs.columns)),
    ("Setup (T-30)",  _Xa, len(_Xa.columns)),
    ("State + Setup", _Xb, len(_Xb.columns)),
]:
    _sc = StandardScaler()
    _lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    _lr.fit(_sc.fit_transform(_X.iloc[_tr].fillna(0).values), _ytr)
    _p  = _lr.predict_proba(_sc.transform(_X.iloc[_te].fillna(0).values))[:, 1]
    model_rows.append({
        "Model": _mname,
        "AUC":      round(roc_auc_score(_yte, _p), 4),
        "Log-loss": round(log_loss(_yte, _p), 4),
        "Brier":    round(brier_score_loss(_yte, _p), 4),
        "Features": _nf,
    })
    _saved_models[_mname] = (_lr, _sc, _X)

# Full-dataset predictions stored in df_sf for the game analyzer
for _key, _mname in [("sf_prob_state", "State (T-60)"), ("sf_prob_both", "State + Setup")]:
    _lr, _sc, _X = _saved_models[_mname]
    df_sf[_key] = _lr.predict_proba(_sc.transform(_X.fillna(0).values))[:, 1]
df_sf["sf_swing_pp"] = (df_sf["sf_prob_both"] - df_sf["sf_prob_state"]) * 100

model_df = pd.DataFrame(model_rows)
print(f"[app] State+Setup AUC = {model_df[model_df['Model']=='State + Setup']['AUC'].values[0]:.3f}")

# SHAP feature importance for State+Setup model
print("[app] Computing SHAP importance…")
import shap as _shap
_lr_sh, _sc_sh, _X_sh = _saved_models["State + Setup"]
_X_sh_scaled = _sc_sh.transform(_X_sh.fillna(0).values)
_shap_exp = _shap.LinearExplainer(_lr_sh, _X_sh_scaled)
_shap_vals = _shap_exp.shap_values(_X_sh_scaled)
_shap_mean = np.abs(_shap_vals).mean(axis=0)
_FEAT_RENAME = {
    "arrived_first":           "Arrived first (T-60)",
    "gold_diff_T_60":          "Gold diff (T-60)",
    "team_alive_T_60":         "Allies alive (T-60)",
    "enemy_alive_T_60":        "Enemies alive (T-60)",
    "nearby_diff_T_30":        "Nearby diff (T-30)",
    "free_setup_T_30":         "Free setup (T-30)",
    "gave_setup_T_30":         "Gave setup (T-30)",
    "numbers_adv_T_30":        "Numbers adv (T-30)",
    "numbers_down_T_30":       "Numbers down (T-30)",
    "team_grouped_T_30":       "Team grouped 3+ (T-30)",
    "team_deaths_90s":         "Allied deaths (last 90s)",
    "enemy_deaths_60s":        "Enemy deaths (last 60s)",
    "objective_number":        "Objective number",
    "previous_same_obj_team":  "Team prev obj taken",
    "previous_same_obj_enemy": "Enemy prev obj taken",
}
_STATE_FEATS = set(_SN) | {c for c in _X_sh.columns if any(
    c.startswith(p) for p in ("objective_type", "rank_bucket", "gold_state", "alive_state", "death_state")
)}
_ACTION_FEATS = set(_SA)
shap_imp_df = (
    pd.DataFrame({"feature": _X_sh.columns, "importance": _shap_mean})
    .nlargest(15, "importance")
    .sort_values("importance")
    .reset_index(drop=True)
)
shap_imp_df["label"] = shap_imp_df["feature"].map(
    lambda c: _FEAT_RENAME.get(c, c.replace("_", " "))
)
shap_imp_df["is_action"] = shap_imp_df["feature"].isin(_ACTION_FEATS)
print(f"[app] SHAP done — top feature: {shap_imp_df['feature'].iloc[-1]}")

# ── definitions ──────────────────────────────────────────────────────────────

OUTCOME_DEFS = [
    ("clean_take",               "good",    "Team secured, no meaningful fight or fight won cleanly."),
    ("clean_give",               "neutral", "Enemy secured; team chose not to contest. May be intentional."),
    ("good_trade",               "good",    "Team gave this objective but gained significant resources elsewhere."),
    ("coinflip",                 "neutral", "Both teams contested; neither had a clear setup advantage beforehand."),
    ("bad_contest",              "bad",     "Team contested from a weak position (low numbers, deaths) and lost."),
    ("won_fight_lost_objective", "bad",     "Team won the fight but failed to secure the objective."),
    ("lost_fight_got_objective", "mixed",   "Team lost the fight but secured the objective anyway (often a steal)."),
    ("throw_setup",              "bad",     "Team had a winning setup but lost a key member just before the window."),
    ("objective_steal",          "mixed",   "Team secured from a heavily disadvantaged state via surprise smite or engage."),
    ("no_meaningful_contest",    "neutral", "Neither team committed. Objective taken without a real fight."),
]

PROFILE_DEFS = [
    ("free_setup",        "good",    "Team present, enemy absent, no recent allied deaths.",
                                     "Full positional control. Highest expected secure rate."),
    ("free_setup_deaths", "mixed",   "Team present, enemy absent, but allied deaths in the prior 60s.",
                                     "Uncontested positionally but weakened by recent losses."),
    ("clean_contest",     "neutral", "Both teams near objective, team not at a numbers or health disadvantage.",
                                     "Even or favorable fight with both sides committed."),
    ("disadvantaged",     "bad",     "Both teams present, but team had recent deaths or fewer alive champions.",
                                     "Contesting from behind. High risk of a bad outcome."),
    ("gave_away",         "bad",     "Enemy present at objective, team absent.",
                                     "Team did not show up to contest. Objective conceded."),
    ("both_absent",       "neutral", "Neither team present near objective at T-30.",
                                     "Neither side committed. Objective taken without setup by one side."),
]

# ── Riot API helpers ─────────────────────────────────────────────────────────

PLATFORM_OPTIONS = [
    {"label": "NA  (North America)",          "value": "na1"},
    {"label": "EUW (Europe West)",            "value": "euw1"},
    {"label": "KR  (Korea)",                  "value": "kr"},
    {"label": "EUN (Europe Nordic & East)",   "value": "eun1"},
    {"label": "JP  (Japan)",                  "value": "jp1"},
    {"label": "BR  (Brazil)",                 "value": "br1"},
]

# Map platform → regional routing host for Match-V5 / Account-V1
_P2R = {k.lower(): v for k, v in PLATFORM_TO_REGION.items()}


def _riot_client_for(platform: str) -> RiotClient | None:
    """Build a RiotClient for a specific platform using the configured API key."""
    try:
        base = load_riot_config()
        region = _P2R.get(platform.lower(), "americas")
        cfg = RiotConfig(api_key=base.api_key, platform=platform.lower(), region=region)
        return RiotClient(config=cfg)
    except RuntimeError:
        return None


# ── design tokens ────────────────────────────────────────────────────────────
_ACCENT = "#3d5af1"
_GOOD   = "#10b981"
_BAD    = "#ef4444"
_MIXED  = "#f59e0b"
_NEUT   = "#94a3b8"
_FONT   = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

_OUTCOME_COLORS = {
    "clean_take": _GOOD, "clean_give": _GOOD, "good_trade": _GOOD,
    "coinflip": _NEUT,
    "bad_contest": _BAD, "won_fight_lost_objective": _BAD, "throw_setup": _BAD,
    "lost_fight_got_objective": _MIXED, "objective_steal": _MIXED,
    "no_meaningful_contest": _NEUT,
}

_BASE = dict(
    paper_bgcolor="white", plot_bgcolor="white",
    font=dict(family=_FONT, size=12, color="#1e293b"),
    margin=dict(l=4, r=52, t=44, b=8),
    hoverlabel=dict(bgcolor="white", font_size=13, font_family=_FONT),
)


def _fig(fig: go.Figure, **layout) -> go.Figure:
    fig.update_layout(**{**_BASE, **layout})
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=False, gridwidth=1)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


# ── figures ───────────────────────────────────────────────────────────────────

def fig_outcomes() -> go.Figure:
    colors = [_OUTCOME_COLORS.get(o, _NEUT) for o in outcome_df["outcome"]]
    fig = go.Figure(go.Bar(
        x=outcome_df["pct"], y=outcome_df["outcome"], orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=outcome_df["pct"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside", textfont=dict(size=11, color="#475569"),
        hovertemplate="<b>%{y}</b><br>%{x:.1f}% of rows<extra></extra>",
    ))
    return _fig(fig,
        title=dict(text="Objective outcome distribution", font=dict(size=14, color="#1e293b")),
        height=360,
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="% of rows", zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False, autorange="reversed"),
        margin=dict(l=4, r=80, t=44, b=8),
    )


def fig_profile_secure() -> go.Figure:
    p = profile_df.sort_values("secure_pct")
    colors = [_GOOD if v >= 70 else _BAD if v < 35 else _ACCENT for v in p["secure_pct"]]
    fig = go.Figure(go.Bar(
        x=p["secure_pct"], y=p["setup_profile"], orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=p["secure_pct"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside", textfont=dict(size=11, color="#475569"),
        hovertemplate="<b>%{y}</b><br>Secure: %{x:.1f}%  (n=%{customdata:,})<extra></extra>",
        customdata=p["n"],
    ))
    return _fig(fig,
        title=dict(text="Secure rate by setup profile", font=dict(size=14, color="#1e293b")),
        height=300,
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", range=[0, 115], title="secure rate (%)"),
        margin=dict(l=4, r=80, t=44, b=8),
    )


def fig_feature_impact() -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Absent / 0", x=feat_df["feature"], y=feat_df["absent"],
        marker=dict(color=_NEUT, line=dict(width=0)),
        hovertemplate="%{x}<br>Absent: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Present / 1", x=feat_df["feature"], y=feat_df["present"],
        marker=dict(color=_ACCENT, line=dict(width=0)),
        hovertemplate="%{x}<br>Present: %{y:.1f}%<extra></extra>",
    ))
    return _fig(fig,
        barmode="group",
        title=dict(text="Secure rate: feature present vs absent", font=dict(size=14, color="#1e293b")),
        height=300,
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", range=[0, 105], title="secure rate (%)"),
        xaxis=dict(showgrid=False),
        showlegend=True,
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center",
                    font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=4, r=20, t=44, b=40),
    )


def fig_deaths_secure() -> go.Figure:
    colors = [_GOOD if i == 0 else _BAD for i in range(len(deaths_df))]
    fig = go.Figure(go.Bar(
        x=deaths_df["deaths"].astype(str), y=deaths_df["secure_pct"],
        marker=dict(color=colors, line=dict(width=0)),
        text=deaths_df["secure_pct"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside", textfont=dict(size=11, color="#475569"),
        hovertemplate="Deaths: %{x}<br>Secure: %{y:.1f}%<extra></extra>",
    ))
    return _fig(fig,
        title=dict(text="Allied deaths in 60s before objective → secure rate", font=dict(size=14, color="#1e293b")),
        height=280,
        xaxis=dict(showgrid=False, title="deaths in T-60s window"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", range=[0, 80], title="secure rate (%)"),
        margin=dict(l=4, r=20, t=44, b=30),
    )


def fig_state_conditioned(tbl: pd.DataFrame, action_label: str) -> go.Figure:
    if tbl.empty:
        return go.Figure()
    plot = tbl.nlargest(15, "n_total").sort_values("diff_pp")
    colors = [_BAD if v < 0 else _ACCENT for v in plot["diff_pp"]]
    fig = go.Figure(go.Bar(
        x=plot["diff_pp"], y=plot["state_bucket"], orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=plot["diff_pp"].apply(lambda x: f"{x:+.1f}pp"),
        textposition="outside", textfont=dict(size=10, color="#475569"),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Absent:  %{customdata[0]:.1f}%  (n=%{customdata[2]:,})<br>"
            "Present: %{customdata[1]:.1f}%  (n=%{customdata[3]:,})<br>"
            "Δ: %{x:+.1f} pp<extra></extra>"
        ),
        customdata=plot[["secure_rate_action_0", "secure_rate_action_1",
                          "n_action_0", "n_action_1"]].values,
    ))
    return _fig(fig,
        title=dict(text=f"State-conditioned effect: {action_label}", font=dict(size=14, color="#1e293b")),
        height=max(340, len(plot) * 30 + 80),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=True,
                   zerolinecolor="#cbd5e1", zerolinewidth=1.5,
                   title="Δ secure rate (pp)  action=1 minus action=0"),
        margin=dict(l=4, r=80, t=44, b=30),
    )


def fig_model_auc() -> go.Figure:
    fig = go.Figure(go.Bar(
        x=model_df["Model"], y=model_df["AUC"],
        marker=dict(color=[_NEUT, _NEUT, _ACCENT], line=dict(width=0)),
        text=model_df["AUC"].apply(lambda x: f"{x:.3f}"),
        textposition="outside", textfont=dict(size=12, color="#475569"),
        hovertemplate="<b>%{x}</b><br>AUC: %{y:.4f}<extra></extra>",
        width=0.45,
    ))
    return _fig(fig,
        title=dict(text="Model AUC: state vs setup vs combined", font=dict(size=14, color="#1e293b")),
        height=280,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", range=[0.75, 0.95], title="AUC (test set)"),
        margin=dict(l=4, r=20, t=44, b=20),
    )


def fig_shap_importance() -> go.Figure:
    colors = [_ACCENT if a else _NEUT for a in shap_imp_df["is_action"]]
    fig = go.Figure(go.Bar(
        x=shap_imp_df["importance"], y=shap_imp_df["label"], orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=shap_imp_df["importance"].apply(lambda x: f"{x:.4f}"),
        textposition="outside", textfont=dict(size=10, color="#475569"),
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
    ))
    return _fig(fig,
        title=dict(text="Feature importance (mean |SHAP|): State + Setup model", font=dict(size=14, color="#1e293b")),
        height=max(360, len(shap_imp_df) * 28 + 80),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="mean |SHAP value|"),
        margin=dict(l=4, r=80, t=44, b=8),
    )


def fig_rank_outcomes() -> go.Figure:
    if rank_outcome_df.empty:
        return go.Figure()
    _rc = {"low": "#94a3b8", "mid": "#60a5fa", "high": "#3d5af1", "elite": "#1e1b4b"}
    fig = px.bar(rank_outcome_df, x="outcome", y="pct", color="rank", barmode="group",
                 color_discrete_map=_rc, labels={"pct": "% of rows", "outcome": "", "rank": "Rank"})
    fig.update_traces(marker_line_width=0)
    return _fig(fig,
        title=dict(text="Outcome distribution by rank bucket", font=dict(size=14, color="#1e293b")),
        height=380,
        xaxis=dict(showgrid=False, tickangle=-35),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="% of rows"),
        showlegend=True,
        legend=dict(orientation="h", y=-0.38, x=0.5, xanchor="center",
                    font=dict(size=12), bgcolor="rgba(0,0,0,0)", title=""),
        margin=dict(l=4, r=20, t=44, b=80),
    )


# ── game analysis builder ────────────────────────────────────────────────────

def _swing_badge(sw: float) -> html.Span:
    if math.isnan(sw):
        return html.Span()
    cls = "swing swing--pos" if sw > 0 else "swing swing--neg" if sw < 0 else "swing swing--zero"
    return html.Span(f"{sw:+.0f} pp", className=cls)


def build_game_analysis(match_id: str) -> html.Div:
    """Return a styled breakdown of each objective in *match_id*."""
    rows = df_sf[df_sf["match_id"] == match_id]
    if rows.empty:
        return html.Div(f"match_id '{match_id}' not found in dataset.", className="ga-error")

    obj_cards = []
    for inst in sorted(rows["objective_instance"].unique()):
        obj = rows[rows["objective_instance"] == inst].sort_values("team_side")
        otype = obj["objective_type"].iloc[0].replace("_", " ").title()
        onum  = int(obj["objective_number"].iloc[0])

        team_panels = []
        for _, r in obj.iterrows():
            side     = str(r.get("team_side", "?")).upper()
            state    = str(r.get("state_bucket_T_60", "N/A"))
            pst      = float(r.get("sf_prob_state", float("nan")))
            pb       = float(r.get("sf_prob_both",  float("nan")))
            sw       = float(r.get("sf_swing_pp",   float("nan")))
            secured  = bool(r.get("secured"))
            outcome  = str(r.get("outcome_label", ""))

            # Action summary
            acts = []
            if r.get("arrived_first") == 1:
                acts.append("arrived first")
            if r.get("free_setup_T_30") == 1:
                acts.append("free setup (enemy absent)")
            elif r.get("gave_setup_T_30") == 1:
                acts.append("gave setup (team absent)")
            if r.get("numbers_adv_T_30") == 1:
                acts.append("numbers advantage")
            elif r.get("numbers_down_T_30") == 1:
                acts.append("numbers down")
            tn = r.get("team_nearby_T_30", "?")
            en = r.get("enemy_nearby_T_30", "?")
            acts.append(f"{int(tn)} allied vs {int(en)} enemy nearby at T-30")

            has_pred = not math.isnan(pst)
            pred_row = html.Div(className="ga-pred-row", children=[
                html.Div(className="ga-pred", children=[
                    html.Span("State only", className="ga-pred-label"),
                    html.Span(f"{pst*100:.0f}%", className="ga-pred-val"),
                ]),
                html.Div("→", className="ga-pred-arrow"),
                html.Div(className="ga-pred", children=[
                    html.Span("State+Setup", className="ga-pred-label"),
                    html.Span(f"{pb*100:.0f}%", className="ga-pred-val"),
                ]),
                html.Div(className="ga-swing-wrap", children=[
                    html.Span("setup swing", className="ga-pred-label"),
                    _swing_badge(sw),
                ]),
            ]) if has_pred else None

            team_panels.append(html.Div(className="ga-team", children=[
                html.Div(className="ga-team-header", children=[
                    html.Span(f"{side} SIDE", className="ga-side"),
                    html.Span(
                        "✓ SECURED" if secured else "✗ not secured",
                        className="ga-result ga-result--yes" if secured else "ga-result ga-result--no",
                    ),
                    html.Span(outcome.replace("_", " "), className="ga-outcome-label"),
                ]),
                html.Div(className="ga-row", children=[
                    html.Span("T-60 state", className="ga-key"),
                    html.Span(state, className="ga-val ga-val--mono"),
                ]),
                html.Div(className="ga-row", children=[
                    html.Span("Setup (T-30)", className="ga-key"),
                    html.Span(", ".join(acts), className="ga-val"),
                ]),
                pred_row,
            ]))

        obj_cards.append(html.Div(className="ga-obj", children=[
            html.Div(f"{otype} #{onum}", className="ga-obj-title"),
            html.Div(className="ga-teams", children=team_panels),
        ]))

    return html.Div(className="ga-root", children=[
        html.Div(className="ga-match-header", children=[
            html.Span(match_id, className="ga-match-id"),
            html.Span(f"{len(rows)//2} objectives", className="ga-match-meta"),
        ]),
        *obj_cards,
    ])


# ── shared layout helpers ─────────────────────────────────────────────────────

def stat_card(label: str, value: str, note: str = "") -> html.Div:
    return html.Div(className="stat-card", children=[
        html.Div(str(value), className="stat-value"),
        html.Div(label, className="stat-label"),
        html.Div(note, className="stat-note") if note else None,
    ])


def section(title: str, subtitle: str, *children) -> html.Div:
    return html.Div(className="section", children=[
        html.H2(title, className="section-title"),
        html.P(subtitle, className="section-sub") if subtitle else None,
        *children,
    ])


def note(text: str) -> html.P:
    return html.P(text, className="note")


def graph(fig: go.Figure) -> dcc.Graph:
    return dcc.Graph(figure=fig, config={"displayModeBar": False}, className="chart")


# ── tab styles ────────────────────────────────────────────────────────────────

_TAB = {
    "padding": "0 2px", "height": "52px", "lineHeight": "52px",
    "border": "none", "borderTop": "none", "borderLeft": "none",
    "borderRight": "none", "borderBottom": "2px solid transparent",
    "borderRadius": "0", "background": "transparent",
    "color": "#64748b", "fontSize": "14px", "fontWeight": "400",
    "fontFamily": _FONT, "marginRight": "28px",
}
_TAB_ACTIVE = {
    **_TAB,
    "color": "#1e293b", "borderBottom": "2px solid #3d5af1", "fontWeight": "500",
}

# ── page: Home ────────────────────────────────────────────────────────────────

def page_home() -> html.Div:
    return html.Div(className="page", children=[
        html.Div(className="hero", children=[
            html.H1("League Objective Setup Analytics", className="hero-title"),
            html.P(
                "A framework for understanding what separates good and bad objective setups "
                "in League of Legends, and how that varies by rank, objective type, and game state.",
                className="hero-sub",
            ),
        ]),
        html.Div(className="stat-row", children=[
            stat_card("Matches analyzed", f"{STATS['matches']:,}"),
            stat_card("Objective instances", f"{STATS['objectives']:,}"),
            stat_card("Objective types", str(STATS["obj_types"])),
            stat_card("Overall secure rate", f"{STATS['secure_rate']}%"),
        ]),
        section("Core research question", "",
            html.P(
                "What conditions make contesting, starting, trading, or giving an objective the right call? "
                "The project focuses on Dragon and Baron first, then extends to Herald, Voidgrubs, Soul, and Elder.",
                className="body-text",
            ),
        ),
        section("Objective outcomes",
                "Each row is classified into one of ten outcome categories, going beyond just secured or not secured.",
            graph(fig_outcomes()),
        ),
        section("What this project does", "",
            html.Ul(className="bullet-list", children=[
                html.Li("Builds a clean objective-window table, one row per team per objective"),
                html.Li("Labels outcomes beyond just secured / not secured"),
                html.Li("Separates the game state at T-60 from what teams actually chose to do by T-30"),
                html.Li("Compares teams in similar situations to see how much setup choices matter"),
                html.Li("Lets you look up your own recent games and see a per-objective breakdown"),
            ]),
        ),
    ])


# ── page: Methods ─────────────────────────────────────────────────────────────

def page_methods() -> html.Div:
    return html.Div(className="page", children=[
        html.H1("Methods", className="page-title"),

        section("Unit of analysis", "Each row is one team's perspective on one objective instance.",
            html.Div(className="mono-block",
                     children=html.Code("match_id  +  objective_instance  +  team_id  →  one row")),
            html.P(
                "For each objective two rows are created, one per team side. Features describe only "
                "that team's state relative to the opponent. Objectives and teams are never mixed within a row.",
                className="body-text",
            ),
        ),

        section("Objective time windows",
                "Features are computed within four windows relative to T=0 (objective take time).",
            html.Div(className="window-row", children=[
                html.Div(className="win-card", children=[
                    html.Div("T−120 → T−90", className="win-label"),
                    html.Div("Early setup", className="win-name"),
                ]),
                html.Div(className="win-card win-card--hi", children=[
                    html.Div("T−90 → T−60", className="win-label"),
                    html.Div("State (pre-commitment)", className="win-name"),
                ]),
                html.Div(className="win-card win-card--hi", children=[
                    html.Div("T−60 → T−30", className="win-label"),
                    html.Div("Setup actions", className="win-name"),
                ]),
                html.Div(className="win-card", children=[
                    html.Div("T−30 → T+30", className="win-label"),
                    html.Div("Contest / take", className="win-name"),
                ]),
                html.Div(className="win-card", children=[
                    html.Div("T+30 → T+120", className="win-label"),
                    html.Div("Aftermath", className="win-name"),
                ]),
            ]),
        ),

        section("State-first analytical framework",
                "The core idea: separate what was already true at T-60 from what teams chose to do between T-60 and T-30.",
            html.Div(className="three-col", children=[
                html.Div(className="method-card", children=[
                    html.Div("01", className="method-num"),
                    html.H3("State", className="method-title"),
                    html.P("Information at T-60 or earlier. Gold difference, alive counts, "
                           "recent deaths. The situation before teams commit to rotating.",
                           className="method-body"),
                ]),
                html.Div(className="method-card", children=[
                    html.Div("02", className="method-num"),
                    html.H3("Action", className="method-title"),
                    html.P("Observed behaviour between T-60 and T-30. Who arrived first, "
                           "how many champions appeared near the objective, whether setup "
                           "was contested or conceded.",
                           className="method-body"),
                ]),
                html.Div(className="method-card", children=[
                    html.Div("03", className="method-num"),
                    html.H3("Outcome", className="method-title"),
                    html.P("Objective secured, outcome label, net objective value. "
                           "Never used as a model feature.",
                           className="method-body"),
                ]),
            ]),
            note("Looking at setup actions within similar T-60 states gives a cleaner comparison than "
                 "looking across all games at once. It holds the starting situation roughly constant. "
                 "Everything here is still observational, not causal."),
        ),

        section("Setup profile definitions",
                "Profiles classify each team's T-30 posture relative to the opponent.",
            html.Div(className="def-grid", children=[
                html.Div(className=f"def-card def-card--{tag}", children=[
                    html.Div(name.replace("_", " "), className="def-name"),
                    html.Div(condition, className="def-condition"),
                    html.Div(meaning, className="def-meaning"),
                ])
                for name, tag, condition, meaning in PROFILE_DEFS
            ]),
        ),

        section("Outcome label definitions",
                "Ten outcome categories describe what happened at and after the objective.",
            html.Div(className="def-grid def-grid--wide", children=[
                html.Div(className=f"def-card def-card--{tag}", children=[
                    html.Div(name.replace("_", " "), className="def-name"),
                    html.Div(desc, className="def-meaning"),
                ])
                for name, tag, desc in OUTCOME_DEFS
            ]),
        ),

        section("Modeling approach",
                "Models are used as analytical tools, not prediction engines.",
            html.Div(className="step-list", children=[
                html.Div(className="step", children=[html.Span("1", className="step-num"),
                    "Descriptive statistics and setup profiles"]),
                html.Div(className="step", children=[html.Span("2", className="step-num"),
                    "State-only logistic regression (T-60 features only)"]),
                html.Div(className="step", children=[html.Span("3", className="step-num"),
                    "Setup-only logistic regression (T-30 action features only)"]),
                html.Div(className="step", children=[html.Span("4", className="step-num"),
                    "Combined model (state + setup): measures how much setup adds on top of state"]),
                html.Div(className="step", children=[html.Span("5", className="step-num"),
                    "State-conditioned within-bucket comparisons for each action"]),
            ]),
            note("Train/test splits are grouped by match_id. All rows from one match stay together "
                 "in train or test, which prevents leakage from objectives within the same game."),
        ),

        section("Data and limitations", "",
            html.Ul(className="bullet-list", children=[
                html.Li("Data source: Riot Games Match-V5 API (match summaries and timeline events)"),
                html.Li("Exact player pathing, wave state, and item timing are not available, so proxies are used"),
                html.Li("Role assignment is inferred from position data and may have errors for off-meta compositions"),
                html.Li("Findings are observational. Teams that arrive first may do so because of champion advantages the model cannot see"),
                html.Li("Language throughout: 'associated with', 'predictive of', 'proxy for'. Not 'proves' or 'causes'"),
            ]),
        ),
    ])


# ── page: Analysis ────────────────────────────────────────────────────────────

def page_analysis() -> html.Div:
    _best = model_df[model_df["Model"] == "State + Setup"]["AUC"].values[0]
    _base = model_df[model_df["Model"] == "State (T-60)"]["AUC"].values[0]
    _lift = round((_best - _base) * 100, 1)

    return html.Div(className="page", children=[
        html.H1("Analysis & Results", className="page-title"),

        section("Setup profiles and secure rates",
                "Profiles classify each team's position at T-30 based on who showed up and whether there were recent deaths.",
            graph(fig_profile_secure()),
            note("'Free setup' and 'gave away' sit at opposite ends of the secure rate distribution, "
                 "which is what you'd expect. See Methods for full profile definitions."),
        ),

        section("Impact of key binary features",
                "Secure rate with and without each binary feature.",
            graph(fig_feature_impact()),
        ),

        section("Deaths before objective",
                "Each allied death in the 60s before the objective substantially reduces the secure rate.",
            graph(fig_deaths_secure()),
            note("Even one death drops the secure rate by roughly 20 percentage points. "
                 "Two deaths brings it below 25%. Dying just before an objective is one of the "
                 "more reliable ways to lose it."),
        ),

        section("State-first: value of arriving first",
                "Within each T-60 state bucket, teams that arrived first vs those that did not.",
            graph(fig_state_conditioned(arrived_tbl, "arrived first")),
            note("The effect is biggest in 'even' and 'behind' states. "
                 "When the gold lead is large enough to decide the fight, arrival timing matters less."),
        ),

        section("State-first: value of numbers advantage at T-30",
                "Same analysis for having more allies than enemies near the objective at T-30.",
            graph(fig_state_conditioned(numbers_tbl, "numbers advantage at T-30")),
        ),

        section("Model comparison",
                f"Three logistic regression models compared on a held-out test set (25% of matches, "
                f"grouped by match_id). State + Setup improves AUC by {_lift:.1f} pp over State alone.",
            graph(fig_model_auc()),
            html.Div(className="table-wrap", children=[
                html.Table(className="data-table", children=[
                    html.Thead(html.Tr([
                        html.Th("Model"), html.Th("AUC"), html.Th("Log-loss"),
                        html.Th("Brier"), html.Th("Features"),
                    ])),
                    html.Tbody([
                        html.Tr(
                            className="row--highlight" if r["Model"] == "State + Setup" else "",
                            children=[
                                html.Td(r["Model"]),
                                html.Td(f"{r['AUC']:.4f}"),
                                html.Td(f"{r['Log-loss']:.4f}"),
                                html.Td(f"{r['Brier']:.4f}"),
                                html.Td(str(r["Features"])),
                            ],
                        )
                        for r in model_rows
                    ]),
                ]),
            ]),
            note("Setup actions add real predictive value beyond T-60 state. "
                 "This is association, not causation."),
        ),

        section("Feature importance (SHAP)",
                "Mean absolute SHAP value per feature in the State + Setup model. "
                "Blue bars are setup actions (T-30); gray bars are state features (T-60 or earlier).",
            graph(fig_shap_importance()),
            note(
                "SHAP values show how much each feature shifts the model's prediction on average. "
                "State features (gray) reflect the situation before teams commit. "
                "Action features (blue) reflect what teams actually did. "
                "High SHAP for an action feature means that behavior consistently moved outcomes, "
                "even after holding the starting state roughly constant."
            ),
        ),

        section("Outcome distribution by rank",
                "How outcomes differ across skill tiers.",
            graph(fig_rank_outcomes()),
        ),

        # ── Player game lookup ───────────────────────────────────────────────
        section("Per-game state-first analysis",
                "Enter a Riot ID to fetch their latest ranked game and run the state-first model on each objective.",
            html.Div(className="lookup-form", children=[
                dcc.Input(
                    id="riot-id-input",
                    type="text",
                    placeholder="GameName#TAG",
                    debounce=False,
                    className="lookup-input",
                ),
                dcc.Dropdown(
                    id="region-select",
                    options=PLATFORM_OPTIONS,
                    value="na1",
                    clearable=False,
                    className="lookup-dropdown",
                ),
                html.Button("Analyze", id="analyze-btn", n_clicks=0, className="lookup-btn"),
            ]),
            note(
                "Searches recent ranked games (queue 420) for a match in the study dataset "
                "(NA, EUW, KR). The Riot API key must be set in your .env file."
            ),
            dcc.Loading(
                id="lookup-loading",
                type="dot",
                color=_ACCENT,
                children=html.Div(id="game-analysis-output", className="ga-container"),
            ),
        ),
    ])


# ── page: Conclusion ──────────────────────────────────────────────────────────

def page_conclusion() -> html.Div:
    _best  = model_df[model_df["Model"] == "State + Setup"]["AUC"].values[0]
    _state = model_df[model_df["Model"] == "State (T-60)"]["AUC"].values[0]
    return html.Div(className="page", children=[
        html.H1("Conclusion", className="page-title"),

        section("Key findings", "",
            html.Div(className="finding-list", children=[
                html.Div(className="finding", children=[
                    html.Div("01", className="finding-num"),
                    html.Div(className="finding-body", children=[
                        html.H3("Setup actions add predictive value beyond pre-objective state",
                                className="finding-title"),
                        html.P(
                            f"The combined model hits AUC {_best:.3f} vs {_state:.3f} for the "
                            "state-only baseline. Teams in similar T-60 situations end up with "
                            "noticeably different outcomes depending on their setup.",
                            className="finding-text",
                        ),
                    ]),
                ]),
                html.Div(className="finding", children=[
                    html.Div("02", className="finding-num"),
                    html.Div(className="finding-body", children=[
                        html.H3("Arrival timing is the strongest single setup signal",
                                className="finding-title"),
                        html.P(
                            "Within similar T-60 state buckets, arriving first is associated with "
                            "40 to 65 percentage point higher secure rates. The pattern holds across "
                            "gold states: behind, even, and ahead.",
                            className="finding-text",
                        ),
                    ]),
                ]),
                html.Div(className="finding", children=[
                    html.Div("03", className="finding-num"),
                    html.Div(className="finding-body", children=[
                        html.H3("Deaths before objective are highly damaging",
                                className="finding-title"),
                        html.P(
                            "A single allied death in the 60s before the objective drops the secure "
                            "rate by ~20 pp. Two deaths brings it below 25%. Throw-setup patterns "
                            "appear to be a primary driver of objective losses.",
                            className="finding-text",
                        ),
                    ]),
                ]),
                html.Div(className="finding", children=[
                    html.Div("04", className="finding-num"),
                    html.Div(className="finding-body", children=[
                        html.H3("Free setup strongly predicts securing; giving setup strongly predicts losing",
                                className="finding-title"),
                        html.P(
                            "Teams that show up with no enemy around secure at very high rates. "
                            "Teams that skip the objective entirely lose it almost every time. "
                            "T-30 presence is the strongest single signal in the data.",
                            className="finding-text",
                        ),
                    ]),
                ]),
            ]),
        ),

        section("Limitations", "",
            html.Ul(className="bullet-list", children=[
                html.Li("All findings are observational. Teams that arrive first may do so because "
                        "of champion kit advantages that also cause them to win fights."),
                html.Li("Wave state, champion scaling, and communication are not captured by the Riot API timeline."),
                html.Li("Results may not generalize across patches with major objective timing changes."),
                html.Li("Role assignment is inferred from position data and may have errors for off-meta compositions."),
            ]),
        ),

        section("Next steps", "",
            html.Ul(className="bullet-list", children=[
                html.Li("Expand to Baron, Herald, Voidgrubs, Soul, and Elder objectives"),
                html.Li("Build rank-stratified models to test whether setup factors differ by tier"),
                html.Li("Add Random Forest + SHAP for non-linear interaction effects"),
                html.Li("Develop setup profile clustering (rule-based or model-assisted)"),
                html.Li("Build the per-game analyzer as a full player-facing diagnostic tool"),
            ]),
        ),

        section("Methodological note", "",
            html.P(
                "State-first analysis should be read as: among teams in similar pre-objective "
                "situations, these setup choices were associated with better or worse secure rates. "
                "This is a cleaner comparison than raw correlation because it holds the starting "
                "situation roughly constant, but it is still observational. It cannot prove that "
                "a team would have secured the objective by making a different choice.",
                className="body-text note",
            ),
        ),
    ])


# ── app ───────────────────────────────────────────────────────────────────────

app = Dash(
    __name__,
    assets_folder=str(Path(__file__).parent / "assets"),
    suppress_callback_exceptions=True,
    title="League Objective Analytics",
)

app.layout = html.Div(className="app-root", children=[
    html.Header(className="site-header", children=[
        html.Div(className="header-inner", children=[
            html.Span("League Objective Analytics", className="site-logo"),
            dcc.Tabs(
                id="tabs", value="home", className="nav-tabs",
                children=[
                    dcc.Tab(label="Home",       value="home",       className="nav-tab", selected_className="nav-tab--on", style=_TAB, selected_style=_TAB_ACTIVE),
                    dcc.Tab(label="Methods",    value="methods",    className="nav-tab", selected_className="nav-tab--on", style=_TAB, selected_style=_TAB_ACTIVE),
                    dcc.Tab(label="Analysis",   value="analysis",   className="nav-tab", selected_className="nav-tab--on", style=_TAB, selected_style=_TAB_ACTIVE),
                    dcc.Tab(label="Conclusion", value="conclusion", className="nav-tab", selected_className="nav-tab--on", style=_TAB, selected_style=_TAB_ACTIVE),
                ],
            ),
        ]),
    ]),
    html.Div(id="content", className="content-root"),
])


@callback(Output("content", "children"), Input("tabs", "value"))
def render(tab: str) -> html.Div:
    if tab == "home":
        return page_home()
    if tab == "methods":
        return page_methods()
    if tab == "analysis":
        return page_analysis()
    return page_conclusion()


@callback(
    Output("game-analysis-output", "children"),
    Input("analyze-btn", "n_clicks"),
    State("riot-id-input", "value"),
    State("region-select", "value"),
    prevent_initial_call=True,
)
def fetch_player_game(n_clicks: int, riot_id: str | None, platform: str | None) -> html.Div:
    """Look up a player's latest ranked game and show state-first analysis."""
    if not riot_id or not riot_id.strip():
        return html.Div("Enter a Riot ID in the form  GameName#TAG", className="ga-msg")

    riot_id = riot_id.strip()
    if "#" not in riot_id:
        return html.Div("Riot ID must include the tag, e.g.  Faker#KR1", className="ga-error")

    game_name, tag_line = riot_id.split("#", 1)
    platform = (platform or "na1").lower()

    client = _riot_client_for(platform)
    if client is None:
        return html.Div(
            "RIOT_API_KEY not found. Set it in your .env file and restart the app.",
            className="ga-error",
        )

    try:
        account = client.get_account_by_riot_id(game_name, tag_line)
    except Exception as exc:
        msg = str(exc)
        if "401" in msg:
            return html.Div(
                "Riot API returned 401 Unauthorized. Development keys expire every 24 hours. "
                "Generate a new key at developer.riotgames.com and update your .env file.",
                className="ga-error",
            )
        if "404" in msg:
            return html.Div(f"Player '{riot_id}' not found on {platform.upper()}.", className="ga-error")
        return html.Div(f"API error: {msg}", className="ga-error")

    puuid = account.get("puuid", "")
    if not puuid:
        return html.Div("Unexpected API response (no PUUID returned).", className="ga-error")

    try:
        match_ids = client.get_match_ids_by_puuid(puuid, count=20, queue=420)
    except Exception as exc:
        msg = str(exc)
        if "401" in msg:
            return html.Div(
                "Riot API returned 401 Unauthorized. The API key may have expired. "
                "Grab a fresh one at developer.riotgames.com.",
                className="ga-error",
            )
        return html.Div(f"API error fetching match list: {exc}", className="ga-error")

    if not match_ids:
        return html.Div(
            f"No recent ranked games found for '{riot_id}'.", className="ga-msg"
        )

    # Find the most recent match that's in our dataset
    found_id: str | None = None
    for mid in match_ids:
        if mid in _MATCH_IDS:
            found_id = mid
            break

    if found_id is None:
        checked = ", ".join(match_ids[:5])
        return html.Div(className="ga-msg-block", children=[
            html.P(f"No recent ranked games for '{riot_id}' found in the study dataset.",
                   className="ga-msg"),
            html.P(
                f"The most recent matches checked: {checked}…",
                className="ga-msg-sub",
            ),
            html.P(
                "The dataset covers NA, EUW, and KR matches from a specific patch window. "
                "Games outside that window won't appear here.",
                className="ga-msg-sub",
            ),
        ])

    return build_game_analysis(found_id)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
