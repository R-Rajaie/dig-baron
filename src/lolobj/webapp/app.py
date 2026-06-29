"""League Objective Setup Analytics — data loading and figure generation."""
from __future__ import annotations

import math
import sys
from pathlib import Path

# ── path setup ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[3]
_SRC  = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from lolobj.config import PROCESSED_DIR
from lolobj.analysis.baseline_report import OUTCOME_ORDER, PROFILE_ORDER, assign_setup_profiles
from lolobj.analysis.state_first import make_state_first_frame, summarize_action_by_state

# ── data ────────────────────────────────────────────────────────────────────
print("[app] Loading data…")
df = pd.read_parquet(PROCESSED_DIR / "objective_windows.parquet")
# Voidgrubs: ordinals 4-6 are a second camp introduced in some patches.
# Only one camp of 3 grubs is meaningful for this analysis.
df = df[~((df["objective_type"] == "HORDE") & (df["objective_number"] > 3))].copy()
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

_outcome_sr = df.groupby("outcome_label")["secured"].mean().reset_index()
_outcome_sr.columns = ["outcome", "secure_rate_val"]
outcome_combo_df = outcome_df.merge(_outcome_sr, on="outcome", how="left")
outcome_combo_df["secure_pct"] = (outcome_combo_df["secure_rate_val"] * 100).round(1)

profile_df = (
    df.groupby("setup_profile")["secured"]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns={"mean": "secure_rate", "count": "n"})
)
profile_df["secure_pct"] = (profile_df["secure_rate"] * 100).round(1)
_pm = {p: i for i, p in enumerate(PROFILE_ORDER)}
profile_df = profile_df.sort_values("setup_profile", key=lambda s: s.map(_pm).fillna(99))
profile_df["prop_pct"] = (profile_df["n"] / _N * 100).round(1)

_FEATURE_LABELS = {
    "arrived_first":      "Arrived first (T-60)",
    "jungler_alive_T_30": "Jungler alive (T-30)",
    "support_alive_T_30": "Support alive (T-30)",
    "numbers_adv_T_30":   "Numbers advantage (T-30)",
    "numbers_down_T_30":  "Numbers down (T-30)",
    "team_grouped_T_30":  "Team grouped 3+ (T-30)",
}
feat_rows = []
for col, label in _FEATURE_LABELS.items():
    src = df_sf if col in df_sf.columns else df
    if col not in src.columns:
        continue
    r1 = round(src[src[col] == 1]["secured"].mean() * 100, 1)
    r0 = round(src[src[col] == 0]["secured"].mean() * 100, 1)
    feat_rows.append({"feature": label, "present": r1, "absent": r0, "diff": round(r1 - r0, 1)})
_FEAT_DISPLAY_ORDER = [
    "Numbers down (T-30)",
    "Numbers advantage (T-30)",
    "Team grouped 3+ (T-30)",
    "Support alive (T-30)",
    "Jungler alive (T-30)",
    "Arrived first (T-60)",
]
feat_df = pd.DataFrame(feat_rows)
_ord_map = {v: i for i, v in enumerate(_FEAT_DISPLAY_ORDER)}
feat_df["_sort"] = feat_df["feature"].map(_ord_map).fillna(99)
feat_df = feat_df.sort_values("_sort").drop("_sort", axis=1).reset_index(drop=True)

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

# SHAP beeswarm — embed the pre-generated notebook output directly
import base64 as _b64
_SHAP_PNG = _ROOT / "notebooks" / "output.png"
_SHAP_IMG_SRC = "data:image/png;base64," + _b64.b64encode(_SHAP_PNG.read_bytes()).decode()
print("[app] SHAP image loaded from notebooks/output.png")

# ── definitions ──────────────────────────────────────────────────────────────

OUTCOME_DEFS = [
    ("clean_take",               "good",    "Team secured. Won the fight or there was no real fight."),
    ("clean_give",               "neutral", "Enemy secured. Team did not contest, which may have been intentional."),
    ("good_trade",               "good",    "Team gave the objective but picked up meaningful value elsewhere."),
    ("coinflip",                 "neutral", "Both teams showed up with no clear advantage. Outcome was roughly 50/50."),
    ("bad_contest",              "bad",     "Team contested from a weak spot (short-handed, recent deaths) and lost."),
    ("won_fight_lost_objective", "bad",     "Team won the fight but still lost the objective."),
    ("lost_fight_got_objective", "mixed",   "Team lost the fight but secured the objective anyway."),
    ("throw_setup",              "bad",     "Team was in good shape but lost a player in the last 30s before the fight."),
    ("objective_steal",          "mixed",   "Team secured from a clearly outnumbered position, usually via smite."),
    ("no_meaningful_contest",    "neutral", "Neither team committed. Objective taken without a real fight."),
]

PROFILE_DEFS = [
    ("free_setup",        "good",    "Team present, enemy absent, no recent allied deaths.",
                                     "Full positional control. Secure rate is highest here."),
    ("free_setup_deaths", "mixed",   "Team present, enemy absent, but allied deaths in the prior 60s.",
                                     "Uncontested positionally but down a player or two."),
    ("clean_contest",     "neutral", "Both teams near objective, team not short-handed or behind on health.",
                                     "Even or favorable fight with both sides present."),
    ("disadvantaged",     "bad",     "Both teams present, but team had recent deaths or fewer alive champions.",
                                     "Contesting from behind."),
    ("gave_away",         "bad",     "Enemy present at objective, team absent.",
                                     "Team did not show up."),
    ("no_early_setup",    "neutral", "Neither team near objective at T-30.",
                                     "Neither side committed. One team eventually took it."),
]

# ── design tokens ────────────────────────────────────────────────────────────
_ACCENT = "#3d5af1"
_GOOD   = "#10b981"
_BAD    = "#ef4444"
_MIXED  = "#f59e0b"
_NEUT   = "#94a3b8"
_FONT   = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

_OUTCOME_COLORS = {
    "clean_take":               "#10b981",  # emerald
    "clean_give":               "#06b6d4",  # cyan
    "good_trade":               "#84cc16",  # lime
    "coinflip":                 "#a78bfa",  # violet
    "bad_contest":              "#ef4444",  # red
    "won_fight_lost_objective": "#f97316",  # orange
    "throw_setup":              "#e11d48",  # rose
    "lost_fight_got_objective": "#f59e0b",  # amber
    "objective_steal":          "#7c3aed",  # purple
    "no_meaningful_contest":    "#64748b",  # slate
}

_PROFILE_COLORS = {
    "free_setup":        "#059669",
    "free_setup_deaths": "#34d399",
    "clean_contest":     "#3d5af1",
    "disadvantaged":     "#ef4444",
    "gave_away":         "#94a3b8",
    "no_early_setup":    "#cbd5e1",
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


# ── pre-compute gold breakeven curves ────────────────────────────────────────
# One logistic regression per setup profile: P(secure) ~ gold_pct_T_60.
# gold_pct_T_60 = gold_diff / estimated_team_gold * 100 (computed in state_first).
# Breakeven = gold_pct where logit = 0  →  -intercept / coef.
_df_drag = df_sf[df_sf["objective_type"] == "DRAGON"].copy()
_g_lo = float(_df_drag["gold_pct_T_60"].quantile(0.02))
_g_hi = float(_df_drag["gold_pct_T_60"].quantile(0.98))
_gold_range = np.linspace(_g_lo, _g_hi, 400)

_BREAKEVEN_DATA: list[tuple] = []
for _prof in PROFILE_ORDER:
    _sub = _df_drag[_df_drag["setup_profile"] == _prof].dropna(
        subset=["gold_pct_T_60", "secured"]
    )
    if len(_sub) < 30:
        continue
    _lr_be = LogisticRegression(max_iter=1000)
    _lr_be.fit(_sub[["gold_pct_T_60"]].values, _sub["secured"].values.astype(int))
    _coef = float(_lr_be.coef_[0][0])
    _be   = (-float(_lr_be.intercept_[0]) / _coef) if _coef != 0 else None
    _y_pred = _lr_be.predict_proba(_gold_range.reshape(-1, 1))[:, 1]
    _BREAKEVEN_DATA.append((
        _prof,
        _PROFILE_COLORS.get(_prof, "#888"),
        _gold_range.copy(),
        _y_pred,
        _be,
        len(_sub),
    ))

# ── pre-compute setup → outcome Sankey edge counts ────────────────────────────
_sankey_counts = (
    df.groupby(["setup_profile", "outcome_label"])
    .size()
    .reset_index(name="n")
)
_sankey_counts = _sankey_counts[_sankey_counts["n"] >= 3].copy()

# ── per-objective pie data ─────────────────────────────────────────────────────
# Voidgrubs: use ordinal 1 (first grub) as representative for camp 1,
# ordinal 4 for camp 2 — avoids triple-counting the three kills per camp.
_is_elder   = df["monster_subtype"] == "ELDER_DRAGON"
_is_reg_drag = (df["objective_type"] == "DRAGON") & ~_is_elder

_OBJ_SLICES: list[tuple[str, pd.Series]] = [
    ("Dragon 1",     _is_reg_drag & (df["objective_number"] == 1)),
    ("Dragon 2",     _is_reg_drag & (df["objective_number"] == 2)),
    ("Dragon 3",     _is_reg_drag & (df["objective_number"] == 3)),
    ("Dragon 4",     _is_reg_drag & (df["objective_number"] == 4)),
    ("Dragon 5",     _is_reg_drag & (df["objective_number"] == 5)),
    ("Dragon 6",     _is_reg_drag & (df["objective_number"] == 6)),
    ("Dragon 7",     _is_reg_drag & (df["objective_number"] == 7)),
    ("Elder Dragon", _is_elder    & (df["objective_number"] == 1)),
    ("Baron 1",      (df["objective_type"] == "BARON_NASHOR") & (df["objective_number"] == 1)),
    ("Baron 2",      (df["objective_type"] == "BARON_NASHOR") & (df["objective_number"] == 2)),
    ("Herald",       (df["objective_type"] == "RIFTHERALD")   & (df["objective_number"] == 1)),
    ("Voidgrubs",    (df["objective_type"] == "HORDE")        & (df["objective_number"] == 1)),
]
_PIE_NCOLS = 6


# ── figures ───────────────────────────────────────────────────────────────────

def _make_pie_grid(
    value_fn,       # callable(sub_df) → (names, values, colors)
    title: str,
    legend_labels: list[str],
    legend_colors: list[str],
) -> go.Figure:
    nrows = math.ceil(len(_OBJ_SLICES) / _PIE_NCOLS)
    specs = [[{"type": "pie"}] * _PIE_NCOLS for _ in range(nrows)]
    titles = [lbl for lbl, _ in _OBJ_SLICES] + [""] * (nrows * _PIE_NCOLS - len(_OBJ_SLICES))
    fig = make_subplots(rows=nrows, cols=_PIE_NCOLS, specs=specs, subplot_titles=titles)

    for i, (lbl, mask) in enumerate(_OBJ_SLICES):
        row, col = i // _PIE_NCOLS + 1, i % _PIE_NCOLS + 1
        sub = df[mask]
        names, values, colors = value_fn(sub)
        fig.add_trace(go.Pie(
            labels=names,
            values=values,
            marker=dict(colors=colors, line=dict(color="white", width=0.3)),
            hovertemplate="%{label}<br>%{percent}<br>n=%{value:,}<extra></extra>",
            textinfo="percent",
            textfont=dict(size=9, family=_FONT),
            showlegend=False,
        ), row=row, col=col)

    # Dummy invisible scatter traces to build a clean shared legend
    for leg_label, leg_color in zip(legend_labels, legend_colors):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=leg_color, symbol="square"),
            name=leg_label, showlegend=True,
        ))

    fig.update_layout(
        **{k: v for k, v in _BASE.items() if k != "margin"},
        title=dict(text=title, font=dict(size=14, color="#1e293b")),
        height=160 * nrows + 80,
        margin=dict(l=4, r=160, t=60, b=8),
        legend=dict(
            orientation="v", x=1.01, xanchor="left", y=0.5, yanchor="middle",
            font=dict(size=11, family=_FONT),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def fig_objective_profile_pies() -> go.Figure:
    ordered = [p for p in PROFILE_ORDER if p in df["setup_profile"].unique()]

    def _values(sub: pd.DataFrame):
        vc = sub["setup_profile"].value_counts()
        names  = [p.replace("_", " ") for p in ordered if p in vc.index]
        values = [int(vc[p]) for p in ordered if p in vc.index]
        colors = [_PROFILE_COLORS.get(p, "#888") for p in ordered if p in vc.index]
        return names, values, colors

    leg_labels = [p.replace("_", " ") for p in ordered]
    leg_colors = [_PROFILE_COLORS.get(p, "#888") for p in ordered]
    return _make_pie_grid(_values, "Setup profile distribution by objective", leg_labels, leg_colors)


def fig_objective_outcome_pies() -> go.Figure:
    ordered = [o for o in OUTCOME_ORDER if o in df["outcome_label"].unique()]

    def _values(sub: pd.DataFrame):
        vc = sub["outcome_label"].value_counts()
        names  = [o.replace("_", " ") for o in ordered if o in vc.index]
        values = [int(vc[o]) for o in ordered if o in vc.index]
        colors = [_OUTCOME_COLORS.get(o, _NEUT) for o in ordered if o in vc.index]
        return names, values, colors

    leg_labels = [o.replace("_", " ") for o in ordered]
    leg_colors = [_OUTCOME_COLORS.get(o, _NEUT) for o in ordered]
    return _make_pie_grid(_values, "Outcome label distribution by objective", leg_labels, leg_colors)


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


def fig_profile_combo() -> go.Figure:
    p = profile_df.copy()
    _cs = [[0, _BAD], [0.5, _MIXED], [1, _GOOD]]
    fig = go.Figure(go.Treemap(
        labels=p["setup_profile"].str.replace("_", " "),
        parents=[""] * len(p),
        values=p["prop_pct"],
        customdata=np.column_stack([p["secure_pct"], p["n"]]),
        marker=dict(
            colors=p["secure_pct"],
            colorscale=_cs,
            cmin=0, cmax=100,
            colorbar=dict(
                title=dict(text="Secure %", font=dict(size=11)),
                tickvals=[0, 50, 100],
                thickness=12, len=0.6, x=1.02,
            ),
            line=dict(width=2, color="white"),
        ),
        texttemplate="<b>%{label}</b><br>%{value:.1f}% of rows<br>%{customdata[0]:.1f}% secure",
        hovertemplate="<b>%{label}</b><br>Proportion: %{value:.1f}%<br>Secure rate: %{customdata[0]:.1f}%<br>n=%{customdata[1]:,}<extra></extra>",
        textfont=dict(family=_FONT, size=12),
    ))
    return _fig(fig,
        title=dict(text="Setup profile: prevalence (size) vs secure rate (color)", font=dict(size=14, color="#1e293b")),
        height=380,
        margin=dict(l=4, r=60, t=44, b=8),
    )


def fig_outcome_combo() -> go.Figure:
    oc = outcome_combo_df.copy()
    _sentiment = {"good": 100, "mixed": 60, "neutral": 45, "bad": 5}
    _tag_map    = {name: _sentiment[tag] for name, tag, _ in OUTCOME_DEFS}
    oc["score"] = oc["outcome"].map(_tag_map).fillna(45)
    _cs = [[0, _BAD], [0.5, _MIXED], [1, _GOOD]]
    fig = go.Figure(go.Treemap(
        labels=oc["outcome"].str.replace("_", " "),
        parents=[""] * len(oc),
        values=oc["pct"],
        customdata=np.column_stack([oc["score"], oc["n"]]),
        marker=dict(
            colors=oc["score"],
            colorscale=_cs,
            cmin=0, cmax=100,
            colorbar=dict(
                title=dict(text="Outcome quality", font=dict(size=11)),
                tickvals=[5, 45, 100],
                ticktext=["bad", "neutral", "good"],
                thickness=12, len=0.6, x=1.02,
            ),
            line=dict(width=2, color="white"),
        ),
        texttemplate="<b>%{label}</b><br>%{value:.1f}% of rows",
        hovertemplate="<b>%{label}</b><br>Proportion: %{value:.1f}%<br>n=%{customdata[1]:,}<extra></extra>",
        textfont=dict(family=_FONT, size=12),
    ))
    return _fig(fig,
        title=dict(text="Outcome label: prevalence (size) vs outcome quality (color)", font=dict(size=14, color="#1e293b")),
        height=380,
        margin=dict(l=4, r=60, t=44, b=8),
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


def fig_gold_breakeven() -> go.Figure:
    """Sigmoid curves: P(secure) vs gold_pct_T_60, one line per setup profile."""
    fig = go.Figure()
    for prof, color, gold_range, y_pred, breakeven, n in _BREAKEVEN_DATA:
        label = prof.replace("_", " ")
        fig.add_trace(go.Scatter(
            x=gold_range, y=y_pred,
            name=label,
            mode="lines",
            line=dict(color=color, width=2.5),
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Gold %: %{x:+.1f}%<br>"
                "P(secure): %{y:.1%}<extra></extra>"
            ),
        ))
        if breakeven is not None:
            fig.add_trace(go.Scatter(
                x=[breakeven], y=[0.5],
                mode="markers+text",
                marker=dict(color=color, size=11, symbol="diamond",
                            line=dict(color="white", width=1.5)),
                text=[f"{breakeven:+.1f}%"],
                textposition="top center",
                textfont=dict(size=9, color=color, family=_FONT),
                showlegend=False,
                hovertemplate=(
                    f"<b>{label}</b> breakeven<br>"
                    f"Gold %: {breakeven:+.1f}%<extra></extra>"
                ),
            ))
    fig.add_hline(
        y=0.5, line_dash="dash", line_color="#94a3b8", line_width=1.2,
        annotation_text="50%", annotation_position="right",
        annotation_font=dict(size=11, color="#64748b", family=_FONT),
    )
    fig.add_vline(x=0, line_dash="dot", line_color="#e2e8f0", line_width=1)
    return _fig(fig,
        title=dict(text="Gold % advantage vs. secure rate by setup profile",
                   font=dict(size=14, color="#1e293b")),
        height=430,
        xaxis=dict(
            showgrid=True, gridcolor="#f1f5f9", zeroline=False,
            title="Gold % advantage at T-60 (team − enemy, % of est. team gold)",
            ticksuffix="%",
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#f1f5f9", zeroline=False,
            title="P(secure objective)",
            tickformat=".0%",
            range=[0, 1],
        ),
        showlegend=True,
        legend=dict(
            orientation="h", y=-0.18, x=0.5, xanchor="center",
            font=dict(size=12), bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=4, r=60, t=44, b=80),
    )


def fig_setup_outcome_sankey() -> go.Figure:
    """Sankey: setup profile → outcome label."""
    p_setups   = [p for p in PROFILE_ORDER if p in _sankey_counts["setup_profile"].unique()]
    p_outcomes = [o for o in OUTCOME_ORDER  if o in _sankey_counts["outcome_label"].unique()]
    all_labels = p_setups + p_outcomes
    node_idx   = {l: i for i, l in enumerate(all_labels)}

    def _rgba(hex_c: str, a: float = 0.28) -> str:
        h = hex_c.lstrip("#")
        return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"

    node_colors = (
        [_rgba(_PROFILE_COLORS.get(p, _NEUT), 1.0) for p in p_setups]
        + [_rgba(_OUTCOME_COLORS.get(o, _NEUT), 1.0) for o in p_outcomes]
    )
    src, tgt, val, lc = [], [], [], []
    for _, row in _sankey_counts.iterrows():
        sp, ol, n = row["setup_profile"], row["outcome_label"], int(row["n"])
        if sp in node_idx and ol in node_idx:
            src.append(node_idx[sp])
            tgt.append(node_idx[ol])
            val.append(n)
            lc.append(_rgba(_PROFILE_COLORS.get(sp, _NEUT)))

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=14, thickness=30,
            label=[l.replace("_", " ") for l in all_labels],
            color=node_colors,
            line=dict(color="rgba(0,0,0,0)", width=0),
            hovertemplate="%{label}<br>Total: %{value:,}<extra></extra>",
        ),
        link=dict(
            source=src, target=tgt, value=val, color=lc,
            hovertemplate="%{source.label} → %{target.label}<br>Count: %{value:,}<extra></extra>",
        ),
    ))
    fig.update_layout(
        paper_bgcolor="white",
        font=dict(family=_FONT, size=11, color="#1e293b"),
        title=dict(
            text="Setup profile → outcome label",
            font=dict(size=14, color="#1e293b"),
        ),
        height=500,
        margin=dict(l=10, r=10, t=44, b=10),
        hoverlabel=dict(bgcolor="white", font_size=13, font_family=_FONT),
    )
    return fig
