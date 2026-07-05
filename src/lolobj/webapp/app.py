"""League Objective Setup Analytics — data loading and figure generation."""
from __future__ import annotations

import colorsys
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
    ("take",            "good", "Team secured the objective (contested or not -- see setup_profile)."),
    ("lost_with_trade", "mixed", "Team lost the objective but gained meaningful value elsewhere in the aftermath."),
    ("lost",            "bad", "Team lost the objective with nothing to show for it."),
]

PROFILE_DEFS = [
    ("free_setup",        "good",    "Team present, enemy absent, no recent allied deaths.",
                                     "Full positional control. Secure rate is highest here."),
    ("free_setup_deaths", "mixed",   "Team present, enemy absent, but allied deaths in the prior 60s.",
                                     "Uncontested positionally but down a player or two."),
    ("clean_contest",     "neutral", "Both teams near objective, team even or ahead in numbers.",
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

# Dark theme matching scripts/plots.R's theme_lol -- used only by the gold-slider
# figures (fig_gold_slider_heatmap / fig_gold_slider_stackedbar), which are meant
# to sit visually alongside the R-rendered charts, not the rest of this
# (light-themed) dashboard.
_DARK_BG   = "#222327"
_DARK_INK  = "#e2e8f0"
_DARK_SUB  = "#94a3b8"
_DARK_GRID = "#334155"
_DARK_PANEL = "#1e293b"
# theme_lol renders with base_family="sans", which resolves to Arial on this
# machine (via ragg) -- match that instead of the dashboard's Inter stack.
_DARK_FONT = "Arial, Helvetica, sans-serif"

# Kept visually in sync with scripts/plots.R's pal_outcome.
_OUTCOME_COLORS = {
    "take":            "#2e8b57",  # green
    "lost_with_trade": "#e8743b",  # orange
    "lost":            "#c0392b",  # red
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

# ── pre-compute rolling gold-window slider frames ────────────────────────────
# Fine-stepped slider (every _SLIDER_STEP pp) where each step re-aggregates a
# rolling window of rows centered on that gold_pct_T_60 value, rather than a
# hard bucket. This is what gives the slider a continuous "scrub" feel while
# keeping enough sample size per frame (window widens automatically if thin).
_SLIDER_STEP     = 2.0    # pp between adjacent slider steps
_SLIDER_HALF_WIN = 5.0    # rolling half-window width (pp) around each center
_SLIDER_MIN_N    = 150    # widen the window until at least this many rows
_SLIDER_MAX_HALF = 20.0

_PROFILES_PRESENT = [p for p in PROFILE_ORDER if p in df["setup_profile"].unique()]
_OUTCOMES_PRESENT = [o for o in OUTCOME_ORDER if o in df["outcome_label"].unique()]

_slider_lo = float(np.ceil(df_sf["gold_pct_T_60"].quantile(0.02) / _SLIDER_STEP) * _SLIDER_STEP)
_slider_hi = float(np.floor(df_sf["gold_pct_T_60"].quantile(0.98) / _SLIDER_STEP) * _SLIDER_STEP)
_slider_centers = np.arange(_slider_lo, _slider_hi + _SLIDER_STEP / 2, _SLIDER_STEP)


def _gold_window_pct(center: float) -> tuple[pd.DataFrame, pd.DataFrame, int, float]:
    """Row-normalized setup_profile x outcome_label % table for a rolling gold window."""
    half = _SLIDER_HALF_WIN
    while True:
        sub = df_sf[(df_sf["gold_pct_T_60"] >= center - half) & (df_sf["gold_pct_T_60"] <= center + half)]
        if len(sub) >= _SLIDER_MIN_N or half >= _SLIDER_MAX_HALF:
            break
        half += 2.5
    ct = pd.crosstab(sub["setup_profile"], sub["outcome_label"])
    ct = ct.reindex(index=_PROFILES_PRESENT, columns=_OUTCOMES_PRESENT, fill_value=0)
    row_n = ct.sum(axis=1)
    pct = (ct.div(row_n.replace(0, np.nan), axis=0) * 100).fillna(0)
    return pct, ct, len(sub), half


_GOLD_SLIDER_FRAMES: list[dict] = []
for _c in _slider_centers:
    _pct, _ct, _n, _half = _gold_window_pct(float(_c))
    _GOLD_SLIDER_FRAMES.append({"center": float(_c), "pct": _pct, "ct": _ct, "n": _n, "half": _half})

# Default slider position: the frame centered closest to 0% (even gold), not an
# extreme end -- users should land on a neutral view and drag from there.
_GOLD_SLIDER_DEFAULT_IDX = int(np.argmin([abs(f["center"]) for f in _GOLD_SLIDER_FRAMES]))

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


def _slider_steps(frames_data: list[dict]) -> list[dict]:
    """go.layout.slider steps that animate to each precomputed gold-window frame."""
    return [
        dict(
            method="animate",
            label="0" if f["center"] == 0 else f"{f['center']:+.0f}",
            args=[[f"_frame_{i}"], dict(mode="immediate",
                  frame=dict(duration=0, redraw=True), transition=dict(duration=150))],
        )
        for i, f in enumerate(frames_data)
    ]


def _frame_annotation(f: dict, color: str = "#64748b", family: str = _FONT, y: float = 1.09) -> dict:
    return dict(
        text=f"n = {f['n']:,} objective windows  ·  ±{f['half']:.1f}pp rolling window",
        xref="paper", yref="paper", x=0.5, y=y, xanchor="center", yanchor="bottom",
        showarrow=False, font=dict(size=11, color=color, family=family),
    )


def _heatmap_cell_alpha(pct: pd.DataFrame) -> pd.DataFrame:
    """Same opacity rule as scripts/plots.R's static heatmap: 0.06 for empty cells,
    else rescaled 0.13-1.0 across this frame's own positive values.

    Returns a DataFrame aligned to pct's own index/columns (not a flat array),
    so lookups are done by (profile, outcome) label rather than by position --
    a flat-array + positional-counter approach previously got the fill order
    out of sync with the (outcome-major) loop order after the axes were
    swapped, silently applying each cell's opacity to the wrong cell.
    """
    flat = pct.values.flatten()
    positive = flat[flat > 0]
    if len(positive) == 0:
        alpha_flat = np.full_like(flat, 0.06)
    else:
        lo, hi = float(positive.min()), float(positive.max())
        span = max(hi - lo, 1.0)
        alpha_flat = np.where(flat == 0, 0.06, 0.13 + 0.87 * (flat - lo) / span)
    return pd.DataFrame(alpha_flat.reshape(pct.shape), index=pct.index, columns=pct.columns)


def _dim_color(hex_color: str, factor: float) -> str:
    """Scale a hex color's HSV value by `factor` (0-1) to encode magnitude.

    Used instead of Plotly's shape opacity for the gold-slider heatmap: opacity
    alpha-blends against the actual page background, which (with the previous
    saturated-navy BG) washed warm hues toward muddy brown. Dimming in HSV value instead of
    alpha-blending avoids that, but for hues in the orange/amber range there's
    a second, unavoidable effect on top: dark + desaturated red-orange is what
    people perceptually call "brown" -- it's not a rendering artifact, it's a
    real category boundary. So for that hue range specifically, nudge hue
    toward true orange/amber and boost saturation as it dims, and keep a
    modest brightness floor so the darkest cells don't get so dark that hue
    stops reading at all. Green/red aren't in that hue range and are
    untouched (pure value-scaling looked fine for those already).
    """
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if 0.02 <= h <= 0.14:
        dim = 1 - factor
        h = min(h + 0.035 * dim, 0.11)
        s = min(1.0, s + 0.30 * dim)
        v = v * (0.24 + 0.76 * factor)
    else:
        v = v * factor
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    return f"#{round(r2 * 255):02x}{round(g2 * 255):02x}{round(b2 * 255):02x}"


def _heatmap_layer(pct: pd.DataFrame, ct: pd.DataFrame):
    """Per-frame (shapes, xs, ys, texts, textcolors, hovertexts) for the gold-slider heatmap.

    x = setup profile, y = outcome. Built from layout shapes + a text/hover
    scatter trace rather than a native Heatmap trace, because a Heatmap can
    only carry one continuous colorscale -- it can't do "hue = outcome
    category, opacity = prevalence" the way scripts/plots.R's static p3
    heatmap does. Shapes give literal per-cell fillcolor (solid, HSV-dimmed by
    prevalence -- see _dim_color) matching that design's intent without the
    hue-muddying that plain shape opacity causes against a colored background.
    """
    alpha = _heatmap_cell_alpha(pct)
    shapes, xs, ys, texts, colors, hovers = [], [], [], [], [], []
    for i, out in enumerate(_OUTCOMES_PRESENT):
        for j, prof in enumerate(_PROFILES_PRESENT):
            p = float(pct.loc[prof, out])
            a = float(alpha.loc[prof, out])
            n = int(ct.loc[prof, out])
            shapes.append(dict(
                type="rect", xref="x", yref="y",
                x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
                fillcolor=_dim_color(_OUTCOME_COLORS.get(out, _NEUT), a), opacity=1.0,
                line=dict(color=_DARK_BG, width=1.5), layer="below",
            ))
            xs.append(j); ys.append(i)
            texts.append(f"{p:.0f}%")
            colors.append("white" if p > 0 else "#3d5470")
            hovers.append(f"<b>{prof.replace('_', ' ')}</b><br>{out.replace('_', ' ')}: "
                          f"{p:.1f}%  (n={n:,})<extra></extra>")
    return shapes, xs, ys, texts, colors, hovers


def fig_gold_slider_heatmap() -> go.Figure:
    """Setup profile x outcome heatmap, scrubbable across gold % advantage at T-60.

    Dark theme_lol style: categorical color by outcome, opacity by prevalence
    within profile -- matches scripts/plots.R's static p3 heatmap design.

    Left = team strongly ahead on gold, right = team strongly behind (matches the
    "left is strong yours, right is strong theirs" framing on the slider labels).
    """
    # Initial trace/active both point at the gold=0% frame -- the slider position
    # on first paint is only reliable when the initial `data` and
    # `sliders[0].active` refer to the same frame.
    frames_data = _GOLD_SLIDER_FRAMES
    default_idx = _GOLD_SLIDER_DEFAULT_IDX
    n_rows, n_cols = len(_OUTCOMES_PRESENT), len(_PROFILES_PRESENT)
    x_lbl = [p.replace("_", " ") for p in _PROFILES_PRESENT]
    y_lbl = [o.replace("_", " ") for o in _OUTCOMES_PRESENT]

    layers = [_heatmap_layer(f["pct"], f["ct"]) for f in frames_data]
    shapes0, xs0, ys0, texts0, colors0, hovers0 = layers[default_idx]

    fig = go.Figure(
        data=[go.Scatter(
            x=xs0, y=ys0, mode="text", text=texts0,
            textfont=dict(color=colors0, size=12, family=_DARK_FONT),
            hovertemplate=hovers0, showlegend=False,
        )],
        frames=[
            go.Frame(
                data=[go.Scatter(text=layer[3], textfont=dict(color=layer[4]),
                                  hovertemplate=layer[5])],
                name=f"_frame_{i}",
                layout=go.Layout(shapes=layer[0],
                                  annotations=[_frame_annotation(f, color=_DARK_SUB, family=_DARK_FONT, y=1.06)]),
            )
            for i, (f, layer) in enumerate(zip(frames_data, layers))
        ],
    )
    fig.update_layout(
        shapes=shapes0,
        paper_bgcolor=_DARK_BG, plot_bgcolor=_DARK_BG,
        font=dict(family=_DARK_FONT, size=12, color=_DARK_INK),
        hoverlabel=dict(bgcolor=_DARK_PANEL, font_size=13, font_family=_DARK_FONT, font_color=_DARK_INK),
        title=dict(
            text="<b>Outcome mix by setup profile, across gold advantage</b>",
            font=dict(size=21, color=_DARK_INK, family=_DARK_FONT),
            subtitle=dict(text="Color = outcome type  ·  Opacity = prevalence within profile",
                         font=dict(size=14, color=_DARK_SUB, family=_DARK_FONT)),
        ),
        height=560,
        xaxis=dict(tickvals=list(range(n_cols)), ticktext=x_lbl, tickangle=-20,
                   range=[-0.5, n_cols - 0.5], showgrid=False, zeroline=False, color=_DARK_SUB),
        yaxis=dict(tickvals=list(range(n_rows)), ticktext=y_lbl,
                   range=[-0.5, n_rows - 0.5], autorange="reversed",
                   showgrid=False, zeroline=False, color=_DARK_SUB),
        annotations=[_frame_annotation(frames_data[default_idx], color=_DARK_SUB, family=_DARK_FONT, y=1.06)],
        margin=dict(l=4, r=20, t=110, b=140),
        sliders=[dict(
            active=default_idx, x=0.02, len=0.96, y=-0.14, pad=dict(t=20, b=10),
            currentvalue=dict(prefix="Gold advantage at T-60 (you → them): ", suffix="%",
                              font=dict(size=13, color=_DARK_INK)),
            font=dict(size=10, color=_DARK_SUB),
            bgcolor=_DARK_PANEL, bordercolor=_DARK_GRID, activebgcolor=_ACCENT,
            steps=_slider_steps(frames_data),
        )],
    )
    return fig


def fig_gold_slider_stackedbar() -> go.Figure:
    """100%-stacked bar version of fig_gold_slider_heatmap: setup profile x outcome mix."""
    # Initial trace/active both point at the gold=0% frame — see
    # fig_gold_slider_heatmap for why active must match the initial data's frame.
    frames_data = _GOLD_SLIDER_FRAMES
    default_idx = _GOLD_SLIDER_DEFAULT_IDX
    x_lbl = [p.replace("_", " ") for p in _PROFILES_PRESENT]

    def _bars(pct: pd.DataFrame):
        return [
            go.Bar(
                name=o.replace("_", " "), x=x_lbl, y=pct[o].values,
                marker=dict(color=_OUTCOME_COLORS.get(o, _NEUT), line=dict(width=0)),
                hovertemplate="<b>%{x}</b><br>" + o.replace("_", " ") + ": %{y:.1f}%<extra></extra>",
            )
            for o in _OUTCOMES_PRESENT
        ]

    fig = go.Figure(
        data=_bars(frames_data[default_idx]["pct"]),
        frames=[
            go.Frame(
                data=_bars(f["pct"]), name=f"_frame_{i}",
                layout=go.Layout(annotations=[_frame_annotation(f, color=_DARK_SUB, family=_DARK_FONT, y=1.04)]),
            )
            for i, f in enumerate(frames_data)
        ],
    )
    fig.update_layout(
        barmode="stack",
        paper_bgcolor=_DARK_BG, plot_bgcolor=_DARK_BG,
        font=dict(family=_DARK_FONT, size=12, color=_DARK_INK),
        hoverlabel=dict(bgcolor=_DARK_PANEL, font_size=13, font_family=_DARK_FONT, font_color=_DARK_INK),
        title=dict(
            text="<b>Outcome mix by setup profile, across gold advantage (stacked)</b>",
            font=dict(size=21, color=_DARK_INK, family=_DARK_FONT),
            subtitle=dict(text="Color = outcome type  ·  Height = share within profile",
                         font=dict(size=14, color=_DARK_SUB, family=_DARK_FONT)),
        ),
        height=640,
        xaxis=dict(tickangle=-15, showgrid=False, zeroline=False, color=_DARK_SUB),
        yaxis=dict(showgrid=True, gridcolor=_DARK_GRID, zeroline=False, color=_DARK_SUB,
                   title=dict(text="% of rows", font=dict(color=_DARK_SUB)), range=[0, 100]),
        showlegend=True,
        legend=dict(orientation="h", y=-0.55, x=0.5, xanchor="center",
                    font=dict(size=10, color=_DARK_SUB), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=4, r=20, t=145, b=220),
        annotations=[_frame_annotation(frames_data[default_idx], color=_DARK_SUB, family=_DARK_FONT, y=1.04)],
        sliders=[dict(
            active=default_idx, x=0.02, len=0.96, y=-0.24, pad=dict(t=20, b=10),
            currentvalue=dict(prefix="Gold advantage at T-60 (you → them): ", suffix="%",
                              font=dict(size=13, color=_DARK_INK)),
            font=dict(size=10, color=_DARK_SUB),
            bgcolor=_DARK_PANEL, bordercolor=_DARK_GRID, activebgcolor=_ACCENT,
            steps=_slider_steps(frames_data),
        )],
    )
    return fig
