"""League Objective Setup Analytics — Dash web application.

Run from project root:
    python src/lolobj/webapp/app.py

Requires:
    pip install dash plotly
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── path setup ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[3]
_SRC  = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dash import Dash, Input, Output, callback, dcc, html
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from lolobj.config import PROCESSED_DIR
from lolobj.analysis.baseline_report import OUTCOME_ORDER, PROFILE_ORDER, assign_setup_profiles
from lolobj.analysis.state_first import make_state_first_frame, summarize_action_by_state

# ── data ────────────────────────────────────────────────────────────────────
print("[app] Loading data…")
df = pd.read_parquet(PROCESSED_DIR / "objective_windows.parquet")
df["setup_profile"] = assign_setup_profiles(df)
df_sf = make_state_first_frame(df)
_N = len(df)
print(f"[app] {_N:,} rows, {df['match_id'].nunique():,} matches")

RANKS = [r for r in ["low", "mid", "high", "elite"] if r in df["rank_bucket"].unique()]

# ── pre-compute ──────────────────────────────────────────────────────────────
STATS = {
    "matches":      df["match_id"].nunique(),
    "objectives":   df.groupby(["match_id", "objective_instance"]).ngroups,
    "secure_rate":  round(df["secured"].mean() * 100, 1),
    "obj_types":    df["objective_type"].nunique(),
}

# Outcome distribution
_vc = df["outcome_label"].value_counts()
outcome_df = pd.DataFrame({
    "outcome": OUTCOME_ORDER,
    "n": [int(_vc.get(o, 0)) for o in OUTCOME_ORDER],
})
outcome_df = outcome_df[outcome_df["n"] > 0].copy()
outcome_df["pct"] = (outcome_df["n"] / _N * 100).round(1)

# Setup profile secure rates
profile_df = (
    df.groupby("setup_profile")["secured"]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns={"mean": "secure_rate", "count": "n"})
)
profile_df["secure_pct"] = (profile_df["secure_rate"] * 100).round(1)
_profile_order_map = {p: i for i, p in enumerate(PROFILE_ORDER)}
profile_df = profile_df.sort_values(
    "setup_profile", key=lambda s: s.map(_profile_order_map).fillna(99)
)

# Binary feature rates
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

# Deaths vs secure rate
deaths_df = (
    df[df["team_deaths_60s"] <= 4]
    .groupby("team_deaths_60s")["secured"]
    .agg(["mean", "count"])
    .reset_index()
)
deaths_df["secure_pct"] = (deaths_df["mean"] * 100).round(1)
deaths_df.columns = ["deaths", "mean", "n", "secure_pct"]

# State-conditioned effects
arrived_tbl = summarize_action_by_state(df_sf, "arrived_first",    min_n=50)
numbers_tbl  = summarize_action_by_state(df_sf, "numbers_adv_T_30", min_n=50)

# Rank outcome breakdown
rank_rows = []
for rank in RANKS:
    sub = df[df["rank_bucket"] == rank]
    for outcome in OUTCOME_ORDER:
        cnt = (sub["outcome_label"] == outcome).sum()
        if cnt > 0:
            rank_rows.append({"rank": rank, "outcome": outcome,
                               "pct": round(cnt / len(sub) * 100, 2)})
rank_outcome_df = pd.DataFrame(rank_rows)

# Model comparison
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

_Xn = df_sf[_SN].fillna(0).astype(float)
_Xc = pd.get_dummies(df_sf[_SC], drop_first=False).astype(float) if _SC else pd.DataFrame(index=df_sf.index)
_Xa = df_sf[_SA].fillna(0).astype(float)
_Xs = pd.concat([_Xn, _Xc], axis=1)
_Xb = pd.concat([_Xn, _Xc, _Xa], axis=1)
_y  = df_sf["secured"].astype(int).values
_g  = df_sf["match_id"].values

_gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
_tr, _te = next(_gss.split(_Xs, _y, _g))
_ytr, _yte = _y[_tr], _y[_te]

model_rows = []
for _name, _X, _nf in [
    ("State (T-60)",  _Xs, len(_Xs.columns)),
    ("Setup (T-30)",  _Xa, len(_Xa.columns)),
    ("State + Setup", _Xb, len(_Xb.columns)),
]:
    _sc = StandardScaler()
    _lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    _lr.fit(_sc.fit_transform(_X.iloc[_tr].fillna(0).values), _ytr)
    _p  = _lr.predict_proba(_sc.transform(_X.iloc[_te].fillna(0).values))[:, 1]
    model_rows.append({
        "Model":    _name,
        "AUC":      round(roc_auc_score(_yte, _p), 4),
        "Log-loss": round(log_loss(_yte, _p), 4),
        "Brier":    round(brier_score_loss(_yte, _p), 4),
        "Features": _nf,
    })
model_df = pd.DataFrame(model_rows)
print(f"[app] Done. State+Setup AUC = {model_df[model_df['Model']=='State + Setup']['AUC'].values[0]:.3f}")

# ── design tokens ────────────────────────────────────────────────────────────
_ACCENT   = "#3d5af1"
_GOOD     = "#10b981"
_BAD      = "#ef4444"
_NEUTRAL  = "#94a3b8"
_FONT     = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

_OUTCOME_COLORS = {
    "clean_take":               _GOOD,
    "clean_give":               _GOOD,
    "good_trade":               _GOOD,
    "coinflip":                 _NEUTRAL,
    "bad_contest":              _BAD,
    "won_fight_lost_objective": _BAD,
    "lost_fight_got_objective": "#f59e0b",
    "throw_setup":              _BAD,
    "objective_steal":          "#f59e0b",
    "no_meaningful_contest":    _NEUTRAL,
}

_BASE = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
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
    colors = [_OUTCOME_COLORS.get(o, _NEUTRAL) for o in outcome_df["outcome"]]
    fig = go.Figure(go.Bar(
        x=outcome_df["pct"],
        y=outcome_df["outcome"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=outcome_df["pct"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
        textfont=dict(size=11, color="#475569"),
        hovertemplate="<b>%{y}</b><br>%{x:.1f}% of objective rows<extra></extra>",
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
        x=p["secure_pct"],
        y=p["setup_profile"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=p["secure_pct"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
        textfont=dict(size=11, color="#475569"),
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
        name="Absent / 0",
        x=feat_df["feature"],
        y=feat_df["absent"],
        marker=dict(color=_NEUTRAL, line=dict(width=0)),
        hovertemplate="%{x}<br>Absent: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Present / 1",
        x=feat_df["feature"],
        y=feat_df["present"],
        marker=dict(color=_ACCENT, line=dict(width=0)),
        hovertemplate="%{x}<br>Present: %{y:.1f}%<extra></extra>",
    ))
    return _fig(fig,
        barmode="group",
        title=dict(text="Secure rate: feature present vs absent", font=dict(size=14, color="#1e293b")),
        height=300,
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", range=[0, 105], title="secure rate (%)"),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center",
                    font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
        showlegend=True,
        margin=dict(l=4, r=20, t=44, b=40),
    )


def fig_deaths_secure() -> go.Figure:
    colors = [_GOOD if i == 0 else _BAD for i in range(len(deaths_df))]
    fig = go.Figure(go.Bar(
        x=deaths_df["deaths"].astype(str),
        y=deaths_df["secure_pct"],
        marker=dict(color=colors, line=dict(width=0)),
        text=deaths_df["secure_pct"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
        textfont=dict(size=11, color="#475569"),
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
        x=plot["diff_pp"],
        y=plot["state_bucket"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=plot["diff_pp"].apply(lambda x: f"{x:+.1f}pp"),
        textposition="outside",
        textfont=dict(size=10, color="#475569"),
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
        title=dict(text=f"State-conditioned effect — {action_label}", font=dict(size=14, color="#1e293b")),
        height=max(340, len(plot) * 30 + 80),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=True, zerolinecolor="#cbd5e1",
                   zerolinewidth=1.5, title="Δ secure rate (pp)  action=1 minus action=0"),
        margin=dict(l=4, r=80, t=44, b=30),
    )


def fig_model_auc() -> go.Figure:
    colors = [_NEUTRAL, _NEUTRAL, _ACCENT]
    fig = go.Figure(go.Bar(
        x=model_df["Model"],
        y=model_df["AUC"],
        marker=dict(color=colors, line=dict(width=0)),
        text=model_df["AUC"].apply(lambda x: f"{x:.3f}"),
        textposition="outside",
        textfont=dict(size=12, color="#475569"),
        hovertemplate="<b>%{x}</b><br>AUC: %{y:.4f}<extra></extra>",
        width=0.45,
    ))
    return _fig(fig,
        title=dict(text="Model AUC — state vs setup vs combined", font=dict(size=14, color="#1e293b")),
        height=280,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", range=[0.75, 0.95], title="AUC (test set)"),
        margin=dict(l=4, r=20, t=44, b=20),
    )


def fig_rank_outcomes() -> go.Figure:
    if rank_outcome_df.empty:
        return go.Figure()
    _rank_colors = {"low": "#94a3b8", "mid": "#60a5fa", "high": "#3d5af1", "elite": "#1e1b4b"}
    fig = px.bar(
        rank_outcome_df,
        x="outcome", y="pct", color="rank",
        barmode="group",
        color_discrete_map=_rank_colors,
        labels={"pct": "% of rows", "outcome": "", "rank": "Rank"},
    )
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


# ── shared helpers ────────────────────────────────────────────────────────────

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
    "padding": "0 2px",
    "height": "52px",
    "lineHeight": "52px",
    "border": "none",
    "borderTop": "none",
    "borderLeft": "none",
    "borderRight": "none",
    "borderBottom": "2px solid transparent",
    "borderRadius": "0",
    "background": "transparent",
    "color": "#64748b",
    "fontSize": "14px",
    "fontWeight": "400",
    "fontFamily": _FONT,
    "marginRight": "28px",
}
_TAB_ACTIVE = {
    **_TAB,
    "color": "#1e293b",
    "borderBottom": "2px solid #3d5af1",
    "fontWeight": "500",
}

# ── page layouts ─────────────────────────────────────────────────────────────

def page_home() -> html.Div:
    return html.Div(className="page", children=[

        html.Div(className="hero", children=[
            html.H1("League Objective Setup Analytics", className="hero-title"),
            html.P(
                "An analytical framework for understanding what makes an objective setup good or bad "
                "in League of Legends — and how those factors differ by rank, objective type, and game state.",
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
                "What conditions make contesting, starting, trading, or giving an objective "
                "a good decision? The project focuses on Dragon 1 and Dragon 2 as the baseline "
                "and can extend to Baron, Herald, Voidgrubs, Soul, and Elder.",
                className="body-text",
            ),
        ),

        section("Objective outcomes", "Each objective row is classified into one of ten categories. "
                "The model goes beyond binary 'did team secure?' to capture richer decision-making signal.",
            graph(fig_outcomes()),
        ),

        section("What this project does", "",
            html.Ul(className="bullet-list", children=[
                html.Li("Creates a clean objective-window table at the team × objective level"),
                html.Li("Labels outcomes beyond binary secured / not-secured"),
                html.Li("Separates pre-objective state (T-60) from realized setup actions (T-30)"),
                html.Li("Compares teams in similar situations to isolate the value of setup choices"),
                html.Li("Explains findings in terms a player can act on — not just model outputs"),
            ]),
        ),
    ])


def page_methods() -> html.Div:
    return html.Div(className="page", children=[
        html.H1("Methods", className="page-title"),

        section("Unit of analysis", "Each row is one team's perspective on one objective instance.",
            html.Div(className="mono-block",
                     children=html.Code("match_id  +  objective_instance  +  team_id  →  one row")),
            html.P(
                "For each objective, two rows are created — one per team side. Features describe only "
                "that team's state relative to the opponent. Objectives and teams are never mixed within a row.",
                className="body-text",
            ),
        ),

        section("Objective time windows", "Features are computed within four windows relative to T=0 (objective take time).",
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
                "Temporal ordering separates what was already true from what teams chose to do.",
            html.Div(className="three-col", children=[
                html.Div(className="method-card", children=[
                    html.Div("01", className="method-num"),
                    html.H3("State", className="method-title"),
                    html.P(
                        "Information at T-60 or earlier. Gold difference, alive counts, "
                        "recent deaths. Describes the situation before teams commit to rotating.",
                        className="method-body",
                    ),
                ]),
                html.Div(className="method-card", children=[
                    html.Div("02", className="method-num"),
                    html.H3("Action", className="method-title"),
                    html.P(
                        "Observed behaviour between T-60 and T-30. Who arrived first, "
                        "how many champions appeared near the objective, whether setup "
                        "was contested or conceded.",
                        className="method-body",
                    ),
                ]),
                html.Div(className="method-card", children=[
                    html.Div("03", className="method-num"),
                    html.H3("Outcome", className="method-title"),
                    html.P(
                        "Objective secured, outcome label (clean take, bad contest, good trade…), "
                        "net objective value. Never used as a model feature.",
                        className="method-body",
                    ),
                ]),
            ]),
            note(
                "Comparing setup actions within similar T-60 states is more informative than global "
                "correlations — it holds the starting situation roughly constant. All findings are "
                "still observational, not causal."
            ),
        ),

        section("Feature categories", "",
            html.Div(className="feat-grid", children=[
                html.Div(className="feat-item", children=[html.Strong("Numbers advantage"), " — alive champion counts at T-30 and T-60"]),
                html.Div(className="feat-item", children=[html.Strong("Tempo & arrival"), " — who arrived first, jungler/support distance to objective"]),
                html.Div(className="feat-item", children=[html.Strong("Combat power"), " — team gold difference, level differences"]),
                html.Div(className="feat-item", children=[html.Strong("Lane priority proxy"), " — laner position before objective, objective-side presence"]),
                html.Div(className="feat-item", children=[html.Strong("Vision"), " — ward placements and kills near objective pre-spawn"]),
                html.Div(className="feat-item", children=[html.Strong("Prior context"), " — previous objective holder, objective number, rank, patch"]),
            ]),
        ),

        section("Modeling approach",
                "Models are used as analytical lenses — not prediction engines.",
            html.Div(className="step-list", children=[
                html.Div(className="step", children=[html.Span("1", className="step-num"), "Descriptive statistics and setup profiles"]),
                html.Div(className="step", children=[html.Span("2", className="step-num"), "State-only logistic regression (T-60 features only)"]),
                html.Div(className="step", children=[html.Span("3", className="step-num"), "Setup-only logistic regression (T-30 action features only)"]),
                html.Div(className="step", children=[html.Span("4", className="step-num"), "Combined model (state + setup) — measures incremental value of actions"]),
                html.Div(className="step", children=[html.Span("5", className="step-num"), "State-conditioned within-bucket comparisons for each action"]),
            ]),
            note(
                "All train/test splits are grouped by match_id — all rows from one match are "
                "entirely in train or test, never split across both. This prevents data leakage."
            ),
        ),

        section("Data and limitations", "",
            html.Ul(className="bullet-list", children=[
                html.Li("Data source: Riot Games Match-V5 API (match summaries and timeline events)"),
                html.Li("Exact player pathing, wave state, and item timing are not available — proxies are used"),
                html.Li("Role assignment is inferred from position data and may have errors for off-meta compositions"),
                html.Li("All findings are observational associations — teams that arrive first may do so because of champion advantages the model cannot see"),
                html.Li("Language used throughout: 'associated with', 'predictive of', 'proxy for' — never 'proves' or 'causes'"),
            ]),
        ),
    ])


def page_analysis() -> html.Div:
    _best_auc = model_df[model_df["Model"] == "State + Setup"]["AUC"].values[0]
    _state_auc = model_df[model_df["Model"] == "State (T-60)"]["AUC"].values[0]
    _auc_lift  = round((_best_auc - _state_auc) * 100, 1)

    return html.Div(className="page", children=[
        html.H1("Analysis & Results", className="page-title"),

        section("Setup profiles and secure rates",
                "Profiles classify each team's setup posture at T-30 based on presence near the objective and recent deaths.",
            graph(fig_profile_secure()),
            note("'Free setup' (only this team near objective) and 'gave away' (only enemy present) show the most extreme secure rates, validating the labeling logic."),
        ),

        section("Impact of key binary features",
                "Secure rate when a binary feature is present (1) vs absent (0).",
            graph(fig_feature_impact()),
        ),

        section("Deaths before objective",
                "Each allied death in the 60s before the objective substantially reduces the secure rate.",
            graph(fig_deaths_secure()),
            note("Even one death drops the rate by ~20 percentage points. Two deaths brings it below 25%. "
                 "Throw-setup patterns — losing a player just before the objective — may be a primary driver of objective losses."),
        ),

        section("State-first: value of arriving first",
                "Within each T-60 state bucket (gold state | alive state | death state), "
                "teams that arrived first are compared to teams that did not.",
            graph(fig_state_conditioned(arrived_tbl, "arrived first")),
            note("Blue bars: arriving first is associated with a higher secure rate in that state bucket. "
                 "Top 15 buckets by sample size shown. Effects are largest in 'even' and 'behind' states, "
                 "suggesting arrival timing matters most when the fight is less predetermined."),
        ),

        section("State-first: value of numbers advantage at T-30",
                "Same analysis for having more allies than enemies near the objective at T-30.",
            graph(fig_state_conditioned(numbers_tbl, "numbers advantage at T-30")),
        ),

        section("Model comparison",
                f"Three logistic regression models compared on a held-out test set (25% of matches, grouped by match_id). "
                f"State + Setup improves AUC by {_auc_lift:.1f} pp over State alone.",
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
            note("Setup actions add meaningful predictive value beyond the T-60 state. "
                 "This is an association, not proof of causality."),
        ),

        section("Outcome distribution by rank",
                "How outcomes differ across skill tiers.",
            graph(fig_rank_outcomes()),
        ),
    ])


def page_conclusion() -> html.Div:
    _best_auc   = model_df[model_df["Model"] == "State + Setup"]["AUC"].values[0]
    _state_auc  = model_df[model_df["Model"] == "State (T-60)"]["AUC"].values[0]

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
                            f"The combined model (State + Setup) achieves AUC {_best_auc:.3f} vs "
                            f"{_state_auc:.3f} for the state-only baseline. Teams in similar T-60 "
                            "situations achieve meaningfully different outcomes based on their setup choices.",
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
                            "Within state buckets where teams start in similar situations, arriving first "
                            "is associated with 40–65 percentage point higher secure rates. The effect is "
                            "consistent across gold states — behind, even, and ahead — suggesting that "
                            "arrival timing adds value regardless of the gold situation.",
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
                            "A single allied death in the 60 seconds before the objective drops the secure "
                            "rate by ~20 percentage points. Two deaths brings it below 25%. Throw-setup "
                            "patterns — losing a player just before the objective window opens — appear "
                            "to be a primary driver of objective losses.",
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
                            "Teams that appear at the objective with no enemy contestation secure at very "
                            "high rates. Teams that cede setup (only enemy present at T-30) secure at very "
                            "low rates. This validates the profile labeling system and confirms that T-30 "
                            "presence is a strong observable signal.",
                            className="finding-text",
                        ),
                    ]),
                ]),
            ]),
        ),

        section("Limitations", "",
            html.Ul(className="bullet-list", children=[
                html.Li(
                    "All findings are observational. Teams that arrive first may do so because "
                    "of champion kit advantages that also cause them to win fights — the arrival "
                    "itself may not be the causal mechanism."
                ),
                html.Li(
                    "Wave state, champion scaling, and communication are not captured by the Riot "
                    "API timeline. These are likely important unmeasured confounders."
                ),
                html.Li(
                    "The dataset covers a specific patch range and region mix. Findings may not "
                    "generalize across patches with major objective timing or respawn changes."
                ),
                html.Li(
                    "Role assignment is inferred from position data and may have errors for "
                    "off-meta or swapped lane compositions."
                ),
            ]),
        ),

        section("Next steps", "",
            html.Ul(className="bullet-list", children=[
                html.Li("Expand to Baron, Herald, Voidgrubs, Soul, and Elder objectives"),
                html.Li("Build rank-stratified models — test whether setup factors differ by tier"),
                html.Li("Add Random Forest + SHAP for non-linear interaction effects"),
                html.Li("Develop setup profile clustering (rule-based or model-assisted)"),
                html.Li("Build the per-game state-first analyzer as a player-facing diagnostic tool"),
            ]),
        ),

        section("Methodological note", "",
            html.P(
                "State-first analysis should be read as: among similar pre-objective situations, "
                "these setup choices were associated with better or worse secure rates. This is "
                "stronger than raw correlation because it respects temporal order and compares more "
                "similar situations — but it is still observational. It cannot prove that a team "
                "would have secured the objective if they had made a different choice.",
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

app.layout = html.Div(
    className="app-root",
    children=[
        # Sticky header
        html.Header(className="site-header", children=[
            html.Div(className="header-inner", children=[
                html.Span("League Objective Analytics", className="site-logo"),
                dcc.Tabs(
                    id="tabs",
                    value="home",
                    className="nav-tabs",
                    children=[
                        dcc.Tab(label="Home",             value="home",       className="nav-tab", selected_className="nav-tab--on", style=_TAB, selected_style=_TAB_ACTIVE),
                        dcc.Tab(label="Methods",          value="methods",    className="nav-tab", selected_className="nav-tab--on", style=_TAB, selected_style=_TAB_ACTIVE),
                        dcc.Tab(label="Analysis",         value="analysis",   className="nav-tab", selected_className="nav-tab--on", style=_TAB, selected_style=_TAB_ACTIVE),
                        dcc.Tab(label="Conclusion",       value="conclusion", className="nav-tab", selected_className="nav-tab--on", style=_TAB, selected_style=_TAB_ACTIVE),
                    ],
                ),
            ]),
        ]),
        # Content rendered by callback
        html.Div(id="content", className="content-root"),
    ],
)


@callback(Output("content", "children"), Input("tabs", "value"))
def render(tab: str) -> html.Div:
    if tab == "home":
        return page_home()
    if tab == "methods":
        return page_methods()
    if tab == "analysis":
        return page_analysis()
    return page_conclusion()


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
