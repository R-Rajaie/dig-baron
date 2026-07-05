"""HTML (Plotly) replicas of scripts/plots.R's static charts in
data/processed/remake/figures/, styled to match theme_lol as closely as
possible. These are static (single-state) charts -- no slider/frames, unlike
the gold-slider dashboard figures in lolobj.webapp.app.

Usage: see exports/export_remake_html.py
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lolobj.config import PROCESSED_DIR, PROJECT_ROOT, RAW_DIR
from lolobj.viz.theme_dark import (
    ALIVE_COLORS, BASE_LAYOUT, BG, FONT, GOLD_COLORS, GRID, INK, OUTCOME_COLORS,
    OUTCOME_ORDER, PANEL, PROFILE_COLORS, PROFILE_ORDER, SUB, dim_color, style_fig,
)


# ── data ──────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Reconstructs scripts/plots.R's `dat` derivation directly from the parquet."""
    need = ["secured", "fight_result", "objective_result", "good_trade", "net_value",
            "objective_time_ms", "team_nearby_T_30", "enemy_nearby_T_30", "team_deaths_60s",
            "team_alive_T_30", "enemy_alive_T_30", "team_alive_T_60", "enemy_alive_T_60",
            "arrived_first", "support_alive_T_30", "jungler_alive_T_30", "gold_diff_T_60"]
    dat = pd.read_parquet(PROCESSED_DIR / "objective_windows.parquet", columns=need)

    tp = dat["team_nearby_T_30"] >= 1
    ep = dat["enemy_nearby_T_30"] >= 1
    deaths = dat["team_deaths_60s"] > 0
    nearby_ge = dat["team_nearby_T_30"] >= dat["enemy_nearby_T_30"]
    dat["setup_profile"] = np.select(
        [tp & ~ep & ~deaths, tp & ~ep & deaths, ~tp & ep, ~tp & ~ep, tp & ep & nearby_ge & ~deaths],
        ["free_setup", "free_setup_deaths", "gave_away", "no_early_setup", "clean_contest"],
        default="disadvantaged",
    )

    minutes = dat["objective_time_ms"] / 60000
    dat["gold_pct"] = dat["gold_diff_T_60"] / (2500 + 850 * minutes) * 100

    alive_diff = dat["team_alive_T_60"] - dat["enemy_alive_T_60"]
    dat["alive_state"] = np.select(
        [alive_diff < 0, alive_diff > 0], ["numbers_down", "numbers_up"], default="even_alive"
    )
    dat["gold_state"] = pd.qcut(
        dat["gold_pct"], 5, labels=["big_behind", "behind", "even", "ahead", "big_ahead"]
    )

    dat["outcome"] = np.select(
        [dat["objective_result"] == "secured",
         (dat["objective_result"] == "lost") & (dat["good_trade"] == 1),
         (dat["objective_result"] == "lost") & (dat["good_trade"] == 0)],
        OUTCOME_ORDER,
        default=None,
    )
    return dat


def _lbl(s: str) -> str:
    return s.replace("_", " ")


# ── 01: setup profile prevalence + secure rate (combo) ───────────────────────

def fig_01_setup_profiles_combo(dat: pd.DataFrame) -> go.Figure:
    s1 = dat.groupby("setup_profile", observed=True).agg(
        n=("secured", "size"), secure=("secured", "mean")
    ).reset_index()
    s1["secure"] *= 100
    s1["prevalence"] = 100 * s1["n"] / s1["n"].sum()
    s1 = s1.sort_values("prevalence")
    order = s1["setup_profile"].tolist()
    colors = [PROFILE_COLORS[p] for p in order]
    ylbl = [_lbl(p) for p in order]

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12)
    fig.add_trace(go.Bar(
        x=s1["prevalence"], y=ylbl, orientation="h", marker=dict(color=colors),
        text=[f"{v:.0f}%" for v in s1["prevalence"]], textposition="outside",
        textfont=dict(color=INK), showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=s1["secure"], y=ylbl, orientation="h", marker=dict(color=colors),
        text=[f"{v:.0f}%" for v in s1["secure"]], textposition="outside",
        textfont=dict(color=INK), showlegend=False,
    ), row=1, col=2)

    fig.update_xaxes(range=[0, 32], title_text=None, row=1, col=1)
    fig.update_xaxes(range=[0, 108], title_text="Secure rate", row=1, col=2)
    fig.add_annotation(text="<b>How common each setup is</b><br>"
                             "<span style='color:%s;font-size:11px'>Share of all objective contests</span>" % SUB,
                        xref="paper", yref="paper", x=0.0, y=1.12, xanchor="left", showarrow=False,
                        font=dict(size=15, color=INK, family=FONT), align="left")
    fig.add_annotation(text="<b>Secure rate by setup profile</b><br>"
                             "<span style='color:%s;font-size:11px'>How often each setup ends "
                             "with the team securing the objective</span>" % SUB,
                        xref="paper", yref="paper", x=0.62, y=1.12, xanchor="left", showarrow=False,
                        font=dict(size=15, color=INK, family=FONT), align="left")

    fig.update_layout(**BASE_LAYOUT, height=460, showlegend=False,
                       margin=dict(l=4, r=20, t=90, b=40))
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, color=SUB, ticksuffix="%")
    fig.update_yaxes(showgrid=False, zeroline=False, color=SUB)
    return fig


# ── 02: outcome label prevalence ──────────────────────────────────────────────

def fig_02_outcome_labels(dat: pd.DataFrame) -> go.Figure:
    s2 = dat.dropna(subset=["outcome"]).groupby("outcome", observed=True).size().reindex(OUTCOME_ORDER)
    prevalence = 100 * s2 / s2.sum()
    order = list(reversed(OUTCOME_ORDER))
    prevalence = prevalence.reindex(order)
    colors = [OUTCOME_COLORS[o] for o in order]

    fig = go.Figure(go.Bar(
        x=prevalence.values, y=[_lbl(o) for o in order], orientation="h",
        marker=dict(color=colors),
        text=[f"{v:.1f}%" for v in prevalence.values], textposition="outside",
        textfont=dict(color=INK, size=13), showlegend=False,
    ))
    style_fig(fig, "How often each outcome occurs", "Share of all objective contests",
              height=420, xaxis=dict(ticksuffix="%", range=[0, max(prevalence.values) * 1.18]),
              margin=dict(l=4, r=40, t=100, b=40))
    return fig


# ── 04: deaths before objective -> secure rate ───────────────────────────────

def _wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z ** 2 / n
    c = p + z ** 2 / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return 100 * (c - h) / d, 100 * (c + h) / d


def fig_04_deaths_to_secure(dat: pd.DataFrame) -> go.Figure:
    d = dat.copy()
    d["deaths"] = d["team_deaths_60s"].clip(upper=4)
    s4 = d.groupby("deaths").agg(n=("secured", "size"), k=("secured", "sum")).reset_index()
    s4["secure"] = 100 * s4["k"] / s4["n"]
    labels = ["0", "1", "2", "3", "4+"]
    death_colors = ["#22c55e", "#84cc16", "#facc15", "#f97316", "#ef4444"]

    fig = go.Figure(go.Bar(
        x=labels, y=s4["secure"], marker=dict(color=death_colors),
        text=[f"{v:.1f}%" for v in s4["secure"]], textposition="outside",
        textfont=dict(color=INK, size=13, family=FONT), showlegend=False,
    ))
    style_fig(fig, "Every death before the objective bleeds the secure",
              "Secure rate vs. allied deaths in the 60s window before the objective is taken",
              height=440,
              xaxis=dict(title="Allied deaths in 60s before objective"),
              yaxis=dict(title="Secure rate", ticksuffix="%", range=[0, 100]),
              margin=dict(l=4, r=20, t=100, b=60))
    return fig


# ── 05: feature present vs absent (dumbbell) ──────────────────────────────────

def fig_05_feature_dumbbell(dat: pd.DataFrame) -> go.Figure:
    flags = {
        "Numbers down (T-30)":       dat["team_nearby_T_30"] < dat["enemy_nearby_T_30"],
        "Numbers advantage (T-30)":  dat["team_nearby_T_30"] > dat["enemy_nearby_T_30"],
        "Team grouped 3+ (T-30)":    dat["team_nearby_T_30"] >= 3,
        "Support dead (T-30)":       dat["support_alive_T_30"] == 0,
        "Jungler dead (T-30)":       dat["jungler_alive_T_30"] == 0,
        "Arrived first (T-60)":      dat["arrived_first"] == 1,
    }
    rows = []
    for feat, flag in flags.items():
        absent = 100 * dat.loc[~flag, "secured"].mean()
        present = 100 * dat.loc[flag, "secured"].mean()
        rows.append({"feature": feat, "absent": absent, "present": present, "gap": present - absent})
    s5 = pd.DataFrame(rows).sort_values("gap", key=lambda s: s.abs())
    feats = s5["feature"].tolist()

    fig = go.Figure()
    for x in (25, 50, 75):
        fig.add_shape(type="line", x0=x, x1=x, y0=-0.5, y1=len(feats) - 0.5,
                      line=dict(color=GRID, width=1), layer="below")
    for i, row in enumerate(s5.itertuples()):
        fig.add_shape(type="line", x0=row.absent, x1=row.present, y0=i, y1=i,
                      line=dict(color="#334155", width=6), layer="below")
    fig.add_trace(go.Scatter(x=s5["absent"], y=feats, mode="markers+text",
                              marker=dict(color="#64748b", size=15),
                              text=[f"{v:.0f}%" for v in s5["absent"]], textposition="top center",
                              textfont=dict(color=SUB, size=11), name="absent",
                              hovertemplate="absent: %{x:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=s5["present"], y=feats, mode="markers+text",
                              marker=dict(color="#60a5fa", size=15),
                              text=[f"{v:.0f}%" for v in s5["present"]], textposition="top center",
                              textfont=dict(color="#60a5fa", size=12), name="present",
                              hovertemplate="present: %{x:.1f}%<extra></extra>"))
    for i, row in enumerate(s5.itertuples()):
        fig.add_annotation(x=(row.absent + row.present) / 2, y=i, text=f"{row.gap:+.0f} pp",
                            showarrow=False, yshift=-18, font=dict(size=11, color=INK), opacity=0.65)

    style_fig(fig, "Variables that strongly influence secure rates",
              "Secure rate when each game-state signal is absent vs present",
              height=460,
              xaxis=dict(title="Secure rate", ticksuffix="%", range=[0, 100], showgrid=False),
              yaxis=dict(showgrid=False),
              showlegend=True,
              legend=dict(orientation="h", y=1.1, x=1, xanchor="right", font=dict(color=SUB, size=11),
                          bgcolor="rgba(0,0,0,0)"),
              margin=dict(l=4, r=20, t=100, b=40))
    return fig


# ── 03: setup profile x outcome heatmap (static; color=outcome, opacity=share) ─

def fig_03_setup_outcome_heatmap(dat: pd.DataFrame) -> go.Figure:
    d = dat.dropna(subset=["outcome"])
    pe = d.groupby("setup_profile", observed=True)["net_value"].apply(lambda s: (s > 0).mean())
    best_first = pe.sort_values(ascending=False).index.tolist()
    worst_first = list(reversed(best_first))
    rest = [p for p in worst_first if p != "no_early_setup"]
    setup_levels = ["no_early_setup"] + rest  # first = bottom of chart, matching R's default factor order

    ct = pd.crosstab(d["setup_profile"], d["outcome"])
    ct = ct.reindex(index=setup_levels, columns=OUTCOME_ORDER, fill_value=0)
    row_n = ct.sum(axis=1)
    pct = (ct.div(row_n.replace(0, np.nan), axis=0) * 100).fillna(0)

    flat = pct.values.flatten()
    positive = flat[flat > 0]
    if len(positive):
        lo, hi = float(positive.min()), float(positive.max())
        span = max(hi - lo, 1.0)
        alpha_flat = np.where(flat == 0, 0.06, 0.13 + 0.87 * (flat - lo) / span)
    else:
        alpha_flat = np.full_like(flat, 0.06)
    alpha = pd.DataFrame(alpha_flat.reshape(pct.shape), index=pct.index, columns=pct.columns)

    shapes, xs, ys, texts, colors, hovers = [], [], [], [], [], []
    for i, prof in enumerate(setup_levels):
        for j, out in enumerate(OUTCOME_ORDER):
            p = float(pct.loc[prof, out])
            a = float(alpha.loc[prof, out])
            n = int(ct.loc[prof, out])
            shapes.append(dict(type="rect", xref="x", yref="y",
                                x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
                                fillcolor=dim_color(OUTCOME_COLORS.get(out, SUB), a), opacity=1.0,
                                line=dict(color=BG, width=1.5), layer="below"))
            xs.append(j); ys.append(i)
            texts.append(f"{p:.0f}%")
            colors.append("white" if p > 0 else "#3d5470")
            hovers.append(f"<b>{_lbl(prof)}</b><br>{_lbl(out)}: {p:.1f}%  (n={n:,})<extra></extra>")

    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="text", text=texts,
        textfont=dict(color=colors, size=13, family=FONT),
        hovertemplate=hovers, showlegend=False,
    ))
    style_fig(fig, "Where each setup ends up",
              "Color = outcome type  ·  Opacity = prevalence within setup",
              height=460, shapes=shapes,
              xaxis=dict(tickvals=list(range(len(OUTCOME_ORDER))),
                         ticktext=[_lbl(o) for o in OUTCOME_ORDER],
                         range=[-0.5, len(OUTCOME_ORDER) - 0.5], showgrid=False),
              yaxis=dict(tickvals=list(range(len(setup_levels))),
                         ticktext=[_lbl(p) for p in setup_levels],
                         range=[-0.5, len(setup_levels) - 0.5], showgrid=False),
              margin=dict(l=4, r=20, t=100, b=40))
    return fig


# ── 06: gold advantage vs secure rate by setup profile (lines + breakeven) ────

def fig_06_gold_by_setup_lines(dat: pd.DataFrame) -> go.Figure:
    from statsmodels.nonparametric.smoothers_lowess import lowess

    d = dat[(dat["gold_pct"] >= -45) & (dat["gold_pct"] <= 38)].copy()
    bins = np.arange(-45, 40.1, 2.5)
    d["bin"] = pd.cut(d["gold_pct"], bins=bins)
    s6 = d.groupby(["setup_profile", "bin"], observed=True).agg(
        x=("gold_pct", "mean"), secure=("secured", "mean"), n=("secured", "size")
    ).reset_index()
    s6["secure"] *= 100
    s6 = s6[s6["n"] >= 120].dropna(subset=["x"])

    fig = go.Figure()
    fig.add_hline(y=50, line=dict(color="#475569", dash="dot"))
    fig.add_vline(x=0, line=dict(color=GRID))

    for prof in PROFILE_ORDER:
        sub = s6[s6["setup_profile"] == prof].sort_values("x")
        if len(sub) < 5:
            continue
        color = PROFILE_COLORS[prof]
        fig.add_trace(go.Scatter(
            x=sub["x"], y=sub["secure"], mode="markers",
            marker=dict(size=np.clip(np.sqrt(sub["n"]) / 3, 3, 14), color=color, opacity=0.22),
            showlegend=False, hoverinfo="skip",
        ))
        smoothed = lowess(sub["secure"], sub["x"], frac=0.5, return_sorted=True)
        sx, sy = smoothed[:, 0], smoothed[:, 1]
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines", line=dict(color=color, width=3),
                                  name=_lbl(prof)))
        diff = sy - 50
        sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
        if len(sign_change):
            i = sign_change[0]
            x0, x1, y0, y1 = sx[i], sx[i + 1], diff[i], diff[i + 1]
            be = x0 - y0 * (x1 - x0) / (y1 - y0)
            fig.add_trace(go.Scatter(
                x=[be], y=[50], mode="markers+text",
                marker=dict(color=color, size=10, line=dict(color=BG, width=1)),
                text=[f"{be:+.0f}%"], textposition="top center",
                textfont=dict(color=color, size=12, family=FONT),
                showlegend=False, hovertemplate=f"{_lbl(prof)} breakeven: {be:+.1f}%<extra></extra>",
            ))

    style_fig(fig, "A good setup is worth its weight in gold",
              "Secure rate vs gold advantage at T-60, by setup profile  ·  ● = 50% breakeven crossing",
              height=580,
              xaxis=dict(title="Gold advantage at T-60 (% of est. team gold)", ticksuffix="%"),
              yaxis=dict(title="Secure rate", ticksuffix="%", range=[0, 100]),
              showlegend=True,
              legend=dict(title=dict(text="Setup profile", font=dict(color="#cbd5e1")),
                          font=dict(color=SUB), bgcolor="rgba(0,0,0,0)"),
              margin=dict(l=4, r=160, t=100, b=60))
    return fig


# ── 07: state-conditioned effect of arriving first (faceted horizontal bars) ─

def _gradient_color(v: float, vmin: float, vmax: float,
                     stops=("#1e3a8a", "#2563eb", "#60a5fa", "#bfdbfe")) -> str:
    t = 0.0 if vmax == vmin else (v - vmin) / (vmax - vmin)
    t = min(max(t, 0.0), 1.0)
    n = len(stops) - 1
    idx = min(int(t * n), n - 1)
    local_t = t * n - idx
    c0, c1 = stops[idx], stops[idx + 1]
    r0, g0, b0 = int(c0[1:3], 16), int(c0[3:5], 16), int(c0[5:7], 16)
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r = round(r0 + (r1 - r0) * local_t)
    g = round(g0 + (g1 - g0) * local_t)
    b = round(b0 + (b1 - b0) * local_t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _effect_table(dat: pd.DataFrame, action: pd.Series) -> pd.DataFrame:
    d = dat.assign(act=action)
    rows = []
    for (gold_state, alive_state), grp in d.groupby(["gold_state", "alive_state"], observed=True):
        n1 = int(grp["act"].sum())
        n0 = int((~grp["act"]).sum())
        if n1 < 40 or n0 < 40:
            continue
        p1 = grp.loc[grp["act"], "secured"].mean()
        p0 = grp.loc[~grp["act"], "secured"].mean()
        se = 100 * np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
        delta = 100 * (p1 - p0)
        rows.append({"gold_state": gold_state, "alive_state": alive_state,
                      "delta": delta, "lo": delta - 1.96 * se, "hi": delta + 1.96 * se})
    return pd.DataFrame(rows)


def fig_07_effect_arrived_first(dat: pd.DataFrame) -> go.Figure:
    e = _effect_table(dat, dat["arrived_first"] == 1)
    alive_order = ["numbers_down", "even_alive", "numbers_up"]
    alive_labels = {"numbers_down": "Numbers down (T-60)", "even_alive": "Even numbers (T-60)",
                     "numbers_up": "Numbers up (T-60)"}
    gold_order = ["big_behind", "behind", "even", "ahead", "big_ahead"]
    dmin, dmax = e["delta"].min(), e["delta"].max()

    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.06,
                         subplot_titles=[alive_labels[a] for a in alive_order])
    for i, a in enumerate(alive_order):
        sub = e[e["alive_state"] == a].set_index("gold_state").reindex(gold_order).dropna(how="all")
        colors = [_gradient_color(v, dmin, dmax) for v in sub["delta"]]
        fig.add_trace(go.Bar(
            x=sub["delta"], y=[_lbl(g) for g in sub.index], orientation="h",
            marker=dict(color=colors),
            error_x=dict(type="data", symmetric=False,
                         array=(sub["hi"] - sub["delta"]).values,
                         arrayminus=(sub["delta"] - sub["lo"]).values,
                         color="#3a4250", thickness=1.2, width=3),
            text=[f"{v:+.0f}" for v in sub["delta"]], textposition="outside",
            textfont=dict(color=INK, size=11), showlegend=False,
        ), row=1, col=i + 1)
        fig.update_xaxes(row=1, col=i + 1, range=[0, e["hi"].max() * 1.25])

    for ann in fig.layout.annotations:
        ann.font = dict(color=INK, size=13, family=FONT)

    style_fig(fig, "How arriving first impacts secure rate by gamestate",
              "Lift in secure rate from arriving first, within each gold x numbers state",
              height=480,
              margin=dict(l=90, r=20, t=110, b=60))
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, color=SUB,
                      title_text="Δ secure rate (pp), action vs no action", row=1, col=2)
    fig.update_yaxes(showgrid=False, zeroline=False, color=SUB)
    return fig


# ── 17: setup profile map states (background map + synthetic positions) ──────

def _load_map_image_uri(darken: float = 0.60) -> str:
    from PIL import Image
    img = Image.open(RAW_DIR / "rift map.png").convert("RGB")
    arr = np.asarray(img).astype(np.float32) * darken
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def fig_17_mapstates() -> go.Figure:
    """Replicates scripts/mapstates_r.R: 6 map panels (annotation_raster-equivalent
    background image + synthetic player positions) plus a legend strip."""
    map_uri = _load_map_image_uri()

    def gg(ny: float) -> float:
        return 1 - ny

    B_JG_DRAG, B_ADC_DRAG = (0.648, gg(0.690)), (0.742, gg(0.770))
    B_SUP_DRAG, B_MID_DRAG = (0.708, gg(0.740)), (0.606, gg(0.674))
    B_TOP_DRAG = (0.633, gg(0.728))
    R_JG_DRAG, R_ADC_DRAG = (0.712, gg(0.606)), (0.682, gg(0.580))
    R_SUP_DRAG, R_MID_DRAG = (0.660, gg(0.624)), (0.610, gg(0.568))
    R_TOP_DRAG = (0.748, gg(0.656))
    B_JG_LANE, B_ADC_LANE = (0.338, gg(0.595)), (0.800, gg(0.848))
    B_SUP_LANE, B_MID_LANE = (0.745, gg(0.900)), (0.460, gg(0.622))
    B_TOP_LANE = (0.128, gg(0.298))
    R_JG_LANE, R_ADC_LANE = (0.742, gg(0.512)), (0.858, gg(0.872))
    R_SUP_LANE, R_MID_LANE = (0.828, gg(0.912)), (0.538, gg(0.448))
    R_TOP_LANE = (0.175, gg(0.192))
    DRAGON_X, DRAGON_Y = 0.666, gg(0.703)

    panels = [
        dict(title="Free Setup", sub="Team present  ·  Enemy absent  ·  No recent allied deaths",
             accent="#10b981",
             ba=[B_JG_DRAG, B_ADC_DRAG, B_SUP_DRAG, B_MID_DRAG, B_TOP_DRAG], bd=[],
             ra=[R_JG_LANE, R_ADC_LANE, R_SUP_LANE, R_MID_LANE, R_TOP_LANE], rd=[]),
        dict(title="Free Setup w/Deaths", sub="Team present  ·  Enemy absent  ·  Allied deaths in prior 60s",
             accent="#f59e0b",
             ba=[B_JG_DRAG, B_ADC_DRAG, B_SUP_DRAG], bd=[B_MID_LANE, B_TOP_LANE],
             ra=[R_JG_LANE, R_ADC_LANE, R_SUP_LANE, R_MID_LANE, R_TOP_LANE], rd=[]),
        dict(title="Clean Contest", sub="Both teams present  ·  Team even or ahead in numbers",
             accent="#a78bfa",
             ba=[B_JG_DRAG, B_ADC_DRAG, B_SUP_DRAG, B_MID_DRAG, B_TOP_DRAG], bd=[],
             ra=[R_JG_DRAG, R_ADC_DRAG, R_SUP_DRAG, R_MID_DRAG, R_TOP_DRAG], rd=[]),
        dict(title="Disadvantaged", sub="3 blue vs 4 red near objective  ·  Locally outnumbered",
             accent="#ef4444",
             ba=[B_JG_DRAG, B_ADC_DRAG, B_SUP_DRAG, B_TOP_LANE], bd=[B_MID_DRAG],
             ra=[R_JG_DRAG, R_ADC_DRAG, R_SUP_DRAG, R_MID_DRAG, R_TOP_LANE], rd=[]),
        dict(title="Gave Away", sub="Enemy present at objective  ·  Team did not show up",
             accent="#64748b",
             ba=[B_JG_LANE, B_ADC_LANE, B_SUP_LANE, B_MID_LANE, B_TOP_LANE], bd=[],
             ra=[R_JG_DRAG, R_ADC_DRAG, R_SUP_DRAG, R_MID_DRAG, R_TOP_DRAG], rd=[]),
        dict(title="No Early Setup", sub="Neither team near objective at T-30  ·  Neither side committed",
             accent="#94a3b8",
             ba=[B_JG_LANE, B_ADC_LANE, B_SUP_LANE, B_MID_LANE, B_TOP_LANE], bd=[],
             ra=[R_JG_LANE, R_ADC_LANE, R_SUP_LANE, R_MID_LANE, R_TOP_LANE], rd=[]),
    ]

    col_x = [(0.005, 0.320), (0.340, 0.655), (0.675, 0.990)]
    row_y = [(0.590, 0.945), (0.115, 0.470)]  # [top row, bottom row]
    legend_dom = (0.30, 0.70), (0.0, 0.075)

    theta = np.linspace(0, 2 * np.pi, 100)
    radius = 2500 / 14870

    fig = go.Figure()
    images: list[dict] = []
    shapes: list[dict] = []
    annotations: list[dict] = []

    def axis_keys(n: int) -> tuple[str, str]:
        return ("x" if n == 1 else f"x{n}"), ("y" if n == 1 else f"y{n}")

    def layout_axis_names(n: int) -> tuple[str, str]:
        return ("xaxis" if n == 1 else f"xaxis{n}"), ("yaxis" if n == 1 else f"yaxis{n}")

    for idx, panel in enumerate(panels):
        row = 0 if idx < 3 else 1
        col = idx % 3
        ax_num = idx + 1
        xkey, ykey = axis_keys(ax_num)
        xaxis_name, yaxis_name = layout_axis_names(ax_num)
        xdom, ydom = col_x[col], row_y[row]

        fig.update_layout(**{
            xaxis_name: dict(domain=list(xdom), range=[0, 1], visible=False),
            yaxis_name: dict(domain=list(ydom), range=[0, 1], visible=False,
                              scaleanchor=xkey, scaleratio=1),
        })

        images.append(dict(source=map_uri, xref=xkey, yref=ykey, x=0, y=1, sizex=1, sizey=1,
                            xanchor="left", yanchor="top", sizing="stretch", layer="below"))

        fig.add_trace(go.Scatter(
            x=DRAGON_X + radius * np.cos(theta), y=DRAGON_Y + radius * np.sin(theta),
            mode="lines", line=dict(color="#fbbf24", width=1.4, dash="dot"), opacity=0.5,
            xaxis=xkey, yaxis=ykey, showlegend=False, hoverinfo="skip",
        ))

        def _scatter(pts, symbol, color, line_color, size, stroke, opacity=1.0):
            if not pts:
                return
            xs, ys = zip(*pts)
            fig.add_trace(go.Scatter(
                x=list(xs), y=list(ys), mode="markers",
                marker=dict(symbol=symbol, size=size, color=color, line=dict(color=line_color, width=stroke)),
                opacity=opacity, xaxis=xkey, yaxis=ykey, showlegend=False, hoverinfo="skip",
            ))

        _scatter(panel["bd"], "x", "#93c5fd", "#93c5fd", 13, 3, opacity=0.55)
        _scatter(panel["rd"], "x", "#fca5a5", "#fca5a5", 13, 3, opacity=0.55)
        _scatter(panel["ra"], "circle", "#ef4444", "#7f1d1d", 17, 2)
        _scatter(panel["ba"], "circle", "#3b82f6", "#1e3a8a", 17, 2)

        fig.add_trace(go.Scatter(x=[DRAGON_X], y=[DRAGON_Y], mode="text", text=["★"],
                                  textfont=dict(size=22, color="#fbbf24"),
                                  xaxis=xkey, yaxis=ykey, showlegend=False, hoverinfo="skip"))

        px0, px1 = xdom
        py0, py1 = ydom
        shapes.append(dict(type="rect", xref="paper", yref="paper", x0=px0, x1=px1, y0=py0, y1=py1,
                            line=dict(color=panel["accent"], width=2.5), fillcolor=BG, layer="below"))
        annotations.append(dict(x=(px0 + px1) / 2, y=py1 + 0.032, xref="paper", yref="paper",
                                 text=f"<b>{panel['title']}</b>", showarrow=False, xanchor="center",
                                 font=dict(size=17, color=panel["accent"], family=FONT)))
        annotations.append(dict(x=(px0 + px1) / 2, y=py1 + 0.010, xref="paper", yref="paper",
                                 text=panel["sub"], showarrow=False, xanchor="center",
                                 font=dict(size=11, color=SUB, family=FONT)))

    # ── legend strip (its own small axis pair, x in [-0.5, 11], y in [-0.2, 1.2]) ─
    leg_num = len(panels) + 1
    lxkey, lykey = axis_keys(leg_num)
    lxaxis_name, lyaxis_name = layout_axis_names(leg_num)
    fig.update_layout(**{
        lxaxis_name: dict(domain=list(legend_dom[0]), range=[-0.5, 11], visible=False),
        lyaxis_name: dict(domain=list(legend_dom[1]), range=[-0.35, 1.2], visible=False),
    })
    leg_items = [(0.5, "Blue (alive)", "circle", "#3b82f6", "#1e3a8a"),
                 (3.5, "Red (alive)", "circle", "#ef4444", "#7f1d1d"),
                 (6.5, "Dead (X)", "x", "#93c5fd", "#93c5fd")]
    for lx, label, symbol, color, line_color in leg_items:
        fig.add_trace(go.Scatter(x=[lx], y=[0.5], mode="markers",
                                  marker=dict(symbol=symbol, size=13, color=color,
                                              line=dict(color=line_color, width=2)),
                                  xaxis=lxkey, yaxis=lykey, showlegend=False, hoverinfo="skip"))
        annotations.append(dict(x=lx, y=0.0, xref=lxkey, yref=lykey, text=label, showarrow=False,
                                 yanchor="top",
                                 font=dict(size=12, color=INK, family=FONT)))
    fig.add_trace(go.Scatter(x=[9.0], y=[0.5], mode="text", text=["★"],
                              textfont=dict(size=17, color="#fbbf24"),
                              xaxis=lxkey, yaxis=lykey, showlegend=False, hoverinfo="skip"))
    annotations.append(dict(x=9.0, y=0.0, xref=lxkey, yref=lykey, text="Dragon", showarrow=False,
                             yanchor="top",
                             font=dict(size=12, color=INK, family=FONT)))

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, images=images, shapes=shapes, annotations=annotations,
        font=dict(family=FONT, size=12, color=INK),
        title=dict(text="<b>Setup Profile Map States</b>  ·  T-30 snapshot  ·  synthetic positions",
                   font=dict(size=19, color=INK, family=FONT), x=0.01, xanchor="left"),
        autosize=True, margin=dict(l=10, r=10, t=60, b=10),
        hoverlabel=dict(bgcolor=PANEL, font_size=13, font_family=FONT, font_color=INK),
    )
    return fig
