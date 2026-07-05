"""Export every Plotly figure from the webapp as a separate PNG.

Usage:
    python exports/export_pngs.py
    python exports/export_pngs.py --out exports/charts
"""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC  = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

print("[png-export] Loading app (this loads data, trains models, computes SHAP)...")
import lolobj.webapp.app as A
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lolobj.viz.mapstates import fig_mapstates_grid
import plotly.graph_objects as go

_W, _H = 2400, 1350

# ── definition table helpers ──────────────────────────────────────────────────

_HEADER_FILL  = "#1e293b"
_HEADER_FONT  = "#ffffff"
_ROW_EVEN     = "#f8fafc"
_ROW_ODD      = "#ffffff"
_BORDER       = "#e2e8f0"

_QUALITY_COLOR = {
    "good":    "#d1fae5",
    "bad":     "#fee2e2",
    "neutral": "#f1f5f9",
    "mixed":   "#fef3c7",
}


def _table_fig(
    headers: list[str],
    rows: list[list[str]],
    col_widths: list[int],
    title: str,
    row_colors: list[str] | None = None,
    height: int = _H,
    fig_width: int | None = None,
) -> go.Figure:
    """Build a styled Plotly table figure."""
    transposed = list(map(list, zip(*rows)))
    n = len(rows)
    if row_colors is None:
        row_colors = [_ROW_EVEN if i % 2 == 0 else _ROW_ODD for i in range(n)]

    # Plotly wants per-column fill colors as lists of length n_rows
    cell_fills = [row_colors for _ in headers]

    fig = go.Figure(go.Table(
        columnwidth=col_widths,
        header=dict(
            values=[f"<b>{h}</b>" for h in headers],
            fill_color=_HEADER_FILL,
            font=dict(color=_HEADER_FONT, size=16, family="Inter, Arial, sans-serif"),
            align="left",
            height=44,
            line_color=_BORDER,
        ),
        cells=dict(
            values=transposed,
            fill_color=cell_fills,
            font=dict(color="#1e293b", size=14, family="Inter, Arial, sans-serif"),
            align="left",
            height=38,
            line_color=_BORDER,
        ),
    ))
    layout_kwargs: dict = dict(
        title=dict(text=title, font=dict(size=20, family="Inter, Arial, sans-serif",
                                          color="#1e293b"), x=0.5, xanchor="center"),
        margin=dict(l=24, r=24, t=56, b=24),
        paper_bgcolor="#ffffff",
        height=height,
    )
    if fig_width is not None:
        layout_kwargs["width"] = fig_width
    fig.update_layout(**layout_kwargs)
    return fig


def fig_setup_profile_defs() -> go.Figure:
    rows, colors = [], []
    for name, tag, condition, meaning in A.PROFILE_DEFS:
        rows.append([name.replace("_", " "), condition, meaning])
        colors.append(_QUALITY_COLOR.get(tag, _ROW_EVEN))
    # col_widths proportional to max content length per column: [17, 72, 53]
    return _table_fig(
        headers=["Profile", "Condition (T-30)", "Strategic meaning"],
        rows=rows,
        col_widths=[17, 72, 53],
        title="Setup Profile Definitions",
        row_colors=colors,
        height=max(320, 100 + len(rows) * 56),
        fig_width=1200,
    )


def fig_outcome_label_defs() -> go.Figure:
    rows, colors = [], []
    for name, tag, desc in A.OUTCOME_DEFS:
        rows.append([name.replace("_", " "), tag, desc])
        colors.append(_QUALITY_COLOR.get(tag, _ROW_EVEN))
    # col_widths proportional to max content length per column: [24, 7, 74]
    return _table_fig(
        headers=["Outcome label", "Quality", "Description"],
        rows=rows,
        col_widths=[24, 7, 74],
        title="Outcome Label Definitions",
        row_colors=colors,
        height=max(320, 100 + len(rows) * 56),
        fig_width=1000,
    )


def fig_state_bucket_defs() -> go.Figure:
    gold_rows = [
        ["big_behind",  "Bottom 20% of gold % advantage"],
        ["behind",      "20th-40th percentile"],
        ["even",        "Middle 20% (40th-60th percentile)"],
        ["ahead",       "60th-80th percentile"],
        ["big_ahead",   "Top 20%"],
    ]
    alive_rows = [
        ["numbers_down",  "Fewer allied champions alive than enemy at T-60"],
        ["even_alive",    "Equal alive counts"],
        ["numbers_up",    "More allied champions alive than enemy at T-60"],
    ]
    death_rows = [
        ["no_recent_deaths", "Neither side lost a champion in the last 90s"],
        ["enemy_pick",       "Enemy had deaths, team did not"],
        ["team_pick",        "Team had deaths, enemy did not"],
        ["trade_deaths",     "Both sides had deaths"],
    ]

    gold_colors  = [_ROW_EVEN if i % 2 == 0 else _ROW_ODD for i in range(len(gold_rows))]
    alive_colors = [_ROW_EVEN if i % 2 == 0 else _ROW_ODD for i in range(len(alive_rows))]
    death_colors = [_ROW_EVEN if i % 2 == 0 else _ROW_ODD for i in range(len(death_rows))]

    gold_fig = _table_fig(
        headers=["Gold state bucket", "Definition (gold % advantage at T-60 vs estimated team gold)"],
        rows=gold_rows, col_widths=[20, 80],
        title="State Bucket Definitions  --  Gold State (T-60)",
        row_colors=gold_colors,
        height=max(320, 100 + len(gold_rows) * 56),
        fig_width=480,
    )
    alive_fig = _table_fig(
        headers=["Alive state bucket", "Definition"],
        rows=alive_rows, col_widths=[30, 70],
        title="State Bucket Definitions  --  Alive State (T-60)",
        row_colors=alive_colors,
        height=max(320, 100 + len(alive_rows) * 56),
        fig_width=430,
    )
    death_fig = _table_fig(
        headers=["Death state", "Definition (standalone feature, not part of combined bucket)"],
        rows=death_rows, col_widths=[22, 78],
        title="State Bucket Definitions  --  Death State (T-90 window)",
        row_colors=death_colors,
        height=max(320, 100 + len(death_rows) * 56),
        fig_width=520,
    )
    return gold_fig, alive_fig, death_fig


# ── chart list (Plotly figures only) ─────────────────────────────────────────

CHARTS: list[tuple[str, object]] = [
    ("01_setup_profiles_combo",         A.fig_profile_combo()),
    ("02_outcome_labels_combo",         A.fig_outcome_combo()),
    ("03_setup_profile_by_objective",   A.fig_objective_profile_pies()),
    ("04_outcome_label_by_objective",   A.fig_objective_outcome_pies()),
    ("05_gold_breakeven",               A.fig_gold_breakeven()),
    ("07_feature_impact",               A.fig_feature_impact()),
    ("08_deaths_before_objective",      A.fig_deaths_secure()),
    ("09_state_conditioned_arrive_first",
        A.fig_state_conditioned(A.arrived_tbl, "arrived first")),
    ("10_state_conditioned_numbers_adv",
        A.fig_state_conditioned(A.numbers_tbl, "numbers advantage at T-30")),
    ("11_model_auc_comparison",         A.fig_model_auc()),
    ("12_rank_outcomes",                A.fig_rank_outcomes()),
    ("14_setup_profile_definitions",    fig_setup_profile_defs()),
    ("15_outcome_label_definitions",    fig_outcome_label_defs()),
]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=_ROOT / "exports" / "charts")
    p.add_argument("--width",  type=int, default=_W)
    p.add_argument("--height", type=int, default=_H)
    p.add_argument("--scale",  type=float, default=3.0)
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    # Figures with their own layout.width use a boosted scale so the output
    # pixel count stays comparable to the standard 2400×scale charts.
    _target_px = args.width * args.scale

    for name, fig in CHARTS:
        out_path = args.out / f"{name}.png"
        print(f"[png-export] {name} ...")
        fig_w = fig.layout.width if fig.layout.width else args.width
        fig_scale = min(round(_target_px / fig_w), 4) if fig.layout.width else args.scale
        fig.write_image(
            str(out_path),
            format="png",
            width=fig_w,
            height=fig.layout.height if fig.layout.height else args.height,
            scale=fig_scale,
        )
        print(f"             saved: {out_path}  ({int(fig_w * fig_scale)}px wide)")

    # SHAP image is stored as a base64 data URI
    shap_src: str = A._SHAP_IMG_SRC
    if shap_src.startswith("data:image/png;base64,"):
        shap_path = args.out / "13_shap_feature_importance.png"
        print("[png-export] 13_shap_feature_importance (base64 decode) ...")
        img_bytes = base64.b64decode(shap_src.split(",", 1)[1])
        shap_path.write_bytes(img_bytes)
        print(f"             saved: {shap_path}")
    else:
        print(f"[png-export] SHAP image is not a base64 PNG, skipping ({shap_src[:60]}...)")

    # State bucket definitions: three separate tables
    # Use a fixed scale (4) so small fig_widths don't get blown up to _target_px.
    _BUCKET_SCALE = 4
    gold_fig, alive_fig, death_fig = fig_state_bucket_defs()
    for suffix, fig in [("gold", gold_fig), ("alive", alive_fig), ("death", death_fig)]:
        name = f"16_state_bucket_definitions_{suffix}"
        out_path = args.out / f"{name}.png"
        print(f"[png-export] {name} ...")
        fig_w = fig.layout.width if fig.layout.width else args.width
        fig_scale = _BUCKET_SCALE
        fig.write_image(
            str(out_path),
            format="png",
            width=fig_w,
            height=fig.layout.height if fig.layout.height else args.height,
            scale=fig_scale,
        )
        print(f"             saved: {out_path}  ({int(fig_w * fig_scale)}px wide)")

    # Setup profile map states (matplotlib figure)
    print("[png-export] 17_setup_profile_mapstates ...")
    mfig = fig_mapstates_grid()
    mout = args.out / "17_setup_profile_mapstates.png"
    mfig.savefig(str(mout), dpi=180, bbox_inches="tight",
                 facecolor=mfig.get_facecolor())
    plt.close(mfig)
    print(f"             saved: {mout}")

    total = len(CHARTS) + 1 + 3 + 1  # plotly + shap + 3 state bucket tables + mapstates
    print(f"\n[png-export] Done -- {total} files written to {args.out}/")


if __name__ == "__main__":
    main()
