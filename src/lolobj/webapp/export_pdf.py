"""Export the webapp as a single-page PDF using headless Chromium.

Requires:
    pip install playwright
    python -m playwright install chromium

Usage:
    python src/lolobj/webapp/export_pdf.py
    python src/lolobj/webapp/export_pdf.py --output results.pdf
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_SRC  = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

print("[pdf] Loading app…")
import lolobj.webapp.app as A
from lolobj.webapp.export_static import (
    _e, _note, _section, _bullets, _stat_card,
    page_methods,
)

_CSS = (Path(__file__).parent / "assets" / "style.css").read_text(encoding="utf-8")

_PDF_CSS = """
@page { size: A4; margin: 16mm 18mm 16mm 18mm; }
body  { font-size: 11pt; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
.pdf-root  { max-width: 100%; }
.pdf-section-break { page-break-before: always; margin-top: 0; }
.page  { gap: 28px; }
.section { gap: 10px; }
.hero  { padding-bottom: 20px; }
.hero-title { font-size: 18pt; }
.stat-row   { grid-template-columns: repeat(2, 1fr); }
.three-col  { grid-template-columns: repeat(3, 1fr); }
.def-grid   { grid-template-columns: repeat(3, 1fr); }
.def-grid--wide { grid-template-columns: repeat(2, 1fr); }
.chart      { page-break-inside: avoid; }
h1.page-title  { font-size: 15pt; }
h2.section-title { font-size: 11pt; }
.section-sub, .note, .body-text { font-size: 10pt; }
.bullet-list li { font-size: 10pt; }
/* hide Plotly mode-bar */
.modebar { display: none !important; }
"""


def _fig(fig) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": False})

def _chart(fig) -> str:
    return f'<div class="chart">{_fig(fig)}</div>'


def page_home_pdf() -> str:
    s = A.STATS
    return (
        '<div class="page">'
        '<div class="hero">'
        '<h1 class="hero-title">League Objective Setup Analytics</h1>'
        '<p class="hero-sub">What separates good and bad objective setups in League of Legends, '
        'and how that varies by rank, objective type, and game state.</p>'
        '</div>'
        '<div class="stat-row">'
        + _stat_card("Matches analyzed", f"{s['matches']:,}")
        + _stat_card("Objective instances", f"{s['objectives']:,}")
        + '</div>'
        + _section("Core research question", "",
            '<p class="body-text">What conditions make contesting, starting, trading, or giving '
            'an objective the right call? The project focuses on Dragon and Baron first.</p>')
        + _section("What this project does", "",
            _bullets([
                "Builds a clean objective-window table, one row per team per objective",
                "Labels outcomes beyond just secured / not secured",
                "Separates the game state at T-60 from what teams actually chose to do by T-30",
                "Compares teams in similar situations to see how much setup choices matter",
            ]))
        + '</div>'
    )


def page_analysis_pdf() -> str:
    best  = A.model_df[A.model_df["Model"] == "State + Setup"]["AUC"].values[0]
    base  = A.model_df[A.model_df["Model"] == "State (T-60)"]["AUC"].values[0]
    lift  = round((best - base) * 100, 1)

    table_rows = "".join(
        f'<tr class="{"row--highlight" if r["Model"] == "State + Setup" else ""}">'
        f'<td>{_e(r["Model"])}</td><td>{r["AUC"]:.4f}</td>'
        f'<td>{r["Log-loss"]:.4f}</td><td>{r["Brier"]:.4f}</td><td>{r["Features"]}</td></tr>'
        for r in A.model_rows
    )

    return (
        '<div class="page">'
        '<h1 class="page-title">Analysis &amp; Results</h1>'
        + _section("Setup profiles: frequency and secure rate",
                   "Tile size = prevalence. Color = secure rate (red to green).",
            _chart(A.fig_profile_combo()))
        + _section("Outcome labels: prevalence and strategic quality",
                   "Tile size = frequency. Color = strategic quality.",
            _chart(A.fig_outcome_combo()),
            _note("lost_fight_got_objective is more common than won_fight_lost_objective because "
                  "if you win the fight you almost always have time to take the objective too."))
        + _section("Impact of key binary features",
                   "Secure rate with and without each binary feature.",
            _chart(A.fig_feature_impact()))
        + _section("Deaths before objective",
                   "Each allied death in the 60s before the objective substantially reduces the secure rate.",
            _chart(A.fig_deaths_secure()),
            _note("Even one death drops the secure rate by roughly 20 percentage points."))
        + _section("State-first: value of arriving first",
                   "Within each T-60 state bucket, teams that arrived first vs those that did not.",
            _chart(A.fig_state_conditioned(A.arrived_tbl, "arrived first")))
        + _section("State-first: value of numbers advantage at T-30",
                   "Same analysis for having more allies than enemies near the objective at T-30.",
            _chart(A.fig_state_conditioned(A.numbers_tbl, "numbers advantage at T-30")))
        + _section("Model comparison",
                   f"State + Setup improves AUC by {lift:.1f} pp over State alone.",
            _chart(A.fig_model_auc()),
            f'<div class="table-wrap"><table class="data-table">'
            f'<thead><tr><th>Model</th><th>AUC</th><th>Log-loss</th><th>Brier</th><th>Features</th></tr></thead>'
            f'<tbody>{table_rows}</tbody></table></div>')
        + _section("Feature importance (SHAP)",
                   "Random Forest model. Each dot is one objective row.",
            f'<img src="{A._SHAP_IMG_SRC}" style="width:100%;max-width:100%;display:block;'
            f'border:1px solid #e2e8f0;border-radius:6px">')
        + _section("Outcome distribution by rank",
                   "How outcomes differ across skill tiers.",
            _chart(A.fig_rank_outcomes()))
        + '</div>'
    )


def build_html() -> str:
    home       = page_home_pdf()
    methods    = page_methods()   # reuse from export_static
    analysis   = page_analysis_pdf()

    # Wrap methods and analysis in page-break divs
    methods_wrapped  = f'<div class="pdf-section-break">{methods}</div>'
    analysis_wrapped = f'<div class="pdf-section-break">{analysis}</div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
{_CSS}
{_PDF_CSS}
</style>
</head>
<body>
<div class="pdf-root content-root">
  {home}
  {methods_wrapped}
  {analysis_wrapped}
</div>
</body>
</html>"""


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path,
                   default=_ROOT / "exports" / "league_objective_analytics.pdf")
    args = p.parse_args(argv)

    print("[pdf] Building HTML…")
    html = build_html()

    # Write to a temp file so Playwright can load it with a file:// URL
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False,
                                     mode="w", encoding="utf-8") as f:
        f.write(html)
        tmp_path = Path(f.name)

    print("[pdf] Launching headless Chromium…")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: playwright not installed.  pip install playwright && python -m playwright install chromium")
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page    = browser.new_page()
        page.goto(tmp_path.as_uri(), wait_until="networkidle", timeout=60_000)
        # Wait for Plotly charts to finish rendering
        page.wait_for_timeout(3000)
        page.pdf(
            path=str(args.output),
            format="A4",
            print_background=True,
            margin={"top": "16mm", "bottom": "16mm",
                    "left": "18mm", "right": "18mm"},
        )
        browser.close()

    tmp_path.unlink(missing_ok=True)
    size_mb = args.output.stat().st_size / 1_048_576
    print(f"[pdf] Written to {args.output}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
