"""Export interactive (slider) Plotly figures as standalone HTML files.

These figures use Plotly frames/sliders and can't be captured as a single
static PNG, so they get their own export path here rather than going through
export_pngs.py. Each output file is a paste-ready snippet (a <div> plus a
couple of <script> tags, not a full <html>/<head>/<body> document) -- open it
in a text editor, copy the whole thing, and paste directly into a WordPress
"Custom HTML" block. No file upload or iframe needed. (It also still renders
fine if you just double-click it to preview locally -- browsers are lenient
about a bare fragment with no <html> wrapper.)

Both figures are built in Python, not R: an earlier attempt built the heatmap
in R (scripts/interactive_gold_slider.R) via ggplot2 + ggplotly() for exact
theme_lol styling, but ggplotly() segfaults on this machine for anything
beyond a single continuous-fill trace (confirmed via many minimal repros --
even a bare plot_ly() or a single annotation with bgcolor crashes it). The
heatmap needs categorical color (by outcome) *and* continuous opacity (by
prevalence) at once, which a native Heatmap trace can't represent anyway
(one colorscale only) -- so it's built here with layout shapes (one rect per
cell, literal fillcolor + opacity) plus a text/hover scatter trace, which
Python's plotly handles natively with no such bug.

Usage:
    python exports/export_html.py
    python exports/export_html.py --out exports/interactive
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC  = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

print("[html-export] Loading app (this loads data, trains models, computes SHAP)...")
import lolobj.webapp.app as A

CHARTS: list[tuple[str, object]] = [
    ("gold_slider_heatmap",    A.fig_gold_slider_heatmap()),
    ("gold_slider_stackedbar", A.fig_gold_slider_stackedbar()),
]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=_ROOT / "exports" / "interactive")
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    for name, fig in CHARTS:
        out_path = args.out / f"{name}.html"
        print(f"[html-export] {name} ...")
        fig.write_html(
            str(out_path),
            include_plotlyjs="cdn",
            full_html=False,
            auto_play=False,
            config={"displayModeBar": False, "responsive": True},
        )
        print(f"              saved: {out_path}")

    print(f"\n[html-export] Done -- {len(CHARTS)} files written to {args.out}/")


if __name__ == "__main__":
    main()
