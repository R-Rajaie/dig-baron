"""Export HTML (Plotly) replicas of scripts/plots.R's static charts.

Each output is a paste-ready snippet (a <div> plus a couple of <script>
tags, not a full <html>/<head>/<body> document) -- open it in a text editor,
copy the whole thing, and paste directly into a WordPress "Custom HTML"
block. No file upload or iframe needed. (It also still renders fine if you
just double-click it to preview locally.)

Usage:
    python exports/export_remake_html.py
    python exports/export_remake_html.py --out exports/interactive/remake
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

print("[remake-html] Loading data...")
from lolobj.viz import remake_html as R

dat = R.load_data()
print(f"[remake-html] {len(dat):,} rows")

CHARTS = [
    ("01_setup_profiles_combo",       R.fig_01_setup_profiles_combo(dat)),
    ("02_outcome_labels_combo",       R.fig_02_outcome_labels(dat)),
    ("03_setup_profile_by_objective", R.fig_03_setup_outcome_heatmap(dat)),
    ("04_deaths_to_secure_bar",       R.fig_04_deaths_to_secure(dat)),
    ("05_feature_present_absent_dumbbell", R.fig_05_feature_dumbbell(dat)),
    ("06_gold_by_setup_lines",        R.fig_06_gold_by_setup_lines(dat)),
    ("07_effect_arrived_first_bars",  R.fig_07_effect_arrived_first(dat)),
    ("17_setup_profile_mapstates",    R.fig_17_mapstates()),
]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=_ROOT / "exports" / "interactive" / "remake")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    for name, fig in CHARTS:
        out_path = args.out / f"{name}.html"
        print(f"[remake-html] {name} ...")
        snippet = fig.to_html(include_plotlyjs="cdn", full_html=False,
                               auto_play=False, config={"displayModeBar": False, "responsive": True})
        if name == "17_setup_profile_mapstates":
            # This one uses autosize (no fixed width/height) since it's a 6-panel
            # image grid designed for a specific ~6:5 aspect ratio -- letting it
            # size purely off the container (as the other, simpler charts do) would
            # let the container's height default to something arbitrary and squish
            # or crop the panels. Locking the wrapper to that aspect ratio via CSS
            # means Plotly's responsive resizing fills a box that's always the
            # right shape, at whatever width the page gives it.
            snippet = (f'<div style="width:100%;max-width:1500px;aspect-ratio:6/5;margin:0 auto;">'
                       f'{snippet}</div>')
        out_path.write_text(snippet, encoding="utf-8")
        print(f"              saved: {out_path}")

    print(f"\n[remake-html] Done -- {len(CHARTS)} files written to {args.out}/")


if __name__ == "__main__":
    main()
