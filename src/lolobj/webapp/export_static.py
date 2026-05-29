"""Export the full webapp as a single self-contained HTML file.

Usage:
    python src/lolobj/webapp/export_static.py
    python src/lolobj/webapp/export_static.py --output my_results.html
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_SRC  = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

print("[export] Loading app (loads data, trains models, computes SHAP)…")
import lolobj.webapp.app as A

_CSS = (Path(__file__).parent / "assets" / "style.css").read_text(encoding="utf-8")


# ── HTML helpers ─────────────────────────────────────────────────────────────

def _e(s: str) -> str:
    """HTML-escape a plain string."""
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))

def _fig(fig) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": False})

def _chart(fig) -> str:
    return f'<div class="chart">{_fig(fig)}</div>'

def _note(text: str) -> str:
    return f'<p class="note">{_e(text)}</p>'

def _section(title: str, subtitle: str, *inner: str) -> str:
    sub = f'<p class="section-sub">{_e(subtitle)}</p>' if subtitle else ""
    return (f'<div class="section">'
            f'<h2 class="section-title">{_e(title)}</h2>{sub}'
            + "".join(inner) + '</div>')

def _bullets(items: list[str]) -> str:
    lis = "".join(f"<li>{_e(i)}</li>" for i in items)
    return f'<ul class="bullet-list">{lis}</ul>'

def _stat_card(label: str, value: str) -> str:
    return (f'<div class="stat-card">'
            f'<div class="stat-value">{_e(value)}</div>'
            f'<div class="stat-label">{_e(label)}</div>'
            f'</div>')


# ── pages ─────────────────────────────────────────────────────────────────────

def page_home() -> str:
    stats = A.STATS
    return (
        '<div class="page">'
        '<div class="hero">'
        '<h1 class="hero-title">League Objective Setup Analytics</h1>'
        '<p class="hero-sub">What separates good and bad objective setups in League of Legends, '
        'and how that varies by rank, objective type, and game state.</p>'
        '</div>'
        '<div class="stat-row">'
        + _stat_card("Matches analyzed", f"{stats['matches']:,}")
        + _stat_card("Objective instances", f"{stats['objectives']:,}")
        + '</div>'
        + _section("Core research question", "",
            '<p class="body-text">What conditions make contesting, starting, trading, or giving '
            'an objective the right call? The project focuses on Dragon and Baron first, then '
            'extends to Herald, Voidgrubs, Soul, and Elder.</p>')
        + _section("What this project does", "",
            _bullets([
                "Builds a clean objective-window table, one row per team per objective",
                "Labels outcomes beyond just secured / not secured",
                "Separates the game state at T-60 from what teams actually chose to do by T-30",
                "Compares teams in similar situations to see how much setup choices matter",
            ]))
        + '</div>'
    )


def page_methods() -> str:
    outcome_cards = "".join(
        f'<div class="def-card def-card--{tag}">'
        f'<div class="def-name">{_e(name.replace("_", " "))}</div>'
        f'<div class="def-meaning">{_e(desc)}</div>'
        f'</div>'
        for name, tag, desc in A.OUTCOME_DEFS
    )
    profile_cards = "".join(
        f'<div class="def-card def-card--{tag}">'
        f'<div class="def-name">{_e(name.replace("_", " "))}</div>'
        f'<div class="def-condition">{_e(condition)}</div>'
        f'<div class="def-meaning">{_e(meaning)}</div>'
        f'</div>'
        for name, tag, condition, meaning in A.PROFILE_DEFS
    )
    return (
        '<div class="page">'
        '<h1 class="page-title">Methods</h1>'
        + _section("Unit of analysis", "Each row is one team's perspective on one objective instance.",
            '<div class="mono-block"><code>match_id  +  objective_instance  +  team_id  →  one row</code></div>'
            '<p class="body-text">For each objective two rows are created, one per team side. '
            'Features describe only that team\'s state relative to the opponent.</p>')
        + _section("Objective time windows",
                   "Features are computed within windows relative to T=0 (objective take time).",
            '<div class="window-row">'
            '<div class="win-card"><div class="win-label">T−120 → T−90</div><div class="win-name">Early setup</div></div>'
            '<div class="win-card win-card--hi"><div class="win-label">T−90 → T−60</div><div class="win-name">State (pre-commitment)</div></div>'
            '<div class="win-card win-card--hi"><div class="win-label">T−60 → T−30</div><div class="win-name">Setup actions</div></div>'
            '<div class="win-card"><div class="win-label">T−30 → T+30</div><div class="win-name">Contest / take</div></div>'
            '<div class="win-card"><div class="win-label">T+30 → T+120</div><div class="win-name">Aftermath</div></div>'
            '</div>')
        + _section("State-first analytical framework",
                   "Separate what was already true at T-60 from what teams chose to do between T-60 and T-30.",
            '<div class="three-col">'
            '<div class="method-card"><div class="method-num">01</div><h3 class="method-title">State</h3>'
            '<p class="method-body">Information at T-60 or earlier. Gold difference, alive counts, recent deaths.</p></div>'
            '<div class="method-card"><div class="method-num">02</div><h3 class="method-title">Action</h3>'
            '<p class="method-body">Observed behaviour between T-60 and T-30. Who arrived first, '
            'how many champions appeared near the objective, whether setup was contested or conceded.</p></div>'
            '<div class="method-card"><div class="method-num">03</div><h3 class="method-title">Outcome</h3>'
            '<p class="method-body">Objective secured, outcome label, net objective value. Never used as a model feature.</p></div>'
            '</div>'
            + _note("Comparing within similar T-60 states is cleaner than looking across all games at once "
                    "because the starting situation is roughly controlled for. Still observational, not causal."))
        + _section("State bucket definitions",
                   "The T-60 state is broken into three dimensions that get combined into one bucket label.",
            '<div class="three-col">'
            '<div class="method-card"><div class="method-title">Gold state (T-60)</div>'
            '<p class="method-body">Gold lead as % of estimated team gold at game time, split into quintiles from the data.</p>'
            '<ul class="bullet-list">'
            '<li><b>big_behind</b>: bottom 20%</li>'
            '<li><b>behind</b>: 20th-40th percentile</li>'
            '<li><b>even</b>: middle 20% (40th-60th)</li>'
            '<li><b>ahead</b>: 60th-80th percentile</li>'
            '<li><b>big_ahead</b>: top 20%</li>'
            '</ul>'
            '<p class="method-body">Normalizing by estimated team gold keeps thresholds comparable across Dragon 1 and Dragon 2.</p>'
            '</div>'
            '<div class="method-card"><div class="method-title">Alive state (T-60)</div>'
            '<ul class="bullet-list">'
            '<li><b>numbers_down</b>: fewer alive than enemy</li>'
            '<li><b>even_alive</b>: equal alive counts</li>'
            '<li><b>numbers_up</b>: more alive than enemy</li>'
            '</ul></div>'
            '<div class="method-card"><div class="method-title">Death state (standalone feature)</div>'
            '<p class="method-body">Kept as a standalone feature but not part of the state bucket. '
            'It overlaps heavily with alive_state, and adding it would create 60 bucket combinations instead of 15.</p>'
            '<ul class="bullet-list">'
            '<li><b>no_recent_deaths</b>: neither side lost a champion in the last 90s</li>'
            '<li><b>enemy_pick</b>: enemy had deaths, team did not</li>'
            '<li><b>team_pick</b>: team had deaths, enemy did not</li>'
            '<li><b>trade_deaths</b>: both sides had deaths</li>'
            '</ul></div>'
            '</div>'
            '<p class="body-text">The combined bucket reads as gold_state | alive_state, e.g. \'even | numbers_up\'. '
            'That gives 15 combinations. Adding death state would push it to 60, '
            'which leaves too few rows per bucket to compare meaningfully.</p>')
        + _section("Setup profile definitions",
                   "Profiles classify each team's T-30 posture relative to the opponent.",
            f'<div class="def-grid">{profile_cards}</div>')
        + _section("Outcome label definitions",
                   "Ten outcome categories describe what happened at and after the objective.",
            f'<div class="def-grid def-grid--wide">{outcome_cards}</div>')
        + _section("Modeling approach", "Models are used as analytical tools, not prediction engines.",
            '<div class="step-list">'
            '<div class="step"><span class="step-num">1</span>Descriptive statistics and setup profiles</div>'
            '<div class="step"><span class="step-num">2</span>State-only logistic regression (T-60 features only)</div>'
            '<div class="step"><span class="step-num">3</span>Setup-only logistic regression (T-30 action features only)</div>'
            '<div class="step"><span class="step-num">4</span>Combined model (state + setup): measures how much setup adds on top of state</div>'
            '<div class="step"><span class="step-num">5</span>State-conditioned within-bucket comparisons for each action</div>'
            '</div>'
            + _note("Train/test splits are grouped by match_id. All rows from one match stay together "
                    "in train or test, which prevents leakage from objectives within the same game."))
        + _section("Data and limitations", "",
            _bullets([
                "Data source: Riot Games Match-V5 API (match summaries and timeline events)",
                "Exact player pathing, wave state, and item timing are not available, so proxies are used",
                "Role assignment is inferred from position data and may have errors for off-meta compositions",
                "Findings are observational. Teams that arrive first may do so because of champion advantages the model cannot see",
                "Language throughout: 'associated with', 'predictive of', 'proxy for'. Not 'proves' or 'causes'",
            ]))
        + '</div>'
    )


def page_analysis() -> str:
    best = A.model_df[A.model_df["Model"] == "State + Setup"]["AUC"].values[0]
    base = A.model_df[A.model_df["Model"] == "State (T-60)"]["AUC"].values[0]
    lift = round((best - base) * 100, 1)

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
                   "Tile size = how common each profile is. Color = secure rate (red to green).",
            _chart(A.fig_profile_combo()))
        + _section("Outcome labels: prevalence and strategic quality",
                   "Tile size = how often each outcome occurred. Color = strategic quality of the outcome.",
            _chart(A.fig_outcome_combo()),
            _note("lost_fight_got_objective is more common than won_fight_lost_objective because if you win "
                  "the fight you almost always have time to take the objective too. 'Won fight, lost objective' "
                  "mostly happens when the enemy gets a last-second smite steal. A team losing the fight will "
                  "often still try a last-second smite, because there is nothing else to do. "
                  "objective_steal is rarer since it requires being behind and winning the smite."))
        + _section("Impact of key binary features",
                   "Secure rate with and without each binary feature.",
            _chart(A.fig_feature_impact()))
        + _section("Deaths before objective",
                   "Each allied death in the 60s before the objective substantially reduces the secure rate.",
            _chart(A.fig_deaths_secure()),
            _note("Even one death drops the secure rate by roughly 20 percentage points. "
                  "Two deaths brings it below 25%. Dying right before the objective is one of the "
                  "most common ways objectives are thrown."))
        + _section("State-first: value of arriving first",
                   "Within each T-60 state bucket, teams that arrived first vs those that did not.",
            _chart(A.fig_state_conditioned(A.arrived_tbl, "arrived first")),
            _note("The effect is biggest in 'even' and 'behind' states. "
                  "When the gold lead is large enough to decide the fight, arrival timing matters less."))
        + _section("State-first: value of numbers advantage at T-30",
                   "Same analysis for having more allies than enemies near the objective at T-30.",
            _chart(A.fig_state_conditioned(A.numbers_tbl, "numbers advantage at T-30")))
        + _section("Model comparison",
                   f"Three logistic regression models compared on a held-out test set (25% of matches, "
                   f"grouped by match_id). State + Setup improves AUC by {lift:.1f} pp over State alone.",
            _chart(A.fig_model_auc()),
            f'<div class="table-wrap"><table class="data-table">'
            f'<thead><tr><th>Model</th><th>AUC</th><th>Log-loss</th><th>Brier</th><th>Features</th></tr></thead>'
            f'<tbody>{table_rows}</tbody></table></div>',
            _note("Setup actions add real predictive value beyond T-60 state. This is association, not causation."))
        + _section("Feature importance (SHAP)",
                   "Random Forest model. Each dot is one objective row. Color shows feature value: blue = low, red = high.",
            f'<img src="{A._SHAP_IMG_SRC}" style="width:100%;max-width:860px;display:block;'
            f'border:1px solid #e2e8f0;border-radius:8px">',
            _note("SHAP values show how much each feature shifts the model's prediction on average. "
                  "Gray dots are state features (T-60 or earlier). Blue dots are setup actions (T-30). "
                  "A wide spread means the feature matters a lot depending on its value."))
        + _section("Outcome distribution by rank",
                   "How outcomes differ across skill tiers.",
            _chart(A.fig_rank_outcomes()))
        + '</div>'
    )


def page_conclusion() -> str:
    best  = A.model_df[A.model_df["Model"] == "State + Setup"]["AUC"].values[0]
    state = A.model_df[A.model_df["Model"] == "State (T-60)"]["AUC"].values[0]
    return (
        '<div class="page">'
        '<h1 class="page-title">Conclusion</h1>'
        + _section("Key findings", "",
            '<div class="finding-list">'
            '<div class="finding"><div class="finding-num">01</div><div class="finding-body">'
            '<h3 class="finding-title">Setup choices matter beyond what the pre-objective state alone predicts</h3>'
            f'<p class="finding-text">The combined model hits AUC {best:.3f} vs {state:.3f} for the '
            'state-only baseline. Teams in similar T-60 situations end up with noticeably different '
            'outcomes depending on their setup.</p></div></div>'
            '<div class="finding"><div class="finding-num">02</div><div class="finding-body">'
            '<h3 class="finding-title">Arriving first is the clearest setup advantage in the data</h3>'
            '<p class="finding-text">Within similar T-60 state buckets, arriving first is associated with '
            '40 to 65 percentage point higher secure rates. The pattern holds across gold states: behind, even, and ahead.</p>'
            '</div></div>'
            '<div class="finding"><div class="finding-num">03</div><div class="finding-body">'
            '<h3 class="finding-title">Dying before the objective is one of the clearest ways to lose it</h3>'
            '<p class="finding-text">A single allied death in the 60s before the objective drops the secure '
            'rate by ~20 pp. Two deaths brings it below 25%. Dying right before the fight is one of the '
            'most common ways objectives are thrown.</p></div></div>'
            '<div class="finding"><div class="finding-num">04</div><div class="finding-body">'
            '<h3 class="finding-title">Showing up uncontested almost always wins. Not showing up almost always loses.</h3>'
            '<p class="finding-text">Teams that arrive with no enemy around secure at very high rates. '
            'Teams that skip the objective entirely lose it almost every time. T-30 presence is the '
            'clearest single signal in the data.</p></div></div>'
            '</div>')
        + _section("Limitations", "",
            _bullets([
                "All findings are observational. Teams that arrive first may do so because of champion kit advantages that also cause them to win fights.",
                "Wave state, champion scaling, and communication are not captured by the Riot API timeline.",
                "Results may not generalize across patches with major objective timing changes.",
                "Role assignment is inferred from position data and may have errors for off-meta compositions.",
            ]))
        + _section("Next steps", "",
            _bullets([
                "Expand to Baron, Herald, Voidgrubs, Soul, and Elder objectives",
                "Build rank-stratified models to test whether setup factors differ by tier",
                "Add Random Forest + SHAP for non-linear interaction effects",
                "Develop setup profile clustering (rule-based or model-assisted)",
            ]))
        + _section("Methodological note", "",
            '<p class="body-text note">State-first results should be read as: among teams in similar '
            'pre-objective situations, these setup choices were associated with better or worse secure rates. '
            'The starting situation is roughly controlled for, which makes it cleaner than raw correlation, '
            'but it is still observational. It does not prove a team would have secured by doing something different.</p>')
        + '</div>'
    )


# ── nav tab CSS override (replaces Dash dcc.Tabs styles) ─────────────────────

_NAV_CSS = """
.export-nav {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 0;
}
.export-tab {
  height: 52px;
  line-height: 52px;
  padding: 0 2px;
  margin-right: 28px;
  border: none;
  border-bottom: 2px solid transparent;
  background: transparent;
  color: #64748b;
  font-size: 14px;
  font-family: var(--font);
  cursor: pointer;
  transition: color 0.15s, border-color 0.15s;
}
.export-tab:hover { color: #1e293b; }
.export-tab.active { color: #1e293b; border-bottom-color: #3d5af1; font-weight: 500; }
"""

_NAV_JS = """
function showTab(name) {
  document.querySelectorAll('.tab-pane').forEach(el => el.style.display = 'none');
  document.querySelectorAll('.export-tab').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + name).style.display = 'block';
  document.getElementById('btn-' + name).classList.add('active');
}
"""

_TABS = [("home", "Home"), ("methods", "Methods"), ("analysis", "Analysis"), ("conclusion", "Conclusion")]


def build_html(pages: dict[str, str]) -> str:
    nav_btns = "".join(
        f'<button id="btn-{k}" class="export-tab{" active" if k == "home" else ""}" '
        f'onclick="showTab(\'{k}\')">{label}</button>'
        for k, label in _TABS
    )
    tab_panes = "".join(
        f'<div id="tab-{k}" class="tab-pane content-root" style="{"" if k == "home" else "display:none"}">'
        f'{pages[k]}</div>'
        for k, _ in _TABS
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>League Objective Analytics</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
{_CSS}
{_NAV_CSS}
</style>
</head>
<body>
<div class="app-root">
  <header class="site-header">
    <div class="header-inner">
      <span class="site-logo">League Objective Analytics</span>
      <nav class="export-nav">{nav_btns}</nav>
    </div>
  </header>
  {tab_panes}
</div>
<script>{_NAV_JS}</script>
</body>
</html>"""


# ── CLI ────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path,
                   default=_ROOT / "exports" / "league_objective_analytics.html")
    args = p.parse_args(argv)

    print("[export] Rendering pages…")
    pages = {
        "home":       page_home(),
        "methods":    page_methods(),
        "analysis":   page_analysis(),
        "conclusion": page_conclusion(),
    }

    print("[export] Assembling HTML…")
    html = build_html(pages)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    size_mb = args.output.stat().st_size / 1_048_576
    print(f"[export] Written to {args.output}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
