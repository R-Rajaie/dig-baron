"""Shared dark theme constants matching scripts/plots.R's theme_lol.

Used by the HTML replicas of the remake/figures charts (see remake_html.py)
and by the gold-slider dashboard figures in lolobj.webapp.app. Kept as its
own module so importing it doesn't pull in app.py's heavy load-data /
train-models side effects.
"""
from __future__ import annotations

import colorsys

BG    = "#222327"
INK   = "#e2e8f0"
SUB   = "#94a3b8"
GRID  = "#334155"
PANEL = "#1e293b"
# theme_lol renders with base_family="sans", which resolves to Arial on this
# machine (via ragg) -- match that instead of a web-font stack.
FONT = "Arial, Helvetica, sans-serif"

# Matches scripts/plots.R's pal_profile (R names -> Python snake_case).
PROFILE_COLORS = {
    "free_setup":        "#2e8b57",
    "free_setup_deaths": "#86c06c",
    "clean_contest":     "#ccb14e",
    "no_early_setup":    "#8c95a3",   # R: "neither team setup"
    "disadvantaged":     "#e8743b",
    "gave_away":         "#c0392b",   # R: "no setup"
}
PROFILE_ORDER = ["free_setup", "free_setup_deaths", "clean_contest",
                  "no_early_setup", "disadvantaged", "gave_away"]

# Matches scripts/plots.R's pal_outcome.
OUTCOME_COLORS = {
    "take":            "#2e8b57",
    "lost_with_trade":  "#e8743b",
    "lost":             "#c0392b",
}
OUTCOME_ORDER = ["take", "lost_with_trade", "lost"]

# Matches scripts/plots.R's pal_gold / pal_alive.
GOLD_COLORS = {
    "big_behind": "#c0392b", "behind": "#e8743b", "even": "#9aa0a6",
    "ahead": "#5aa469", "big_ahead": "#2e8b57",
}
ALIVE_COLORS = {"numbers_down": "#E69F00", "even_alive": "#0072B2", "numbers_up": "#009E73"}


def dim_color(hex_color: str, factor: float) -> str:
    """Scale a hex color's HSV value by `factor` (0-1) to encode magnitude.

    See lolobj.webapp.app._dim_color for the full rationale: plain opacity
    alpha-blends against the page background and washes warm hues toward
    muddy brown, and dark+desaturated red-orange reads as "brown" regardless
    of exact hue (a real perceptual category boundary, not a rendering bug),
    so orange/amber hues get an extra hue/saturation nudge as they dim.
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


BASE_LAYOUT = dict(
    paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(family=FONT, size=12, color=INK),
    hoverlabel=dict(bgcolor=PANEL, font_size=13, font_family=FONT, font_color=INK),
)


def style_fig(fig, title: str, subtitle: str | None = None, **layout):
    """Apply the shared dark theme_lol look: title (bold), optional subtitle,
    axis colors, background. `layout` overrides/extends on top."""
    title_dict = dict(text=f"<b>{title}</b>", font=dict(size=21, color=INK, family=FONT))
    if subtitle:
        title_dict["subtitle"] = dict(text=subtitle, font=dict(size=14, color=SUB, family=FONT))
    fig.update_layout(**{**BASE_LAYOUT, "title": title_dict, **layout})
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, color=SUB, gridwidth=0.7)
    fig.update_yaxes(showgrid=False, zeroline=False, color=SUB)
    return fig
