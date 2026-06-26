"""Minimap snapshot figures for each setup profile.

Generates archetypal T-30 champion-position snapshots for the six setup
profiles. Positions are synthetic (not drawn from real matches) and are
designed to clearly illustrate each profile's defining condition.

Coordinate system: normalized image space where (0, 0) = top-left,
(1, 1) = bottom-right. Derived from LoL game coordinates via:
    nx = game_x / 14820
    ny = 1 - game_y / 14881          (y is flipped: high game-y = top of map)

Key locations:
    Dragon pit  ≈ game (9866, 4414)  → (0.666, 0.703)
    Baron pit   ≈ game (5007, 10471) → (0.338, 0.297)
    Blue nexus  ≈ game  (500,  500)  → (0.034, 0.966)
    Red  nexus  ≈ game (14320, 14380)→ (0.966, 0.034)

Usage:
    python -m lolobj.viz.mapstates               # → exports/charts/
    python -m lolobj.viz.mapstates --out <dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_ROOT   = Path(__file__).resolve().parents[3]
_MAP    = _ROOT / "data" / "raw" / "rift map.png"
_OUTDIR = _ROOT / "exports" / "charts"

# ── key map locations (normalized, y = 0 at top) ─────────────────────────────
DRAGON: tuple[float, float] = (0.666, 0.703)

# ── individual role positions ─────────────────────────────────────────────────
# Blue side: base is bottom-left; bot lane runs along bottom/right edges.
# Red side:  base is top-right;  bot lane runs along top/left edges.
#
# "Botside" = lower half of image (ny > 0.5) = dragon side.
# "Topside" = upper half (ny < 0.5) = baron / red base side.

# Blue at dragon — jungler at pit, others rotating in from their angles
_B_JG_DRAG  = (0.648, 0.690)
_B_ADC_DRAG = (0.742, 0.770)
_B_SUP_DRAG = (0.708, 0.740)
_B_MID_DRAG = (0.606, 0.674)
_B_TOP_DRAG = (0.633, 0.728)

# Red at dragon — approaching from their side (northeast of pit)
_R_JG_DRAG  = (0.712, 0.606)
_R_ADC_DRAG = (0.682, 0.580)
_R_SUP_DRAG = (0.660, 0.624)
_R_MID_DRAG = (0.610, 0.568)
_R_TOP_DRAG = (0.748, 0.656)

# Blue in lanes / jungle — not rotating to dragon
# ADC is ahead in bot lane; support is back in the same lane, not touching.
_B_JG_LANE  = (0.338, 0.595)   # blue-side jungle (wolves / blue buff area)
_B_ADC_LANE = (0.800, 0.848)   # farther up bot lane (toward dragon side)
_B_SUP_LANE = (0.745, 0.900)   # further back in bot lane, near tower
_B_MID_LANE = (0.460, 0.622)   # blue-side mid lane
_B_TOP_LANE = (0.128, 0.298)   # blue top lane (left edge)

# Red in lanes / jungle
# Red's bot lane (ADC + support) is in the top-left of the map.
# Red's top laner and jungler are on botside (bottom half of image):
#   top laner is on the right edge (red's top lane = blue's bot side),
#   jungler is near red buff which sits mid-right, just below centre.
_R_JG_LANE  = (0.742, 0.512)   # near red buff — botside, right of centre
_R_ADC_LANE = (0.240, 0.118)   # red bot lane, further up
_R_SUP_LANE = (0.168, 0.205)   # red bot lane, further back toward tower
_R_MID_LANE = (0.538, 0.448)   # red-side mid lane
_R_TOP_LANE = (0.858, 0.752)   # red top lane — botside, right edge

# Convenience lists
_BLUE_AT_DRAGON = [_B_JG_DRAG, _B_ADC_DRAG, _B_SUP_DRAG, _B_MID_DRAG, _B_TOP_DRAG]
_RED_AT_DRAGON  = [_R_JG_DRAG, _R_ADC_DRAG, _R_SUP_DRAG, _R_MID_DRAG, _R_TOP_DRAG]
_BLUE_IN_LANES  = [_B_JG_LANE, _B_ADC_LANE, _B_SUP_LANE, _B_MID_LANE, _B_TOP_LANE]
_RED_IN_LANES   = [_R_JG_LANE, _R_ADC_LANE, _R_SUP_LANE, _R_MID_LANE, _R_TOP_LANE]

# Ward positions near dragon (vision control)
_WARDS_FULL = [(0.700, 0.645), (0.758, 0.618), (0.612, 0.758)]
_WARDS_THIN = [(0.700, 0.645), (0.758, 0.618)]
_WARDS_ONE  = [(0.700, 0.645)]
_WARDS_NONE: list = []

# ── profile definitions ───────────────────────────────────────────────────────
# (key, display_title, subtitle, accent_color,
#  blue_alive, blue_dead, red_alive, red_dead, wards)
#
# For free_setup_deaths: mid/top died in their lanes before rotating — dead
# markers appear at their lane positions, not at dragon.
# For disadvantaged: mid/top were at dragon but died in the pre-fight — dead
# markers appear near the pit.
PROFILES = [
    (
        "free_setup",
        "Free Setup",
        "Team present · Enemy absent · No recent allied deaths",
        "#10b981",
        _BLUE_AT_DRAGON, [],
        _RED_IN_LANES,   [],
        _WARDS_FULL,
    ),
    (
        "free_setup_deaths",
        "Free Setup (with Deaths)",
        "Team present · Enemy absent · Allied deaths in the prior 60s",
        "#f59e0b",
        [_B_JG_DRAG, _B_ADC_DRAG, _B_SUP_DRAG],   # JG+ADC+support alive at dragon
        [_B_MID_LANE, _B_TOP_LANE],                 # mid/top died in their lanes
        _RED_IN_LANES, [],
        _WARDS_THIN,
    ),
    (
        "clean_contest",
        "Clean Contest",
        "Both teams present · Team not short-handed",
        "#a78bfa",
        _BLUE_AT_DRAGON, [],
        _RED_AT_DRAGON,  [],
        _WARDS_ONE,
    ),
    (
        "disadvantaged",
        "Disadvantaged",
        "Both teams present · Team had recent deaths or fewer alive",
        "#ef4444",
        [_B_JG_DRAG, _B_ADC_DRAG, _B_SUP_DRAG],   # 3 alive at dragon
        [_B_MID_DRAG, _B_TOP_DRAG],                 # mid/top died near pit
        _RED_AT_DRAGON, [],
        _WARDS_NONE,
    ),
    (
        "gave_away",
        "Gave Away",
        "Enemy present at objective · Team did not show up",
        "#64748b",
        _BLUE_IN_LANES, [],
        _RED_AT_DRAGON, [],
        _WARDS_NONE,
    ),
    (
        "no_early_setup",
        "No Early Setup",
        "Neither team near objective at T-30 · Neither side committed",
        "#94a3b8",
        _BLUE_IN_LANES, [],
        _RED_IN_LANES,  [],
        _WARDS_NONE,
    ),
]


def _to_px(
    positions: list[tuple[float, float]], w: int, h: int
) -> tuple[list[float], list[float]]:
    if not positions:
        return [], []
    xs = [nx * w for nx, _ in positions]
    ys = [ny * h for _, ny in positions]
    return xs, ys


def _draw_scene(
    ax: plt.Axes,
    img,
    blue_alive: list[tuple[float, float]],
    blue_dead:  list[tuple[float, float]],
    red_alive:  list[tuple[float, float]],
    red_dead:   list[tuple[float, float]],
    wards:      list[tuple[float, float]],
    title: str,
    subtitle: str,
    accent: str,
) -> None:
    h, w = img.shape[:2]
    ax.imshow(img)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)   # y=0 at top, y=h at bottom
    ax.set_aspect("equal")
    ax.axis("off")

    # Dragon objective marker
    dx, dy = DRAGON[0] * w, DRAGON[1] * h
    ax.scatter(dx, dy, s=350, marker="*", color="#fbbf24",
               edgecolors="#92400e", linewidths=1.8, zorder=10)

    # Vision wards
    wxs, wys = _to_px(wards, w, h)
    if wxs:
        ax.scatter(wxs, wys, s=80, marker="D", color="#fde68a",
                   edgecolors="#d97706", linewidths=1.2, zorder=8)

    # Red alive
    rxs, rys = _to_px(red_alive, w, h)
    if rxs:
        ax.scatter(rxs, rys, s=260, color="#ef4444",
                   edgecolors="#7f1d1d", linewidths=1.8, zorder=6)

    # Red dead
    rxs, rys = _to_px(red_dead, w, h)
    if rxs:
        ax.scatter(rxs, rys, s=260, marker="X", color="#fca5a5",
                   edgecolors="#ef4444", linewidths=1.2, alpha=0.55, zorder=6)

    # Blue alive
    bxs, bys = _to_px(blue_alive, w, h)
    if bxs:
        ax.scatter(bxs, bys, s=260, color="#3b82f6",
                   edgecolors="#1e3a8a", linewidths=1.8, zorder=7)

    # Blue dead
    bxs, bys = _to_px(blue_dead, w, h)
    if bxs:
        ax.scatter(bxs, bys, s=260, marker="X", color="#93c5fd",
                   edgecolors="#3b82f6", linewidths=1.2, alpha=0.55, zorder=7)

    ax.set_title(
        title,
        fontsize=10.5,
        fontweight="bold",
        color="white",
        pad=5,
        bbox=dict(boxstyle="square,pad=0.35", facecolor=accent, edgecolor="none"),
    )
    ax.text(
        0.5, -0.02, subtitle,
        transform=ax.transAxes,
        fontsize=7.8, color="#475569",
        ha="center", va="top",
    )


def fig_mapstates_grid() -> plt.Figure:
    """Return a 2×3 figure with all six setup profile map snapshots."""
    img = mpimg.imread(str(_MAP))

    fig, axes = plt.subplots(2, 3, figsize=(15, 11.5))
    fig.patch.set_facecolor("#f8fafc")

    for ax, (key, title, subtitle, accent,
              blue_alive, blue_dead, red_alive, red_dead, wards) in zip(
        axes.flat, PROFILES
    ):
        _draw_scene(ax, img, blue_alive, blue_dead,
                    red_alive, red_dead, wards,
                    title, subtitle, accent)

    # Legend
    legend_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor="#3b82f6", markersize=11,
               markeredgecolor="#1e3a8a", markeredgewidth=1.5,
               label="Blue team (alive)"),
        Line2D([0], [0], marker="X", color="none",
               markerfacecolor="#93c5fd", markersize=10,
               markeredgecolor="#3b82f6", markeredgewidth=1.2,
               alpha=0.7, label="Blue team (dead)"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor="#ef4444", markersize=11,
               markeredgecolor="#7f1d1d", markeredgewidth=1.5,
               label="Red team (alive)"),
        Line2D([0], [0], marker="X", color="none",
               markerfacecolor="#fca5a5", markersize=10,
               markeredgecolor="#ef4444", markeredgewidth=1.2,
               alpha=0.7, label="Red team (dead)"),
        Line2D([0], [0], marker="*", color="none",
               markerfacecolor="#fbbf24", markersize=14,
               markeredgecolor="#92400e", markeredgewidth=1.5,
               label="Dragon (objective)"),
        Line2D([0], [0], marker="D", color="none",
               markerfacecolor="#fde68a", markersize=9,
               markeredgecolor="#d97706", markeredgewidth=1.2,
               label="Vision ward"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=6,
        fontsize=9,
        frameon=True,
        framealpha=0.95,
        edgecolor="#e2e8f0",
        bbox_to_anchor=(0.5, 0.005),
    )

    fig.suptitle(
        "Setup Profile Map States  ·  T-30 snapshot  ·  synthetic positions",
        fontsize=13,
        fontweight="bold",
        color="#1e293b",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.99])
    fig.subplots_adjust(hspace=0.28)
    return fig


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=_OUTDIR)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    print("[mapstates] generating setup profile map snapshots ...")
    fig = fig_mapstates_grid()
    out = args.out / "17_setup_profile_mapstates.png"
    fig.savefig(str(out), dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[mapstates] saved: {out}")


if __name__ == "__main__":
    main()
