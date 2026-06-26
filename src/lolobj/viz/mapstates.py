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

# ── archetypal champion positions ─────────────────────────────────────────────
# Blue team approaching dragon from their side (south/southwest of Dragon pit)
_BLUE_AT_DRAGON = [
    (0.645, 0.685),  # Jungler — at pit edge
    (0.740, 0.780),  # ADC — coming up from bot lane
    (0.715, 0.745),  # Support — flanking with ADC
    (0.605, 0.675),  # Mid — rotated down
    (0.635, 0.730),  # Top — rotated in
]

# Red team approaching dragon from their side (north/northeast of Dragon pit)
_RED_AT_DRAGON = [
    (0.710, 0.600),  # Red jungler
    (0.658, 0.575),  # Red ADC
    (0.685, 0.625),  # Red support
    (0.610, 0.570),  # Red mid
    (0.745, 0.655),  # Red top
]

# Blue team scattered in their lanes / jungle — far from dragon
_BLUE_IN_LANES = [
    (0.770, 0.870),  # ADC — blue bot lane
    (0.740, 0.835),  # Support — with ADC
    (0.385, 0.635),  # Jungler — blue-side jungle
    (0.500, 0.550),  # Mid — mid lane
    (0.140, 0.240),  # Top — blue top lane (top-left of map)
]

# Red team scattered in their lanes / jungle — far from dragon
_RED_IN_LANES = [
    (0.220, 0.130),  # Red ADC — red bot = top-left of map
    (0.170, 0.185),  # Red support
    (0.630, 0.380),  # Red jungler
    (0.520, 0.465),  # Red mid
    (0.855, 0.755),  # Red top — red top = bottom-right of map
]

# Ward positions near dragon (vision control)
_WARDS_FULL  = [(0.700, 0.645), (0.760, 0.720), (0.615, 0.760)]
_WARDS_THIN  = [(0.700, 0.645), (0.760, 0.720)]
_WARDS_ONE   = [(0.700, 0.645)]
_WARDS_NONE  = []

# ── profile definitions ───────────────────────────────────────────────────────
# (key, display_title, subtitle_line1, subtitle_line2, accent_color,
#  blue_alive, blue_dead, red_alive, red_dead, wards)
PROFILES = [
    (
        "free_setup",
        "Free Setup",
        "Team present · Enemy absent",
        "No recent allied deaths",
        "#10b981",
        _BLUE_AT_DRAGON, [],
        _RED_IN_LANES,   [],
        _WARDS_FULL,
    ),
    (
        "free_setup_deaths",
        "Free Setup (with Deaths)",
        "Team present · Enemy absent",
        "Allied deaths in the prior 60s",
        "#f59e0b",
        _BLUE_AT_DRAGON[:3], _BLUE_AT_DRAGON[3:],  # 3 alive, 2 dead
        _RED_IN_LANES,        [],
        _WARDS_THIN,
    ),
    (
        "clean_contest",
        "Clean Contest",
        "Both teams present",
        "Team not short-handed",
        "#a78bfa",
        _BLUE_AT_DRAGON, [],
        _RED_AT_DRAGON,  [],
        _WARDS_ONE,
    ),
    (
        "disadvantaged",
        "Disadvantaged",
        "Both teams present",
        "Team had recent deaths / fewer alive",
        "#ef4444",
        _BLUE_AT_DRAGON[:3], _BLUE_AT_DRAGON[3:],  # 3 alive, 2 dead
        _RED_AT_DRAGON,      [],
        _WARDS_NONE,
    ),
    (
        "gave_away",
        "Gave Away",
        "Enemy present at objective",
        "Team did not show up",
        "#64748b",
        _BLUE_IN_LANES, [],
        _RED_AT_DRAGON, [],
        _WARDS_NONE,
    ),
    (
        "no_early_setup",
        "No Early Setup",
        "Neither team near objective at T-30",
        "Neither side committed",
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
    sub1: str,
    sub2: str,
    accent: str,
) -> None:
    h, w = img.shape[:2]
    ax.imshow(img)          # default: origin='upper', pixel coords
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)       # y=0 at top, y=h at bottom
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
        0.5, -0.02, f"{sub1}\n{sub2}",
        transform=ax.transAxes,
        fontsize=7.8, color="#475569",
        ha="center", va="top", linespacing=1.5,
    )


def fig_mapstates_grid() -> plt.Figure:
    """Return a 2×3 figure with all six setup profile map snapshots."""
    img = mpimg.imread(str(_MAP))

    fig, axes = plt.subplots(2, 3, figsize=(15, 11.5))
    fig.patch.set_facecolor("#f8fafc")

    for ax, (key, title, sub1, sub2, accent,
              blue_alive, blue_dead, red_alive, red_dead, wards) in zip(
        axes.flat, PROFILES
    ):
        _draw_scene(ax, img, blue_alive, blue_dead,
                    red_alive, red_dead, wards,
                    title, sub1, sub2, accent)

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
