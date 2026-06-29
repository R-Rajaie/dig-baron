"""Mapstate visualization — V2 (matplotlib + scipy).

Improvements over V1:
  • Map darkened slightly for contrast
  • Objective influence radius (translucent gold circle)
  • Glow rings on player markers via layered scatter
  • Role abbreviation badges (JG / ADC / SUP / MID / TOP)
  • Team territory convex hull (≥3 alive at objective, scipy.spatial)
  • Ward pulse rings (concentric transparent circles)
  • Ghost-ring marker for dead players

Usage:
    python -m lolobj.viz.mapstates_v2
    python -m lolobj.viz.mapstates_v2 --out <dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

_ROOT   = Path(__file__).resolve().parents[3]
_MAP    = _ROOT / "data" / "raw" / "rift map.png"
_OUTDIR = _ROOT / "exports" / "charts"

DRAGON = (0.666, 0.703)

# ── position constants ────────────────────────────────────────────────────────
_B_JG_DRAG  = (0.648, 0.690); _B_ADC_DRAG = (0.742, 0.770)
_B_SUP_DRAG = (0.708, 0.740); _B_MID_DRAG = (0.606, 0.674)
_B_TOP_DRAG = (0.633, 0.728)
_R_JG_DRAG  = (0.712, 0.606); _R_ADC_DRAG = (0.682, 0.580)
_R_SUP_DRAG = (0.660, 0.624); _R_MID_DRAG = (0.610, 0.568)
_R_TOP_DRAG = (0.748, 0.656)
_B_JG_LANE  = (0.338, 0.595); _B_ADC_LANE = (0.800, 0.848)
_B_SUP_LANE = (0.745, 0.900); _B_MID_LANE = (0.460, 0.622)
_B_TOP_LANE = (0.128, 0.298)
_R_JG_LANE  = (0.742, 0.512); _R_ADC_LANE = (0.858, 0.872)
_R_SUP_LANE = (0.828, 0.912); _R_MID_LANE = (0.538, 0.448)
_R_TOP_LANE = (0.175, 0.192)

_WARDS_FULL = [(0.700, 0.645), (0.758, 0.618), (0.612, 0.758)]
_WARDS_THIN = [(0.700, 0.645), (0.758, 0.618)]
_WARDS_ONE  = [(0.700, 0.645)]
_WARDS_NONE: list = []

def _ann(positions: list, roles: list[str]) -> list[tuple]:
    return list(zip(positions, roles))

_ALL = ["JG", "ADC", "SUP", "MID", "TOP"]
_JAS = ["JG", "ADC", "SUP"]
_MT  = ["MID", "TOP"]

PROFILES: list[tuple] = [
    ("free_setup", "Free Setup",
     "Team present  ·  Enemy absent  ·  No recent allied deaths", "#10b981",
     _ann([_B_JG_DRAG,_B_ADC_DRAG,_B_SUP_DRAG,_B_MID_DRAG,_B_TOP_DRAG], _ALL), [],
     _ann([_R_JG_LANE,_R_ADC_LANE,_R_SUP_LANE,_R_MID_LANE,_R_TOP_LANE], _ALL), [],
     _WARDS_FULL),
    ("free_setup_deaths", "Free Setup (Deaths)",
     "Team present  ·  Enemy absent  ·  Allied deaths in prior 60s", "#f59e0b",
     _ann([_B_JG_DRAG,_B_ADC_DRAG,_B_SUP_DRAG], _JAS),
     _ann([_B_MID_LANE,_B_TOP_LANE], _MT),
     _ann([_R_JG_LANE,_R_ADC_LANE,_R_SUP_LANE,_R_MID_LANE,_R_TOP_LANE], _ALL), [],
     _WARDS_THIN),
    ("clean_contest", "Clean Contest",
     "Both teams present  ·  Team not short-handed", "#a78bfa",
     _ann([_B_JG_DRAG,_B_ADC_DRAG,_B_SUP_DRAG,_B_MID_DRAG,_B_TOP_DRAG], _ALL), [],
     _ann([_R_JG_DRAG,_R_ADC_DRAG,_R_SUP_DRAG,_R_MID_DRAG,_R_TOP_DRAG], _ALL), [],
     _WARDS_ONE),
    ("disadvantaged", "Disadvantaged",
     "Both teams present  ·  Team had recent deaths or fewer alive", "#ef4444",
     _ann([_B_JG_DRAG,_B_ADC_DRAG,_B_SUP_DRAG], _JAS),
     _ann([_B_MID_DRAG,_B_TOP_DRAG], _MT),
     _ann([_R_JG_DRAG,_R_ADC_DRAG,_R_SUP_DRAG,_R_MID_DRAG,_R_TOP_DRAG], _ALL), [],
     _WARDS_NONE),
    ("gave_away", "Gave Away",
     "Enemy present at objective  ·  Team did not show up", "#64748b",
     _ann([_B_JG_LANE,_B_ADC_LANE,_B_SUP_LANE,_B_MID_LANE,_B_TOP_LANE], _ALL), [],
     _ann([_R_JG_DRAG,_R_ADC_DRAG,_R_SUP_DRAG,_R_MID_DRAG,_R_TOP_DRAG], _ALL), [],
     _WARDS_NONE),
    ("no_early_setup", "No Early Setup",
     "Neither team near objective at T-30  ·  Neither side committed", "#94a3b8",
     _ann([_B_JG_LANE,_B_ADC_LANE,_B_SUP_LANE,_B_MID_LANE,_B_TOP_LANE], _ALL), [],
     _ann([_R_JG_LANE,_R_ADC_LANE,_R_SUP_LANE,_R_MID_LANE,_R_TOP_LANE], _ALL), [],
     _WARDS_NONE),
]


def _to_px(annotated: list[tuple], w: int, h: int):
    if not annotated:
        return [], [], []
    xs    = [a[0][0] * w for a in annotated]
    ys    = [a[0][1] * h for a in annotated]
    roles = [a[1]        for a in annotated]
    return xs, ys, roles


def _darken_hex(h: str, f: float = 0.45) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16)
    return f"#{int(r*f):02x}{int(g*f):02x}{int(b*f):02x}"


def _glow(ax, xs, ys, color: str, edge: str, s: float = 270, zorder: int = 6) -> None:
    for mult, alpha in [(4.5, 0.04), (2.8, 0.09), (1.7, 0.17)]:
        ax.scatter(xs, ys, s=s*mult, color=color, alpha=alpha,
                   linewidths=0, zorder=zorder)
    ax.scatter(xs, ys, s=s, color=color, edgecolors=edge,
               linewidths=1.9, zorder=zorder + 1)


def _ghost(ax, xs, ys, color: str, s: float = 270, zorder: int = 6) -> None:
    ax.scatter(xs, ys, s=s * 2.0, color="none", edgecolors=color,
               linewidths=1.6, alpha=0.30, linestyle="--", zorder=zorder)
    ax.scatter(xs, ys, s=s * 0.55, marker="x", color=color,
               linewidths=2.2, alpha=0.42, zorder=zorder + 1)


def _role_badges(ax, xs, ys, roles, color: str) -> None:
    for x, y, role in zip(xs, ys, roles):
        ax.annotate(
            role, (x, y), xytext=(10, -10), textcoords="offset points",
            fontsize=6.2, color="white", fontweight="bold", zorder=22,
            bbox=dict(boxstyle="round,pad=0.20", facecolor="#0f172a",
                      edgecolor=color, linewidth=0.75, alpha=0.88),
        )


def _team_zone(ax, annotated, color: str, w: int, h: int) -> None:
    if len(annotated) < 3:
        return
    try:
        from scipy.spatial import ConvexHull
        pts = np.array([[a[0][0] * w, a[0][1] * h] for a in annotated])
        hull = ConvexHull(pts)
        v = np.append(hull.vertices, hull.vertices[0])
        ax.fill(pts[v, 0], pts[v, 1], alpha=0.09, color=color, zorder=3)
        ax.plot(pts[v, 0], pts[v, 1], alpha=0.22, color=color,
                linewidth=1.1, linestyle="--", zorder=4)
    except Exception:
        pass


def _obj_zone(ax, w: int, h: int) -> None:
    cx, cy = DRAGON[0] * w, DRAGON[1] * h
    for r_frac, alpha in [(0.108, 0.05), (0.062, 0.11)]:
        ax.add_patch(mpatches.Circle(
            (cx, cy), r_frac * w,
            facecolor="#fbbf24", alpha=alpha,
            edgecolor="#fbbf24", linewidth=0.9, linestyle="--", zorder=2,
        ))


def _ward_rings(ax, wards, w: int, h: int) -> None:
    for nx, ny in wards:
        x, y = nx * w, ny * h
        for r_frac, alpha in [(0.036, 0.08), (0.024, 0.16)]:
            ax.add_patch(mpatches.Circle(
                (x, y), r_frac * w,
                facecolor="none", edgecolor="#fde68a",
                linewidth=0.8, alpha=alpha, zorder=7,
            ))
        ax.scatter([x], [y], s=75, marker="D", color="#fde68a",
                   edgecolors="#d97706", linewidths=1.1, zorder=8)


def _draw(ax, img_dark, blue_alive, blue_dead,
          red_alive, red_dead, wards, title, subtitle, accent) -> None:
    h, w = img_dark.shape[:2]
    ax.imshow(img_dark)
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.set_aspect("equal"); ax.axis("off")

    _obj_zone(ax, w, h)
    _team_zone(ax, red_alive,  "#ef4444", w, h)
    _team_zone(ax, blue_alive, "#3b82f6", w, h)

    rxs, rys, rr = _to_px(red_alive, w, h)
    if rxs:
        _glow(ax, rxs, rys, "#ef4444", _darken_hex("#ef4444"), zorder=6)
        _role_badges(ax, rxs, rys, rr, "#ef4444")
    rxd, ryd, _ = _to_px(red_dead, w, h)
    if rxd:
        _ghost(ax, rxd, ryd, "#ef4444", zorder=6)

    bxs, bys, br = _to_px(blue_alive, w, h)
    if bxs:
        _glow(ax, bxs, bys, "#3b82f6", _darken_hex("#3b82f6"), zorder=7)
        _role_badges(ax, bxs, bys, br, "#3b82f6")
    bxd, byd, _ = _to_px(blue_dead, w, h)
    if bxd:
        _ghost(ax, bxd, byd, "#3b82f6", zorder=7)

    dx, dy = DRAGON[0] * w, DRAGON[1] * h
    for s, alpha in [(1400, 0.06), (800, 0.11)]:
        ax.scatter([dx], [dy], s=s, color="#fbbf24", alpha=alpha, linewidths=0, zorder=9)
    ax.scatter([dx], [dy], s=410, marker="*", color="#fbbf24",
               edgecolors="#92400e", linewidths=1.9, zorder=10)

    ax.set_title(title, fontsize=10.5, fontweight="bold", color="white", pad=5,
                 bbox=dict(boxstyle="square,pad=0.38", facecolor=accent, edgecolor="none"))
    ax.text(0.5, -0.022, subtitle, transform=ax.transAxes,
            fontsize=7.6, color="#475569", ha="center", va="top")


def fig_mapstates_v2() -> plt.Figure:
    raw = mpimg.imread(str(_MAP))
    # PNG returns float RGBA [0,1]; darken while preserving alpha channel
    img_dark = raw.copy()
    img_dark[..., :3] = np.clip(raw[..., :3] * 0.80, 0, 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 11.5))
    fig.patch.set_facecolor("#f8fafc")

    for ax, (_, title, subtitle, accent,
             ba, bd, ra, rd, wards) in zip(axes.flat, PROFILES):
        _draw(ax, img_dark, ba, bd, ra, rd, wards, title, subtitle, accent)

    legend_handles = [
        Line2D([0],[0], marker="o", color="none", markerfacecolor="#3b82f6",
               markersize=11, markeredgecolor="#1e3a8a", markeredgewidth=1.5,
               label="Blue (alive)"),
        Line2D([0],[0], marker="o", color="none", markerfacecolor="none",
               markersize=10, markeredgecolor="#3b82f6", markeredgewidth=1.5,
               alpha=0.45, label="Blue (dead — ghost ring)"),
        Line2D([0],[0], marker="o", color="none", markerfacecolor="#ef4444",
               markersize=11, markeredgecolor="#7f1d1d", markeredgewidth=1.5,
               label="Red (alive)"),
        Line2D([0],[0], marker="o", color="none", markerfacecolor="none",
               markersize=10, markeredgecolor="#ef4444", markeredgewidth=1.5,
               alpha=0.45, label="Red (dead — ghost ring)"),
        Line2D([0],[0], marker="*", color="none", markerfacecolor="#fbbf24",
               markersize=14, markeredgecolor="#92400e", markeredgewidth=1.5,
               label="Dragon"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=6, fontsize=9,
        frameon=True, framealpha=0.95, edgecolor="#e2e8f0",
        bbox_to_anchor=(0.5, 0.004),
    )
    fig.suptitle(
        "Setup Profile Map States  ·  T-30 snapshot  ·  synthetic positions\n"
        "Enhancements: glow markers  ·  role labels  ·  team zones  ·  objective radius",
        fontsize=12, fontweight="bold", color="#1e293b", y=0.998,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.995])
    fig.subplots_adjust(hspace=0.28)
    return fig


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=_OUTDIR)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    print("[mapstates_v2] generating enhanced map snapshots ...")
    fig = fig_mapstates_v2()
    out = args.out / "17b_mapstates_glow_zones.png"
    fig.savefig(str(out), dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[mapstates_v2] saved: {out}")


if __name__ == "__main__":
    main()
