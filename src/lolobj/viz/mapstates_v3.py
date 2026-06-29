"""Mapstate visualization — V3 (Pillow / PIL image compositing).

Technique: pixel-level rendering via Pillow. All marks are drawn directly
onto RGBA image layers and composited, giving a distinct look from the
matplotlib vector-graphics approach of V1/V2.

Aesthetic: dark-tinted map with soft-glow player icons — esports broadcast style.
  • Map darkened to ~60% brightness with a subtle cool tint
  • Glow via Gaussian-blurred RGBA layers composited before solid circles
  • Neon color palette: sky-blue team vs hot-red team
  • Dead players shown as dim X marks on ghost rings
  • Ward: diamond with blurred gold glow
  • Dragon: orange starburst with concentric glow layers
  • Role badges rendered using system font (Calibri / Arial / fallback)
  • Six panels assembled into a 3×2 grid with colored title strips

Usage:
    python -m lolobj.viz.mapstates_v3
    python -m lolobj.viz.mapstates_v3 --out <dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageEnhance

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

def _ann(pos: list, roles: list[str]) -> list[tuple]:
    return list(zip(pos, roles))

_ALL = ["JG", "ADC", "SUP", "MID", "TOP"]
_JAS = ["JG", "ADC", "SUP"]
_MT  = ["MID", "TOP"]

PROFILES: list[tuple] = [
    ("Free Setup",
     "Team present  ·  Enemy absent  ·  No recent allied deaths", "#10b981",
     _ann([_B_JG_DRAG,_B_ADC_DRAG,_B_SUP_DRAG,_B_MID_DRAG,_B_TOP_DRAG], _ALL), [],
     _ann([_R_JG_LANE,_R_ADC_LANE,_R_SUP_LANE,_R_MID_LANE,_R_TOP_LANE], _ALL), [],
     _WARDS_FULL),
    ("Free Setup (Deaths)",
     "Team present  ·  Enemy absent  ·  Allied deaths in prior 60s", "#f59e0b",
     _ann([_B_JG_DRAG,_B_ADC_DRAG,_B_SUP_DRAG], _JAS),
     _ann([_B_MID_LANE,_B_TOP_LANE], _MT),
     _ann([_R_JG_LANE,_R_ADC_LANE,_R_SUP_LANE,_R_MID_LANE,_R_TOP_LANE], _ALL), [],
     _WARDS_THIN),
    ("Clean Contest",
     "Both teams present  ·  Team not short-handed", "#a78bfa",
     _ann([_B_JG_DRAG,_B_ADC_DRAG,_B_SUP_DRAG,_B_MID_DRAG,_B_TOP_DRAG], _ALL), [],
     _ann([_R_JG_DRAG,_R_ADC_DRAG,_R_SUP_DRAG,_R_MID_DRAG,_R_TOP_DRAG], _ALL), [],
     _WARDS_ONE),
    ("Disadvantaged",
     "Both teams present  ·  Team had recent deaths or fewer alive", "#ef4444",
     _ann([_B_JG_DRAG,_B_ADC_DRAG,_B_SUP_DRAG], _JAS),
     _ann([_B_MID_DRAG,_B_TOP_DRAG], _MT),
     _ann([_R_JG_DRAG,_R_ADC_DRAG,_R_SUP_DRAG,_R_MID_DRAG,_R_TOP_DRAG], _ALL), [],
     _WARDS_NONE),
    ("Gave Away",
     "Enemy present at objective  ·  Team did not show up", "#64748b",
     _ann([_B_JG_LANE,_B_ADC_LANE,_B_SUP_LANE,_B_MID_LANE,_B_TOP_LANE], _ALL), [],
     _ann([_R_JG_DRAG,_R_ADC_DRAG,_R_SUP_DRAG,_R_MID_DRAG,_R_TOP_DRAG], _ALL), [],
     _WARDS_NONE),
    ("No Early Setup",
     "Neither team near objective at T-30  ·  Neither side committed", "#94a3b8",
     _ann([_B_JG_LANE,_B_ADC_LANE,_B_SUP_LANE,_B_MID_LANE,_B_TOP_LANE], _ALL), [],
     _ann([_R_JG_LANE,_R_ADC_LANE,_R_SUP_LANE,_R_MID_LANE,_R_TOP_LANE], _ALL), [],
     _WARDS_NONE),
]

# Neon palette
_BLUE_C = (30, 144, 255)    # dodger blue
_RED_C  = (255, 50,  80)    # vivid red
_DEAD_C = (110, 120, 140)   # muted slate
_WARD_C = (255, 215,  50)   # gold
_DRAG_C = (255, 165,  10)   # orange-gold
_DARK   = (15,  23,  42)    # near-black bg


def _hex2rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16)


def _get_fonts() -> tuple:
    """Return (font_title, font_role, font_sub) using system fonts if available."""
    paths = [
        "C:/Windows/Fonts/calibrib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/verdanab.ttf",
    ]
    for path in paths:
        try:
            return (
                ImageFont.truetype(path, 18),
                ImageFont.truetype(path, 12),
                ImageFont.truetype(path.replace("b.ttf", ".ttf").replace("bd.ttf", ".ttf"), 12),
            )
        except OSError:
            continue
    default = ImageFont.load_default()
    return default, default, default


def _blur_glow(size: tuple, positions: list[tuple[int, int]],
               color: tuple[int, int, int], r: int, alpha: int = 160) -> Image.Image:
    """Gaussian-blurred glow layer for a set of positions."""
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    draw  = ImageDraw.Draw(layer)
    for cx, cy in positions:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color + (alpha,))
    return layer.filter(ImageFilter.GaussianBlur(radius=r * 0.75))


def _solid_circle(draw: ImageDraw.Draw, cx: int, cy: int, r: int,
                  fill: tuple, edge: tuple, edge_w: int = 3) -> None:
    draw.ellipse([cx-r, cy-r, cx+r, cy+r],
                 fill=fill + (255,), outline=edge + (200,), width=edge_w)
    # inner highlight
    hi_r = max(2, r // 3)
    draw.ellipse([cx - hi_r, cy - r + 3, cx + hi_r, cy - r + 3 + hi_r * 2],
                 fill=(255, 255, 255, 80))


def _dead_mark(draw: ImageDraw.Draw, cx: int, cy: int, r: int) -> None:
    draw.ellipse([cx-r, cy-r, cx+r, cy+r],
                 fill=None, outline=_DEAD_C + (110,), width=2)
    m = int(r * 0.55)
    draw.line([(cx-m, cy-m), (cx+m, cy+m)], fill=_DEAD_C + (140,), width=2)
    draw.line([(cx+m, cy-m), (cx-m, cy+m)], fill=_DEAD_C + (140,), width=2)


def _ward_mark(draw: ImageDraw.Draw, glow_layer: Image.Image,
               cx: int, cy: int, r: int, size: tuple) -> None:
    # Draw glow onto layer directly
    gd = ImageDraw.Draw(glow_layer)
    gd.ellipse([cx-r*2, cy-r*2, cx+r*2, cy+r*2], fill=_WARD_C + (120,))
    # Solid diamond
    pts = [(cx, cy-r), (cx+r, cy), (cx, cy+r), (cx-r, cy)]
    draw.polygon(pts, fill=_WARD_C + (230,))


def _dragon_mark(draw: ImageDraw.Draw, cx: int, cy: int) -> None:
    r = 14
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=_DRAG_C + (255,))
    # Star points
    for angle in range(0, 360, 45):
        rad = np.radians(angle)
        px = int(cx + np.cos(rad) * r * 1.8)
        py = int(cy + np.sin(rad) * r * 1.8)
        draw.line([(cx, cy), (px, py)], fill=_DRAG_C + (160,), width=2)
    # White centre dot
    draw.ellipse([cx-4, cy-4, cx+4, cy+4], fill=(255, 255, 255, 200))


def _role_badge(draw: ImageDraw.Draw, cx: int, cy: int,
                role: str, team_color: tuple, font) -> None:
    ox, oy = cx + 14, cy - 14
    try:
        bbox = draw.textbbox((ox, oy), role, font=font)
        pad = 3
        draw.rectangle([bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad],
                        fill=_DARK + (200,))
        draw.text((ox, oy), role, fill=team_color + (230,), font=font)
    except Exception:
        draw.text((ox, oy), role, fill=team_color + (220,))


def _obj_radius(canvas: Image.Image, cx: int, cy: int, r: int) -> Image.Image:
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw  = ImageDraw.Draw(layer)
    draw.ellipse([cx-r, cy-r, cx+r, cy+r],
                 fill=_DRAG_C + (18,), outline=_DRAG_C + (50,), width=2)
    draw.ellipse([cx-int(r*1.7), cy-int(r*1.7), cx+int(r*1.7), cy+int(r*1.7)],
                 fill=None, outline=_DRAG_C + (22,), width=1)
    return Image.alpha_composite(canvas, layer)


def _make_panel(map_dark: Image.Image,
                title: str, subtitle: str, accent: str,
                blue_alive: list, blue_dead: list,
                red_alive:  list, red_dead:  list,
                wards: list,
                font_title, font_role, font_sub,
                title_h: int = 38) -> Image.Image:

    w, h = map_dark.size
    canvas: Image.Image = map_dark.convert("RGBA")

    def px(pos): return int(pos[0] * w), int(pos[1] * h)

    # ── glow layers ──────────────────────────────────────────────────────────
    alive_r = [px(a[0]) for a in red_alive]
    alive_b = [px(a[0]) for a in blue_alive]
    drag_px = px(DRAGON)

    if alive_r:
        canvas = Image.alpha_composite(canvas, _blur_glow(canvas.size, alive_r, _RED_C, r=24))
    if alive_b:
        canvas = Image.alpha_composite(canvas, _blur_glow(canvas.size, alive_b, _BLUE_C, r=24))
    canvas = Image.alpha_composite(canvas,
                _blur_glow(canvas.size, [drag_px], _DRAG_C, r=30, alpha=150))

    # Objective radius circle
    canvas = _obj_radius(canvas, drag_px[0], drag_px[1], r=int(0.065 * w))

    draw = ImageDraw.Draw(canvas)

    # ── dead players ──────────────────────────────────────────────────────────
    for pos, _ in red_dead:
        _dead_mark(draw, *px(pos), r=10)
    for pos, _ in blue_dead:
        _dead_mark(draw, *px(pos), r=10)

    # ── alive circles ─────────────────────────────────────────────────────────
    red_edge  = (140, 10, 30)
    blue_edge = (10,  50, 140)
    for pos, _ in red_alive:
        _solid_circle(draw, *px(pos), r=14, fill=_RED_C, edge=red_edge)
    for pos, _ in blue_alive:
        _solid_circle(draw, *px(pos), r=14, fill=_BLUE_C, edge=blue_edge)

    # ── dragon ────────────────────────────────────────────────────────────────
    _dragon_mark(draw, drag_px[0], drag_px[1])

    # ── role labels (second pass, on top) ─────────────────────────────────────
    for pos, role in red_alive:
        _role_badge(draw, *px(pos), role, _RED_C, font_role)
    for pos, role in blue_alive:
        _role_badge(draw, *px(pos), role, _BLUE_C, font_role)

    # ── title strip ───────────────────────────────────────────────────────────
    result_rgb = canvas.convert("RGB")
    title_strip = Image.new("RGB", (w, title_h), _hex2rgb(accent))
    td = ImageDraw.Draw(title_strip)
    try:
        bbox = td.textbbox((0, 0), title, font=font_title)
        tx = (w - (bbox[2] - bbox[0])) // 2
        ty = (title_h - (bbox[3] - bbox[1])) // 2
        td.text((tx, ty), title, fill=(255, 255, 255), font=font_title)
    except Exception:
        td.text((8, 10), title, fill=(255, 255, 255))

    # ── subtitle strip ────────────────────────────────────────────────────────
    sub_h = 28
    sub_strip = Image.new("RGB", (w, sub_h), (248, 250, 252))
    sd = ImageDraw.Draw(sub_strip)
    try:
        bbox = sd.textbbox((0, 0), subtitle, font=font_sub)
        sx = (w - (bbox[2] - bbox[0])) // 2
        sy = (sub_h - (bbox[3] - bbox[1])) // 2
        sd.text((sx, sy), subtitle, fill=(71, 85, 105), font=font_sub)
    except Exception:
        sd.text((8, 8), subtitle, fill=(71, 85, 105))

    # ── assemble panel ────────────────────────────────────────────────────────
    panel = Image.new("RGB", (w, title_h + h + sub_h), (248, 250, 252))
    panel.paste(title_strip, (0, 0))
    panel.paste(result_rgb,  (0, title_h))
    panel.paste(sub_strip,   (0, title_h + h))
    return panel


def _legend_strip(w: int, h: int = 52) -> Image.Image:
    strip = Image.new("RGB", (w, h), (248, 250, 252))
    draw  = ImageDraw.Draw(strip)
    items = [
        (_BLUE_C, "Blue team (alive)"),
        (_RED_C,  "Red team (alive)"),
        (_DEAD_C, "Dead (ghost ring)"),
        (_DRAG_C, "Dragon"),
    ]
    x = 30
    for color, label in items:
        draw.ellipse([x, h//2-8, x+16, h//2+8], fill=color + (255,), outline=None)
        x += 22
        try:
            draw.text((x, h//2-8), label, fill=(30, 41, 59))
        except Exception:
            pass
        x += len(label) * 7 + 20
    return strip


def _title_banner(w: int, h: int = 52) -> Image.Image:
    banner = Image.new("RGB", (w, h), (30, 41, 59))
    draw   = ImageDraw.Draw(banner)
    text   = ("Setup Profile Map States  ·  T-30 snapshot  ·  synthetic positions  "
               "·  V3: Pillow glow compositing")
    try:
        draw.text((w // 2, h // 2), text, fill=(226, 232, 240), anchor="mm")
    except Exception:
        draw.text((16, 16), text, fill=(226, 232, 240))
    return banner


def fig_mapstates_v3() -> Image.Image:
    raw = Image.open(str(_MAP)).convert("RGBA")
    # Darken and apply a very subtle cool tint
    r_arr, g_arr, b_arr, a_arr = raw.split()
    r_arr = r_arr.point(lambda p: int(p * 0.62))
    g_arr = g_arr.point(lambda p: int(p * 0.62))
    b_arr = b_arr.point(lambda p: int(p * 0.70))   # slightly more blue
    map_dark = Image.merge("RGBA", (r_arr, g_arr, b_arr, a_arr))

    font_title, font_role, font_sub = _get_fonts()

    panels = []
    for title, subtitle, accent, ba, bd, ra, rd, wards in PROFILES:
        panel = _make_panel(
            map_dark, title, subtitle, accent,
            ba, bd, ra, rd, wards,
            font_title, font_role, font_sub,
        )
        panels.append(panel)

    pw, ph = panels[0].size
    gap  = 10
    cols = 3
    rows = 2

    top_h = 56
    leg_h = 52
    total_w = cols * pw + (cols - 1) * gap + 40
    total_h = top_h + rows * ph + (rows - 1) * gap + leg_h + 20

    canvas = Image.new("RGB", (total_w, total_h), (240, 242, 247))

    title_banner = _title_banner(total_w, h=top_h)
    canvas.paste(title_banner, (0, 0))

    for i, panel in enumerate(panels):
        row, col = divmod(i, cols)
        x = 20 + col * (pw + gap)
        y = top_h + 10 + row * (ph + gap)
        canvas.paste(panel, (x, y))

    legend = _legend_strip(total_w, h=leg_h)
    canvas.paste(legend, (0, total_h - leg_h))
    return canvas


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=_OUTDIR)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    print("[mapstates_v3] generating neon-HUD map snapshots ...")
    img = fig_mapstates_v3()
    out = args.out / "17c_mapstates_neon_hud.png"
    img.save(str(out), dpi=(180, 180))
    print(f"[mapstates_v3] saved: {out}")


if __name__ == "__main__":
    main()
