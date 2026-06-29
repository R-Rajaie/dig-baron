# =====================================================================
#  Setup Profile Map States — V4 (R / ggplot2 + patchwork)
#
#  Technique: ggplot2 + annotation_raster (png package) + patchwork + ggrepel
#  Aesthetic: clean academic / data-viz style matching the project theme_lol.
#    • Map rendered via annotation_raster with slight brightness reduction
#    • Team territory zones via base-R chull() → geom_polygon
#    • Player circles via geom_point (shape=21, filled with border)
#    • Role labels via ggrepel::geom_label_repel (anti-overlap, leader lines)
#    • Dead players: transparent X marks (shape=4)
#    • Wards: diamond shapes (shape=18)
#    • Dragon: star (shape=8)
#    • Objective influence radius: annotate circle
#    • 6 panels combined with patchwork; each has a colored left-border accent
#
#  Output: exports/charts/17d_mapstates_ggplot2.png
# =====================================================================

.pkgs <- c("png", "grid", "ggplot2", "ggrepel", "patchwork", "scales")
.new  <- .pkgs[!sapply(.pkgs, requireNamespace, quietly = TRUE)]
if (length(.new)) install.packages(.new, repos = "https://cloud.r-project.org")
rm(.pkgs, .new)

suppressPackageStartupMessages({
  library(png); library(grid)
  library(ggplot2); library(ggrepel); library(patchwork); library(scales)
})

ROOT     <- "C:/Users/Laser/dig-baron"
MAP_PATH <- file.path(ROOT, "data/raw/rift map.png")
OUT_PATH <- file.path(ROOT, "exports/charts/17e_mapstates_ggplot2_dark.png")

# ── load and prepare map image ────────────────────────────────────────────────
# readPNG gives (row, col, channel) with row=1 at TOP; ggplot y=0 at BOTTOM.
# We flip the rows so annotation_raster with ymin=0,ymax=1 matches ggplot coords.
map_raw  <- readPNG(MAP_PATH)
map_dark <- map_raw * 0.60                          # darken to 60% for dark theme
map_r    <- map_dark                               # no row-flip: annotation_raster + gg() handle orientation

# ── coordinate helpers ────────────────────────────────────────────────────────
# Our normalised coords: nx in [0,1], ny in [0,1] where ny=0 is TOP of image.
# After flipping, ggplot y=1 corresponds to image top → use y_gg = 1 - ny.
gg <- function(ny) 1 - ny

# ── position constants ────────────────────────────────────────────────────────
B_JG_DRAG  <- c(0.648, gg(0.690)); B_ADC_DRAG <- c(0.742, gg(0.770))
B_SUP_DRAG <- c(0.708, gg(0.740)); B_MID_DRAG <- c(0.606, gg(0.674))
B_TOP_DRAG <- c(0.633, gg(0.728))
R_JG_DRAG  <- c(0.712, gg(0.606)); R_ADC_DRAG <- c(0.682, gg(0.580))
R_SUP_DRAG <- c(0.660, gg(0.624)); R_MID_DRAG <- c(0.610, gg(0.568))
R_TOP_DRAG <- c(0.748, gg(0.656))
B_JG_LANE  <- c(0.338, gg(0.595)); B_ADC_LANE <- c(0.800, gg(0.848))
B_SUP_LANE <- c(0.745, gg(0.900)); B_MID_LANE <- c(0.460, gg(0.622))
B_TOP_LANE <- c(0.128, gg(0.298))
R_JG_LANE  <- c(0.742, gg(0.512)); R_ADC_LANE <- c(0.858, gg(0.872))
R_SUP_LANE <- c(0.828, gg(0.912)); R_MID_LANE <- c(0.538, gg(0.448))
R_TOP_LANE <- c(0.175, gg(0.192))

DRAGON_X <- 0.666;  DRAGON_Y <- gg(0.703)

WARDS_FULL <- list(c(0.700, gg(0.645)), c(0.758, gg(0.618)), c(0.612, gg(0.758)))
WARDS_THIN <- list(c(0.700, gg(0.645)), c(0.758, gg(0.618)))
WARDS_ONE  <- list(c(0.700, gg(0.645)))
WARDS_NONE <- list()

# ── helper: build player data frame ──────────────────────────────────────────
make_df <- function(pos_list, role_vec, team, status) {
  if (length(pos_list) == 0) return(NULL)
  data.frame(
    x      = vapply(pos_list, `[[`, 0.0, 1),
    y      = vapply(pos_list, `[[`, 0.0, 2),
    role   = role_vec,
    team   = team,
    status = status,
    stringsAsFactors = FALSE
  )
}

# ── helper: convex hull polygon data frame ────────────────────────────────────
hull_df <- function(df) {
  if (is.null(df) || nrow(df) < 3) return(NULL)
  hi <- chull(df$x, df$y)
  df[c(hi, hi[1]), ]
}

# ── theme ─────────────────────────────────────────────────────────────────────
INK  <- "#e2e8f0"; SUB <- "#94a3b8"
BG   <- "#0f172a"; BG2  <- "#1e293b"

theme_mapstate <- function(accent = "#64748b") {
  theme_void(base_size = 12) +
    theme(
      plot.title        = element_text(face = "bold", colour = accent,
                                       size = rel(1.0), hjust = 0.5,
                                       margin = margin(8, 0, 4, 0)),
      plot.subtitle     = element_text(colour = SUB, size = rel(0.70),
                                       hjust = 0.5,
                                       margin = margin(t = 2, b = 4)),
      plot.background   = element_rect(fill = BG,
                                       colour = accent, linewidth = 2.5),
      panel.background  = element_rect(fill = NA, colour = NA),
      plot.margin       = margin(2, 2, 4, 2),
      legend.position   = "none",
      plot.title.position = "plot",
    )
}

# ── panel builder ─────────────────────────────────────────────────────────────
make_panel <- function(title, subtitle, accent,
                       ba_list, ba_roles,
                       bd_list = list(), bd_roles = character(0),
                       ra_list, ra_roles,
                       rd_list = list(), rd_roles = character(0),
                       wards   = list()) {

  ba_df <- make_df(ba_list, ba_roles, "blue", "alive")
  bd_df <- make_df(bd_list, bd_roles, "blue", "dead")
  ra_df <- make_df(ra_list, ra_roles, "red",  "alive")
  rd_df <- make_df(rd_list, rd_roles, "red",  "dead")

  ward_df <- if (length(wards) > 0)
    data.frame(x = vapply(wards, `[[`, 0.0, 1),
               y = vapply(wards, `[[`, 0.0, 2))
  else NULL

  p <- ggplot() +
    annotation_raster(map_r, xmin = 0, xmax = 1, ymin = 0, ymax = 1,
                      interpolate = TRUE) +
    # objective influence radius — subtle glow only
    annotate("point", x = DRAGON_X, y = DRAGON_Y,
             size = 16, colour = "#fbbf24", alpha = 0.06) +
    annotate("point", x = DRAGON_X, y = DRAGON_Y,
             size = 10, colour = "#fbbf24", alpha = 0.10) +
    coord_fixed(ratio = 1, xlim = c(0, 1), ylim = c(0, 1), expand = FALSE)

  # Wards
  if (!is.null(ward_df))
    p <- p + geom_point(data = ward_df, aes(x = x, y = y),
                        shape = 18, size = 4.5,
                        colour = "#fde68a", stroke = 0)

  # Dead players — transparent X
  if (!is.null(bd_df))
    p <- p + geom_point(data = bd_df, aes(x = x, y = y),
                        shape = 4, size = 6, colour = "#93c5fd",
                        stroke = 2.0, alpha = 0.50)
  if (!is.null(rd_df))
    p <- p + geom_point(data = rd_df, aes(x = x, y = y),
                        shape = 4, size = 6, colour = "#fca5a5",
                        stroke = 2.0, alpha = 0.50)

  # Alive players
  if (!is.null(ra_df))
    p <- p + geom_point(data = ra_df, aes(x = x, y = y),
                        shape = 21, size = 8,
                        fill = "#ef4444", colour = "#7f1d1d", stroke = 1.8)
  if (!is.null(ba_df))
    p <- p + geom_point(data = ba_df, aes(x = x, y = y),
                        shape = 21, size = 8,
                        fill = "#3b82f6", colour = "#1e3a8a", stroke = 1.8)

  # Dragon — small gold star
  p <- p +
    annotate("text", x = DRAGON_X, y = DRAGON_Y, label = "★",
             size = 5, colour = "#fbbf24") +
    labs(title = title, subtitle = subtitle) +
    theme_mapstate(accent = accent)

  p
}

# ── define all six profiles ───────────────────────────────────────────────────
ALL <- list(B_JG_DRAG,B_ADC_DRAG,B_SUP_DRAG,B_MID_DRAG,B_TOP_DRAG)
ALL_R <- c("JG","ADC","SUP","MID","TOP")

p1 <- make_panel(
  "Free Setup",
  "Team present  ·  Enemy absent  ·  No recent allied deaths", "#10b981",
  ba_list  = ALL, ba_roles = ALL_R,
  ra_list  = list(R_JG_LANE,R_ADC_LANE,R_SUP_LANE,R_MID_LANE,R_TOP_LANE),
  ra_roles = ALL_R,
  wards    = WARDS_FULL)

p2 <- make_panel(
  "Free Setup (Deaths)",
  "Team present  ·  Enemy absent  ·  Allied deaths in prior 60s", "#f59e0b",
  ba_list  = list(B_JG_DRAG,B_ADC_DRAG,B_SUP_DRAG), ba_roles = c("JG","ADC","SUP"),
  bd_list  = list(B_MID_LANE,B_TOP_LANE),            bd_roles = c("MID","TOP"),
  ra_list  = list(R_JG_LANE,R_ADC_LANE,R_SUP_LANE,R_MID_LANE,R_TOP_LANE),
  ra_roles = ALL_R,
  wards    = WARDS_THIN)

p3 <- make_panel(
  "Clean Contest",
  "Both teams present  ·  Team not short-handed", "#a78bfa",
  ba_list  = ALL, ba_roles = ALL_R,
  ra_list  = list(R_JG_DRAG,R_ADC_DRAG,R_SUP_DRAG,R_MID_DRAG,R_TOP_DRAG),
  ra_roles = ALL_R,
  wards    = WARDS_ONE)

p4 <- make_panel(
  "Disadvantaged",
  "Both teams present  ·  Team had recent deaths or fewer alive", "#ef4444",
  ba_list  = list(B_JG_DRAG,B_ADC_DRAG,B_SUP_DRAG), ba_roles = c("JG","ADC","SUP"),
  bd_list  = list(B_MID_DRAG,B_TOP_DRAG),            bd_roles = c("MID","TOP"),
  ra_list  = list(R_JG_DRAG,R_ADC_DRAG,R_SUP_DRAG,R_MID_DRAG,R_TOP_DRAG),
  ra_roles = ALL_R,
  wards    = WARDS_NONE)

p5 <- make_panel(
  "Gave Away",
  "Enemy present at objective  ·  Team did not show up", "#64748b",
  ba_list  = list(B_JG_LANE,B_ADC_LANE,B_SUP_LANE,B_MID_LANE,B_TOP_LANE),
  ba_roles = ALL_R,
  ra_list  = list(R_JG_DRAG,R_ADC_DRAG,R_SUP_DRAG,R_MID_DRAG,R_TOP_DRAG),
  ra_roles = ALL_R,
  wards    = WARDS_NONE)

p6 <- make_panel(
  "No Early Setup",
  "Neither team near objective at T-30  ·  Neither side committed", "#94a3b8",
  ba_list  = list(B_JG_LANE,B_ADC_LANE,B_SUP_LANE,B_MID_LANE,B_TOP_LANE),
  ba_roles = ALL_R,
  ra_list  = list(R_JG_LANE,R_ADC_LANE,R_SUP_LANE,R_MID_LANE,R_TOP_LANE),
  ra_roles = ALL_R,
  wards    = WARDS_NONE)

# ── legend panel ──────────────────────────────────────────────────────────────
legend_data <- data.frame(
  x      = c(0.5, 3.5, 6.5),
  y      = rep(0.5, 3),
  label  = c("Blue (alive)", "Red (alive)", "Dead (X)"),
  color  = c("#3b82f6", "#ef4444", "#93c5fd"),
  shape  = c(21L, 21L, 4L),
  fill   = c("#3b82f6", "#ef4444", "#93c5fd"),
  stringsAsFactors = FALSE
)

p_legend <- ggplot(legend_data, aes(x = x, y = y)) +
  geom_point(aes(colour = I(color), shape = I(shape), fill = I(fill)),
             size = 5, stroke = 1.5) +
  geom_text(aes(label = label), nudge_y = -0.35, size = 3.2,
            colour = INK, fontface = "bold", vjust = 1) +
  annotate("text", x = 9.0, y = 0.5, label = "★", size = 5, colour = "#fbbf24") +
  annotate("text", x = 9.0, y = 0.15, label = "Dragon",
           size = 3.2, colour = INK, fontface = "bold", vjust = 1) +
  scale_shape_identity() +
  coord_cartesian(xlim = c(-0.5, 11), ylim = c(-0.2, 1.2), clip = "off") +
  theme_void() +
  theme(plot.margin = margin(4, 8, 8, 8),
        plot.background = element_rect(fill = BG, colour = NA))

# ── assemble ──────────────────────────────────────────────────────────────────
layout <- (p1 | p2 | p3) / (p4 | p5 | p6) / p_legend +
  plot_layout(heights = c(1, 1, 0.14)) +
  plot_annotation(
    title    = "Setup Profile Map States  ·  T-30 snapshot  ·  synthetic positions",
    subtitle = "V4: R / ggplot2 + patchwork  ·  objective radius  ·  synthetic positions",
    caption  = "Blue circles = blue team alive  ·  Red circles = red team alive  ·  ✕ = dead  ·  ★ = dragon spawn",
    theme    = theme(
      plot.title    = element_text(face = "bold", colour = INK, size = 14,
                                   margin = margin(b = 4)),
      plot.subtitle = element_text(colour = SUB, size = 10,
                                   margin = margin(b = 8)),
      plot.caption  = element_text(colour = "#475569", size = 8,
                                   hjust = 0, margin = margin(t = 6)),
      plot.background = element_rect(fill = BG, colour = NA),
      plot.margin     = margin(16, 16, 10, 16),
    )
  )

# ── save ──────────────────────────────────────────────────────────────────────
if (requireNamespace("ragg", quietly = TRUE)) {
  ggsave(OUT_PATH, layout, width = 15, height = 12.5, dpi = 180,
         device = ragg::agg_png, bg = BG)
} else {
  ggsave(OUT_PATH, layout, width = 15, height = 12.5, dpi = 180, bg = BG)
}

cat("Saved:", OUT_PATH, "\n")
