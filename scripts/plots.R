# =====================================================================
#  "Let's Just Flip It" — remade figures (ggplot2)
#  Reads objective_windows.parquet, reconstructs the derived buckets,
#  and renders all 9 figures to remake/figures/*.png
# =====================================================================

# ----- package bootstrap ---------------------------------------------------
.pkgs <- c("arrow","dplyr","tidyr","forcats","stringr",
           "ggplot2","ggrepel","ggforce","scales","readr","patchwork")
.new  <- .pkgs[!sapply(.pkgs, requireNamespace, quietly = TRUE)]
if (length(.new)) install.packages(.new, repos = "https://cloud.r-project.org")
rm(.pkgs, .new)

suppressPackageStartupMessages({
  library(arrow); library(dplyr); library(tidyr); library(forcats)
  library(stringr); library(ggplot2); library(ggrepel); library(ggforce)
  library(scales); library(readr); library(patchwork)
})

ROOT <- "C:/Users/Laser/dig-baron/data/processed"
FIG  <- file.path(ROOT, "remake", "figures")
dir.create(FIG, showWarnings = FALSE, recursive = TRUE)

# ----- look & feel ----------------------------------------------------
BG   <- "#0f172a"
INK  <- "#e2e8f0"; SUB <- "#94a3b8"; GRID <- "#334155"
theme_lol <- function(base = 16) {
  theme_minimal(base_size = base, base_family = "sans") +
    theme(
      plot.title       = element_text(face = "bold", colour = INK, size = rel(1.30),
                                      margin = margin(b = 2)),
      plot.subtitle    = element_text(colour = SUB, size = rel(0.92),
                                      margin = margin(b = 14), lineheight = 1.05),
      plot.caption     = element_text(colour = "#64748b", size = rel(0.70),
                                      hjust = 0, margin = margin(t = 14)),
      axis.title       = element_text(colour = "#cbd5e1", size = rel(0.92)),
      axis.title.x     = element_text(margin = margin(t = 9)),
      axis.title.y     = element_text(margin = margin(r = 9)),
      axis.text        = element_text(colour = "#94a3b8"),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = GRID, linewidth = 0.45),
      plot.title.position   = "plot",
      plot.caption.position = "plot",
      legend.title     = element_text(size = rel(0.82), colour = "#cbd5e1"),
      legend.text      = element_text(size = rel(0.80), colour = "#94a3b8"),
      strip.text       = element_text(face = "bold", colour = INK, size = rel(0.92)),
      plot.margin      = margin(18, 20, 12, 16),
      plot.background  = element_rect(fill = BG, colour = NA),
      panel.background = element_rect(fill = BG, colour = NA),
      legend.background = element_rect(fill = BG, colour = NA),
      legend.key       = element_rect(fill = BG, colour = NA)
    )
}
# secure-rate diverging scale, anchored 0-100 with neutral at 50
RYG <- c("#c0392b", "#e8743b", "#f4d35e", "#86c06c", "#2e8b57")
fill_secure  <- function(...) scale_fill_gradientn (colours = RYG, limits = c(0,100),
                                                    name = "Secure %", ...)
col_secure   <- function(...) scale_colour_gradientn(colours = RYG, limits = c(0,100),
                                                    name = "Secure %", ...)
prof_levels <- c("free setup","free setup deaths","clean contest",
                 "neither team setup","disadvantaged","no setup")
pal_profile <- c("free setup"="#2e8b57","free setup deaths"="#86c06c",
                 "clean contest"="#ccb14e","neither team setup"="#8c95a3",
                 "disadvantaged"="#e8743b","no setup"="#c0392b")
pal_fight  <- c("won" = "#2e8b57", "draw" = "#64748b", "lost" = "#c0392b", "none" = "#334155")
fight_levels <- c("won", "draw", "lost", "none")
outcome_levels <- c(
  "uncontested take","steal","contested take",
  "gave, got trade","clean give",
  "contested loss + trade","contested loss")
pal_outcome <- c(
  "uncontested take"       = "#2e8b57",
  "steal"                  = "#0891b2",
  "contested take"         = "#22c55e",
  "clean give"             = "#8c95a3",
  "gave, got trade"        = "#7fb3d3",
  "contested loss + trade" = "#e8743b",
  "contested loss"         = "#c0392b"
)
pal_gold <- c("big behind"="#c0392b","behind"="#e8743b","even"="#9aa0a6",  # behind=red, ahead=green
              "ahead"="#5aa469","big ahead"="#2e8b57")
pal_alive <- c("Numbers down"="#E69F00","Even numbers"="#0072B2","Numbers up"="#009E73")
CAP <- NULL
desir <- function(secure, net)                                    # equal-weight blend
  as.numeric(scale(secure)) + as.numeric(scale(net))              # of secure rate + net value
pretty_feat <- function(x) {                                       # SHAP feature names
  x <- gsub("_T_(\\d+)", " T-\\1", x)
  x <- gsub("_(\\d+)s", " \\1s", x)
  gsub("_", " ", x)
}

savep <- function(p, file, w, h) {
  if (requireNamespace("ragg", quietly = TRUE))
    ggsave(file.path(FIG, file), p, width = w, height = h, dpi = 200,
           device = ragg::agg_png, bg = BG)
  else
    ggsave(file.path(FIG, file), p, width = w, height = h, dpi = 200, bg = BG)
}

wilson <- function(k, n, z = 1.96) {           # returns pct lo/hi
  p <- k / n; d <- 1 + z^2/n
  c <- p + z^2/(2*n); h <- z * sqrt(p*(1-p)/n + z^2/(4*n^2))
  list(lo = 100*(c - h)/d, hi = 100*(c + h)/d)
}

# ----- load & derive --------------------------------------------------
need <- c("secured","fight_result","objective_result","good_trade","steal","net_value",
          "objective_time_ms","team_nearby_T_30","enemy_nearby_T_30","team_deaths_60s",
          "team_alive_T_30","enemy_alive_T_30","team_alive_T_60","enemy_alive_T_60",
          "arrived_first","support_alive_T_30","jungler_alive_T_30","gold_diff_T_60")
dat <- read_parquet(file.path(ROOT, "objective_windows.parquet"), col_select = all_of(need)) |>
  mutate(
    tp = team_nearby_T_30 >= 1, ep = enemy_nearby_T_30 >= 1,
    deaths = team_deaths_60s > 0,
    alive_ge = (team_alive_T_30 - enemy_alive_T_30) >= 0,
    setup_profile = case_when(
      tp & !ep & !deaths              ~ "free setup",
      tp & !ep &  deaths              ~ "free setup deaths",
      !tp &  ep                       ~ "no setup",
      !tp & !ep                       ~ "neither team setup",
      tp &  ep &  alive_ge & !deaths  ~ "clean contest",
      TRUE                            ~ "disadvantaged"),
    setup_profile = factor(setup_profile, levels = prof_levels),
    min  = objective_time_ms / 60000,
    gold_pct = gold_diff_T_60 / (2500 + 850 * min) * 100,
    alive_state = factor(case_when(
      team_alive_T_60 < enemy_alive_T_60 ~ "numbers_down",
      team_alive_T_60 > enemy_alive_T_60 ~ "numbers_up",
      TRUE                               ~ "even_alive"),
      levels = c("numbers_down","even_alive","numbers_up")),
    gold_state = factor(c("big_behind","behind","even","ahead","big_ahead")[ntile(gold_pct,5)],
                        levels = c("big_behind","behind","even","ahead","big_ahead")),
    outcome = factor(case_when(
      steal == 1                                                               ~ "steal",
      fight_result == "none" & objective_result == "secured"                   ~ "uncontested take",
      fight_result == "none" & objective_result == "lost" & good_trade == 0    ~ "clean give",
      fight_result == "none" & objective_result == "lost" & good_trade == 1    ~ "gave, got trade",
      objective_result == "secured"                                             ~ "contested take",
      objective_result == "lost"    & good_trade == 1                          ~ "contested loss + trade",
      objective_result == "lost"    & good_trade == 0                          ~ "contested loss",
      TRUE                                                                      ~ NA_character_),
      levels = outcome_levels))
N <- nrow(dat)

# =====================================================================
# 1 — Setup profile: prevalence vs secure rate   (was: treemap)
# =====================================================================
s1 <- dat |> group_by(setup_profile) |>
  summarise(n = n(), secure = 100*mean(secured), net = mean(net_value), .groups = "drop") |>
  mutate(prevalence = 100*n/sum(n))
s1_ord <- s1 |> arrange(desc(prevalence)) |>
  mutate(setup_profile = factor(as.character(setup_profile), levels = as.character(setup_profile)))
p1a <- ggplot(s1_ord, aes(prevalence, fct_reorder(as.character(setup_profile), prevalence),
                          fill = setup_profile)) +
  geom_col(width = 0.72) +
  geom_text(aes(label = sprintf("%.0f%%", prevalence)), hjust = -0.2, size = 3.6, colour = INK) +
  scale_fill_manual(values = pal_profile, guide = "none") +
  scale_x_continuous(labels = label_percent(scale = 1), limits = c(0, 30),
                     expand = expansion(mult = c(0, .10))) +
  labs(title = "How common each setup is",
       subtitle = "Share of all objective contests") +
  theme_lol() + theme(panel.grid.major.y = element_blank(), axis.title.y = element_blank(),
                      axis.title.x = element_blank())
p1b <- ggplot(s1, aes(secure, fct_reorder(as.character(setup_profile), secure), fill = setup_profile)) +
  geom_col(width = 0.72) +
  geom_text(aes(label = sprintf("%.0f%%", secure)), hjust = -0.2, size = 3.6, colour = INK) +
  scale_fill_manual(values = pal_profile, guide = "none") +
  scale_x_continuous(labels = label_percent(scale = 1), limits = c(0, 100),
                     expand = expansion(mult = c(0, .10))) +
  labs(title = "Secure rate by setup profile",
       subtitle = "How often each setup ends with the team securing the objective",
       caption = CAP) +
  theme_lol() + theme(panel.grid.major.y = element_blank(), axis.title.y = element_blank(),
                      plot.title = element_text(face = "bold", colour = INK, size = rel(1.08)))
p1c <- ggplot(s1, aes(prevalence, secure, colour = setup_profile, label = as.character(setup_profile))) +
  geom_point(size = 5) +
  geom_text_repel(size = 3.4, colour = INK, fontface = "bold",
                  box.padding = 0.5, max.overlaps = 20) +
  scale_colour_manual(values = pal_profile, guide = "none") +
  scale_x_continuous(labels = label_percent(scale = 1), limits = c(0, 35)) +
  scale_y_continuous(labels = label_percent(scale = 1), limits = c(0, 100)) +
  labs(title = "Prevalence vs secure rate",
       subtitle = "Each dot = one setup profile",
       x = "Prevalence (%)", y = "Secure rate (%)") +
  theme_lol()
p1 <- (p1a + p1b + p1c + plot_layout(widths = c(1, 1, 1))) &
  theme(plot.background = element_rect(fill = BG, colour = NA))
savep(p1, "01_setup_profiles_combo.png", 22, 6.6)

# =====================================================================
# 2 — Outcome label prevalence
# =====================================================================
s2 <- dat |>
  filter(!is.na(outcome)) |>
  group_by(outcome) |>
  summarise(n = n(), .groups = "drop") |>
  mutate(prevalence = 100*n/sum(n),
         outcome = factor(outcome, levels = rev(outcome_levels)))
p2 <- ggplot(s2, aes(prevalence, outcome, fill = outcome)) +
  geom_col(width = 0.72) +
  geom_text(aes(label = sprintf("%.1f%%", prevalence)), hjust = -0.15,
            colour = INK, size = 3.5, fontface = "bold") +
  scale_fill_manual(values = pal_outcome, guide = "none") +
  scale_x_continuous(labels = label_percent(scale = 1),
                     expand = expansion(mult = c(0, .14))) +
  labs(title = "How often each outcome occurs",
       subtitle = "Share of all objective contests",
       x = "Share", y = NULL) +
  theme_lol() +
  theme(panel.grid.major.y = element_blank())
savep(p2, "02_outcome_labels_combo.png", 11, 7.0)

# =====================================================================
# 3 — Setup profile × outcome label heatmap
# =====================================================================
setup_order <- dat |> group_by(setup_profile) |> summarise(pe = mean(net_value > 0)) |>
  arrange(desc(pe)) |> pull(setup_profile) |> as.character()
s3 <- dat |>
  filter(!is.na(outcome)) |>
  mutate(setup_profile = factor(as.character(setup_profile), levels = rev(setup_order))) |>
  group_by(setup_profile, outcome) |>
  summarise(n = n(), .groups = "drop_last") |>
  mutate(p = 100 * n / sum(n)) |>
  ungroup() |>
  complete(setup_profile, outcome, fill = list(n = 0, p = 0))
p3 <- ggplot(s3, aes(outcome, setup_profile, fill = outcome, alpha = p)) +
  geom_tile(colour = "#0f172a", linewidth = 1.2) +
  geom_text(aes(label = sprintf("%.0f%%", p)), size = 3.2, colour = "white", alpha = 1) +
  scale_fill_manual(values = pal_outcome, guide = "none") +
  scale_alpha_continuous(range = c(0.12, 1), name = "Share of\nsetup's contests",
                         labels = label_percent(scale = 1)) +
  scale_x_discrete(limits = outcome_levels) +
  coord_cartesian(expand = FALSE) +
  labs(title = "Where each setup ends up",
       subtitle = "Color = outcome type  ·  Opacity = how often that setup produces it  ·  Value = share of setup's contests",
       x = NULL, y = NULL) +
  theme_lol() +
  theme(axis.text.x  = element_text(angle = 32, hjust = 1, size = rel(0.82)),
        axis.text.y  = element_text(face = "bold", colour = INK),
        panel.grid.major = element_blank(),
        legend.position  = "right")
savep(p3, "03_setup_profile_by_objective.png", 15, 6.2)

# =====================================================================
# 4 — Allied deaths in 60s before objective -> secure rate   (bar)
# =====================================================================
s4 <- dat |> mutate(deaths = pmin(team_deaths_60s, 4)) |>
  group_by(deaths) |> summarise(n = n(), k = sum(secured), secure = 100*mean(secured), .groups="drop")
ci4 <- wilson(s4$k, s4$n); s4$lo <- ci4$lo; s4$hi <- ci4$hi
s4 <- s4 |> mutate(lab = c("0","1","2","3","4+")[deaths + 1])
p4 <- ggplot(s4, aes(factor(deaths), secure)) +
  geom_col(aes(fill = secure), width = 0.72) +
  geom_text(aes(label = sprintf("%.1f%%", secure)), vjust = -0.5, fontface = "bold",
            colour = INK, size = 4.4) +
  fill_secure(guide = "none") +
  scale_x_discrete(labels = s4$lab) +
  scale_y_continuous(labels = label_percent(scale = 1), limits = c(0, 100),
                     expand = expansion(c(0,.08))) +
  labs(title = "Every death before the objective bleeds the secure",
       subtitle = "Secure rate vs. allied deaths in the 60s window before the objective is taken",
       x = "Allied deaths in 60s before objective", y = "Secure rate", caption = CAP) +
  theme_lol()
savep(p4, "04_deaths_to_secure_bar.png", 8.8, 6.0)

# =====================================================================
# 5 — Secure rate: feature present vs absent   (was: grouped bar -> dumbbell)
# =====================================================================
binflags <- tibble(
  feature = c("Numbers down (T-30)","Numbers advantage (T-30)","Team grouped 3+ (T-30)",
              "Support dead (T-30)","Jungler dead (T-30)","Arrived first (T-60)"),
  flag = list(dat$team_nearby_T_30 <  dat$enemy_nearby_T_30,
              dat$team_nearby_T_30 >  dat$enemy_nearby_T_30,
              dat$team_nearby_T_30 >= 3,
              dat$support_alive_T_30 == 0,
              dat$jungler_alive_T_30 == 0,
              dat$arrived_first == 1))
s5 <- binflags |> rowwise() |>
  mutate(absent  = 100*mean(dat$secured[!flag]),
         present = 100*mean(dat$secured[ flag]),
         gap = present - absent) |> ungroup() |>
  mutate(feature = fct_reorder(feature, abs(gap)))
p5 <- ggplot(s5, aes(y = feature)) +
  geom_segment(data = data.frame(x = c(25, 50, 75),
                                 y    = levels(s5$feature)[1],
                                 yend = levels(s5$feature)[nlevels(s5$feature)]),
               aes(x = x, xend = x, y = y, yend = yend),
               colour = GRID, linewidth = 0.45, inherit.aes = FALSE) +
  geom_segment(aes(x = absent, xend = present, yend = feature),
               colour = "#334155", linewidth = 2.4, lineend = "round") +
  geom_point(aes(x = absent),  colour = "#64748b", size = 5.5) +
  geom_point(aes(x = present), colour = "#60a5fa", size = 5.5) +
  geom_text(aes(x = absent,  label = sprintf("%.0f%%", absent)),  vjust = -1.4,
            colour = "#94a3b8", size = 3.2) +
  geom_text(aes(x = present, label = sprintf("%.0f%%", present)), vjust = -1.4,
            colour = "#60a5fa", size = 3.4, fontface = "bold") +
  geom_text(aes(x = (absent+present)/2,
                label = sprintf("%+.0f pp", gap)), vjust = 2.0, size = 3.2, colour = INK) +
  annotate("point", x = 65, y = 7.05, colour = "#64748b", size = 4) +
  annotate("text",  x = 67, y = 7.05, label = "absent",  hjust = 0, size = 3.3, colour = "#94a3b8") +
  annotate("point", x = 80, y = 7.05, colour = "#60a5fa", size = 4) +
  annotate("text",  x = 82, y = 7.05, label = "present", hjust = 0, size = 3.3, colour = "#60a5fa") +
  scale_x_continuous(labels = label_percent(scale = 1), limits = c(0, 100)) +
  coord_cartesian(clip = "off", ylim = c(0.6, 6.5)) +
  labs(title = "Variables that strongly influence secure rates",
       subtitle = "Secure rate when each game-state signal is absent vs present",
       x = "Secure rate", y = NULL, caption = CAP) +
  theme_lol() + theme(panel.grid.major.y = element_blank(),
                      panel.grid.major.x = element_blank())
savep(p5, "05_feature_present_absent_dumbbell.png", 10, 6.2)

# =====================================================================
# 6 — Gold advantage vs secure rate, by setup profile   (lines)
# =====================================================================
s6 <- dat |> filter(gold_pct >= -45, gold_pct <= 38) |>
  mutate(bin = cut(gold_pct, breaks = seq(-45, 40, 2.5))) |>
  group_by(setup_profile, bin) |>
  summarise(x = mean(gold_pct), secure = 100*mean(secured), n = n(), .groups = "drop") |>
  filter(n >= 120)
p6 <- ggplot(s6, aes(x, secure, colour = setup_profile)) +
  geom_hline(yintercept = 50, linetype = "22", colour = "#475569") +
  geom_vline(xintercept = 0, colour = GRID) +
  geom_point(aes(size = n), alpha = 0.22) +
  geom_smooth(aes(weight = n), method = "loess", se = FALSE, span = 0.9, linewidth = 1.35) +
  scale_colour_manual(values = pal_profile, breaks = prof_levels, name = "Setup profile") +
  scale_size_area(max_size = 3.4, guide = "none") +
  scale_x_continuous(labels = label_percent(scale = 1)) +
  scale_y_continuous(labels = label_percent(scale = 1), limits = c(0, 100)) +
  labs(title = "A good setup is worth its weight in gold",
       subtitle = "LOESS fits of P(secure) vs gold advantage at T-60, weighted by sample size.\nSetup sets the baseline; gold only slides you along it.",
       x = "Gold advantage at T-60 (% of est. team gold)", y = "Secure rate", caption = CAP) +
  theme_lol() + theme(legend.position = "right")
savep(p6, "06_gold_by_setup_lines.png", 10.5, 6.6)


# =====================================================================
# 7 & 8 — State-conditioned effect of an action  (faceted horizontal bars)
# =====================================================================
effect_plot <- function(action, title, sub, style) {
  e <- dat |> mutate(act = action) |>
    group_by(gold_state, alive_state) |>
    summarise(n1 = sum(act), n0 = sum(!act),
              p1 = mean(secured[act]), p0 = mean(secured[!act]), .groups = "drop") |>
    filter(n1 >= 40, n0 >= 40) |>
    mutate(delta = 100*(p1 - p0),
           se = 100*sqrt(p1*(1-p1)/n1 + p0*(1-p0)/n0),
           lo = delta - 1.96*se, hi = delta + 1.96*se)
  ylab <- "Δ secure rate (pp), action vs no action"
  pd <- position_dodge(0.28)

  if (style == "bars") {                                   # (a) original faceted bars
    e <- e |> mutate(alive_state = factor(recode(alive_state,
           numbers_down = "Numbers down (T-60)", even_alive = "Even numbers (T-60)",
           numbers_up = "Numbers up (T-60)"),
           levels = c("Numbers down (T-60)","Even numbers (T-60)","Numbers up (T-60)")))
    ggplot(e, aes(delta, gold_state, fill = delta)) +
      geom_col(width = 0.72) +
      geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0.18, colour = "#3a4250", linewidth = 0.4) +
      geom_text(aes(x = hi, label = sprintf("%+.0f", delta)), hjust = -0.35,
                size = 3.2, colour = INK, fontface = "bold") +
      facet_wrap(~alive_state) +
      scale_fill_gradientn(colours = c("#1e3a8a","#2563eb","#60a5fa","#bfdbfe"), guide = "none") +
      scale_y_discrete(labels = function(x) gsub("_", " ", x)) +
      scale_x_continuous(expand = expansion(mult = c(.02,.22))) +
      labs(title = title, subtitle = sub, x = ylab, y = "Gold state (T-60)", caption = CAP) +
      theme_lol(15) + theme(panel.grid.major.y = element_blank(), panel.spacing = unit(14, "pt"))

  } else if (style == "gold_x") {                          # (b) gold on x, lines = numbers state
    e <- e |> mutate(alive_state = factor(recode(alive_state,
           numbers_down = "Numbers down", even_alive = "Even numbers", numbers_up = "Numbers up"),
           levels = c("Numbers down","Even numbers","Numbers up")))
    ggplot(e, aes(gold_state, delta, colour = alive_state, group = alive_state)) +
      geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.16, linewidth = 0.45, alpha = 0.55, position = pd) +
      geom_line(linewidth = 1.25, position = pd) + geom_point(size = 3, position = pd) +
      scale_colour_manual(values = pal_alive, name = "Numbers state (T-60)") +
      scale_x_discrete(labels = function(x) gsub("_", " ", x)) +
      scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, .08))) +
      labs(title = title, subtitle = sub, x = "Gold state (T-60)", y = ylab, caption = CAP) +
      theme_lol(15) + theme(legend.position = "right", panel.grid.major.x = element_blank())

  } else if (style == "numbers_x") {                       # (c) numbers on x, lines = gold state
    e <- e |> mutate(
      alive_state = factor(recode(alive_state, numbers_down = "numbers down",
             even_alive = "even", numbers_up = "numbers up"),
             levels = c("numbers down","even","numbers up")),
      gold_state = factor(gsub("_", " ", as.character(gold_state)),
             levels = c("big behind","behind","even","ahead","big ahead")))
    ggplot(e, aes(alive_state, delta, colour = gold_state, group = gold_state)) +
      geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.14, linewidth = 0.45, alpha = 0.5, position = pd) +
      geom_line(linewidth = 1.25, position = pd) + geom_point(size = 3, position = pd) +
      scale_colour_manual(values = pal_gold, name = "Gold state (T-60)") +
      scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, .08))) +
      labs(title = title, subtitle = sub, x = "Numbers state (T-60)", y = ylab, caption = CAP) +
      theme_lol(15) + theme(legend.position = "right", panel.grid.major.x = element_blank())

  } else if (style == "gold_cont") {                        # (d) loess delta on raw gold values
    gold_grid <- seq(-42, 37, by = 0.5)
    e_c <- dat |>
      mutate(act = action) |>
      filter(gold_pct >= -44, gold_pct <= 38) |>
      group_by(alive_state) |>
      group_modify(~{
        df1 <- .x[.x$act,  ]; df0 <- .x[!.x$act, ]
        if (nrow(df1) < 100 || nrow(df0) < 100) return(tibble())
        fit1 <- loess(secured ~ gold_pct, data = df1, span = 0.65)
        fit0 <- loess(secured ~ gold_pct, data = df0, span = 0.65)
        grd  <- data.frame(gold_pct = gold_grid)
        tibble(x = gold_grid,
               delta = 100 * (suppressWarnings(predict(fit1, grd)) -
                               suppressWarnings(predict(fit0, grd))))
      }) |>
      ungroup() |>
      filter(!is.na(delta)) |>
      mutate(alive_state = factor(recode(alive_state,
             numbers_down = "Numbers down", even_alive = "Even numbers", numbers_up = "Numbers up"),
             levels = c("Numbers down","Even numbers","Numbers up")))
    ggplot(e_c, aes(x, delta, colour = alive_state)) +
      geom_hline(yintercept = 0, linetype = "22", colour = "#475569") +
      geom_vline(xintercept = 0, colour = GRID) +
      geom_line(linewidth = 1.3) +
      scale_colour_manual(values = pal_alive, name = "Numbers state (T-60)") +
      scale_x_continuous(labels = label_percent(scale = 1)) +
      scale_y_continuous(expand = expansion(mult = c(.04, .08))) +
      labs(title = title, subtitle = sub,
           x = "Gold advantage at T-60 (% of est. team gold)", y = ylab, caption = CAP) +
      theme_lol(15) + theme(legend.position = "right")
  }
}
af <- dat$arrived_first == 1
afT <- "How arriving first impacts secure rate by gamestate"
afS <- "Lift in secure rate from arriving first, within each gold x numbers state"
savep(effect_plot(af, afT, afS, "bars"),           "07a_effect_arrived_first_bars.png",          11, 6.0)

na <- dat$team_nearby_T_30 > dat$enemy_nearby_T_30
naT <- "How numbers advantage influences secure rates by gamestate"
naS <- "Lift in secure rate from a T-30 numbers advantage, within each gold x numbers state"
savep(effect_plot(na, naT, naS, "bars"),      "08a_effect_numbers_advantage_bars.png",      11, 6.0)

# =====================================================================
# 9 — SHAP beeswarm (RandomForest)   (from shap_compute.py output)
# =====================================================================
shp_path <- file.path(ROOT, "remake", "shap_long.csv")
if (file.exists(shp_path)) {
  imp <- read_csv(file.path(ROOT,"remake","shap_importance.csv"), show_col_types = FALSE)
  auc <- imp$test_auc[1]
  topf <- head(imp$feature, 18)
  shp <- read_csv(shp_path, show_col_types = FALSE) |>
    filter(feature %in% topf) |>
    mutate(feature = factor(pretty_feat(feature), levels = rev(pretty_feat(topf))))
  p9 <- ggplot(shp, aes(shap, feature, colour = value_rank)) +
    geom_vline(xintercept = 0, colour = "#334155") +
    geom_sina(size = 0.55, alpha = 0.45, maxwidth = 0.82, scale = "width") +
    scale_colour_gradient(low = "#3b4cc0", high = "#e8482b",
                          breaks = c(0.04, 0.96), labels = c("low","high"),
                          name = "Feature\nvalue") +
    labs(title = "What the model leans on to predict a secure",
         subtitle = sprintf("SHAP value per contest (RandomForest, test AUC = %.2f). Right = pushes toward securing.", auc),
         x = "SHAP value (impact on secure probability)", y = NULL, caption = CAP) +
    theme_lol(14) +
    theme(panel.grid.major.y = element_blank(),
          axis.text.y = element_text(colour = INK))
  savep(p9, "09_shap_beeswarm.png", 9.8, 8.2)
  cat("plot 9 done\n")
} else cat("!! shap_long.csv not found — run shap_compute.py first; skipping plot 9\n")

cat("ALL FIGURES WRITTEN TO", FIG, "\n")
