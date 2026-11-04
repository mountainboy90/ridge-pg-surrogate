#=====================================================================
# Step 4: robustness figure - KL divergence, error to the SHAP slopes, local SHAP R2 and
# importance-rank agreement of each surrogate against the hard-label cut-off (maize, original
# split), with the mean and 95% range over the 30 random splits at the 70th percentile.
# Run from the "PG paper" folder:  Rscript "R code/04_robustness_figure.R"
#=====================================================================
suppressPackageStartupMessages({ library(ggplot2); library(dplyr); library(tidyr) })
r <- read.csv("results/robustness_runs.csv")
r$Surrogate <- factor(r$Surrogate, levels = c("LR", "Ridge", "Ridge-PG (no BB term)", "Ridge-PG"))
labs <- c(KL = "KL to black box (lower is better)", MAE_slope = "Error to SHAP slopes (lower is better)",
          Local_R2 = "Local SHAP R2, clipped at -2 (higher is better)",
          Spearman_imp = "Rank agreement with SHAP (higher is better)")
long <- r %>% mutate(Local_R2 = pmax(Local_R2, -2), Pct = 100 * Cutoff) %>%
  select(Scenario, Pct, Seed, Surrogate, all_of(names(labs))) %>%
  pivot_longer(all_of(names(labs)), names_to = "Metric") %>%
  mutate(Metric = factor(Metric, levels = names(labs), labels = labs))
cut <- filter(long, Scenario == "B: cut-off")
spl <- filter(long, Scenario == "A: splits") %>% group_by(Surrogate, Metric) %>%
  summarise(m = mean(value), lo = quantile(value, 0.025), hi = quantile(value, 0.975), .groups = "drop")
cols <- c(LR = "#E69F00", Ridge = "#56B4E9", "Ridge-PG (no BB term)" = "#999999", "Ridge-PG" = "#009E73")
fig <- ggplot(cut, aes(Pct, value, colour = Surrogate)) +
  geom_line(linewidth = 0.7) + geom_point(size = 1.4) +
  geom_pointrange(data = spl, aes(x = 70 + c(-1.5, -0.5, 0.5, 1.5)[as.integer(Surrogate)], y = m, ymin = lo, ymax = hi),
                  shape = 23, fill = "white", size = 0.35) +
  facet_wrap(~Metric, scales = "free_y", ncol = 2) + scale_colour_manual(values = cols) +
  scale_x_continuous(breaks = seq(50, 90, 10)) +
  labs(x = "Cut-off percentile for the black-box labels", y = NULL) +
  theme_bw(base_size = 10) + theme(legend.position = "bottom")
ggsave("figures/fig_robustness.pdf", fig, width = 8.5, height = 6)
ggsave("figures/fig_robustness.png", fig, width = 8.5, height = 6, dpi = 200)
