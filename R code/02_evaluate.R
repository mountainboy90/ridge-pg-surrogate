#=====================================================================
# Step 2: evaluation - sampler diagnostics, fidelity to the black box, accuracy on the
# ground truth, and agreement of every surrogate with TreeSHAP (slopes, signs, importance
# ranks, local contributions), plus SHAP dependence plots with surrogate overlays.
# Run from the "PG paper" folder:  Rscript "R code/02_evaluate.R"
#=====================================================================
source("R code/00_setup.R")
suppressPackageStartupMessages({ library(shapviz); library(ggplot2); library(patchwork) })
sr <- readRDS("results/surrogates.rds"); keep <- (sr$burn + 1):sr$iters
pgn <- names(sr$pg)

#------------------ sampler diagnostics ------------------------------------------------
diag_tab <- do.call(rbind, lapply(pgn, function(v) {
  ch <- sr$pg[[v]]; d <- chain_diag(lapply(ch, `[[`, "beta"), sr$burn); rt <- sapply(ch, `[[`, "runtime")
  data.frame(Sampler = v, Rhat_max = max(d$rhat), ESS_min = min(d$ess_bulk), ESS_med = median(d$ess_bulk),
             Time_chain_s = mean(rt), ESS_med_per_s = median(d$ess_bulk) / sum(rt),
             s1s2_mean = mean(unlist(lapply(ch, function(x) x$s12[keep]))))
}))
print(diag_tab, digits = 3)

draws <- lapply(sr$pg, function(ch) do.call(rbind, lapply(ch, function(x) x$beta[keep, ])))
post <- lapply(draws, function(D) list(mean = colMeans(D), q = apply(D, 2, quantile, c(0.025, 0.975))))
alg1 <- post[["PG (Alg-1 prior)"]]$mean                 # check of the reference prior scale only
cat("Reference prior scale (g = 1): max |beta_j| =", signif(max(abs(alg1)), 3), "| train log-loss on y =",
    round(calc_ce(ytr, sigmoid(Xtr %*% alg1)), 3), "\n")
abl <- post[["Ridge-PG (no BB term)"]]                  # ablation: black-box cross-entropy cancelled
draws <- draws["Ridge-PG"]; post <- post["Ridge-PG"]; pgn <- "Ridge-PG"
diag_tab <- diag_tab[diag_tab$Sampler == "Ridge-PG", ]

#------------------ black box vs logistic regression on the true labels y ---------------
# each model labels a record positive above the 70th percentile of its own training scores
cls_metrics <- function(model, set, y, p, cut) {
  cl <- as.integer(p > cut)
  data.frame(Model = model, Set = set, Recall = sum(cl & y) / sum(y), Specificity = sum(!cl & !y) / sum(!y),
             Accuracy = mean(cl == y), CE = mean(cl != y), LogLoss = calc_ce(y, p),
             AUC = as.numeric(auc(y, p, quiet = TRUE)))
}
lr_y <- suppressWarnings(glm(ytr ~ Xtr - 1, family = binomial))
plr_tr <- as.numeric(sigmoid(Xtr %*% coef(lr_y))); plr_te <- as.numeric(sigmoid(Xte %*% coef(lr_y)))
bb_tab <- rbind(cls_metrics("XGBoost", "Train", ytr, pbb_tr, cut_bb), cls_metrics("XGBoost", "Test", yte, pbb_te, cut_bb),
                cls_metrics("LR", "Train", ytr, plr_tr, quantile(plr_tr, 0.7)),
                cls_metrics("LR", "Test", yte, plr_te, quantile(plr_tr, 0.7)))
print(bb_tab, digits = 3)
cat("5-fold CV log-loss (mean, SD):\n"); print(round(rbind(mean = colMeans(s1$cv), sd = apply(s1$cv, 2, sd)), 3))

#------------------ fidelity (black box) and accuracy (ground truth) -------------------
betas <- c(list(LR = b_lr, Ridge = sr$b_ridge), lapply(post, `[[`, "mean"))
fid_tab <- do.call(rbind, lapply(names(betas), function(nm)
  rbind(cbind(Set = "Train", evaluate(nm, betas[[nm]], Xtr, ytr, pbb_tr, Y1_tr)),
        cbind(Set = "Test",  evaluate(nm, betas[[nm]], Xte, yte, pbb_te, Y1_te)))))
print(fid_tab, digits = 3)
cat("XGBoost on y (test): Acc", round(mean(as.integer(pbb_te > cut_bb) == yte), 3),
    "| AUC", round(as.numeric(auc(yte, pbb_te, quiet = TRUE)), 3), "| log-loss", round(calc_ce(yte, pbb_te), 3), "\n")

#------------------ black-box explanations: exact TreeSHAP ------------------------------
bbm <- s1$model
shap_of <- function(d) predict(bbm, xgb.DMatrix(as.matrix(d[, ff_bb])), predcontrib = TRUE)
phi_tr_all <- shap_of(train_data); phi_te_all <- shap_of(test_data)
phi_tr <- phi_tr_all[, ff]; phi_te <- phi_te_all[, ff]
mean_abs_shap <- colMeans(abs(phi_tr))
shap_slope <- sapply(ff, function(f) coef(lm(phi_tr[, f] ~ train_data[, f]))[2]); names(shap_slope) <- ff
used <- ff[mean_abs_shap > 1e-8]                       # features XGBoost actually splits on
cat("Features unused by XGBoost:", setdiff(ff, used), "\n")
sdx <- apply(train_data[, ff], 2, sd); xbar <- colMeans(train_data[, ff])

# agreement of a coefficient vector with TreeSHAP, all on the training data (what the black box learned from)
#   MAE / Pearson to the SHAP slopes and sign agreement (used features),
#   Spearman between |beta_j| sd(x_j) and mean|phi_j| (all features),
#   R2 of the surrogate's local contributions beta_j (x_ij - mean x_j) as predictors of phi_ij
shap_agree <- function(beta, X = as.matrix(train_data[, ff]), phi = phi_tr) {
  b <- beta[ff]; C <- sweep(sweep(X, 2, xbar), 2, b, `*`)
  c(MAE_slope = mean(abs(b[used] - shap_slope[used])),
    Pearson_slope = cor(b[used], shap_slope[used]),
    Sign_agree = sum(sign(b[used]) == sign(shap_slope[used])),
    Spearman_imp = cor(abs(b) * sdx, mean_abs_shap, method = "spearman"),
    Local_R2 = 1 - sum((phi - C)^2) / sum(sweep(phi, 2, colMeans(phi))^2),
    Local_cor = mean(sapply(used, function(f) if (sd(C[, f]) > 0) cor(C[, f], phi[, f]) else 0)))
}
agree_tab <- do.call(rbind, lapply(names(betas), function(nm) data.frame(Surrogate = nm, t(shap_agree(betas[[nm]])))))

# interval-based checks: coverage of the SHAP slope, and features unused by XGBoost flagged
ints <- c(list(LR = rbind(b_lr - 1.96 * se_lr, b_lr + 1.96 * se_lr)), lapply(post, `[[`, "q"))
unused <- setdiff(ff, used)
int_tab <- do.call(rbind, lapply(names(ints), function(nm) {
  q <- ints[[nm]][, ff]
  data.frame(Surrogate = nm,
             Slope_coverage = mean(q[1, used] <= shap_slope[used] & shap_slope[used] <= q[2, used]),
             Unused_flagged = sum(q[1, unused] > 0 | q[2, unused] < 0),
             Mean_CrI_width = mean(q[2, used] - q[1, used]))
}))
agree_tab <- merge(agree_tab, int_tab, all.x = TRUE, sort = FALSE)
print(agree_tab, digits = 3)

# posterior uncertainty of the SHAP-agreement metrics (every 25th pooled draw)
agree_post <- do.call(rbind, lapply(pgn, function(v) {
  A <- t(apply(draws[[v]][seq(1, nrow(draws[[v]]), by = 25), ], 1, shap_agree))
  data.frame(Surrogate = v, Metric = colnames(A), Mean = colMeans(A),
             Lo = apply(A, 2, quantile, 0.025), Hi = apply(A, 2, quantile, 0.975), row.names = NULL)
}))
print(agree_post, digits = 3)

#------------------ coefficient table -------------------------------------------------
coef_tab <- data.frame(Feature = ff, LR = b_lr[ff], LR_lo = ints$LR[1, ff], LR_hi = ints$LR[2, ff],
                       Ridge = sr$b_ridge[ff])
for (v in pgn) {
  coef_tab[[v]] <- post[[v]]$mean[ff]
  coef_tab[[paste0(v, "_lo")]] <- post[[v]]$q[1, ff]; coef_tab[[paste0(v, "_hi")]] <- post[[v]]$q[2, ff]
}
coef_tab$SHAP_slope <- shap_slope; coef_tab$Mean_abs_SHAP <- mean_abs_shap
coef_tab <- coef_tab[order(-mean_abs_shap), ]

#------------------ ablation: Ridge-PG with and without the black-box cross-entropy ----------
abl_tab <- do.call(rbind, lapply(list("Ridge-PG" = post[["Ridge-PG"]], "Ridge-PG (no BB term)" = abl), function(ps) {
  q <- ps$q[, ff]
  cbind(evaluate("", ps$mean, Xte, yte, pbb_te, Y1_te)[, c("KL", "Dloss", "LogLoss_y", "Acc_y", "AUC_y", "Fid_Acc")],
        t(shap_agree(ps$mean)),
        Slope_coverage = mean(q[1, used] <= shap_slope[used] & shap_slope[used] <= q[2, used]),
        Unused_flagged = sum(q[1, unused] > 0 | q[2, unused] < 0))
}))
abl_tab <- cbind(Variant = c("Ridge-PG", "Ridge-PG (no BB term)"), abl_tab)
print(abl_tab, digits = 3)
write.csv(abl_tab, "results/ablation_bb_term.csv", row.names = FALSE)

write.csv(bb_tab, "results/blackbox_vs_lr.csv", row.names = FALSE)
write.csv(diag_tab, "results/diagnostics.csv", row.names = FALSE)
write.csv(fid_tab, "results/fidelity_accuracy.csv", row.names = FALSE)
write.csv(agree_tab, "results/shap_agreement.csv", row.names = FALSE)
write.csv(agree_post, "results/shap_agreement_posterior.csv", row.names = FALSE)
write.csv(coef_tab, "results/coefficients.csv", row.names = FALSE)

#------------------ figures -----------------------------------------------------------
best <- "Ridge-PG"
cols <- c(LR = "#E69F00", Ridge = "#56B4E9", "Ridge-PG" = "#009E73")

# (1) SHAP dependence plots (shapviz, exact TreeSHAP) with each surrogate's local effect
#     beta_j (x - mean x_j) and the 95% posterior band of the Ridge-PG surrogate
sv <- shapviz(bbm, X_pred = as.matrix(train_data[, ff_bb]), X = train_data[, ff_bb])
cont <- ff[!grepl("^Soilcolor", ff)]
top <- cont[order(-mean_abs_shap[cont])][1:6]
dep_plot <- function(f) {
  xs <- seq(min(train_data[, f]), max(train_data[, f]), length.out = 100)
  lines <- do.call(rbind, lapply(c("LR", "Ridge", best), function(nm)
    data.frame(x = xs, y = betas[[nm]][f] * (xs - xbar[f]), Surrogate = nm)))
  band <- t(sapply(xs, function(x) quantile(draws[[best]][, f] * (x - xbar[f]), c(0.025, 0.975))))
  rng <- range(phi_tr[, f])
  sv_dependence(sv, v = f, color_var = NULL, alpha = 0.3, color = "grey45") +
    geom_ribbon(data = data.frame(x = xs, lo = band[, 1], hi = band[, 2]),
                aes(x = x, ymin = lo, ymax = hi), inherit.aes = FALSE, fill = cols[best], alpha = 0.25) +
    geom_line(data = lines, aes(x = x, y = y, colour = Surrogate), inherit.aes = FALSE, linewidth = 0.8) +
    scale_colour_manual(values = cols) + coord_cartesian(ylim = rng + c(-0.2, 0.2) * diff(rng)) +
    labs(y = "SHAP value (log-odds)", title = f) + theme_bw(base_size = 9)
}
fig_dep <- wrap_plots(lapply(top, dep_plot), ncol = 3) + plot_layout(guides = "collect") &
  theme(legend.position = "bottom")
ggsave("figures/fig_shap_dependence.pdf", fig_dep, width = 9, height = 6)
ggsave("figures/fig_shap_dependence.png", fig_dep, width = 9, height = 6, dpi = 200)

# (2) coefficients (95% intervals) against the TreeSHAP slopes, features ordered by mean|SHAP|
cf <- coef_tab[coef_tab$Feature %in% used, ]
long <- rbind(
  data.frame(Feature = cf$Feature, Model = "LR", est = cf$LR, lo = cf$LR_lo, hi = cf$LR_hi),
  data.frame(Feature = cf$Feature, Model = "Ridge", est = cf$Ridge, lo = NA, hi = NA),
  data.frame(Feature = cf$Feature, Model = best, est = cf[[best]], lo = cf[[paste0(best, "_lo")]], hi = cf[[paste0(best, "_hi")]]))
long$Feature <- factor(long$Feature, levels = rev(cf$Feature))
fig_coef <- ggplot(long, aes(est, Feature, colour = Model)) +
  geom_vline(xintercept = 0, colour = "grey70") +
  geom_pointrange(aes(xmin = lo, xmax = hi), position = position_dodge(width = 0.6), size = 0.25, na.rm = TRUE) +
  geom_point(data = data.frame(Feature = factor(cf$Feature, levels = levels(long$Feature)), est = cf$SHAP_slope),
             aes(est, Feature), inherit.aes = FALSE, shape = 4, size = 2.5) +
  scale_colour_manual(values = cols) + coord_cartesian(xlim = c(-3, 3)) +
  labs(x = "Coefficient (log-odds per unit); x = TreeSHAP slope", y = NULL) + theme_bw(base_size = 9) +
  theme(legend.position = "bottom")
ggsave("figures/fig_coef_vs_shap.pdf", fig_coef, width = 6, height = 5.5)
ggsave("figures/fig_coef_vs_shap.png", fig_coef, width = 6, height = 5.5, dpi = 200)
cat("Done.\n")
