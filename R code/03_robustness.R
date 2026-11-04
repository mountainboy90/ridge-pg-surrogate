#=====================================================================
# Step 3: robustness - the whole pipeline (XGBoost black box, hard labels, LR, ridge,
# Ridge-PG, TreeSHAP) re-run for maize on
#   (A) 30 random stratified 80/20 splits (70th-percentile cut-off)
#   (B) cut-offs from the 50th to the 90th percentile in steps of 5 (original split)
# The feature set is the 22-feature set of the main analysis (data/step1.rds).
# Run from the "PG paper" folder:  Rscript "R code/03_robustness.R"
#=====================================================================
source("R code/00_setup.R")
library(parallel)

pool <- rbind(train_data, test_data)                   # same row order: train, then test
orig_tr <- seq_len(nrow(train_data))

strat_split <- function(y, seed) {
  set.seed(seed)
  sort(unlist(lapply(split(seq_along(y), y), function(ix) sample(ix, round(0.8 * length(ix))))))
}
cases <- c(
  lapply(1:30, function(s) list(scenario = "A: splits", q = 0.7, seed = s, tr = strat_split(pool$target, s))),
  lapply(seq(0.5, 0.9, by = 0.05), function(q) list(scenario = "B: cut-off", q = q, seed = 0, tr = orig_tr)))

run_case <- function(cs) {
  source("R code/00_setup.R"); source("R code/pg_gibbs.R")
  pool <- rbind(train_data, test_data)
  y <- pool$target; tr <- cs$tr; te <- setdiff(seq_len(nrow(pool)), tr)
  ytr <- y[tr]; yte <- y[te]; q <- cs$q
  # black box: XGBoost with the settings of the EP paper
  set.seed(101112)
  bbm <- xgb.train(params = list(objective = "binary:logistic", eval_metric = "logloss", max_depth = 3,
                                 eta = 0.03, subsample = 0.8, colsample_bytree = 0.8, min_child_weight = 5,
                                 gamma = 1, nthread = 1),
                   data = xgb.DMatrix(as.matrix(pool[tr, ff_bb]), label = ytr), nrounds = 400, verbose = 0)
  pbb_tr <- predict(bbm, xgb.DMatrix(as.matrix(pool[tr, ff_bb])))
  pbb_te <- predict(bbm, xgb.DMatrix(as.matrix(pool[te, ff_bb])))
  cut <- quantile(pbb_tr, q); Y1tr <- as.integer(pbb_tr > cut); Y1te <- as.integer(pbb_te > cut)

  # surrogate design: drop soil colours absent from this training set
  feats <- ff[apply(pool[tr, ff], 2, sd) > 0]
  Xtr <- cbind("(Intercept)" = 1, as.matrix(pool[tr, feats])); Xte <- cbind("(Intercept)" = 1, as.matrix(pool[te, feats]))
  n <- nrow(Xtr); p <- ncol(Xtr)
  hardq <- function(beta, X) as.integer(sigmoid(X %*% beta) > quantile(sigmoid(Xtr %*% beta), q))

  # benchmarks
  lr <- suppressWarnings(glm(Y1tr ~ Xtr - 1, family = binomial))
  b_lr <- coef(lr); b_lr[is.na(b_lr)] <- 0; names(b_lr) <- colnames(Xtr)
  se_lr <- rep(NA_real_, p); se_lr[!is.na(coef(lr))] <- sqrt(diag(vcov(lr))); names(se_lr) <- colnames(Xtr)
  set.seed(2026); foldid <- sample(rep(1:10, length.out = n))
  cv_r <- cv.glmnet(Xtr[, -1], Y1tr, family = "binomial", alpha = 0, foldid = foldid)
  b_ridge <- as.numeric(coef(cv_r, s = "lambda.min"))
  # stage 1: ridge anchor (least held-out log-loss on y)
  lam <- cv_r$lambda
  cv_y <- sapply(1:10, function(k) {
    fit <- glmnet(Xtr[foldid != k, -1], Y1tr[foldid != k], family = "binomial", alpha = 0, lambda = lam)
    apply(predict(fit, Xtr[foldid == k, -1], type = "response"), 2, function(pk) calc_ce(ytr[foldid == k], pk))
  })
  lam_a <- lam[which.min(rowMeans(cv_y))]
  b_anchor <- as.numeric(coef(glmnet(Xtr[, -1], Y1tr, family = "binomial", alpha = 0, lambda = lam), s = lam_a))
  names(b_ridge) <- names(b_anchor) <- colnames(Xtr)

  # stage 2: Ridge-PG with the entropy-based hyperparameters; g = 1/(lambda* E[s1]) leaves the
  # black-box cross-entropy in the prior variance (about H(y, p_BB)/lambda*). Ablation: g also
  # divided by E[s2], which cancels the black-box term.
  a <- (n - p) / 2
  hp <- list(a = a, c = a, b = a * calc_ce(ytr, sigmoid(Xtr %*% b_lr)), d = a * calc_ce(ytr, pbb_tr))
  Es1 <- hp$b / (a - 1); Es2 <- hp$d / (a - 1)
  set.seed(99); starts <- list(b_anchor, b_anchor + rnorm(p))
  fit_pg <- function(g) {
    ch <- lapply(1:2, function(k) pg_gibbs(Xtr, Y1tr, g, rep(0, p), hp, starts[[k]], 3000, 2025 + k)$beta[501:3000, ])
    arr <- aperm(simplify2array(ch), c(1, 3, 2)); dimnames(arr)[[3]] <- colnames(Xtr)
    D <- do.call(rbind, ch)
    list(rhat = max(summarise_draws(as_draws_array(arr), "rhat")$rhat), b = colMeans(D), q = apply(D, 2, quantile, c(0.025, 0.975)))
  }
  g <- 1 / (lam_a * Es1)
  pg_main <- fit_pg(g); pg_nobb <- fit_pg(1 / (lam_a * Es1 * Es2))
  rhat <- max(pg_main$rhat, pg_nobb$rhat); b_pg <- pg_main$b; q_pg <- pg_main$q

  # TreeSHAP of the black box
  phi_tr <- predict(bbm, xgb.DMatrix(as.matrix(pool[tr, ff_bb])), predcontrib = TRUE)[, feats]
  phi_te <- predict(bbm, xgb.DMatrix(as.matrix(pool[te, ff_bb])), predcontrib = TRUE)[, feats]
  mabs <- colMeans(abs(phi_tr)); used <- feats[mabs > 1e-8]; unused <- setdiff(feats, used)
  slope <- sapply(feats, function(f) coef(lm(phi_tr[, f] ~ pool[tr, f]))[2]); names(slope) <- feats
  sdx <- apply(pool[tr, feats], 2, sd); xbar <- colMeans(pool[tr, feats]); Xf <- as.matrix(pool[tr, feats])

  score <- function(nm, beta, ints = NULL) {
    ps <- as.numeric(sigmoid(Xte %*% beta)); cls <- hardq(beta, Xte); b <- beta[feats]
    C <- sweep(sweep(Xf, 2, xbar), 2, b, `*`)
    data.frame(Surrogate = nm, Fid_Acc = mean(cls == Y1te), KL = mean(kl_bern(pbb_te, ps)),
               LogLoss_y = calc_ce(yte, ps), Acc_y = mean(cls == yte), AUC_y = as.numeric(auc(yte, ps, quiet = TRUE)),
               MAE_slope = mean(abs(b[used] - slope[used])), Pearson_slope = cor(b[used], slope[used]),
               Sign_agree = mean(sign(b[used]) == sign(slope[used])),
               Spearman_imp = cor(abs(b) * sdx, mabs, method = "spearman"),
               Local_R2 = 1 - sum((phi_tr - C)^2) / sum(sweep(phi_tr, 2, colMeans(phi_tr))^2),
               Coverage = if (is.null(ints)) NA else mean(ints[1, used] <= slope[used] & slope[used] <= ints[2, used], na.rm = TRUE),
               Unused_flagged = if (is.null(ints) || !length(unused)) NA else sum(ints[1, unused] > 0 | ints[2, unused] < 0, na.rm = TRUE))
  }
  out <- rbind(score("LR", b_lr, rbind(b_lr - 1.96 * se_lr, b_lr + 1.96 * se_lr)[, feats, drop = FALSE]),
               score("Ridge", b_ridge), score("Ridge-PG", b_pg, q_pg[, feats, drop = FALSE]),
               score("Ridge-PG (no BB term)", pg_nobb$b, pg_nobb$q[, feats, drop = FALSE]))
  cbind(Scenario = cs$scenario, Cutoff = q, Seed = cs$seed, Prevalence = mean(ytr),
        BB_AUC = as.numeric(auc(yte, pbb_te, quiet = TRUE)), BB_LogLoss = calc_ce(yte, pbb_te),
        Lambda_ridge = cv_r$lambda.min, Lambda_anchor = lam_a, g = g, Rhat = rhat, N_used = length(used), out)
}

cl <- makeCluster(min(8, max(1, detectCores() - 1)))
invisible(clusterCall(cl, setwd, getwd()))
t0 <- Sys.time()
runs <- do.call(rbind, parLapply(cl, cases, run_case))
stopCluster(cl)
cat("Elapsed:", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), "min\n")
write.csv(runs, "results/robustness_runs.csv", row.names = FALSE)

#------------------ summaries --------------------------------------------------------
mets <- c("KL", "LogLoss_y", "Acc_y", "AUC_y", "Fid_Acc", "MAE_slope", "Pearson_slope", "Sign_agree",
          "Spearman_imp", "Local_R2", "Coverage", "Unused_flagged")
sur <- c("LR", "Ridge", "Ridge-PG", "Ridge-PG (no BB term)")
splits <- runs[runs$Scenario == "A: splits", ]
summ <- do.call(rbind, lapply(sur, function(s) {
  d <- splits[splits$Surrogate == s, ]
  data.frame(Surrogate = s, Runs = nrow(d), t(sapply(mets, function(m)
    sprintf("%.3f (%.3f)", mean(d[[m]], na.rm = TRUE), sd(d[[m]], na.rm = TRUE)))), check.names = FALSE)
}))
print(summ, row.names = FALSE)
write.csv(summ, "results/robustness_splits_summary.csv", row.names = FALSE)

cut_tab <- runs[runs$Scenario == "B: cut-off", c("Cutoff", "Surrogate", "Lambda_anchor", "g", mets)]
print(cut_tab, digits = 3, row.names = FALSE)
write.csv(cut_tab, "results/robustness_cutoffs.csv", row.names = FALSE)

# head-to-head over all runs: Ridge-PG better / tied / worse than each benchmark,
# with a Clopper-Pearson 95% interval for the share of runs it wins
runs$id <- paste(runs$Scenario, runs$Cutoff, runs$Seed)
h2h <- do.call(rbind, lapply(c("KL", "LogLoss_y", "MAE_slope", "Local_R2", "Spearman_imp", "Sign_agree", "AUC_y", "Fid_Acc"), function(m) {
  lower <- m %in% c("KL", "LogLoss_y", "MAE_slope")
  w <- reshape(runs[, c("id", "Surrogate", m)], idvar = "id", timevar = "Surrogate", direction = "wide")
  pg <- w[[paste0(m, ".Ridge-PG")]]
  do.call(rbind, lapply(c("LR", "Ridge", "Ridge-PG (no BB term)"), function(s) {
    d <- (pg - w[[paste0(m, ".", s)]]) * ifelse(lower, -1, 1)
    win <- sum(d > 1e-12); ci <- binom.test(win, length(d))$conf.int
    data.frame(Metric = m, Versus = s, Better = win, Tied = sum(abs(d) <= 1e-12), Worse = sum(d < -1e-12),
               Win_lo = ci[1], Win_hi = ci[2])
  }))
}))
print(h2h, digits = 3, row.names = FALSE); write.csv(h2h, "results/robustness_head_to_head.csv", row.names = FALSE)
cat("Max R-hat over runs:", round(max(runs$Rhat), 3), "\n")
print(summary(unique(runs[runs$Scenario == "A: splits", c("Seed", "BB_AUC", "Lambda_ridge", "Lambda_anchor", "g", "N_used")])))
