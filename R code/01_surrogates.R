#=====================================================================
# Step 1: surrogates of the black-box hard labels Y1
#   Benchmarks : LR (MLE) and ridge logistic regression (lambda by 10-fold CV deviance on Y1)
#   Stage 1    : ridge anchor - ridge logistic on Y1 with lambda chosen to give the least
#                held-out log-loss on the ground truth y (replaces Algorithm 1, SHC)
#   Stage 2    : Polya-Gamma Gibbs sampler with the hierarchical prior of the reference paper
#                beta | s1, s2 ~ N(0, g s1 s2 (X_c'X_c)^-1), s1 ~ IG(a, b), s2 ~ IG(c, d),
#                b = a H(y, p^s_LR), d = a H(y, p_BB)
#                (replaces Algorithm 2, Monte Carlo composition)
# Run from the "PG paper" folder:  Rscript "R code/01_surrogates.R"
#=====================================================================
source("R code/00_setup.R")
cat("Black box:", bb, "| mismatch(y, Y1) =", round(ce_bb, 4), "| H(y, p^s_LR) =", round(h_sur, 4),
    "| H(y, p_BB) =", round(h_bb, 4), "\n")
cat("Hyperprior: a = c =", hp$a, "| E[s1] =", round(hp$b / (hp$a - 1), 4),
    "| E[s2] =", round(hp$d / (hp$c - 1), 4), "\n")
Xg_tr <- Xtr[, -1]                                     # glmnet adds its own intercept

#------------------ benchmark: standard ridge (CV deviance on Y1) ----------------------
set.seed(2026)
foldid <- sample(rep(1:10, length.out = n))
t0 <- Sys.time()
cv_r <- cv.glmnet(Xg_tr, Y1_tr, family = "binomial", alpha = 0, foldid = foldid, type.measure = "deviance")
t_ridge <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
b_ridge <- as.numeric(coef(cv_r, s = "lambda.min")); names(b_ridge) <- colnames(Xtr)

#------------------ stage 1: ridge anchor (least held-out log-loss on y) ---------------
lam <- cv_r$lambda
t0 <- Sys.time()
cv_y <- sapply(1:10, function(k) {
  tr <- foldid != k
  fit <- glmnet(Xg_tr[tr, ], Y1_tr[tr], family = "binomial", alpha = 0, lambda = lam)
  P <- predict(fit, Xg_tr[!tr, ], type = "response")
  apply(P, 2, function(pk) calc_ce(ytr[!tr], pk))
})
lam_anchor <- lam[which.min(rowMeans(cv_y))]
fit_a <- glmnet(Xg_tr, Y1_tr, family = "binomial", alpha = 0, lambda = lam)
b_anchor <- as.numeric(coef(fit_a, s = lam_anchor)); names(b_anchor) <- colnames(Xtr)
t_anchor <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("Ridge benchmark lambda = %.4g | anchor lambda (min CV log-loss on y) = %.4g\n",
            cv_r$lambda.min, lam_anchor))

#------------------ stage 2: Polya-Gamma Gibbs sampler ----------------------------------
source("R code/pg_gibbs.R")

iters <- 6000; burn <- 1000; n_chains <- 4; seeds <- 2026 + 0:(n_chains - 1)
set.seed(99)
starts <- c(list(b_anchor), lapply(2:n_chains, function(k) b_anchor + rnorm(p)))
# glmnet's standardised ridge penalty (lambda/2) sum_j sd_j^2 beta_j^2 equals the g-prior penalty
# beta' X_c'X_c beta / (2 g s1 s2) up to feature correlations when lambda = 1/(g s1 s2).
# Proposed: g = 1/(lambda* E[s1]) replaces the LR-surrogate scale s1 by the stage-1 scale 1/lambda*,
# so the prior variance is about H(y, p_BB) / lambda*: the black box's cross-entropy enters the prior
# and the posterior mode is a ridge fit with penalty lambda* / E[s2].
# Ablation: g = 1/(lambda* E[s1 s2]) cancels both cross-entropies (posterior mode = stage-1 anchor).
Es1 <- hp$b / (hp$a - 1); Es2 <- hp$d / (hp$c - 1)
g_ridge <- 1 / (lam_anchor * Es1)
g_nobb  <- 1 / (lam_anchor * Es1 * Es2)
cat(sprintf("E[s1] = %.4f | E[s2] = %.4f | g (proposed) = %.2f | g (no black-box term) = %.2f\n", Es1, Es2, g_ridge, g_nobb))
variants <- list(
  # check only: prior scale exactly as in the reference paper (g = 1, intercept penalised)
  "PG (Alg-1 prior)"      = list(g = 1, flat = FALSE),
  # ablation: black-box cross-entropy cancelled
  "Ridge-PG (no BB term)" = list(g = g_nobb, flat = TRUE),
  # proposed: prior variance = H(y, p_BB) / lambda*
  "Ridge-PG"              = list(g = g_ridge, flat = TRUE))

pg <- lapply(names(variants), function(v) {
  lapply(seq_len(n_chains), function(k) {
    t0 <- Sys.time()
    out <- pg_gibbs(Xtr, Y1_tr, variants[[v]]$g, rep(0, p), hp, starts[[k]], iters, seeds[k],
                    flat_intercept = variants[[v]]$flat)
    out$runtime <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    cat(sprintf("%-18s chain %d: %.1f s\n", v, k, out$runtime)); out
  })
})
names(pg) <- names(variants)

saveRDS(list(b_ridge = b_ridge, b_anchor = b_anchor, lam_ridge = cv_r$lambda.min, lam_anchor = lam_anchor,
             t_ridge = t_ridge, t_anchor = t_anchor, g_ridge = g_ridge, g_nobb = g_nobb, pg = pg, iters = iters, burn = burn),
        "results/surrogates.rds")
