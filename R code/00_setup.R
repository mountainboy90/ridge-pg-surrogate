#=====================================================================
# Shared setup for the ridge + Polya-Gamma surrogate paper (sourced by every script).
#   * data, black box and surrogate design are identical to the EP and MALA papers:
#     data/*.csv and data/step1.rds are unchanged copies of "MALA paper/data"
#   * surrogates are fitted to the black-box HARD labels Y1 = I(p_BB > 70th percentile)
# Run every script from the "PG paper" folder, e.g.
#   Rscript "R code/01_surrogates.R"
#=====================================================================
suppressPackageStartupMessages({
  library(xgboost); library(pROC); library(posterior); library(glmnet); library(BayesLogit)
})
dir.create("results", showWarnings = FALSE); dir.create("figures", showWarnings = FALSE)

calc_ce <- function(actual, predicted) {
  eps <- 1e-15
  predicted <- pmax(pmin(predicted, 1 - eps), eps)
  -mean(actual * log(predicted) + (1 - actual) * log(1 - predicted))
}
sigmoid <- function(z) 1 / (1 + exp(-z))
kl_bern <- function(p, q, eps = 1e-15) {
  p <- pmin(pmax(p, eps), 1 - eps); q <- pmin(pmax(q, eps), 1 - eps)
  p * log(p / q) + (1 - p) * log((1 - p) / (1 - q))
}

s1 <- readRDS("data/step1.rds")
ff <- s1$final_features; bb <- s1$best_bb            # 22 black-box features, "XGB"
train_data <- read.csv("data/train_data_maize.csv")
test_data  <- read.csv("data/test_data_maize.csv")
ytr <- train_data$target; yte <- test_data$target
# surrogate design: intercept + 21 features, soil colour "brown" = reference
ff_bb <- ff; ff <- setdiff(ff_bb, "Soilcolorbrown")
Xtr <- cbind("(Intercept)" = 1, as.matrix(train_data[, ff]))
Xte <- cbind("(Intercept)" = 1, as.matrix(test_data[, ff]))
n <- nrow(Xtr); p <- ncol(Xtr)

# black-box outputs: probabilities and hard labels Y1 = I(p_BB > 70th percentile)
pbb_tr <- s1$probs[[bb]]$train; pbb_te <- s1$probs[[bb]]$test
cut_bb <- quantile(pbb_tr, 0.7)
Y1_tr <- as.integer(pbb_tr > cut_bb); Y1_te <- as.integer(pbb_te > cut_bb)

# hard labels of a surrogate: its own 70th-percentile cut-off on the training scores
hard <- function(beta, X = Xtr) {
  ps <- as.numeric(sigmoid(X %*% beta)); as.integer(ps > quantile(sigmoid(Xtr %*% beta), 0.7))
}

# LR surrogate (MLE on Y1): benchmark, and H(y, p^s_LR) for the hyperprior
lr_sur <- suppressWarnings(glm(Y1_tr ~ Xtr - 1, family = binomial))
b_lr <- coef(lr_sur); names(b_lr) <- colnames(Xtr)
se_lr <- sqrt(diag(vcov(lr_sur))); names(se_lr) <- colnames(Xtr)

# hyperparameters of the reference (SHC) paper with the classification errors replaced by
# the cross-entropies on the true labels (the two terms of the surrogate loss):
#   a = c = (n - p)/2,  b = a * H(y, p^s_LR),  d = a * H(y, p_BB)
ce_bb  <- mean(Y1_tr != ytr)                         # black box vs ground truth (reported only)
h_sur <- calc_ce(ytr, as.numeric(sigmoid(Xtr %*% b_lr)))
h_bb  <- calc_ce(ytr, pbb_tr)
hp <- list(a = (n - p) / 2, c = (n - p) / 2, b = (n - p) / 2 * h_sur, d = (n - p) / 2 * h_bb)

# fidelity to the black box and accuracy on the ground truth for a coefficient vector
evaluate <- function(name, beta, X, y, pbb, Y1) {
  ps <- as.numeric(sigmoid(X %*% beta)); cls <- hard(beta, X)
  data.frame(Surrogate = name,
             Fid_Acc = mean(cls == Y1),
             Fid_Recall = sum(cls & Y1) / sum(Y1),
             Fid_Spec = sum(!cls & !Y1) / sum(!Y1),
             Fid_LogLoss = calc_ce(Y1, ps),                  # H(Y1, p^s): cross-entropy on the black-box labels
             KL = mean(kl_bern(pbb, ps)),
             Dloss = calc_ce(y, ps) - calc_ce(y, pbb),
             Acc_y = mean(cls == y),
             Recall_y = sum(cls & y) / sum(y),
             Spec_y = sum(!cls & !y) / sum(!y),
             AUC_y = as.numeric(auc(y, ps, quiet = TRUE)),
             LogLoss_y = calc_ce(y, ps))
}

# multi-chain diagnostics (posterior package): bulk ESS and rank-normalised R-hat
chain_diag <- function(chains, burn) {
  arr <- simplify2array(lapply(chains, function(B) B[(burn + 1):nrow(B), , drop = FALSE]))
  arr <- aperm(arr, c(1, 3, 2))                       # iterations x chains x parameters
  dimnames(arr)[[3]] <- colnames(Xtr)
  s <- summarise_draws(as_draws_array(arr), "rhat", "ess_bulk")
  s$variable <- colnames(Xtr); s
}
