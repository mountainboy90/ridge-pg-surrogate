#=====================================================================
# Polya-Gamma Gibbs sampler for the hierarchical g-prior surrogate (sourced by 01 and 03)
#=====================================================================
# Y ~ Bern(sigmoid(X beta)); omega_i | beta ~ PG(1, x_i' beta);
# beta | omega, s ~ N(m, V), V = (X' Omega X + G / (g s1 s2))^-1, m = V (X' kappa + G mu0 / (g s1 s2))
# s1 | beta, s2 ~ IG(a + r/2, b + Q / (2 g s2)), s2 | beta, s1 ~ IG(c + r/2, d + Q / (2 g s1)),
# with kappa = Y - 1/2, Q = (beta - mu0)' G (beta - mu0) and r = rank(G).
# G = X'X as stated in Algorithm 1 (intercept penalised), or the centred X_c'X_c with a flat
# prior on the intercept (as in glmnet's ridge, which never penalises the intercept)
pg_gibbs <- function(X, Y, g, mu0, hp, start, iters, seed, flat_intercept = TRUE) {
  set.seed(seed)
  if (flat_intercept) {
    Xc <- scale(X[, -1], scale = FALSE); G <- matrix(0, ncol(X), ncol(X)); G[-1, -1] <- crossprod(Xc)
  } else G <- crossprod(X)
  r <- ncol(X) - flat_intercept
  Gmu <- G %*% mu0; Xk <- crossprod(X, Y - 0.5)
  s1 <- hp$b / (hp$a - 1); s2 <- hp$d / (hp$c - 1); beta <- start
  B <- matrix(NA_real_, iters, ncol(X), dimnames = list(NULL, colnames(X))); S <- numeric(iters)
  for (t in seq_len(iters)) {
    om <- rpg(nrow(X), 1, as.numeric(X %*% beta))
    s <- g * s1 * s2
    R <- chol(crossprod(X * om, X) + G / s)
    m <- backsolve(R, forwardsolve(t(R), Xk + Gmu / s))
    beta <- as.numeric(m + backsolve(R, rnorm(ncol(X))))
    Q <- as.numeric(crossprod(beta - mu0, G %*% (beta - mu0)))
    s1 <- 1 / rgamma(1, hp$a + r / 2, rate = hp$b + Q / (2 * g * s2))
    s2 <- 1 / rgamma(1, hp$c + r / 2, rate = hp$d + Q / (2 * g * s1))
    B[t, ] <- beta; S[t] <- s1 * s2
  }
  list(beta = B, s12 = S)
}
