#=====================================================================
# Step 0: rebuild data/ from the raw dataset (Alemu 2024, https://doi.org/10.17632/8v757rr4st.1)
#   (A) clean soil colours, one-hot encode, standardise, stratified 80/20 split (seed 123)
#       -> data/train_data_maize.csv, data/test_data_maize.csv
#   (B) select the black box by 5-fold CV log-loss among a random forest, XGBoost and a
#       neural network, refit it and a logistic regression -> data/step1.rds
# Follows "IML AgroStats 2.R" and "01_blackbox_models.R" of the EP paper, with the same
# settings and seeds.
# Put the raw file at data/raw/Crop Recommendation using Soil Properties and Weather Prediction.csv
# and run from the repository root:  Rscript "R code/00_prepare_data.R"
# An optional argument gives another output folder, e.g. Rscript "R code/00_prepare_data.R" check
#=====================================================================
suppressPackageStartupMessages({
  library(caret); library(randomForest); library(xgboost); library(torch); library(pROC); library(car)
})
args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args)) args[1] else "data"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
raw_file <- "data/raw/Crop Recommendation using Soil Properties and Weather Prediction.csv"
if (!file.exists(raw_file)) stop("Raw data not found: ", raw_file, "\nDownload it from https://doi.org/10.17632/8v757rr4st.1")

calc_ce <- function(actual, predicted) {
  eps <- 1e-15
  predicted <- pmax(pmin(predicted, 1 - eps), eps)
  -mean(actual * log(predicted) + (1 - actual) * log(1 - predicted))
}

#------------------ (A) cleaning, encoding, standardisation, split -------------------
crop_data <- read.csv(raw_file, stringsAsFactors = FALSE)

# harmonise the free-text soil colours (spelling variants and notes)
clean_labels <- function(x) {
  x <- tolower(x)
  x <- gsub(";.* ", "", x)
  x <- gsub("\\(.*\\)", "", x)
  x <- gsub("replacement of inaccessible target", "", x)
  x <- trimws(x)
  x <- gsub("black;vertisol", "black", x)
  x <- gsub("lihgtish", "lightish", x)
  x <- gsub("red;luvisols", "red", x)
  x <- gsub("reddis", "reddish", x)
  x <- gsub("redish", "reddish", x)
  x <- gsub("broown", "brown", x)
  x <- gsub("perl", "pearl", x)
  x <- gsub("darkbrown", "dark brown", x)
  x <- gsub("dark greyish brown", "dark grayish brown", x)
  x <- gsub("reddishh", "reddish", x)
  x <- gsub("reddishh brown", "reddish brown", x)
  x <- gsub("reddishbrown", "reddish brown", x)
  x <- gsub("reddishh brown;moist", "reddish brown moinst", x)
  x <- gsub("reddish brown;moist", "reddish brown moinst", x)
  x <- gsub("dark reddishh brow", "reddish brown", x)
  x
}
crop_data$Soilcolor <- clean_labels(crop_data$Soilcolor)

onehot <- as.data.frame(predict(dummyVars(~ ., data = crop_data), newdata = crop_data))
names(onehot) <- gsub(" ", "_", names(onehot))
soil_cols <- grep("^Soilcolor", names(onehot), value = TRUE)
cont_cols <- setdiff(names(onehot), c(soil_cols, grep("^label", names(onehot), value = TRUE)))
stopifnot(length(soil_cols) == 23, length(cont_cols) == 27, "labelMaize" %in% names(onehot))

# continuous variables are standardised over all records, as in the original analysis
standardize <- function(x) (x - mean(x)) / sd(x)
dt_std <- as.data.frame(lapply(onehot[, cont_cols], standardize))
dt_std <- cbind(dt_std, onehot[, soil_cols])
dt_std$target <- onehot$labelMaize

set.seed(123)
train_index <- createDataPartition(dt_std$target, p = 0.8, list = FALSE)
train_data <- dt_std[train_index, ]
test_data  <- dt_std[-train_index, ]
write.csv(train_data, file.path(out_dir, "train_data_maize.csv"), row.names = FALSE)
write.csv(test_data,  file.path(out_dir, "test_data_maize.csv"),  row.names = FALSE)
cat("(A) train:", nrow(train_data), "records | test:", nrow(test_data), "records | maize share:",
    round(mean(dt_std$target), 3), "\n")

#------------------ (B) black-box selection -----------------------------------------
train_data <- read.csv(file.path(out_dir, "train_data_maize.csv"))
test_data  <- read.csv(file.path(out_dir, "test_data_maize.csv"))

soil <- grep("^Soilcolor", names(train_data), value = TRUE)
soil <- soil[colSums(train_data[, soil]) >= 5]                    # drop sparse soil colours
cont <- c("Ph", "K", "P", "N", "Zn", "S", "QV2M.W", "T2M_MAX.W", "T2M_MAX.Su",
          "WD10M", "CLOUD_AMT")                                    # variance-inflation-filtered set
final_features <- c(soil, cont)
cat("(B) VIF of the continuous features:\n")
print(round(vif(lm(target ~ ., data = train_data[, c(cont, "target")])), 2))

Xtr <- train_data[, final_features]; ytr <- train_data$target
Xte <- test_data[, final_features];  yte <- test_data$target

fit_rf <- function(X, y, mtry = 5, nodesize = 35, ntree = 350) {
  set.seed(100)
  randomForest(x = X, y = factor(y), ntree = ntree, mtry = mtry, nodesize = nodesize)
}
pred_rf <- function(m, X) predict(m, X, type = "prob")[, "1"]

fit_xgb <- function(X, y, max_depth = 3, eta = 0.03, nrounds = 400) {
  set.seed(101112)
  xgb.train(params = list(objective = "binary:logistic", eval_metric = "logloss",
                          max_depth = max_depth, eta = eta, subsample = 0.8,
                          colsample_bytree = 0.8, min_child_weight = 5, gamma = 1, nthread = 2),
            data = xgb.DMatrix(as.matrix(X), label = y), nrounds = nrounds, verbose = 0)
}
pred_xgb <- function(m, X) predict(m, xgb.DMatrix(as.matrix(X)))

fit_dnn <- function(X, y, epochs = 150, lr = 1e-3, wd = 1e-4) {
  torch_manual_seed(2026); set.seed(2026)
  net <- nn_sequential(nn_linear(ncol(X), 64), nn_relu(), nn_dropout(0.3),
                       nn_linear(64, 32), nn_relu(), nn_dropout(0.3), nn_linear(32, 1))
  opt <- optim_adam(net$parameters, lr = lr, weight_decay = wd)
  idx <- sample(nrow(X)); nv <- floor(0.15 * nrow(X))             # inner split for early stopping
  va <- idx[1:nv]; tr <- idx[-(1:nv)]
  xt <- torch_tensor(as.matrix(X[tr, ]), dtype = torch_float()); yt <- torch_tensor(matrix(y[tr]), dtype = torch_float())
  xv <- torch_tensor(as.matrix(X[va, ]), dtype = torch_float()); yv <- torch_tensor(matrix(y[va]), dtype = torch_float())
  lossf <- nn_bce_with_logits_loss()
  best <- Inf; best_state <- NULL; patience <- 0
  for (ep in 1:epochs) {
    net$train()
    for (b in split(sample(length(tr)), ceiling(seq_along(tr) / 128))) {
      opt$zero_grad(); l <- lossf(net(xt[b, ]), yt[b, ]); l$backward(); opt$step()
    }
    net$eval()
    vl <- with_no_grad(lossf(net(xv), yv)$item())
    if (vl < best - 1e-5) { best <- vl; patience <- 0
      best_state <- lapply(net$state_dict(), function(t) t$clone())
    } else if ((patience <- patience + 1) >= 15) break
  }
  net$load_state_dict(best_state); net$eval(); net
}
pred_dnn <- function(m, X) {
  with_no_grad(as.numeric(torch_sigmoid(m(torch_tensor(as.matrix(X), dtype = torch_float())))))
}

learners <- list(RF  = list(fit = fit_rf,  pred = pred_rf),
                 XGB = list(fit = fit_xgb, pred = pred_xgb),
                 DNN = list(fit = fit_dnn, pred = pred_dnn))

# 5-fold cross-validated log-loss for model selection
set.seed(123)
folds <- createFolds(factor(ytr), k = 5)
cv <- sapply(names(learners), function(nm) sapply(folds, function(f) {
  m <- learners[[nm]]$fit(Xtr[-f, ], ytr[-f])
  calc_ce(ytr[f], learners[[nm]]$pred(m, Xtr[f, ]))
}))
print(round(rbind(mean = colMeans(cv), sd = apply(cv, 2, sd)), 3))
best_bb <- colnames(cv)[which.min(colMeans(cv))]
cat("Selected black box (lowest CV log-loss):", best_bb, "\n")

# refit on all training data; logistic regression on the true labels as a benchmark
models <- lapply(learners, function(L) L$fit(Xtr, ytr))
logit_model <- glm(target ~ ., data = train_data[, c(final_features, "target")], family = binomial)
probs <- lapply(names(learners), function(nm) list(train = learners[[nm]]$pred(models[[nm]], Xtr),
                                                    test  = learners[[nm]]$pred(models[[nm]], Xte)))
names(probs) <- names(learners)
probs$LR <- list(train = as.numeric(predict(logit_model, train_data, type = "response")),
                 test  = as.numeric(predict(logit_model, test_data,  type = "response")))

saveRDS(list(final_features = final_features, best_bb = best_bb,
             model = if (best_bb == "DNN") NULL else models[[best_bb]],
             probs = probs, logit_model = logit_model, cv = cv),
        file.path(out_dir, "step1.rds"))
cat("Wrote", file.path(out_dir, c("train_data_maize.csv", "test_data_maize.csv", "step1.rds")), sep = "\n  ")
