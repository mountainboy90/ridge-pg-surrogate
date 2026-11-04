# Ridge-anchored Pólya–Gamma surrogates for interpreting black-box classifiers

Code and results for the paper:

> Sharma, A.S.; Rajesh; Sangal, P.K. *Ridge-Anchored Pólya–Gamma Surrogates for Interpreting
> Black-Box Classifiers: A Surrogate Loss Linking Fidelity and Ground Truth.* Submitted to *Entropy* (MDPI), 2026.

The study explains an XGBoost model of maize suitability with three global surrogates fitted to
the black box's hard labels: logistic regression, ridge regression, and the proposed Ridge-PG
surrogate. Ridge-PG chooses a ridge penalty by the cross-entropy on the true labels (stage 1) and then
samples a hierarchical posterior with a Pólya–Gamma Gibbs sampler (stage 2), whose prior variance
carries the black box's cross-entropy. The surrogates are compared with TreeSHAP, the true labels and
the black box (KL divergence and the surrogate loss), with an ablation and a 39-run robustness study.

## Repository layout

| Path | Contents |
|---|---|
| `R code/00_prepare_data.R` | Rebuilds `data/` from the raw dataset: cleaning, encoding, split, black-box selection |
| `R code/00_setup.R` | Shared setup: data, black-box outputs, hard labels, hyperparameters, metric helpers |
| `R code/pg_gibbs.R` | Pólya–Gamma Gibbs sampler for the hierarchical g-prior surrogate |
| `R code/01_surrogates.R` | Ridge benchmark, stage-1 ridge anchor, Ridge-PG and its ablation (4 chains each) |
| `R code/02_evaluate.R` | Black box vs. logistic regression, fidelity and accuracy, agreement with TreeSHAP, ablation, Figures 1–2 |
| `R code/03_robustness.R` | 39 re-runs: 30 random splits and 9 label cut-offs |
| `R code/04_robustness_figure.R` | Figure 3 |
| `results/` | All tables of the paper as CSV files, run logs and the saved posterior draws (`surrogates.rds`) |
| `figures/` | Figures 1–3 (PDF and PNG) |
| `data/` | **Not included** (see below) |

## Data

The data are the *Crop Recommendation using Soil Properties and Weather Prediction* dataset:

> Alemu, S. *Crop Recommendation using Soil Properties and Weather Prediction Dataset*. Mendeley Data, V1, 2024.
> https://doi.org/10.17632/8v757rr4st.1

The dataset is licensed **CC BY-NC-ND 4.0**, which does not allow redistribution of modified copies.
This repository therefore includes neither the raw nor the processed data. To rebuild them:

1. Download `Crop Recommendation using Soil Properties and Weather Prediction.csv` from the DOI above.
2. Save it, under that name, as `data/raw/Crop Recommendation using Soil Properties and Weather Prediction.csv`.
3. Run `Rscript "R code/00_prepare_data.R"` from the repository root.

The script writes three files to `data/`:

| File | Contents |
|---|---|
| `data/train_data_maize.csv` | Training records (80%, stratified, seed 123): standardised continuous features, one-hot soil colours, `target` = 1 for maize |
| `data/test_data_maize.csv` | Test records (20%), same format |
| `data/step1.rds` | List with `final_features` (22 features), `best_bb` (`"XGB"`), the fitted XGBoost `model`, the training and test probabilities of every candidate model (`probs`) and their 5-fold CV log-losses (`cv`) |

The steps are those of Section 3.1 of the paper: the soil-colour spellings are harmonised and one-hot
encoded, the continuous variables are standardised, the records are split 80/20 by stratified sampling,
soil colours with fewer than five training records are dropped, and a variance-inflation filter keeps
11 continuous variables. A random forest, XGBoost and a neural network are then compared by 5-fold
cross-validated log-loss, and the best one (XGBoost) is refitted on all training records.
With the package versions below, the script reproduces the files used in the paper exactly.

## Requirements

R 4.5.1 with these packages (versions used):

| Package | Version | Package | Version |
|---|---|---|---|
| xgboost | 3.1.2.1 | pROC | 1.19.0.1 |
| glmnet | 4.1.10 | shapviz | 0.10.3 |
| BayesLogit | 2.1 | ggplot2 | 4.0.0 |
| posterior | 1.7.0 | patchwork | 1.3.2 |
| dplyr | 1.1.4 | tidyr | 1.3.1 |
| caret | 7.0.1 | randomForest | 4.7.1.2 |
| torch | 0.16.1 | car | 3.1.3 |

`caret`, `randomForest`, `torch` and `car` are needed only by `00_prepare_data.R`.

## How to run

Run the scripts in order from the repository root (the folder that contains `R code/`):

```bash
Rscript "R code/00_prepare_data.R"
Rscript "R code/01_surrogates.R"
Rscript "R code/02_evaluate.R"
Rscript "R code/03_robustness.R"
Rscript "R code/04_robustness_figure.R"
```

| Script | Writes | Approximate time |
|---|---|---|
| `00_prepare_data.R` | `data/train_data_maize.csv`, `data/test_data_maize.csv`, `data/step1.rds` | 11 min (mostly the neural network's cross-validation) |
| `01_surrogates.R` | `results/surrogates.rds` | 4 min |
| `02_evaluate.R` | `results/*.csv` (main tables), `figures/fig_coef_vs_shap.*`, `figures/fig_shap_dependence.*` | 1 min |
| `03_robustness.R` | `results/robustness_*.csv`, `results/03_log.txt` | 6 min (runs in parallel) |
| `04_robustness_figure.R` | `figures/fig_robustness.*` | seconds |

All random seeds are fixed in the scripts.

## Results files

| File | Paper |
|---|---|
| `blackbox_vs_lr.csv` | Table 1: XGBoost vs. logistic regression on the true labels |
| `diagnostics.csv` | Gibbs sampler diagnostics (R-hat, ESS, run time) |
| `fidelity_accuracy.csv` | Table 2: fidelity to the black box and accuracy on the true labels |
| `shap_agreement.csv`, `shap_agreement_posterior.csv` | Table 3: agreement with TreeSHAP, with posterior intervals |
| `ablation_bb_term.csv` | Table 4: Ridge-PG with and without the black box's cross-entropy |
| `coefficients.csv` | Table 5: coefficients, credible intervals and TreeSHAP summaries |
| `robustness_runs.csv` | Every measure for every surrogate in all 39 re-runs |
| `robustness_splits_summary.csv`, `robustness_head_to_head.csv`, `robustness_cutoffs.csv` | Tables 6–7 and Figure 3 |

## License

- **Code** (`R code/`): MIT License, see [`LICENSE`](LICENSE).
- **Results and figures** (`results/`, `figures/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Data**: not included; the original dataset is licensed CC BY-NC-ND 4.0 by its author (see *Data*).

## Citation

If you use this code, please cite the paper above. A Zenodo DOI for this repository will be added here.
