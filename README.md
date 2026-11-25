# What Is the Best Repair for Rare Positives?

A cost-sensitive comparison of imbalance-handling strategies for credit card fraud detection using logistic regression. This project investigates whether class weighting, random undersampling, or random oversampling improves upon a simple threshold-adjusted baseline when fraudulent transactions are rare.

## Dataset

The experiments use the [Credit Card Fraud Detection dataset (OpenML ID 1597)](https://www.openml.org/d/1597).

| Statistic | Original | After deduplication |
| --- | ---: | ---: |
| Transactions | 284,807 | 283,726 |
| Fraudulent transactions | 492 | 473 |
| Legitimate transactions | 284,315 | 283,253 |
| Fraud prevalence | 0.173% | 0.167% |

Exact duplicate records are removed before splitting. The model uses 29 predictors (`V1`–`V28` and `Amount`). `Time` is excluded from the predictors but retained for chronological evaluation. `Class` is the binary target (0 = legitimate, 1 = fraud).

## Methods

Four logistic regression models are compared through nine decision strategies:

- Flag nothing (non-model reference).
- Unrepaired logistic regression at threshold 0.5 and the theoretical cost-based threshold.
- Balanced class-weighted logistic regression at threshold 0.5 and an out-of-fold (OOF) selected threshold.
- Random undersampling (1:1 class ratio) at threshold 0.5 and an OOF-selected threshold.
- Random oversampling (1:1 class ratio) at threshold 0.5 and an OOF-selected threshold.

The threshold-adjusted unrepaired model is the main comparison reference. For a false-negative to false-positive cost ratio of `r:1`, its theoretical threshold is `1 / (1 + r)`, assuming calibrated probabilities. Repaired models instead select thresholds by minimizing cost on training-only OOF predictions.

## Experimental design

- Stratified 80/20 train/test splits with seeds `42`, `123`, `2026`, `423`, and `99`.
- Preliminary nested five-fold cross-validation on the seed-42 training set.
- Regularization grid: `C = [0.001, 0.01, 0.1, 1, 10, 100]`, selected by average precision (AP).
- Scaling and resampling are performed inside training folds using pipelines.
- Final held-out evaluation on all five splits, with seed 42 as the primary comparison.
- Chronological robustness check using the earliest approximately 80% of transactions for training and the latest 20% for testing, with expanding-window cross-validation.
- 1,000 stratified bootstrap resamples for 95% confidence intervals, including paired cost differences against the threshold-adjusted baseline.

### Evaluation metrics

The primary metric is misclassification cost per 1,000 transactions:

$$
\mathrm{Cost}_{1000}=\frac{1000}{N}(FP+r\,FN)
$$

where `FP` and `FN` are false-positive and false-negative counts, respectively, and `r` is 5, 10, or 20. Lower cost is better. Average precision is a secondary metric of ranking performance.

## Results

### Mean held-out cost across five random splits

| Strategy | 5:1 | 10:1 | 20:1 |
| --- | ---: | ---: | ---: |
| Flag nothing | 8.3706 | 16.7413 | 33.4825 |
| Baseline (0.5) | 3.5914 | 7.0454 | 13.9534 |
| Threshold-adjusted baseline | 2.6997 | 4.4514 | 7.1970 |
| Class weighting (0.5) | 25.0485 | 25.8415 | 27.4275 |
| Class weighting (OOF) | 2.0865 | 3.8699 | 7.2780 |
| Undersampling (0.5) | 2.0830 | 3.6866 | 6.8939 |
| Undersampling (OOF) | 2.1006 | **3.6549** | **6.7282** |
| Oversampling (0.5) | 24.8652 | 25.6406 | 27.1913 |
| Oversampling (OOF) | **2.0794** | 3.8734 | 7.3098 |

Bold values indicate the lowest observed mean cost, not statistically established winners.

### Key findings

- Threshold adjustment consistently improves the unrepaired baseline across the five splits and three cost ratios.
- Class weighting and oversampling at threshold 0.5 generate excessive false positives; OOF threshold selection substantially reduces their cost.
- Random undersampling is competitive across random and chronological evaluations, but no single repair is conclusively superior across all settings.
- Ranking performance and operational cost are distinct objectives.

## Reproducing the experiments

1. Clone the repository and enter its directory:

   ```bash
   git clone https://github.com/kims12uci/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. Create and activate a virtual environment (the recorded experiment used Python 3.14.6), then install dependencies:

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   macOS/Linux:

   ```bash
   source .venv/bin/activate
   ```

   ```bash
   python -m pip install -r requirements.txt
   python -m pip install notebook
   ```

3. Download the ARFF dataset from [OpenML](https://www.openml.org/d/1597), rename it to `creditcard.arff` if necessary, and place it in the repository root alongside the notebook. The dataset is not required to be committed to GitHub.
4. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open `proj_code.ipynb` (the final notebook, renamed from `proj_code(8).ipynb`) and run all cells in order. The notebook generates result tables, figures, prediction arrays, and environment metadata. If dependencies are already installed, the notebook's installation cell can be skipped.

## Repository contents

| Path | Description |
| --- | --- |
| `proj_code.ipynb` | Complete experimental workflow |
| `requirements.txt` | Recorded package versions |
| `preliminary_results/` | Nested-CV results and fold-level diagnostics |
| `final_results/` | Held-out costs, AP, thresholds, chronological results, configuration and saved predictions |
| `final_cost_comparison.pdf` | Primary cost figure with bootstrap intervals |
| `five_seed_robustness.pdf` | Mean costs and standard deviations across splits |
| `final_pr_curves.pdf` | Primary test precision-recall curves |

The notebook writes figures to its working directory; they may be organized into a `figures/` folder for publication. Results directories are generated automatically when the notebook is run.

## Limitations

The five random test sets overlap, so across-seed variability is descriptive rather than independent replication. Bootstrap intervals condition on fitted predictions and observed class proportions and are not adjusted for multiple comparisons. The sampling ratios are fixed at 1:1, probability calibration is not guaranteed, and the dataset spans only two days. Exact-record deduplication does not eliminate all repeated predictor vectors.

## References

- [OpenML: Credit Card Fraud Detection (1597)](https://www.openml.org/d/1597)
- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [imbalanced-learn documentation](https://imbalanced-learn.org/stable/)
