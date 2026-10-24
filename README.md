# GDSC gene-set study

Compare 50, 200, and 1,000 expression genes for predicting one anticancer drug's
LN_IC50 in cancer cell lines. Implements elastic net, random forest, and a
training-mean baseline with shared five-fold outer and three-fold inner CV.

**Status:** real-data preparation is complete. MG-132 (drug ID 1862) was selected
by maximum coverage: 944 matched cell lines. The five-fold experiment is running.
The frozen design and checksums are in `results/study_design/`.

## Setup

Python 3.10 or newer:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[notebook]"
python -m unittest discover -s tests -v
gene-study demo --output outputs/demo
```

The demo deliberately uses synthetic samples and smaller gene counts (5, 20,
50), three outer folds, two inner folds, and 20 trees for a fast smoke test.
Its performance must not be reported as GDSC research findings.

## Data and execution

See [data preparation](docs/data-preparation.md) for the input contract and
provenance requirements. Store downloads under `data/raw/` (ignored by Git).
Use one screening dataset and record its release. Do not pool GDSC1 and GDSC2.

```bash
gene-study coverage --expression data/processed/expression.csv --responses data/processed/responses.csv --dataset GDSC2
gene-study run --expression data/processed/expression.csv --responses data/processed/responses.csv --dataset GDSC2 --drug DRUG_ID --release RELEASE_LABEL --output outputs/first-study
```

Select and record the drug based on matched sample coverage before running model
comparisons. `DRUG_ID` and `RELEASE_LABEL` are placeholders for custom runs.
For this study, the frozen choice is MG-132, GDSC2 release 27Oct23.
Output folders must be empty to prevent accidental overwrites. The real study
uses 200 trees; `--jobs` controls parallel search workers (default one).

## Method

Each inner training fold independently fits missingness filtering (maximum 20%),
median imputation, zero-variance filtering, and absolute Pearson gene ranking.
Elastic net also fits standardization within that fold. A grid search selects
parameters by inner-fold RMSE, refits on the outer training set, and predicts
the untouched outer test set. Gene counts fail explicitly if a training fold
does not contain enough eligible genes. Exact correlation ties retain input
gene order. Cell lines occur in exactly one outer test fold.

Elastic net searches alpha 0.01/0.1/1 and l1 ratio 0.1/0.5/0.9. Random forest
searches max features 0.3/1.0 and minimum leaf size 2/5. These are initial,
prespecified small grids. Default random seed is 42.

Outputs include fold RMSE/MAE/R2, mean and sample standard deviation, out-of-fold
predictions, fold assignments, selected genes, selection frequency, tuning
parameters, software versions, input SHA-256 hashes, and an RMSE plot. Fold-level
metrics support paired comparisons; standard deviation is not a confidence
interval. Elastic-net weights refer to standardized expression; forest weights
are impurity importances and can be biased. Interpretation remains exploratory.

## Current study

The downloaded expression archive is `rnaseq_merged_20250922.zip`.
Use the adapter described in [data preparation](docs/data-preparation.md).
The response workbook uses Sanger model IDs directly. Raw and normalized matrices
remain in ignored local directories; source hashes and compact research results
are versioned. The September 2025 expression release is intentional.

After preparing the local inputs and freezing the selection, run the complete
training, report, and executed-notebook workflow with:

```bash
python scripts/finish_study.py
```

Progress is recorded in `outputs/study_status.json`. Each completed model/fold
configuration is saved under `results/mg132_complete/checkpoints/`; rerunning
the command restores matching checkpoints. Changes to the data, configuration,
or recorded software versions require a fresh output directory. Checkpoints
and raw data are excluded from Git. The final notebook checks shared test folds,
prediction coverage, the training-mean baseline, and recomputed RMSE values.

## Next milestones

- Run the full 50/200/1,000-gene study and inspect convergence warnings.
- Compare paired fold errors and investigate tissue composition as a limitation.
- Review selected genes against drug biology and write the final report.

The notebook in `notebooks/` provides a starting workflow. This study evaluates
cell-line prediction within one dataset; it does not establish patient utility.
