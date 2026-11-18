# Freight Rate Prediction

Predict total `posted_rate` dollars for 12,000 November–December 2025 loads using
48,000 labeled January–October loads. The final model averages two two-stage
models: a robust linear time-trend model followed by LightGBM on log-rate-per-mile
residuals. City names are not model inputs.

## Run

From this repository's root, with Python 3.13:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v

# Copy the supplied assessment datasets into these paths first:
# data/train_test.csv and data/validation.csv
python train.py
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```

The small December input and prediction template are included in `data/`.
The full training and validation datasets are intentionally excluded from Git.
Place your supplied assessment files at `data/train_test.csv` and
`data/validation.csv` (rename hyphens to underscores if necessary). Training reads
all input CSVs without modifying them. It writes:

- `validation_predictions.csv`: exactly `load_id,predicted_rate`, all 12,000 IDs.
- `december_predictions.csv`: the fixed 31 December rows with predicted rates.
- `reports/cv_results.csv`: rolling chronological validation metrics.

The unmodified assessment `score.py` validates both prediction files and creates
`scorer_results/candidate_december.png`. It does **not** calculate hidden-test
accuracy. The current report is [Freight_Rate_Report.pdf](Freight_Rate_Report.pdf).

## Validation and modeling choices

- Train January–June → test July; expand training one month at a time through
  January–September → October. Weight-fill medians are fitted within each fold.
- Negative weights are assumed to be sign errors and converted to absolute
  values. Missing weights use training-only equipment medians plus a flag.
  These are modeling assumptions, not verified corrections to ground truth.
- Same-day market means use input features only, including the held-out batch's
  inputs. This fits the supplied batch-prediction task; it assumes those inputs
  are available and is not a real-time forecasting evaluation.
- All labeled rows remain in training. The `clean` metric rows mean the
  label-defined diagnostic subset with $1 < rate/mile < $4; that name does not
  establish that excluded observations are corrupted. All-row MAE is primary.
- Initial selection used July–September. October was subsequently examined in
  finalist and blend comparisons, so it is not an untouched final test. Search
  and selection histories are preserved in `reports/`.
- The linear trend may extrapolate incorrectly. The trees do not receive explicit
  date but can still learn time-correlated covariates. Neither guarantees stable
  future relationships.

The December input lacks coordinates and market/quote fields. Coordinates are
looked up from the training city table. Per-date market and quote means come from
the supplied validation inputs, without labels. Thus December predictions can
vary with these recovered inputs as well as the date. No external geocoding is
used, and the supplied coordinates have not been geographically verified.

## Project layout

```text
train.py                     end-to-end CV, fit and predictions
src/                         cleaning, features, model, config, CV and metrics
experiments/                 historical feature/model search scripts
reports/                     experiment records, figures and report builder
data/                        December input/template; add raw train/validation locally
scorer_results/              provided scorer's December chart
validation_predictions.csv   required 12,000-load submission
december_predictions.csv     completed fixed December inputs
Freight_Rate_Report.pdf       validation, approach and December chart
```

Rebuild the PDF after training and scoring:

```bash
python -m pip install -r requirements-report.txt
python reports/make_report.py
```

Optional experiments run from the repository root, for example:

```bash
PYTHONPATH=. python experiments/compare_final.py
PYTHONPATH=. python experiments/blend_test.py
PYTHONPATH=. python experiments/shap_analysis.py
```

Historical feature-search logs are retained as exploration evidence; they are not
fresh independent test results. The finalist/blend checks and main pipeline use
fold-specific preprocessing. Three seeds are averaged within each model; the
final prediction is the 50/50 dollar-space average of models A and D. SHAP figures
summarize component log predictions rather than exact additive contributions to
that final dollar-space average.
