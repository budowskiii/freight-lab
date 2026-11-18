# Freight Rate Prediction

Predict total load prices for 12,000 November–December 2025 loads using
48,000 labeled January–October loads.

## Run

Use Python 3.13 and run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place the supplied assessment datasets at `data/train_test.csv` and
`data/validation.csv`. These raw datasets stay local and are excluded from Git.
The December input and prediction template are included in `data/`.

```bash
python train.py
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```

Training reads the input files without modifying them. It produces:

- `validation_predictions.csv`: exactly `load_id,predicted_rate`, all 12,000 IDs.
- `december_predictions.csv`: the 31 fixed December rows with predicted prices.
- `reports/cv_results.csv`: chronological validation results.

The provided scorer validates both prediction files and generates
[the December chart](scorer_results/candidate_december.png). It does not calculate
accuracy against the hidden validation labels.

## Model and validation

The final model averages two models in dollars. Each combines a robust linear
model with a time trend and LightGBM trained on log-rate-per-mile residuals.
Each member averages three random seeds. City names are not model features.

Validation trains on January–June and tests July, then expands training through
September to test October. Missing weights use equipment medians learned only
from each training fold. Average all-row error is approximately **$95.46 per
load**. October was reused during final model selection, so it is not an
untouched final test.

Negative weights are treated as possible sign errors. All labeled rows remain
in training; `clean` in the CV CSV is only a diagnostic subset with
$1 < rate/mile < $4, not proof that excluded prices are incorrect.

Daily market features use the supplied batch's input columns, never its target
prices. This assumes the same-day batch is available. December coordinates are
looked up from training cities; missing market and quote inputs use that day's
validation-input averages. No external geocoding is used. The linear trend can
extrapolate incorrectly, and coordinates are not geographically verified.

## Submission files

- [Report](Freight_Rate_Report.pdf): approach, validation and December chart.
- [Validation predictions](validation_predictions.csv).
- [December predictions](december_predictions.csv).
- `train.py` and `src/`: runnable solution.
- `requirements.txt`: training and scoring dependencies.
