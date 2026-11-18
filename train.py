"""Validate the blend chronologically, then fit all labels and export predictions.

Run ``python train.py`` from the repository root. Original data files are read
only; completed December predictions are saved in ``december_predictions.csv``.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.cv import prepare_fold, time_folds
from src.data import TARGET, clean, load
from src.features import build, daily_market
from src.metrics import report
from src.model import BlendModel

ROOT = Path(__file__).resolve().parent


def main():
    """Run fold-safe CV and write predictions without overwriting input CSVs."""
    reports = ROOT / 'reports'
    reports.mkdir(exist_ok=True)
    raw_training = load(ROOT / 'data/train_test.csv')
    rows = []
    for month, training_index, test_index in time_folds(raw_training):
        training, test, inputs, test_inputs = prepare_fold(raw_training, training_index, test_index)
        model = BlendModel().fit(inputs, training.date, training.distance, training[TARGET])
        prediction = model.predict(test_inputs, test.date, test.distance)
        non_extreme = ((test[TARGET] / test.distance > 1) & (test[TARGET] / test.distance < 4)).to_numpy()
        rows.append({'test_month': month, 'set': 'all', **report(test[TARGET], prediction)})
        # Diagnostic only: this label-defined subset is never used to remove training rows.
        rows.append({'test_month': month, 'set': 'clean', **report(test[TARGET].to_numpy()[non_extreme], prediction[non_extreme])})
        print(f'Finished chronological CV month {month}', flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(reports / 'cv_results.csv', index=False)
    print(results.round(2).to_string(index=False))
    print(results.groupby('set').mean(numeric_only=True).drop(columns='test_month').round(2))

    training, medians = clean(raw_training)
    validation, _ = clean(load(ROOT / 'data/validation.csv'), medians)
    daily = daily_market([training, validation])
    inputs = build(training, daily)
    final = BlendModel().fit(inputs, training.date, training.distance, training[TARGET])
    for index, member in enumerate(final.members):
        print(f'Member {index} stage-1 coefficients:', dict(zip(member.cols_, member.beta_.round(4))))

    prediction = final.predict(build(validation, daily), validation.date, validation.distance)
    template = pd.read_csv(ROOT / 'data/validation_predictions_template.csv')
    prediction_by_id = pd.Series(prediction, index=validation.load_id)
    template['predicted_rate'] = template.load_id.map(prediction_by_id).round(2)
    if template.predicted_rate.isna().any() or not np.isfinite(template.predicted_rate).all() or (template.predicted_rate <= 0).any():
        raise ValueError('Validation predictions must be finite, positive and cover every load ID')
    template.to_csv(ROOT / 'validation_predictions.csv', index=False)

    december = pd.read_csv(ROOT / 'data/december_chart_inputs.csv', parse_dates=['date'])
    city_coordinates = pd.concat([
        training[['pickup', 'pickup_lat', 'pickup_lon']].rename(columns=lambda column: column.replace('pickup', 'city')),
        training[['delivery', 'delivery_lat', 'delivery_lon']].rename(columns=lambda column: column.replace('delivery', 'city')),
    ]).drop_duplicates('city').set_index('city')
    december_inputs = december.copy()
    for side in ['pickup', 'delivery']:
        for coordinate in ['lat', 'lon']:
            december_inputs[f'{side}_{coordinate}'] = december_inputs[side].map(city_coordinates[f'city_{coordinate}'])
    daily_december = daily.reindex(december_inputs.date)
    december_inputs['market_index'] = daily_december.mi_day.to_numpy()
    december_inputs['quote_signal'] = daily_december.qs_day.to_numpy()
    december_inputs['weight_missing'] = 0
    december_inputs['market_index_missing'] = 0
    december['predicted_rate'] = final.predict(build(december_inputs, daily), december_inputs.date, december_inputs.distance).round(2)
    if not np.isfinite(december.predicted_rate).all() or (december.predicted_rate <= 0).any():
        raise ValueError('December predictions must be finite and positive')
    december['date'] = december.date.dt.strftime('%Y-%m-%d')
    december.to_csv(ROOT / 'december_predictions.csv', index=False)
    print('Saved 12,000 validation predictions and 31 December predictions.', flush=True)


if __name__ == '__main__':
    main()
