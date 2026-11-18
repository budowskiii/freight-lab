"""Rolling-origin (expanding window) time splits: always train on the past, test on the next month."""
import pandas as pd

TEST_MONTHS = [7, 8, 9, 10]  # train = Jan..(m-1), test = m


def time_folds(df: pd.DataFrame):
    month = df["date"].dt.month
    for m in TEST_MONTHS:
        yield m, df.index[month < m], df.index[month == m]


def prepare_fold(frame, training_index, test_index):
    """Clean a chronological fold with training-only imputation statistics.

    Daily covariate means use the training and held-out month inputs, never
    their labels. This matches the assessment's batch-prediction setting.
    """
    from src.data import clean
    from src.features import build, daily_market

    training, medians = clean(frame.loc[training_index])
    test, _ = clean(frame.loc[test_index], medians)
    if training.date.max() >= test.date.min():
        raise ValueError('Training dates must precede test dates')
    daily = daily_market([training, test])
    return training, test, build(training, daily), build(test, daily)
