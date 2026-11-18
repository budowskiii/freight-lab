"""Loading + cleaning. Uses no labels; weight-fill medians must be learned on each training fold."""
from __future__ import annotations
import numpy as np
import pandas as pd

TARGET = "posted_rate"


def load(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def clean(df: pd.DataFrame, weight_medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Return (cleaned frame, weight medians by equipment).

    Pass the medians learned on train when cleaning validation so nothing leaks.
    """
    df = df.copy()

    # Treat negative weights as sign errors, an assumption rather than verified ground truth.
    df["weight"] = df["weight"].abs()

    # 2) Missing weight: flag it, then fill with the equipment median.
    df["weight_missing"] = df["weight"].isna().astype(int)
    if weight_medians is None:
        weight_medians = df.groupby("equipment")["weight"].median()
    df["weight"] = df["weight"].fillna(df["equipment"].map(weight_medians))

    # 3) Missing market_index: it is a *daily* level (within-day std ~0.025),
    #    so fill from other loads on the same date; interpolate if the whole date is empty.
    df["market_index_missing"] = df["market_index"].isna().astype(int)
    daily = df.groupby("date")["market_index"].mean()
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max())).interpolate(limit_direction="both")
    df["market_index"] = df["market_index"].fillna(df["date"].map(daily))

    return df, weight_medians
