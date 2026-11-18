"""Label-free features for batch prediction with same-day covariates available."""
from __future__ import annotations
import numpy as np
import pandas as pd

EQUIP = {"Dry Van": 0, "Flatbed": 1, "Reefer": 2}


def daily_market(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Daily market level from the (cleaned) market_index + quote_signal of every row we have.
    No labels are pooled. This assumes same-day batch inputs are available; it
    is not a claim of suitability for real-time prediction before those inputs exist."""
    allr = pd.concat([f[["date", "market_index", "quote_signal"]] for f in frames])
    d = allr.groupby("date").mean()
    d = d.reindex(pd.date_range(d.index.min(), d.index.max())).interpolate(limit_direction="both")
    d.columns = ["mi_day", "qs_day"]
    d["mi_roll7"] = d["mi_day"].rolling(7, min_periods=1).mean()
    d["mi_roll14"] = d["mi_day"].rolling(14, min_periods=1).mean()
    d["mi_diff7"] = d["mi_day"] - d["mi_day"].shift(7).bfill()
    d["mi_diff1"] = d["mi_day"] - d["mi_day"].shift(1).bfill()
    return d


def build(df: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["distance"] = df["distance"]
    X["log_dist"] = np.log(df["distance"])
    X["inv_dist"] = 1.0 / df["distance"]          # short hauls: rate-per-mile ~ a + b/distance
    X["equip"] = df["equipment"].map(EQUIP)
    X["weight"] = df["weight"]
    X["weight_missing"] = df["weight_missing"]
    X["market_index"] = df["market_index"]
    X["quote_signal"] = df["quote_signal"]
    d = daily.reindex(df["date"]).reset_index(drop=True)
    d.index = df.index
    X = X.join(d)
    X["mi_dev"] = X["market_index"] - X["mi_day"]
    X["qs_dev"] = X["quote_signal"] - X["qs_day"]
    # geography: coordinates are noisy -> use them for direction/region only
    dlat = df["delivery_lat"] - df["pickup_lat"]
    dlon = df["delivery_lon"] - df["pickup_lon"]
    X["dlat"], X["dlon"] = dlat, dlon
    X["bearing"] = np.degrees(np.arctan2(dlon, dlat))
    for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        X[c] = df[c]
    return X
