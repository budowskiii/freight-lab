"""Two-stage model on log(rate per mile), with configurable feature sets.

Stage 1: robust (Huber/IRLS) linear model, optionally WITH a time trend. A linear term can extrapolate
         forward in time, which trees cannot -- and the target set is after all training data.
Stage 2: LightGBM (Huber loss) on the stage-1 residuals. It never sees the date.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb

T0 = pd.Timestamp("2025-01-01")

# stage-1 columns the linear model can use (names are the pieces you can switch on/off)
S1_DEFAULT = ["log_dist", "inv_dist", "equipment", "weight", "market_index", "quote_signal", "t"]
LGB_DEFAULT = dict(objective="huber", alpha=0.15, n_estimators=700, learning_rate=0.03, num_leaves=15,
                   min_child_samples=60, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=2, verbose=-1)


def design(X: pd.DataFrame, dates: pd.Series, cols: list[str]) -> pd.DataFrame:
    D = pd.DataFrame({"c": 1.0}, index=X.index)
    for c in cols:
        if c == "equipment":
            D["reef"] = (X["equip"] == 2) * 1.0
            D["flat"] = (X["equip"] == 1) * 1.0
        elif c == "t":
            D["t"] = (dates - T0).dt.days.astype(float).values / 100.0
        elif c == "weight":
            D["weight"] = X["weight"] / 1e4
        elif c == "distance":
            D["distance"] = X["distance"] / 1000.0
        else:
            D[c] = X[c]
    return D


def huber_irls(A: np.ndarray, y: np.ndarray, iters: int = 8, delta: float = 0.08) -> np.ndarray:
    w = np.ones(len(y))
    for _ in range(iters):
        sw = np.sqrt(w)
        beta = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
        r = np.abs(y - A @ beta)
        w = np.where(r <= delta, 1.0, delta / np.maximum(r, 1e-9))
    return beta


class FreightModel:
    def __init__(self, s1_cols=None, s2_cols=None, lgb_params=None, delta=0.08, seeds=(0, 1, 2)):
        self.s1_cols = list(S1_DEFAULT if s1_cols is None else s1_cols)
        self.s2_cols = s2_cols            # None = all columns passed in
        self.params = {**LGB_DEFAULT, **(lgb_params or {})}
        self.delta, self.seeds = delta, seeds

    def _x2(self, X):
        return X if self.s2_cols is None else X[self.s2_cols]

    def fit(self, X, dates, distance, rate):
        y = np.log(rate / distance).values
        D = design(X, dates, self.s1_cols)
        self.beta_ = huber_irls(D.values, y, delta=self.delta)
        resid = y - D.values @ self.beta_
        self.trees_ = [lgb.LGBMRegressor(**self.params, random_state=s).fit(self._x2(X), resid) for s in self.seeds]
        self.cols_ = list(D.columns)
        return self

    def predict(self, X, dates, distance) -> np.ndarray:
        base = design(X, dates, self.s1_cols).values @ self.beta_
        corr = np.mean([m.predict(self._x2(X)) for m in self.trees_], axis=0)
        return np.exp(base + corr) * distance.values


class BlendModel:
    """Final model: average of the dollar predictions of model A (full features) and model D (selected + tuned)."""
    def __init__(self, seeds=(0, 1, 2)):
        from src.config import A, D
        self.members = [FreightModel(seeds=seeds, **A), FreightModel(seeds=seeds, **D)]

    def fit(self, X, dates, distance, rate):
        for m in self.members:
            m.fit(X, dates, distance, rate)
        return self

    def predict(self, X, dates, distance) -> np.ndarray:
        return np.mean([m.predict(X, dates, distance) for m in self.members], axis=0)
