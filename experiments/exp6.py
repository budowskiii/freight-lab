import numpy as np, pandas as pd, lightgbm as lgb
from src.data import load, clean, TARGET
from src.features import daily_market, build
from src.cv import time_folds
from src.metrics import report

tr, med = clean(load("data/train_test.csv")); va, _ = clean(load("data/validation.csv"), med)
daily = daily_market([tr, va]); X = build(tr, daily); y = tr[TARGET]; dist = tr["distance"]
rpm = y / dist; cm_all = (rpm > 1) & (rpm < 4); lrpm = np.log(rpm)
T0 = pd.Timestamp("2025-01-01"); t = (tr["date"] - T0).dt.days.astype(float)

def lin_design(X, t, with_t=True):
    D = pd.DataFrame({"c": 1.0, "logd": X.log_dist, "invd": X.inv_dist, "reef": (X.equip == 2) * 1.0, "flat": (X.equip == 1) * 1.0,
                      "w": X.weight / 1e4, "mi": X.market_index, "qs": X.quote_signal})
    if with_t: D["t"] = t / 100.0
    return D

def lgbm(**kw):
    p = dict(objective="huber", alpha=0.15, n_estimators=700, learning_rate=0.03, num_leaves=15, min_child_samples=60,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1); p.update(kw); return lgb.LGBMRegressor(**p)

def huber_ols(D, yv, iters=8, delta=0.08):
    """IRLS Huber regression: robust to the wild-outlier rows."""
    w = np.ones(len(yv)); A = D.values
    for _ in range(iters):
        beta = np.linalg.lstsq(A * np.sqrt(w)[:, None], yv * np.sqrt(w), rcond=None)[0]
        r = yv - A @ beta; w = np.where(np.abs(r) <= delta, 1.0, delta / np.abs(r))
    return beta

res = []
for m, i_tr, i_te in time_folds(tr):
    cm = cm_all.loc[i_te]; yt = y.loc[i_te]; dte = dist.loc[i_te]
    out = {}
    out["A lgbm only"] = np.exp(lgbm().fit(X.loc[i_tr], lrpm.loc[i_tr]).predict(X.loc[i_te])) * dte
    for tag, wt in [("noT", False), ("T", True)]:
        Dtr, Dte = lin_design(X.loc[i_tr], t.loc[i_tr], wt), lin_design(X.loc[i_te], t.loc[i_te], wt)
        beta = huber_ols(Dtr, lrpm.loc[i_tr].values)
        base_tr, base_te = Dtr.values @ beta, Dte.values @ beta
        out[f"OLS {tag}"] = np.exp(base_te) * dte
        # stage 2: tree on the residual of stage 1 (time itself is NOT given to the tree)
        mdl = lgbm().fit(X.loc[i_tr], lrpm.loc[i_tr] - base_tr)
        out[f"OLS {tag} + LGBM"] = np.exp(base_te + mdl.predict(X.loc[i_te])) * dte
        if wt: coef = dict(zip(Dtr.columns, beta.round(4)))
    for name, p in out.items():
        res.append({"model": name, "fold": m, "set": "all", **report(yt, p)})
        res.append({"model": name, "fold": m, "set": "clean", **report(yt[cm], p[cm])})
    print(m, "coef:", coef)
r = pd.DataFrame(res)
print(r.groupby(["set", "model"]).mean(numeric_only=True).drop(columns="fold").round(2))
print(r[r.set == "clean"].pivot(index="fold", columns="model", values="MedAPE%").round(2))
