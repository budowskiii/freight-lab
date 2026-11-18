import numpy as np, pandas as pd, lightgbm as lgb
from src.data import load, clean, TARGET
from src.features import daily_market, build
from src.cv import time_folds
from src.metrics import report

tr, med = clean(load("data/train_test.csv")); va, _ = clean(load("data/validation.csv"), med)
daily = daily_market([tr, va]); X = build(tr, daily); y = tr[TARGET]; dist = tr["distance"]
rpm = y / dist; cm_all = (rpm > 1) & (rpm < 4)

def fixed_charge(idx):
    """Per-equipment intercept of rate = a + b*distance, fit on clean training rows only."""
    out = {}
    for e, g in tr.loc[idx][cm_all.loc[idx]].groupby("equipment"):
        out[e] = np.linalg.lstsq(np.c_[np.ones(len(g)), g.distance], g[TARGET], rcond=None)[0][0]
    return out

def lgbm(leaves=15, n=900, **kw):
    p = dict(objective="huber", alpha=0.15, n_estimators=n, learning_rate=0.03, num_leaves=leaves, min_child_samples=60,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1); p.update(kw)
    return lgb.LGBMRegressor(**p)

res = []
for m, i_tr, i_te in time_folds(tr):
    F = fixed_charge(i_tr); fe_tr = tr.loc[i_tr, "equipment"].map(F); fe_te = tr.loc[i_te, "equipment"].map(F)
    cm = cm_all.loc[i_te]
    # A) previous best: log rate-per-mile
    pa = np.exp(lgbm().fit(X.loc[i_tr], np.log(rpm.loc[i_tr])).predict(X.loc[i_te])) * dist.loc[i_te]
    # B) subtract fixed charge, model log per-mile slope
    slope = ((y.loc[i_tr] - fe_tr) / dist.loc[i_tr]).clip(lower=0.2)
    Xb = X.copy(); Xb["fixed"] = tr["equipment"].map(F)
    pb = fe_te + np.exp(lgbm().fit(X.loc[i_tr], np.log(slope)).predict(X.loc[i_te])) * dist.loc[i_te]
    for name, p in [("A log-rpm", pa), ("B fixed+slope", pb)]:
        res.append({"model": name, "fold": m, "set": "all", **report(y.loc[i_te], p)})
        res.append({"model": name, "fold": m, "set": "clean", **report(y.loc[i_te][cm], p[cm])})
    print(m, {k: round(v, 1) for k, v in F.items()})
r = pd.DataFrame(res)
print(r.groupby(["set", "model"]).mean(numeric_only=True).drop(columns="fold").round(2))
print(r[r.set == "clean"].pivot(index="fold", columns="model", values="MedAPE%").round(2))
