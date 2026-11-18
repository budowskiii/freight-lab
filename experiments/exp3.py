import numpy as np, pandas as pd, lightgbm as lgb
from src.data import load, clean, TARGET
from src.features import daily_market, build
from src.cv import time_folds
from src.metrics import report

tr, med = clean(load("data/train_test.csv")); va, _ = clean(load("data/validation.csv"), med)
daily = daily_market([tr, va]); X = build(tr, daily); y = tr[TARGET]; rpm = y / tr["distance"]
cm_all = (rpm > 1) & (rpm < 4)

def make(mono=False, **over):
    kw = dict(objective="huber", alpha=0.15, n_estimators=700, learning_rate=0.03, num_leaves=31, min_child_samples=60,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
    kw.update(over)
    if mono:
        up = {"weight", "market_index", "mi_day", "mi_roll7"}
        kw["monotone_constraints"] = [1 if c in up else 0 for c in X.columns]
    return lgb.LGBMRegressor(**kw)

oof = pd.Series(np.nan, index=tr.index); res = []
for name, kwargs in [("huber", {}), ("huber+mono", {"mono": True}), ("huber+mono, leaves15", {"mono": True, "num_leaves": 15, "n_estimators": 900})]:
    rows = []
    for m, i_tr, i_te in time_folds(tr):
        mdl = make(**kwargs).fit(X.loc[i_tr], np.log(rpm.loc[i_tr]))
        p = np.exp(mdl.predict(X.loc[i_te])) * tr.loc[i_te, "distance"]
        if name == "huber+mono": oof.loc[i_te] = p
        cm = cm_all.loc[i_te]
        rows.append({**{f"all_{k}": v for k, v in report(y.loc[i_te], p).items() if k in ("MAE", "MedAPE%")},
                     **{f"clean_{k}": v for k, v in report(y.loc[i_te][cm], p[cm]).items() if k in ("MAE", "MAPE%", "MedAPE%")}})
    res.append({"model": name, **pd.DataFrame(rows).mean().round(2).to_dict()})
print(pd.DataFrame(res).to_string(index=False))

d = tr[oof.notna() & cm_all].copy(); d["res"] = np.log(y[d.index]) - np.log(oof[d.index])
print("\nOOF residual std (clean):", d.res.std().round(4), " mean:", d.res.mean().round(4))
for c, q in [("weight", 6), ("market_index", 6), ("quote_signal", 6), ("distance", 6)]:
    print(f"\nresidual by {c}"); print(d.groupby(pd.qcut(d[c], q), observed=True).res.mean().round(4).to_string())
print("\nresidual by month"); print(d.groupby(d.date.dt.month).res.agg(["mean", "std"]).round(4))
print("\nresidual by equipment"); print(d.groupby("equipment").res.agg(["mean", "std"]).round(4))
