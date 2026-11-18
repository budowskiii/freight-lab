import numpy as np, pandas as pd, lightgbm as lgb
from src.data import load, clean, TARGET
from src.features import daily_market, build
from src.cv import time_folds
from src.metrics import report

tr, med = clean(load("data/train_test.csv"))
va, _ = clean(load("data/validation.csv"), med)
daily = daily_market([tr, va])
X = build(tr, daily); y = tr[TARGET]; rpm = y / tr["distance"]
clean_mask = (rpm > 1) & (rpm < 4)

def make(obj):
    kw = dict(n_estimators=700, learning_rate=0.03, num_leaves=31, min_child_samples=60, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
    if obj == "huber": kw.update(objective="huber", alpha=0.15)   # log-scale delta ~ 15%
    elif obj == "l1": kw.update(objective="l1")
    else: kw.update(objective="regression")
    return lgb.LGBMRegressor(**kw)

res = []
for m, i_tr, i_te in time_folds(tr):
    a, b = X.loc[i_tr], X.loc[i_te]
    ya = np.log(rpm.loc[i_tr]); cm = clean_mask.loc[i_te]
    for obj in ["l2", "huber", "l1"]:
        mdl = make(obj).fit(a, ya)
        p = np.exp(mdl.predict(b)) * tr.loc[i_te, "distance"]
        yt = y.loc[i_te]
        res.append({"obj": obj, "fold": m, "set": "all", **report(yt, p)})
        res.append({"obj": obj, "fold": m, "set": "clean", **report(yt[cm], p[cm])})
r = pd.DataFrame(res)
print(r.groupby(["set", "obj"]).mean(numeric_only=True).drop(columns="fold").round(2))

# B2 baseline on clean rows for reference
BANDS = [0, 250, 500, 1000, 1500, 2000, 5000]; tr["band"] = pd.cut(tr["distance"], BANDS); tr["rpm"] = rpm
bl = []
for m, i_tr, i_te in time_folds(tr):
    a, b = tr.loc[i_tr], tr.loc[i_te]
    md = a.groupby(["equipment", "band"], observed=True)["rpm"].median()
    p = b["distance"].values * md.reindex(pd.MultiIndex.from_frame(b[["equipment", "band"]])).values
    cm = clean_mask.loc[i_te].values
    bl.append(report(b[TARGET][cm], p[cm]))
print("B2 baseline, clean rows:", pd.DataFrame(bl).mean().round(2).to_dict())

mdl = make("huber").fit(X, np.log(rpm))
imp = pd.Series(mdl.booster_.feature_importance("gain"), X.columns).sort_values(ascending=False)
print((100 * imp / imp.sum()).round(1).head(12))
