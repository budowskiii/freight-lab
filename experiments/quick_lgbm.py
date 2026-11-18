import numpy as np, pandas as pd, lightgbm as lgb
from src.data import load, clean, TARGET
from src.cv import time_folds
from src.metrics import report

tr, _ = clean(load("data/train_test.csv"))
tr["rpm"] = tr[TARGET] / tr["distance"]
tr["equip"] = tr["equipment"].astype("category").cat.codes
tr["log_dist"] = np.log(tr["distance"])
tr["doy"] = tr["date"].dt.dayofyear
FEATS = ["log_dist","distance","equip","weight","market_index","quote_signal",
         "pickup_lat","pickup_lon","delivery_lat","delivery_lon","doy","weight_missing","market_index_missing"]

# how much of RMSE is just outlier labels?
hi = (tr.rpm > 4) | (tr.rpm < 1)
print(f"label outliers (rpm<1 or >4): {hi.mean()*100:.2f}% of rows, {((tr[TARGET]-tr.groupby(['equipment',pd.cut(tr.distance,[0,250,500,1000,1500,2000,5000])],observed=True)[TARGET].transform('median'))**2)[hi].sum()/((tr[TARGET]-tr.groupby(['equipment',pd.cut(tr.distance,[0,250,500,1000,1500,2000,5000])],observed=True)[TARGET].transform('median'))**2).sum()*100:.0f}% of squared-error mass (crude)")

res = []
for m, i_tr, i_te in time_folds(tr):
    a, b = tr.loc[i_tr], tr.loc[i_te]
    for name, obj in [("L2 on log-rpm", "regression"), ("Huber on log-rpm", "huber")]:
        mdl = lgb.LGBMRegressor(objective=obj, n_estimators=600, learning_rate=0.04, num_leaves=31,
                                min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
        mdl.fit(a[FEATS], np.log(a["rpm"]))
        p = np.exp(mdl.predict(b[FEATS])) * b["distance"]
        res.append({"model": name, "fold": m, **report(b[TARGET], p)})
    # also: score on "clean" rows only, to show the noise floor
r = pd.DataFrame(res); print(r.round(2).to_string(index=False))
print(r.groupby("model").mean(numeric_only=True).drop(columns="fold").round(2))
