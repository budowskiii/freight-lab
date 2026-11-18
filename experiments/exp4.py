import numpy as np, pandas as pd, lightgbm as lgb
from src.data import load, clean, TARGET
from src.features import daily_market, build
from src.cv import time_folds
pd.set_option("display.width", 200)
tr, med = clean(load("data/train_test.csv")); va, _ = clean(load("data/validation.csv"), med)
daily = daily_market([tr, va]); X = build(tr, daily); y = tr[TARGET]; rpm = y / tr["distance"]
oof = pd.Series(np.nan, index=tr.index)
for m, i_tr, i_te in time_folds(tr):
    mdl = lgb.LGBMRegressor(objective="huber", alpha=0.15, n_estimators=700, learning_rate=0.03, num_leaves=15, min_child_samples=60,
                            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1).fit(X.loc[i_tr], np.log(rpm.loc[i_tr]))
    oof.loc[i_te] = np.exp(mdl.predict(X.loc[i_te])) * tr.loc[i_te, "distance"]
tr["res"] = np.log(y) - np.log(oof); c = tr[oof.notna() & (rpm > 1) & (rpm < 4)].copy()
a = c[c.date.dt.month == 8]
print("Aug residual by day:"); print(a.groupby(a.date.dt.day).res.agg(["mean", "std", "size"]).round(3).T.to_string())
print("\nAug by equipment x distance band:")
print(a.groupby(["equipment", pd.cut(a.distance, [0, 300, 700, 1500, 5000])], observed=True).res.agg(["mean", "std", "size"]).round(3))
print("\nAug by market_index bucket:"); print(a.groupby(pd.qcut(a.market_index, 5), observed=True).res.agg(["mean", "std"]).round(3))
print("\nDaily market_index around Jul-Sep (mi_day):")
print(daily.loc["2025-07-20":"2025-09-10", "mi_day"].round(3).values)
print("\nDaily median rpm residual by date (all months, top |mean|):")
g = c.groupby("date").res.mean(); print(g.abs().sort_values(ascending=False).head(10).round(3))
print("\nrpm daily median vs mi_day corr:", np.corrcoef(c.groupby("date").apply(lambda d: np.log(d.posted_rate/d.distance).median()).values, daily.loc[c.date.unique().sort_values()].mi_day.values)[0,1].round(3))
