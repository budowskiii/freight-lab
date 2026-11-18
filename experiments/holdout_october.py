"""Untouched October holdout: every model trained on Jan-Sep, tested on Oct, full settings (3 seeds)."""
import json, numpy as np, pandas as pd
from src.data import load, clean, TARGET
from src.features import daily_market, build
from src.model import FreightModel, S1_DEFAULT
from src.metrics import report
sel = json.load(open("reports/feature_selection.json")); grid = json.load(open("reports/grid_results.json"))
tr, med = clean(load("data/train_test.csv")); va, _ = clean(load("data/validation.csv"), med)
daily = daily_market([tr, va]); X = build(tr, daily); y = tr[TARGET]; dist = tr["distance"]; month = tr["date"].dt.month
a, b = month < 10, month == 10; cm = ((y / dist > 1) & (y / dist < 4))[b].values
RAW_S1 = ["distance", "equipment", "weight", "market_index", "quote_signal"]
RAW_S2 = ["distance", "equip", "weight", "market_index", "quote_signal", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
cfgs = {
 "B0 raw cols, LightGBM only (default params)": dict(s1_cols=[], s2_cols=RAW_S2),
 "B1 raw cols, linear + LightGBM (default params)": dict(s1_cols=RAW_S1, s2_cols=RAW_S2),
 "Earlier hand-picked features (default params)": dict(s1_cols=S1_DEFAULT, s2_cols=list(X.columns)),
 "Selected features (default params)": dict(s1_cols=sel["s1"], s2_cols=sel["s2"]),
 "Selected features + grid-searched params": dict(s1_cols=sel["s1"], s2_cols=sel["s2"], lgb_params=grid["params"], delta=grid["delta"]),
}
rows = []
for name, kw in cfgs.items():
    mdl = FreightModel(seeds=(0, 1, 2), **kw).fit(X[a], tr.date[a], dist[a], y[a]); p = mdl.predict(X[b], tr.date[b], dist[b])
    ra, rc = report(y[b], p), report(y[b][cm], p[cm])
    rows.append({"model": name, "MAE_all": ra["MAE"], "MedAPE_all": ra["MedAPE%"], "MAE_clean": rc["MAE"], "MedAPE_clean": rc["MedAPE%"]})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("reports/october_holdout.csv", index=False)
