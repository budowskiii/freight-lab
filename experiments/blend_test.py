"""Does averaging config A (full set) and config D (selected + tuned) beat either alone? All 4 folds, 3 seeds."""
import json, numpy as np, pandas as pd
from src.data import load, clean, TARGET
from src.features import daily_market, build
from src.model import FreightModel, S1_DEFAULT
from src.metrics import report
from src.cv import prepare_fold
sel = json.load(open("reports/feature_selection.json")); grid = json.load(open("reports/grid_results.json"))
raw = load("data/train_test.csv")
tr, med = clean(raw); va, _ = clean(load("data/validation.csv"), med)
daily = daily_market([tr, va]); X = build(tr, daily); y = tr[TARGET]; dist = tr["distance"]; month = tr["date"].dt.month
clean_rows = (y / dist > 1) & (y / dist < 4)
A = dict(s1_cols=S1_DEFAULT, s2_cols=list(X.columns)); D = dict(s1_cols=sel["s1"], s2_cols=sel["s2"], lgb_params=grid["params"], delta=grid["delta"])
rows = []
for m in [7, 8, 9, 10]:
    a, b = month < m, month == m; cm = clean_rows[b].values
    fold_train, fold_test, fold_inputs, fold_test_inputs = prepare_fold(raw, raw.index[a], raw.index[b])
    pa = FreightModel(seeds=(0, 1, 2), **A).fit(fold_inputs, fold_train.date, fold_train.distance, fold_train[TARGET]).predict(fold_test_inputs, fold_test.date, fold_test.distance)
    pd_ = FreightModel(seeds=(0, 1, 2), **D).fit(fold_inputs, fold_train.date, fold_train.distance, fold_train[TARGET]).predict(fold_test_inputs, fold_test.date, fold_test.distance)
    for name, p in [("A full set", pa), ("D selected+tuned", pd_), ("Blend 50/50 (A+D)", 0.5 * pa + 0.5 * pd_)]:
        ra, rc = report(y[b], p), report(y[b][cm], p[cm]); rows.append({"model": name, "fold": m, "MAE_all": ra["MAE"], "MedAPE_all": ra["MedAPE%"], "MAE_clean": rc["MAE"]})
    print("fold", m, "done", flush=True)
r = pd.DataFrame(rows); r.to_csv("reports/blend_test.csv", index=False)
print(r.pivot(index="model", columns="fold", values="MAE_all").round(1).assign(mean=lambda d: d.mean(axis=1).round(2)))
print(r.groupby("model")[["MAE_all", "MedAPE_all", "MAE_clean"]].mean().round(2))
