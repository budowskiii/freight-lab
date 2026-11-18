"""Time-aware grid search on the selected feature set. Folds: test Jul, Aug, Sep. October stays untouched."""
import itertools, json, numpy as np, pandas as pd
from src.data import load, clean, TARGET
from src.features import daily_market, build
from src.model import FreightModel
from src.metrics import report

sel = json.load(open("reports/feature_selection.json")); S1, S2 = sel["s1"], sel["s2"]
tr, med = clean(load("data/train_test.csv")); va, _ = clean(load("data/validation.csv"), med)
daily = daily_market([tr, va]); X = build(tr, daily); y = tr[TARGET]; dist = tr["distance"]; month = tr["date"].dt.month
FOLDS = [7, 8, 9]
def log(*a): print(*a, flush=True)

def cv(params, delta=0.08):
    mae, med_ = [], []
    for m in FOLDS:
        a, b = month < m, month == m
        mdl = FreightModel(S1, S2, params, delta=delta, seeds=(0,)).fit(X[a], tr.date[a], dist[a], y[a])
        r = report(y[b], mdl.predict(X[b], tr.date[b], dist[b])); mae.append(r["MAE"]); med_.append(r["MedAPE%"])
    return float(np.mean(mae)), float(np.mean(med_))

rows = []
# ---- grid 1: LightGBM (3 x 3 x 4 x 2 = 72 configs) ----
for leaves, mcs, alpha, (lr, n) in itertools.product([7, 15, 31], [30, 60, 120], [0.05, 0.10, 0.15, 0.25], [(0.05, 300), (0.03, 600)]):
    p = dict(num_leaves=leaves, min_child_samples=mcs, alpha=alpha, learning_rate=lr, n_estimators=n)
    mae, mp = cv(p); rows.append({**p, "delta": 0.08, "MAE": mae, "MedAPE": mp}); log(f"{p} -> MAE {mae:.2f}  MedAPE {mp:.2f}%")
best = min(rows, key=lambda r: r["MAE"]); log("BEST GRID 1:", best)
bp = {k: best[k] for k in ["num_leaves", "min_child_samples", "alpha", "learning_rate", "n_estimators"]}
# ---- grid 2: stage-1 Huber delta ----
for d in [0.03, 0.05, 0.08, 0.12, 0.20]:
    mae, mp = cv(bp, d); rows.append({**bp, "delta": d, "MAE": mae, "MedAPE": mp}); log(f"delta={d} -> MAE {mae:.2f}  MedAPE {mp:.2f}%")
bd = min([r for r in rows if r["num_leaves"] == bp["num_leaves"] and r["min_child_samples"] == bp["min_child_samples"] and r["alpha"] == bp["alpha"]
          and r["learning_rate"] == bp["learning_rate"]], key=lambda r: r["MAE"])["delta"]
# ---- grid 3: row/column sampling ----
for ss, cs in [(0.8, 0.8), (1.0, 1.0), (0.7, 0.6), (0.9, 0.5)]:
    p = {**bp, "subsample": ss, "colsample_bytree": cs}; mae, mp = cv(p, bd); rows.append({**p, "delta": bd, "MAE": mae, "MedAPE": mp}); log(f"sub={ss} col={cs} -> MAE {mae:.2f}")
best = min(rows, key=lambda r: r["MAE"]); log("OVERALL BEST:", best)
keep = ["num_leaves", "min_child_samples", "alpha", "learning_rate", "n_estimators", "subsample", "colsample_bytree"]
json.dump({"params": {k: best[k] for k in keep if k in best}, "delta": best["delta"], "score": [best["MAE"], best["MedAPE"]], "all": rows},
          open("reports/grid_results.json", "w"), indent=1)
