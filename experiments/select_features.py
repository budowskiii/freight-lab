"""Forward feature-group selection (start from raw columns), then backward pruning.
Selection folds: test Jul, Aug, Sep (train = everything before). October is NOT touched here."""
import json, sys, numpy as np, pandas as pd
from src.data import load, clean, TARGET
from src.features import daily_market, build
from src.model import FreightModel
from src.metrics import report

tr, med = clean(load("data/train_test.csv")); va, _ = clean(load("data/validation.csv"), med)
daily = daily_market([tr, va]); X = build(tr, daily); y = tr[TARGET]; dist = tr["distance"]; month = tr["date"].dt.month
clean_rows = (y / dist > 1) & (y / dist < 4)
FOLDS = [7, 8, 9]
FAST = dict(n_estimators=400, learning_rate=0.05)
cache = {}

def score(s1, s2):
    key = (tuple(s1), tuple(s2))
    if key in cache: return cache[key]
    rows = []
    for m in FOLDS:
        a, b = month < m, month == m
        mdl = FreightModel(s1_cols=s1, s2_cols=list(s2), lgb_params=FAST, seeds=(0,)).fit(X[a], tr.date[a], dist[a], y[a])
        p = mdl.predict(X[b], tr.date[b], dist[b]); cm = clean_rows[b].values
        ra, rc = report(y[b], p), report(y[b][cm], p[cm])
        rows.append((ra["MAE"], ra["MedAPE%"], rc["MAE"]))
    out = tuple(np.mean(rows, axis=0)); cache[key] = out; return out

RAW_S2 = ["distance", "equip", "weight", "market_index", "quote_signal", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
CANDS = {
 "S1 linear block on raw cols (distance, equipment, weight, market_index, quote_signal)": (["distance", "equipment", "weight", "market_index", "quote_signal"], []),
 "S1 time trend (t)": (["t"], []),
 "S1 log distance": (["log_dist"], []),
 "S1 inverse distance": (["inv_dist"], []),
 "S2 daily market level (mi_day, qs_day)": ([], ["mi_day", "qs_day"]),
 "S2 market rolling/changes (roll7, roll14, diff1, diff7)": ([], ["mi_roll7", "mi_roll14", "mi_diff1", "mi_diff7"]),
 "S2 deviation from daily (mi_dev, qs_dev)": ([], ["mi_dev", "qs_dev"]),
 "S2 geo deltas + bearing (dlat, dlon, bearing)": ([], ["dlat", "dlon", "bearing"]),
 "S2 weight_missing flag": ([], ["weight_missing"]),
 "S2 log/inv distance (tree-redundant check)": ([], ["log_dist", "inv_dist"]),
}
def log(*a):
    print(*a, flush=True)

s1, s2 = [], list(RAW_S2)
cur = score(s1, s2); log(f"BASELINE raw features, LightGBM only: MAE={cur[0]:.2f} MedAPE={cur[1]:.2f}% cleanMAE={cur[2]:.2f}")
history = [("baseline: raw features, LightGBM only", *cur)]
remaining = dict(CANDS)
while remaining:
    trials = {}
    for name, (a1, a2) in remaining.items():
        n1 = s1 + [c for c in a1 if c not in s1]; n2 = s2 + [c for c in a2 if c not in s2]
        trials[name] = (score(n1, n2), n1, n2)
        log(f"   try {name}: MAE={trials[name][0][0]:.2f}")
    best = min(trials, key=lambda k: trials[k][0][0])
    (bm, bmed, bc), n1, n2 = trials[best]
    if bm < cur[0] * (1 - 0.003):
        log(f"ADD  {best}: MAE {cur[0]:.2f} -> {bm:.2f}  MedAPE {bmed:.2f}%  cleanMAE {bc:.2f}")
        s1, s2, cur = n1, n2, (bm, bmed, bc); history.append(("+ " + best, *cur)); remaining.pop(best)
    else:
        log(f"STOP: best remaining ({best}) only gives MAE {bm:.2f} vs {cur[0]:.2f} (<0.3% gain)"); break
log("FORWARD RESULT  S1:", s1, " S2:", s2)

# ---- backward pruning: drop anything that doesn't matter (within 0.1% of current MAE) ----
for _ in range(5):
    trials = {}
    for c in s1: trials[("S1", c)] = score([x for x in s1 if x != c], s2)
    for c in s2: trials[("S2", c)] = score(s1, [x for x in s2 if x != c])
    k = min(trials, key=lambda k: trials[k][0])
    if trials[k][0] <= cur[0] * 1.001:
        log(f"PRUNE {k}: MAE {cur[0]:.2f} -> {trials[k][0]:.2f}")
        if k[0] == "S1": s1 = [x for x in s1 if x != k[1]]
        else: s2 = [x for x in s2 if x != k[1]]
        cur = trials[k]; history.append((f"- drop {k[1]} ({k[0]})", *cur))
    else:
        log("PRUNE done: every remaining feature hurts if removed"); break
log("FINAL  S1:", s1, " S2:", s2, " score:", cur)
json.dump({"s1": s1, "s2": s2, "history": history, "final": cur}, open("reports/feature_selection.json", "w"), indent=1)
