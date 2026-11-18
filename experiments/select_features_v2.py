"""Feature selection v2. Selection folds: test Jul, Aug, Sep. October is untouched.
 (a) FORWARD from a raw-features baseline (raw linear stage + raw LightGBM), adding groups one at a time
 (b) BACKWARD from the full feature set, dropping groups that don't matter
 Then compare both against my earlier hand-picked setup, all on the same folds."""
import json, numpy as np, pandas as pd
from src.data import load, clean, TARGET
from src.features import daily_market, build
from src.model import FreightModel, S1_DEFAULT
from src.metrics import report

tr, med = clean(load("data/train_test.csv")); va, _ = clean(load("data/validation.csv"), med)
daily = daily_market([tr, va]); X = build(tr, daily); y = tr[TARGET]; dist = tr["distance"]; month = tr["date"].dt.month
clean_rows = (y / dist > 1) & (y / dist < 4)
FOLDS, FAST, cache = [7, 8, 9], dict(n_estimators=400, learning_rate=0.05), {}
def log(*a): print(*a, flush=True)

def score(s1, s2):
    key = (tuple(s1), tuple(s2))
    if key in cache: return cache[key]
    rows = []
    for m in FOLDS:
        a, b = month < m, month == m
        mdl = FreightModel(s1_cols=list(s1), s2_cols=list(s2), lgb_params=FAST, seeds=(0,)).fit(X[a], tr.date[a], dist[a], y[a])
        p = mdl.predict(X[b], tr.date[b], dist[b]); cm = clean_rows[b].values
        ra, rc = report(y[b], p), report(y[b][cm], p[cm]); rows.append((ra["MAE"], ra["MedAPE%"], rc["MAE"]))
    cache[key] = tuple(float(v) for v in np.mean(rows, axis=0)); return cache[key]

RAW_S1 = ["distance", "equipment", "weight", "market_index", "quote_signal"]
RAW_S2 = ["distance", "equip", "weight", "market_index", "quote_signal", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
ALL_S2 = list(X.columns)
# groups: name -> (stage1 cols, stage2 cols)
G = {
 "time trend t (S1)": (["t"], []),
 "log distance (S1)": (["log_dist"], []),
 "inverse distance (S1)": (["inv_dist"], []),
 "daily market level mi_day, qs_day (S2)": ([], ["mi_day", "qs_day"]),
 "market rolling/changes x4 (S2)": ([], ["mi_roll7", "mi_roll14", "mi_diff1", "mi_diff7"]),
 "deviation from daily mi_dev, qs_dev (S2)": ([], ["mi_dev", "qs_dev"]),
 "geo deltas + bearing (S2)": ([], ["dlat", "dlon", "bearing"]),
 "weight_missing flag (S2)": ([], ["weight_missing"]),
 "log/inv distance (S2, tree-redundant)": ([], ["log_dist", "inv_dist"]),
}
def fmt(s): return f"MAE={s[0]:.2f}  MedAPE={s[1]:.2f}%  cleanMAE={s[2]:.2f}"

hand_s1, hand_s2 = list(S1_DEFAULT), ALL_S2
hand = score(hand_s1, hand_s2)
log("REFERENCE  earlier hand-picked (all 23 tree features + full linear stage w/ trend):", fmt(hand))
b0 = score([], RAW_S2); log("B0  raw features, LightGBM only:", fmt(b0))
s1, s2 = list(RAW_S1), list(RAW_S2); cur = score(s1, s2); log("B1  raw features, linear stage + LightGBM (START):", fmt(cur))
hist = [["B0 raw cols, LightGBM only", *b0], ["B1 raw cols, linear stage + LightGBM (start)", *cur]]

# ---------- (a) forward ----------
rem = dict(G)
while rem:
    tri = {}
    for n, (a1, a2) in rem.items():
        n1 = s1 + [c for c in a1 if c not in s1]; n2 = s2 + [c for c in a2 if c not in s2]
        tri[n] = (score(n1, n2), n1, n2); log(f"   try +{n}: MAE={tri[n][0][0]:.2f}")
    best = min(tri, key=lambda k: tri[k][0][0]); sc, n1, n2 = tri[best]
    if sc[0] < cur[0] * 0.997:
        log(f"ADD  {best}: MAE {cur[0]:.2f} -> {sc[0]:.2f}  ({fmt(sc)})"); s1, s2, cur = n1, n2, sc; hist.append(["+ " + best, *cur]); rem.pop(best)
    else:
        log(f"STOP forward: best leftover ({best}) MAE {sc[0]:.2f} vs {cur[0]:.2f}, under 0.3% gain"); break
fwd = (list(s1), list(s2), cur); log("FORWARD RESULT:", fwd[0], fwd[1], fmt(cur))

# ---------- (b) backward from everything (group level) ----------
BG = {  # droppable groups for the full set
 "time trend t (S1)": ("S1", ["t"]), "log distance (S1)": ("S1", ["log_dist"]), "inverse distance (S1)": ("S1", ["inv_dist"]),
 "S1 equipment": ("S1", ["equipment"]), "S1 weight": ("S1", ["weight"]), "S1 market_index": ("S1", ["market_index"]), "S1 quote_signal": ("S1", ["quote_signal"]),
 "S2 distance transforms (log,inv)": ("S2", ["log_dist", "inv_dist"]), "S2 weight_missing": ("S2", ["weight_missing"]),
 "S2 daily level (mi_day,qs_day)": ("S2", ["mi_day", "qs_day"]), "S2 rolling/changes x4": ("S2", ["mi_roll7", "mi_roll14", "mi_diff1", "mi_diff7"]),
 "S2 deviations (mi_dev,qs_dev)": ("S2", ["mi_dev", "qs_dev"]), "S2 geo deltas+bearing": ("S2", ["dlat", "dlon", "bearing"]),
 "S2 raw coordinates x4": ("S2", ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]),
 "S2 quote_signal": ("S2", ["quote_signal"]), "S2 market_index": ("S2", ["market_index"]), "S2 weight": ("S2", ["weight"]), "S2 equip": ("S2", ["equip"]),
}
bs1, bs2, bcur = list(S1_DEFAULT), list(ALL_S2), hand; bh = [["FULL set (start)", *hand]]; bg = dict(BG)
for _ in range(10):
    tri = {}
    for n, (st, cols) in bg.items():
        n1 = [c for c in bs1 if not (st == "S1" and c in cols)]; n2 = [c for c in bs2 if not (st == "S2" and c in cols)]
        tri[n] = (score(n1, n2), n1, n2)
    k = min(tri, key=lambda k: tri[k][0][0]); sc, n1, n2 = tri[k]
    if sc[0] <= bcur[0] * 1.001:
        log(f"BACKWARD drop {k}: MAE {bcur[0]:.2f} -> {sc[0]:.2f}"); bs1, bs2, bcur = n1, n2, sc; bh.append(["- drop " + k, *sc]); bg.pop(k)
    else:
        log(f"STOP backward: cheapest remaining drop ({k}) costs MAE {sc[0]:.2f} vs {bcur[0]:.2f}"); break
log("BACKWARD RESULT:", bs1, bs2, fmt(bcur))

cands = {"forward": fwd, "backward": (bs1, bs2, bcur), "hand-picked": (hand_s1, hand_s2, hand)}
win = min(cands, key=lambda k: cands[k][2][0]); log("WINNER on Jul-Sep folds:", win, fmt(cands[win][2]))
json.dump({"forward": {"s1": fwd[0], "s2": fwd[1], "score": fwd[2], "history": hist},
           "backward": {"s1": bs1, "s2": bs2, "score": bcur, "history": bh},
           "hand": {"s1": hand_s1, "s2": hand_s2, "score": hand}, "b0": b0, "winner": win,
           "s1": cands[win][0], "s2": cands[win][1]}, open("reports/feature_selection.json", "w"), indent=1)
