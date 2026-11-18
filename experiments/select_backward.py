"""Backward elimination from the FULL feature set (cross-check for greedy forward selection).
Same folds/settings as select_features_v2.py (test Jul, Aug, Sep). Writes reports/feature_selection.json."""
import json, re, numpy as np, pandas as pd
from src.data import load, clean, TARGET
from src.features import daily_market, build
from src.model import FreightModel, S1_DEFAULT
from src.metrics import report

tr, med = clean(load("data/train_test.csv")); va, _ = clean(load("data/validation.csv"), med)
daily = daily_market([tr, va]); X = build(tr, daily); y = tr[TARGET]; dist = tr["distance"]; month = tr["date"].dt.month
clean_rows = (y / dist > 1) & (y / dist < 4); FOLDS, FAST, cache = [7, 8, 9], dict(n_estimators=400, learning_rate=0.05), {}
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
fmt = lambda s: f"MAE={s[0]:.2f} MedAPE={s[1]:.2f}% cleanMAE={s[2]:.2f}"

# forward results parsed from the saved forward log
fl = open("reports/select_log_forward.txt").read()
fwd_hist = [["B0 raw cols, LightGBM only", *[float(v) for v in re.search(r"B0.*?MAE=([\d.]+)\s+MedAPE=([\d.]+)%\s+cleanMAE=([\d.]+)", fl).groups()]],
            ["B1 raw cols, linear stage + LightGBM (start)", *[float(v) for v in re.search(r"B1.*?MAE=([\d.]+)\s+MedAPE=([\d.]+)%\s+cleanMAE=([\d.]+)", fl).groups()]]]
for m in re.finditer(r"ADD\s+(.*?): MAE [\d.]+ -> [\d.]+\s+\(MAE=([\d.]+)\s+MedAPE=([\d.]+)%\s+cleanMAE=([\d.]+)\)", fl):
    fwd_hist.append(["+ " + m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))])
fm = re.search(r"FORWARD RESULT: (\[.*?\]) (\[.*?\]) MAE=([\d.]+)\s+MedAPE=([\d.]+)%\s+cleanMAE=([\d.]+)", fl)
f1, f2 = eval(fm.group(1)), eval(fm.group(2)); fscore = tuple(float(fm.group(i)) for i in (3, 4, 5))
hand_s1, hand_s2 = list(S1_DEFAULT), list(X.columns); hand = score(hand_s1, hand_s2); log("hand-picked:", fmt(hand))

BG = {
 "time trend t (S1)": ("S1", ["t"]), "log distance (S1)": ("S1", ["log_dist"]), "inverse distance (S1)": ("S1", ["inv_dist"]),
 "S2 distance transforms (log,inv)": ("S2", ["log_dist", "inv_dist"]), "S2 weight_missing": ("S2", ["weight_missing"]),
 "S2 daily level (mi_day,qs_day)": ("S2", ["mi_day", "qs_day"]), "S2 rolling/changes x4": ("S2", ["mi_roll7", "mi_roll14", "mi_diff1", "mi_diff7"]),
 "S2 deviations (mi_dev,qs_dev)": ("S2", ["mi_dev", "qs_dev"]), "S2 geo deltas+bearing": ("S2", ["dlat", "dlon", "bearing"]),
 "S2 raw coordinates x4": ("S2", ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]),
 "S2 quote_signal": ("S2", ["quote_signal"]), "S2 market_index": ("S2", ["market_index"]), "S2 weight": ("S2", ["weight"]),
}
bs1, bs2, bcur = list(S1_DEFAULT), list(X.columns), hand; bh = [["FULL set (start)", *hand]]; bg = dict(BG)
for _ in range(9):
    tri = {}
    for n, (st, cols) in bg.items():
        n1 = [c for c in bs1 if not (st == "S1" and c in cols)]; n2 = [c for c in bs2 if not (st == "S2" and c in cols)]
        tri[n] = (score(n1, n2), n1, n2); log(f"   try -{n}: MAE={tri[n][0][0]:.2f}")
    k = min(tri, key=lambda k: tri[k][0][0]); sc, n1, n2 = tri[k]
    if sc[0] <= bcur[0] * 1.001:
        log(f"DROP {k}: MAE {bcur[0]:.2f} -> {sc[0]:.2f}"); bs1, bs2, bcur = n1, n2, sc; bh.append(["- drop " + k, *sc]); bg.pop(k)
    else:
        log(f"STOP: cheapest remaining drop ({k}) would cost MAE {sc[0]:.2f} vs {bcur[0]:.2f}"); break
log("BACKWARD RESULT:", bs1, bs2, fmt(bcur))
# forward set evaluated under the same scorer for a fair head-to-head
fs = score(f1, f2); log("FORWARD set re-scored:", fmt(fs))
cands = {"forward": (f1, f2, fs), "backward": (bs1, bs2, bcur), "hand-picked": (hand_s1, hand_s2, hand)}
win = min(cands, key=lambda k: cands[k][2][0]); log("WINNER on Jul-Sep folds:", win, fmt(cands[win][2]))
json.dump({"forward": {"s1": f1, "s2": f2, "score": fs, "history": fwd_hist}, "backward": {"s1": bs1, "s2": bs2, "score": bcur, "history": bh},
           "hand": {"s1": hand_s1, "s2": hand_s2, "score": hand}, "winner": win, "s1": cands[win][0], "s2": cands[win][1]},
          open("reports/feature_selection.json", "w"), indent=1)
log("DONE")
