import numpy as np, pandas as pd
from src.data import load, clean, TARGET
from src.cv import time_folds
from src.metrics import report

tr_raw = load("data/train_test.csv")
tr, _ = clean(tr_raw)
tr["rpm"] = tr[TARGET] / tr["distance"]
BANDS = [0, 250, 500, 1000, 1500, 2000, 5000]
tr["band"] = pd.cut(tr["distance"], BANDS)

rows = {"B0 global rpm": [], "B1 equip rpm": [], "B2 equip x band rpm": []}
for m, i_tr, i_te in time_folds(tr):
    a, b = tr.loc[i_tr], tr.loc[i_te]
    p0 = b["distance"] * a["rpm"].median()
    p1 = b["distance"] * b["equipment"].map(a.groupby("equipment")["rpm"].median())
    med = a.groupby(["equipment", "band"], observed=True)["rpm"].median()
    key = pd.MultiIndex.from_frame(b[["equipment", "band"]])
    p2 = b["distance"].values * med.reindex(key).values
    for name, p in zip(rows, [p0, p1, p2]):
        rows[name].append({"fold": f"test-m{m}", **report(b[TARGET], p)})

out = []
for name, r in rows.items():
    d = pd.DataFrame(r); print(f"\n{name}"); print(d.round(2).to_string(index=False))
    out.append({"model": name, **d.drop(columns="fold").mean().round(2).to_dict()})
print("\nAVERAGE ACROSS FOLDS"); print(pd.DataFrame(out).to_string(index=False))
