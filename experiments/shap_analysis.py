"""How does each feature move the prediction? SHAP values (LightGBM's built-in TreeSHAP) for stage 2 plus exact
contributions of the linear stage 1, summed per feature family, in log-rate-per-mile units (0.01 ~ 1%)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.data import load, clean, TARGET
from src.features import daily_market, build
from src.model import BlendModel, design
tr, med = clean(load("data/train_test.csv")); va, _ = clean(load("data/validation.csv"), med)
daily = daily_market([tr, va]); X = build(tr, daily); Xv = build(va, daily)
blend = BlendModel().fit(X, tr["date"], tr["distance"], tr[TARGET])

FAM = {"Distance": ["distance", "log_dist", "inv_dist"], "Equipment": ["equipment", "equip", "reef", "flat"], "Weight": ["weight", "w", "weight_missing"],
       "Market level (market_index, daily avg)": ["market_index", "mi", "mi_day", "mi_dev", "mi_roll7", "mi_roll14", "mi_diff1", "mi_diff7"],
       "quote_signal": ["quote_signal", "qs", "qs_day", "qs_dev"], "Geography (lat/lon, bearing)": ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "dlat", "dlon", "bearing"],
       "Time trend (date)": ["t"]}
fam_of = {c: f for f, cs in FAM.items() for c in cs}
contrib = []   # per member: DataFrame rows x families
for m in blend.members:
    D = design(Xv, va["date"], m.s1_cols); mu = design(X, tr["date"], m.s1_cols).mean()
    parts = {}
    for j, c in enumerate(m.cols_):
        if c == "c": continue
        parts.setdefault(fam_of[c], 0); parts[fam_of[c]] = parts[fam_of[c]] + m.beta_[j] * (D[c].values - mu[c])
    shp = np.mean([t.booster_.predict(m._x2(Xv), pred_contrib=True) for t in m.trees_], axis=0)
    for j, c in enumerate(m._x2(Xv).columns):
        parts.setdefault(fam_of[c], 0); parts[fam_of[c]] = parts[fam_of[c]] + shp[:, j]
    contrib.append(pd.DataFrame(parts, index=Xv.index))
fam = sum(contrib) / len(contrib)                      # blend ~ average of the two members, in log space
imp = fam.abs().mean().sort_values(); imp.to_csv("reports/shap_family_importance.csv", header=["mean_abs_log_effect"])
print((imp * 100).round(2).sort_values(ascending=False).to_string())

fig, ax = plt.subplots(figsize=(8.2, 3.6), dpi=200)
ax.barh(imp.index, imp.values * 100, color="#064A56"); ax.set_xlabel("Average effect on predicted rate per mile (%)  [mean |SHAP|, Nov-Dec loads]")
for i, v in enumerate(imp.values * 100): ax.text(v + .1, i, f"{v:.1f}%", va="center", fontsize=8)
ax.spines[["top", "right"]].set_visible(False); ax.set_title("What drives the predictions", loc="left", fontweight="bold", fontsize=11)
fig.tight_layout(); fig.savefig("reports/shap_importance.png"); plt.close(fig)

fig, axs = plt.subplots(2, 2, figsize=(8.6, 5.4), dpi=200)
def dep(ax, x, f, title, xl, bins=14):
    d = pd.DataFrame({"x": x.values, "y": fam[f].values * 100}); d["b"] = pd.qcut(d.x, bins, duplicates="drop")
    g = d.groupby("b", observed=True).agg(x=("x", "median"), y=("y", "mean")); ax.plot(g.x, g.y, "-o", color="#064A56", ms=3.5)
    ax.axhline(0, color="#9DAFB3", lw=.8); ax.set_title(title, fontsize=9.5, loc="left", fontweight="bold"); ax.set_xlabel(xl, fontsize=8); ax.set_ylabel("effect on rate/mile (%)", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=7.5)
dep(axs[0, 0], va["distance"], "Distance", "Distance: short hauls cost more per mile", "distance (miles)")
dep(axs[0, 1], va["weight"], "Weight", "Weight: heavier loads cost a bit more", "weight (lb)")
dep(axs[1, 0], Xv["mi_day"], "Market level (market_index, daily avg)", "Market index: higher market, higher rate", "daily market_index")
d = pd.DataFrame({"date": va["date"].values, "y": fam["Time trend (date)"].values * 100}).groupby("date").y.mean()
axs[1, 1].plot(d.index, d.values, color="#064A56"); axs[1, 1].set_title("Time trend: slow drift upward into Nov-Dec", fontsize=9.5, loc="left", fontweight="bold")
axs[1, 1].tick_params(labelsize=7.5, axis="x", rotation=30); axs[1, 1].set_ylabel("effect on rate/mile (%)", fontsize=8); axs[1, 1].spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("reports/shap_dependence.png"); plt.close(fig)
# equipment effect
print((fam["Equipment"].groupby(va["equipment"]).mean() * 100).round(2).to_dict())
print("trend range Nov-Dec (%):", round(d.min(), 2), round(d.max(), 2))
