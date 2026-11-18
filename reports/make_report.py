"""Builds the PDF report. Needs reportlab (not required for training)."""
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

TEAL = colors.HexColor("#064A56")
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, textColor=TEAL, spaceBefore=10, spaceAfter=5, keepWithNext=1)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.6, leading=13.2, spaceAfter=5)
S = ParagraphStyle("S", parent=B, fontSize=8.4, leading=11, textColor=colors.HexColor("#455A60"))
T = ParagraphStyle("T", parent=ss["Title"], fontSize=21, textColor=TEAL, alignment=0, spaceAfter=2)
cell = ParagraphStyle("c", parent=B, fontSize=8.6, leading=11, spaceAfter=0)
cellb = ParagraphStyle("cb", parent=cell, textColor=colors.white, fontName="Helvetica-Bold")


def table(data, widths, hl_last=False):
    rows = [[Paragraph(str(c), cellb if i == 0 else cell) for c in r] for i, r in enumerate(data)]
    t = Table(rows, colWidths=widths, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), TEAL), ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#C9D6D9")),
          ("VALIGN", (0, 0), (-1, -1), "TOP"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F7F8")]),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if hl_last: st.append(("BACKGROUND", (0, len(data) - 1), (-1, len(data) - 1), colors.HexColor("#DCEBEE")))
    t.setStyle(TableStyle(st)); return t



import json
cv = pd.read_csv("reports/cv_results.csv")
fs = json.load(open("reports/feature_selection.json")); cmp_ = pd.read_csv("reports/compare_final.csv"); bl = pd.read_csv("reports/blend_test.csv")
mon = {7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct"}
fold_rows = [["Test month", "Train on", "MAE (all)", "MedAPE (all)", "MAE (clean)", "MedAPE (clean)"]]
for m in [7, 8, 9, 10]:
    a = cv[(cv.test_month == m) & (cv.set == "all")].iloc[0]; c = cv[(cv.test_month == m) & (cv.set == "clean")].iloc[0]
    fold_rows.append([mon[m], "Jan to " + ("Jun" if m == 7 else mon[m - 1]), f"${a.MAE:.0f}", f"{a['MedAPE%']:.2f}%", f"${c.MAE:.0f}", f"{c['MedAPE%']:.2f}%"])
fa = cv[cv.set == "all"].mean(numeric_only=True); fc = cv[cv.set == "clean"].mean(numeric_only=True)
fold_rows.append(["Average", "", f"${fa.MAE:.0f}", f"{fa['MedAPE%']:.2f}%", f"${fc.MAE:.0f}", f"{fc['MedAPE%']:.2f}%"])
cm = cmp_.groupby("config")[["MAE_all", "MedAPE_all", "MAE_clean"]].mean()
A_, D_ = cm.loc["A full feature set, default params"], cm.loc["D selected features, tuned params"]; B_ = bl[bl.model.str.startswith("Blend")][["MAE_all", "MedAPE_all", "MAE_clean"]].mean()
cmp_rows = [["Model (average of 4 time folds, 3 seeds)", "MAE, all", "MedAPE, all", "MAE, clean"],
            ["Rate per mile by equipment and distance band (baseline)", "$139", "3.4%", "$86"],
            ["LightGBM on log rate per mile, no time trend", "$121", "2.7%", "$67"],
            ["A: trend + LightGBM, full feature set", f"${A_.MAE_all:.0f}", f"{A_.MedAPE_all:.2f}%", f"${A_.MAE_clean:.0f}"],
            ["D: selected features + grid-searched parameters", f"${D_.MAE_all:.0f}", f"{D_.MedAPE_all:.2f}%", f"${D_.MAE_clean:.0f}"],
            ["<b>Final: 50/50 blend of A and D</b>", f"<b>${B_.MAE_all:.0f}</b>", f"<b>{B_.MedAPE_all:.2f}%</b>", f"<b>${B_.MAE_clean:.0f}</b>"]]
sel_rows = [["Step (selection folds: test Jul, Aug, Sep)", "MAE (all)", "MedAPE", "MAE (clean)"]]
for name, mae, med, cl in fs["forward"]["history"]:
    sel_rows.append([name.replace("S2", "LightGBM").replace("S1", "linear stage"), f"${mae:.1f}", f"{med:.2f}%", f"${cl:.1f}"])
bh = fs["backward"]["history"]
sel_rows.append(["Cross-check: backward elimination from the full set (start)", f"${bh[0][1]:.1f}", f"{bh[0][2]:.2f}%", f"${bh[0][3]:.1f}"])
sel_rows.append(["Backward: after dropping 7 groups that did not help", f"${bh[-1][1]:.1f}", f"{bh[-1][2]:.2f}%", f"${bh[-1][3]:.1f}"])

doc = SimpleDocTemplate("Freight_Rate_Report.pdf", pagesize=letter, leftMargin=.8 * inch, rightMargin=.8 * inch,
                        topMargin=.7 * inch, bottomMargin=.7 * inch, title="Freight Rate Prediction - Report")
s = []
s += [Paragraph("Freight Rate Prediction: Validation and Approach", T),
      Paragraph("Machine Learning Engineer assessment. Labeled data: 48,000 loads, Jan 1 to Oct 31, 2025. Target set: 12,000 loads, Nov 1 to Dec 31, 2025.", S),
      Spacer(1, 6), Paragraph("1. Summary", H1),
      Paragraph("The final model averages two two-stage models on log(rate per mile). Each is a robust linear model with a time trend, followed by LightGBM on its residuals; they differ in features and tuning. "
                f"On rolling time-based validation it reaches a mean absolute error of about <b>${B_.MAE_all:.0f} on all loads</b> and <b>${B_.MAE_clean:.0f} on the diagnostic non-extreme-price subset</b>, "
                f"with a <b>median error of about {B_.MedAPE_all:.1f}%</b>. A distance-and-equipment baseline scores $139 and 3.4%.", B)]

s += [Paragraph("2. How the data was split and validated", H1),
      Paragraph("The target set is entirely in the future: it starts the day after the last labeled load. A random split would mix months, let the model see "
                "neighbouring days of the same market, and overstate accuracy. I used <b>rolling-origin (expanding window) time validation</b>: "
                "train on every month before month <i>m</i>, test on month <i>m</i>, for m = Jul, Aug, Sep, Oct. Every fold predicts forward in time, as the real task does. "
                "After validation the model is refit on all 10 months.", B),
      Image("reports/split_diagram.png", width=6.4 * inch, height=2.15 * inch),
      Paragraph("Feature selection and the grid search used only the Jul, Aug and Sep folds, with October excluded from the initial feature search. The final candidates were then compared on all four folds with three random seeds each. "
                "Two views of every metric are reported. <b>All</b> rows include everything. <b>Clean</b> excludes the 1.4% of loads whose rate per mile is below $1 or above $4; "
                "this is a label-defined diagnostic subset, not proof of corrupt labels and not an inference-time filter. All rows remain in training. Weight medians are learned separately within each training fold in the refreshed main, finalist and blend checks. Historical feature-search logs used development-wide weight medians, so those exploratory scores are not equally strict. Same-day market means use batch input features, never target prices. This assumes the held-out batch of covariates is available at prediction time.", B)]

s += [Paragraph("3. Key findings and data-quality fixes", H1),
      table([["Finding", "What I did"],
             ["292 negative weights (145 in target set). Treated as possible sign errors; the correct values are unverified.", "Assumed a sign error and took the absolute value; this is not a verified correction."],
             ["300 missing weights (165 in target set).", "Filled with the equipment median and added a missing flag."],
             ["374 missing market_index (249 in target set). It is a daily level (within-day spread about 0.025).", "Filled with the same-day mean; interpolated if a whole day was empty."],
             ["1.4% extreme-price rows (rate per mile under $1 or over $4); corruption is unverified.", "Kept every row and used Huber loss to reduce extreme observations' influence; reported all-row and diagnostic subset metrics."],
             ["Distance is retained as supplied. Coordinates have not been geographically verified.", "Kept distance and tested coordinate-derived features for predictive usefulness; no geographic correction was performed."],
             ["8 target-set cities never appear in training.", "No city-name features. Geography and distance generalise to new cities."],
             ["Pricing is roughly a fixed charge ($246 to $278 by equipment) plus a per-mile part; only the per-mile part follows the market.", "Modelled rate per mile with distance-dependent terms so short hauls are handled."],
             ["At the same market_index, later months price higher (a time trend).", "Added a linear trend in stage 1; this was the single largest improvement."]],
            [3.5 * inch, 3.4 * inch]), Spacer(1, 4)]

s += [Paragraph("4. Feature selection", H1),
      Paragraph("I started from the raw columns (distance, equipment, weight, market_index, quote_signal and the four coordinates) and added one engineered feature group at a time, keeping a group only if it cut MAE by at least 0.3%. "
                "As a cross-check I also ran backward elimination from the full feature set. Time-trend and market-level features gave the big gains; the rest did little or nothing.", B),
      table(sel_rows, [4.0 * inch, .95 * inch, .9 * inch, 1.05 * inch]), Spacer(1, 4),
      Paragraph("<b>What helped:</b> the linear time trend (about 13% lower MAE) and the daily market level (about 5% lower). <b>What did not:</b> rolling averages and changes of the market index made results worse (they act as a proxy for the date), "
                "and coordinate deltas, bearing and a missing-weight flag were neutral. Removing the raw coordinates costs about 15% in MAE, so geography matters. Log and inverse distance did not help: trees ignore monotone transforms of a feature, and in the linear stage they were neutral to slightly harmful.", B),
      Paragraph("<b>A caution on the selection itself:</b> a first greedy pass got stuck, because the time trend only works when market_index, weight and equipment sit next to it in the linear stage. "
                "The best set found on Jul to Sep scored worse on the later-reused October fold than my original set (103 vs 96 MAE), so differences of about 2% between feature sets are within noise. That is why the final model blends the two.", B)]

s += [Paragraph("5. Model", H1),
      Paragraph("<b>Stage 1</b> is a Huber-loss (IRLS) linear regression on log(rate per mile) with a time trend. Ordinary constant-leaf trees do not extend a slope beyond their date thresholds. A linear trend is an explicit forecasting assumption, not the only possible solution. <b>Stage 2</b> is LightGBM with Huber loss on the stage-1 residuals, averaged over three seeds. It receives no explicit date; its other covariates can still correlate with time, so stability is not guaranteed.", B),
      Paragraph("<b>Model A</b> uses the full engineered feature set with default parameters. <b>Model D</b> uses the backward-selected features "
                "(stage 1: equipment, weight, market_index, quote_signal, time; stage 2: distance, equipment, market_index, quote_signal, deviation of each load from its day's market level, four coordinates) with grid-searched parameters. "
                "The final prediction is the average of A and D in dollars.", B),
      Paragraph("<b>Grid search</b> (time-aware, Jul to Sep folds): 72 LightGBM configurations (leaves 7/15/31, min child samples 30/60/120, Huber alpha 0.05/0.10/0.15/0.25, learning rate and trees 0.05x300 or 0.03x600), "
                "then the stage-1 Huber delta (0.03 to 0.20) and row/column sampling. Best: 31 leaves, min child 30, alpha 0.10, learning rate 0.03, 600 trees, delta 0.12, sampling 0.8. "
                "Tuning improved MAE by only about 1.7%, and the best values sit at the edge of the grid.", B)]

s += [Paragraph("6. Results", H1), table(fold_rows, [.85 * inch, 1.1 * inch, 1.0 * inch, 1.1 * inch, 1.0 * inch, 1.15 * inch], hl_last=True), Spacer(1, 6),
      table(cmp_rows, [3.6 * inch, 1.0 * inch, 1.15 * inch, 1.1 * inch], hl_last=True),
      Paragraph("RMSE stays near $620 in these experiments because extreme-price rows dominate it. All-row metrics remain the primary results; non-extreme-row metrics are supplementary diagnostics. "
                "The blend was chosen after seeing these four folds, so its numbers are slightly optimistic.", S)]

s += [PageBreak(), Paragraph("7. How each feature moves the predictions (SHAP)", H1),
      Paragraph("Stage-2 TreeSHAP and linear-stage contributions were summed in log-rate-per-mile space, then averaged across blend members. These explain the component log predictions, not an exact additive explanation of the final dollar-space average. The summaries were computed per feature family on the November and December loads. "
                "Distance matters most, then equipment, the time trend, the market level and weight. quote_signal adds almost nothing.", B),
      Image("reports/shap_importance.png", width=5.8 * inch, height=2.55 * inch),
      Image("reports/shap_dependence.png", width=6.0 * inch, height=3.77 * inch),
      Paragraph("Effects are relative to the average load. Dry Van is about 4% cheaper per mile, Flatbed about 3% and Reefer about 7% pricier. The time trend adds about 3 to 4.5% across November and December, measured against the average training date.", B)]

s += [Paragraph("8. Fixed December prediction chart (produced by score.py)", H1),
      Image("scorer_results/candidate_december.png", width=6.5 * inch, height=2.87 * inch),
      Paragraph("Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb. Only the date changes. The prediction stays between about $829 and $852 and moves in a weekly cycle that follows the daily market index. "
                "For reference, real Dry Van loads on this lane in the training data had a median of $808 (monthly medians $778 to $916). "
                "City coordinates came from the training data; market_index and quote_signal for each December date are that date's average in validation.csv (features only, no labels).", B),
      Paragraph("9. Limitations", H1),
      Paragraph("The time trend is extrapolated about 60 days beyond the last labeled load, and its estimate fell from 0.040 to 0.021 per 100 days as data accumulated. If the trend is flattening, "
                "November and December predictions may run roughly 1% high. Differences of a few percent between feature sets are within fold-to-fold noise, so the choice of the blend is a variance-reduction decision more than a proven winner. "
                "The final scoring metric is hidden; the model is tuned for median-style accuracy, and extreme-price rows dominate observed RMSE, but this does not prove they are unpredictable or that another model cannot improve them.", B)]
doc.build(s)
print("built")
