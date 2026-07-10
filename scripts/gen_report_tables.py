"""Generate LaTeX fragments for the report from result CSVs."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"

df = pd.read_csv(ROOT / "outputs/results/phase3_results.csv")
df = df.sort_values("f1", ascending=False)

rows = []
for _, r in df.iterrows():
    m = r["model"].replace(" ", r"\_")
    v = r["vectorizer"].replace(" ", r"\_")
    rows.append(
        rf"\lr{{{m}}} & \lr{{{v}}} & \lr{{{r['accuracy']:.4f}}} & "
        rf"\lr{{{r['precision']:.4f}}} & \lr{{{r['recall']:.4f}}} & \lr{{{r['f1']:.4f}}} \\"
    )
(REPORT / "phase3_table_rows.tex").write_text("\n".join(rows), encoding="utf-8")

avg = df.groupby("vectorizer")[["accuracy", "precision", "recall", "f1"]].mean()
avg = avg.sort_values("f1", ascending=False)
avg_rows = []
for v, r in avg.iterrows():
    vv = v.replace(" ", r"\_")
    avg_rows.append(
        rf"\lr{{{vv}}} & \lr{{{r['accuracy']:.4f}}} & \lr{{{r['precision']:.4f}}} & "
        rf"\lr{{{r['recall']:.4f}}} & \lr{{{r['f1']:.4f}}} \\"
    )
(REPORT / "vec_avg_rows.tex").write_text("\n".join(avg_rows), encoding="utf-8")

with open(ROOT / "outputs/results/best_vectorizer.json", encoding="utf-8") as f:
    bv = json.load(f)
peak_rows = []
for v, f1 in sorted(bv["peak_f1_per_vec"].items(), key=lambda x: -x[1]):
    vv = v.replace(" ", r"\_")
    peak_rows.append(rf"\lr{{{vv}}} & \lr{{{f1:.4f}}} \\")
(REPORT / "peak_f1_rows.tex").write_text("\n".join(peak_rows), encoding="utf-8")

print("wrote", len(rows), "phase3 rows,", len(avg_rows), "avg rows")
