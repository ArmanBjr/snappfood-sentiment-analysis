"""Find stacking misclassifications and label-noise patterns for the report."""
from pathlib import Path
import os
import sys
import warnings

warnings.filterwarnings("ignore")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

DATA = os.path.join(REPO, "data", "processed")
OUT = os.path.join(REPO, "report", "error_analysis.txt")

X_train = sp.load_npz(os.path.join(DATA, "X_train_hybrid.npz"))
X_test = sp.load_npz(os.path.join(DATA, "X_test_hybrid.npz"))
y_train = np.load(os.path.join(DATA, "y_train.npy"))
y_test = np.load(os.path.join(DATA, "y_test.npy"))
test_df = pd.read_csv(os.path.join(DATA, "test.csv"))

# Same stacking config as 05_stacking.ipynb (sparse / Hybrid)
stack = StackingClassifier(
    estimators=[
        ("nb", ComplementNB(alpha=0.1)),
        (
            "svm",
            CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=4000)),
        ),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", n_jobs=-1)),
        ("rf", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)),
    ],
    final_estimator=LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"),
    cv=5,
    stack_method="auto",
    n_jobs=-1,
)
print("Training stacking...")
stack.fit(X_train, y_train)
y_pred = stack.predict(X_test)

wrong = np.where(y_test != y_pred)[0]
print(f"Total wrong: {len(wrong)} / {len(y_test)} ({len(wrong)/len(y_test)*100:.2f}%)")

LABEL = {0: "Sad", 1: "Happy"}
POS = ("خوب", "عالی", "خوشمزه", "ممنون", "عالیه", "تازه", "سریع", "مناسب")
NEG = ("بد", "افتضاح", "سرد", "کم", "گرون", "ناراضی", "اشتباه", "موند", "کهنه")
CONTRAST = ("ولی", "اما", "ولی ", " اما ")

known = {
    "کیفیت خوب بود ولی نسبت به قیمت",
    "هسته زردآلو خیلی بد بود",
    "آدرس اشتباه فرستاد",
    "از نظر مزه خوشمزه بود",
    "سالاد سزار کاهو موند",
}

rows = []
for i in wrong:
    text = test_df.iloc[i]["comment"]
    cleaned = test_df.iloc[i].get("cleaned", text)
    t = str(text)
    c = str(cleaned)
    has_contrast = any(x in t for x in CONTRAST)
    pos_hits = sum(1 for w in POS if w in t)
    neg_hits = sum(1 for w in NEG if w in t)
    rows.append(
        {
            "idx": i,
            "true": LABEL[int(y_test[i])],
            "pred": LABEL[int(y_pred[i])],
            "contrast": has_contrast,
            "pos": pos_hits,
            "neg": neg_hits,
            "text": t,
            "cleaned": c[:200],
        }
    )

df = pd.DataFrame(rows)
contrast_df = df[df["contrast"]].copy()
mixed_df = df[(df["pos"] > 0) & (df["neg"] > 0)].copy()

lines = []
lines.append("=== STACKING ERROR SUMMARY ===")
lines.append(f"wrong={len(wrong)} contrast_errors={len(contrast_df)} mixed_polarity={len(mixed_df)}")
lines.append("")

lines.append("=== NOTEBOOK 05 EXAMPLES (verify) ===")
for s in known:
    hit = df[df["text"].str.contains(s[:20], na=False)]
    for _, r in hit.iterrows():
        lines.append(f"[{r['true']}->{r['pred']}] {r['text'][:140]}")
lines.append("")

lines.append("=== EXTRA CONTRAST ERRORS (beyond notebook 5) ===")
extra = contrast_df[~contrast_df["text"].apply(lambda t: any(k in t for k in known))]
for _, r in extra.head(15).iterrows():
    lines.append(f"[{r['true']}->{r['pred']}] {r['text'][:160]}")
lines.append("")

lines.append("=== LABEL NOISE CANDIDATES (raw train, contrast + polarity clash) ===")
train_df = pd.read_csv(os.path.join(DATA, "train.csv"))
for label_name, label_val, favor_pos in [("SAD with strong positive", "SAD", True), ("HAPPY with strong negative", "HAPPY", False)]:
    sub = train_df[train_df["label"] == label_val]
    for _, r in sub.sample(min(20000, len(sub)), random_state=42).iterrows():
        t = str(r["comment"])
        if not any(x in t for x in CONTRAST):
            continue
        pos = sum(1 for w in POS if w in t)
        neg = sum(1 for w in NEG if w in t)
        if favor_pos and pos >= 2 and neg >= 1:
            lines.append(f"[{label_val}] {t[:160]}")
            break
    for _, r in sub.sample(min(20000, len(sub)), random_state=7).iterrows():
        t = str(r["comment"])
        if not any(x in t for x in CONTRAST):
            continue
        pos = sum(1 for w in POS if w in t)
        neg = sum(1 for w in NEG if w in t)
        if (not favor_pos) and neg >= 2 and pos >= 1:
            lines.append(f"[{label_val}] {t[:160]}")
            break

Path(OUT).write_text("\n".join(lines), encoding="utf-8")
print("Wrote", OUT)
