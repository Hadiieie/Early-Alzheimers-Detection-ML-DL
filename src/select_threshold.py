from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    recall_score, precision_score, f1_score,
    confusion_matrix, roc_auc_score,
    average_precision_score
)

root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / "results/three_slice_oof_predictions.csv")

y = (df["Target"] == "Early_Demented").astype(int).values
p = df["Probability"].values


def evaluate(t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

    return {
        "threshold": float(t),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity": recall_score(y, pred),
        "specificity": tn / (tn + fp),
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred),
        "TN": int(tn), "FP": int(fp),
        "FN": int(fn), "TP": int(tp)
    }


thresholds = np.arange(0.20, 0.81, 0.01)
results = [evaluate(t) for t in thresholds]

best = max(
    results,
    key=lambda x: (
        x["balanced_accuracy"],
        -abs(x["threshold"] - 0.50)
    )
)

base = evaluate(0.50)

best["roc_auc"] = roc_auc_score(y, p)
best["pr_auc"] = average_precision_score(y, p)

with open(root / "three_slice_threshold.json", "w") as f:
    json.dump(best, f, indent=2)

print("THRESHOLD 0.50")
print(base)

print("\nSELECTED THRESHOLD")
print(best)

print("\nSaved: three_slice_threshold.json")