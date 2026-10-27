from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import *

root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root/"results/oasis_oof_predictions.csv")

y = (df.Target == "Early_Demented").astype(int).values
p = df.Probability.values

def metrics(t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return [
        accuracy_score(y,pred), balanced_accuracy_score(y,pred),
        recall_score(y,pred), tn/(tn+fp),
        precision_score(y,pred,zero_division=0), f1_score(y,pred)
    ]

base = metrics(.50)
thresholds = np.arange(.20, .81, .01)
scores = [metrics(t)[1] for t in thresholds]
best_t = thresholds[np.argmax(scores)]
best = metrics(best_t)

print("Threshold 0.50:", [round(x,3) for x in base])
print("ROC-AUC:", round(roc_auc_score(y,p),3))
print("PR-AUC :", round(average_precision_score(y,p),3))
print("Best threshold:", round(best_t,2))
print("Optimized:", [round(x,3) for x in best])