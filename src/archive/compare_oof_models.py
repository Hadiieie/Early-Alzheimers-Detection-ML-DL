from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import *

root = Path(__file__).resolve().parents[1]

def evaluate(file):
    df = pd.read_csv(file)
    y = (df.Target == "Early_Demented").astype(int).values
    p = df.Probability.values
    pred = (p >= .5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

    return {
        "Accuracy": accuracy_score(y, pred),
        "BA": balanced_accuracy_score(y, pred),
        "Sensitivity": recall_score(y, pred),
        "Specificity": tn / (tn + fp),
        "F1": f1_score(y, pred),
        "ROC_AUC": roc_auc_score(y, p),
        "PR_AUC": average_precision_score(y, p)
    }

image = evaluate(root/"results/oasis_oof_predictions.csv")
hybrid = evaluate(root/"results/hybrid_oof_predictions.csv")

print("\nIMAGE-ONLY")
for k,v in image.items(): print(k, round(v,3))

print("\nHYBRID")
for k,v in hybrid.items(): print(k, round(v,3))