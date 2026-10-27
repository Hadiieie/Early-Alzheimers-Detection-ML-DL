from pathlib import Path
import json
import pandas as pd
import torch

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    recall_score, precision_score, f1_score,
    confusion_matrix, roc_auc_score,
    average_precision_score
)

from patient_dataset import PatientDataset
from patient_model import PatientModel
from patient_transforms import patient_transform


root = Path(__file__).resolve().parents[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(root / "oasis_model_manifest.csv")
test = df[df["Split"] == "Final_Test"].copy().reset_index(drop=True)

with open(root / "three_slice_threshold.json") as f:
    threshold = float(json.load(f)["threshold"])


def make_loader():
    transform = lambda x: patient_transform(x, train=False)

    dataset = PatientDataset(
        test,
        root / "oasis_roi",
        transform,
        n_slices=3
    )

    return DataLoader(
        dataset,
        batch_size=2,
        shuffle=False
    )


model = PatientModel(
    root / "models/kaggle_pretrained_resnet18.pt"
)

state = torch.load(
    root / "models/final_three_slice_model.pt",
    map_location=device
)

model.load_state_dict(state)
model = model.to(device)
model.eval()


probabilities = []

with torch.no_grad():
    for images, _ in make_loader():
        images = images.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]

        probabilities.extend(probs.cpu().tolist())


test["Probability"] = probabilities

test["Prediction"] = [
    "Early_Demented" if p >= threshold else "Normal"
    for p in probabilities
]

test["Correct"] = (
    test["Prediction"] == test["Target"]
)


y_true = (
    test["Target"] == "Early_Demented"
).astype(int).values

y_pred = (
    test["Prediction"] == "Early_Demented"
).astype(int).values


tn, fp, fn, tp = confusion_matrix(
    y_true, y_pred
).ravel()


metrics = {
    "Threshold": threshold,
    "Patients": len(test),

    "Accuracy":
        accuracy_score(y_true, y_pred),

    "Balanced_Accuracy":
        balanced_accuracy_score(y_true, y_pred),

    "Sensitivity":
        recall_score(y_true, y_pred),

    "Specificity":
        tn / (tn + fp),

    "Precision":
        precision_score(
            y_true, y_pred,
            zero_division=0
        ),

    "F1":
        f1_score(y_true, y_pred),

    "ROC_AUC":
        roc_auc_score(
            y_true,
            probabilities
        ),

    "PR_AUC":
        average_precision_score(
            y_true,
            probabilities
        ),

    "TN": int(tn),
    "FP": int(fp),
    "FN": int(fn),
    "TP": int(tp)
}


out = root / "results" / "FINAL_TEST"
out.mkdir(parents=True, exist_ok=True)

test.to_csv(
    out / "final_test_predictions.csv",
    index=False
)

with open(
    out / "final_test_metrics.json",
    "w"
) as f:
    json.dump(metrics, f, indent=2)


print("\nFINAL LOCKED TEST")
print("-----------------")

for key, value in metrics.items():

    if isinstance(value, float):
        print(f"{key}: {value:.4f}")

    else:
        print(f"{key}: {value}")


print("\nPER-PATIENT RESULTS")

print(
    test[
        [
            "Subject",
            "Target",
            "Probability",
            "Prediction",
            "Correct"
        ]
    ].to_string(index=False)
)