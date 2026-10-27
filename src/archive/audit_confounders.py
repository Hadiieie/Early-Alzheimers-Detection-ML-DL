from pathlib import Path
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


root = Path(__file__).resolve().parents[1]

df = pd.read_csv(root / "oasis_model_manifest.csv")
vol = pd.read_csv(root / "oasis_volume_features.csv")

df = df[df["Split"] == "Development"].copy()
df = df.merge(vol, on="Subject", how="left")

y = (df["Target"] == "Early_Demented").astype(int)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

tests = {
    "Age only":
        ["Age"],

    "Hippocampus only":
        ["LeftNorm", "RightNorm", "Asymmetry"],

    "Age + Hippocampus":
        ["Age", "LeftNorm", "RightNorm", "Asymmetry"]
}


for name, columns in tests.items():

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000
        )
    )

    probability = cross_val_predict(
        model,
        df[columns],
        y,
        cv=cv,
        method="predict_proba"
    )[:, 1]

    prediction = (
        probability >= 0.5
    ).astype(int)

    ba = balanced_accuracy_score(
        y,
        prediction
    )

    auc = roc_auc_score(
        y,
        probability
    )

    print(
        f"{name:22s} | "
        f"BA={ba:.3f} | "
        f"ROC-AUC={auc:.3f}"
    )