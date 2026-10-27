from pathlib import Path
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

root = Path(__file__).resolve().parents[1]

df = pd.read_csv(root / "oasis_model_manifest.csv")
hip = pd.read_csv(root / "oasis_volume_features.csv")
mtl = pd.read_csv(root / "mtl_features.csv")

dev = df[df.Split == "Development"].copy()
dev = dev.merge(hip, on="Subject").merge(mtl, on="Subject")

y = (dev.Target == "Early_Demented").astype(int)

sets = {
    "HIP-3": ["LeftNorm", "RightNorm", "Asymmetry"],
    "MTL-12": [c for c in mtl.columns if c != "Subject"]
}

models = {
    "Logistic": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    ),

    "SVM-RBF": make_pipeline(
        StandardScaler(),
        SVC(C=1, kernel="rbf")
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=3,
        random_state=42
    )
}

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)

for set_name, cols in sets.items():
    print(f"\n{set_name}")

    for model_name, model in models.items():
        r = cross_validate(
            model,
            dev[cols],
            y,
            cv=cv,
            scoring=["balanced_accuracy", "roc_auc"]
        )

        print(
            f"{model_name:13s} | "
            f"BA={r['test_balanced_accuracy'].mean():.3f} "
            f"+/- {r['test_balanced_accuracy'].std():.3f} | "
            f"AUC={r['test_roc_auc'].mean():.3f} "
            f"+/- {r['test_roc_auc'].std():.3f}"
        )