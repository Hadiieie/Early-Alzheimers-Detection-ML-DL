from pathlib import Path
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

root = Path(__file__).resolve().parents[1]

df = pd.read_csv(root/"oasis_model_manifest.csv")
vol = pd.read_csv(root/"oasis_volume_features.csv")

df = df[df.Split=="Development"].merge(vol, on="Subject")

X = df[["LeftNorm","RightNorm","Asymmetry"]]
y = (df.Target=="Early_Demented").astype(int)

cv = RepeatedStratifiedKFold(
    n_splits=5, n_repeats=10, random_state=42
)

models = {
    "Logistic": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    ),

    "SVM-RBF": make_pipeline(
        StandardScaler(),
        SVC(C=1, kernel="rbf", probability=True)
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=3,
        random_state=42
    )
}

for name, model in models.items():
    r = cross_validate(
        model, X, y, cv=cv,
        scoring=["balanced_accuracy","roc_auc"]
    )

    print(
        f"{name:13s} | "
        f"BA={r['test_balanced_accuracy'].mean():.3f} "
        f"+/- {r['test_balanced_accuracy'].std():.3f} | "
        f"AUC={r['test_roc_auc'].mean():.3f} "
        f"+/- {r['test_roc_auc'].std():.3f}"
    )