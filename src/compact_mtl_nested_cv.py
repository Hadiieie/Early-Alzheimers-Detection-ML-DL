from pathlib import Path
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold, RepeatedStratifiedKFold,
    GridSearchCV, cross_validate
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC

root = Path(__file__).resolve().parents[1]

df = pd.read_csv(root / "oasis_model_manifest.csv")
mtl = pd.read_csv(root / "mtl_features.csv")

dev = df[df.Split == "Development"].merge(mtl, on="Subject")

features = ["HipL", "HipR", "AmyL", "AmyR"]
X = dev[features]
y = (dev.Target == "Early_Demented").astype(int)

inner = StratifiedKFold(
    n_splits=4,
    shuffle=True,
    random_state=42
)

outer = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=42
)

model = Pipeline([
    ("scale", StandardScaler()),
    ("select", SelectKBest(f_classif)),
    ("clf", SVC(
        kernel="rbf",
        class_weight="balanced"
    ))
])

grid = {
    "select__k": [2, 3, 4],
    "clf__C": [0.1, 1],
    "clf__gamma": ["scale"]
}

search = GridSearchCV(
    model,
    grid,
    scoring="balanced_accuracy",
    cv=inner
)

result = cross_validate(
    search,
    X,
    y,
    cv=outer,
    scoring={
        "BA": "balanced_accuracy",
        "AUC": "roc_auc"
    }
)

print(
    "Compact MTL-SVM | "
    f"BA={result['test_BA'].mean():.3f} "
    f"+/- {result['test_BA'].std():.3f} | "
    f"AUC={result['test_AUC'].mean():.3f} "
    f"+/- {result['test_AUC'].std():.3f}"
)