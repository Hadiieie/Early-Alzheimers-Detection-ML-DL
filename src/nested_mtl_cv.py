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
from sklearn.ensemble import RandomForestClassifier


root = Path(__file__).resolve().parents[1]

df = pd.read_csv(root / "oasis_model_manifest.csv")
mtl = pd.read_csv(root / "mtl_features.csv")

dev = (
    df[df.Split == "Development"]
    .merge(mtl, on="Subject")
)

features = [
    c for c in mtl.columns
    if c != "Subject"
]

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


svm = Pipeline([
    ("scale", StandardScaler()),
    ("select", SelectKBest(f_classif)),
    ("clf", SVC(
        kernel="rbf",
        class_weight="balanced"
    ))
])

svm_grid = {
    "select__k": [3, 5, 8, 12],
    "clf__C": [0.1, 1, 10],
    "clf__gamma": ["scale", 0.01, 0.1]
}


rf = Pipeline([
    ("select", SelectKBest(f_classif)),
    ("clf", RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    ))
])

rf_grid = {
    "select__k": [3, 5, 8, 12],
    "clf__n_estimators": [200, 500],
    "clf__max_depth": [2, 3, None],
    "clf__min_samples_leaf": [1, 2, 4]
}


def test(name, model, grid):

    search = GridSearchCV(
        model,
        grid,
        scoring="balanced_accuracy",
        cv=inner,
        n_jobs=-1
    )

    result = cross_validate(
        search,
        X,
        y,
        cv=outer,
        scoring={
            "BA": "balanced_accuracy",
            "AUC": "roc_auc"
        },
        n_jobs=1
    )

    print(
        f"{name:12s} | "
        f"BA={result['test_BA'].mean():.3f} "
        f"+/- {result['test_BA'].std():.3f} | "
        f"AUC={result['test_AUC'].mean():.3f} "
        f"+/- {result['test_AUC'].std():.3f}"
    )


test("SVM-RBF", svm, svm_grid)
test("RandomForest", rf, rf_grid)