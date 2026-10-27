from pathlib import Path
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC

root = Path(__file__).resolve().parents[1]

df = pd.read_csv(root / "oasis_model_manifest.csv")
mtl = pd.read_csv(root / "mtl_features.csv")

dev = df[df.Split == "Development"].merge(mtl, on="Subject")

features = [c for c in mtl.columns if c != "Subject"]
X = dev[features]
y = (dev.Target == "Early_Demented").astype(int)

outer = StratifiedKFold(5, shuffle=True, random_state=42)
inner = StratifiedKFold(4, shuffle=True, random_state=42)

for fold, (tr, va) in enumerate(outer.split(X, y), 1):

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif)),
        ("clf", SVC(kernel="rbf", class_weight="balanced"))
    ])

    grid = {
        "select__k": [3, 5, 8, 12],
        "clf__C": [0.1, 1, 10],
        "clf__gamma": ["scale", 0.01, 0.1]
    }

    search = GridSearchCV(
        pipe,
        grid,
        scoring="balanced_accuracy",
        cv=inner
    )

    search.fit(X.iloc[tr], y.iloc[tr])

    mask = search.best_estimator_["select"].get_support()
    selected = [f for f, keep in zip(features, mask) if keep]

    print(f"\nFold {fold}")
    print("Best:", search.best_params_)
    print("Features:", selected)