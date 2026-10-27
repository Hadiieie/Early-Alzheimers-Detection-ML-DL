from pathlib import Path
import json, joblib, pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC

root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root/"oasis_model_manifest.csv")
mtl = pd.read_csv(root/"mtl_features.csv")
df = df[df.Split=="Development"].merge(mtl,on="Subject")

features = ["HipL","HipR","AmyL","AmyR"]
X = df[features]
y = (df.Target=="Early_Demented").astype(int)

model = Pipeline([
    ("scale", StandardScaler()),
    ("select", SelectKBest(f_classif)),
    ("svm", SVC(kernel="rbf", class_weight="balanced", probability=True))
])

grid = {
    "select__k":[2,3,4],
    "svm__C":[0.1,1],
    "svm__gamma":["scale"]
}

cv = StratifiedKFold(5, shuffle=True, random_state=42)
search = GridSearchCV(model, grid, scoring="balanced_accuracy", cv=cv)
search.fit(X,y)

joblib.dump(search.best_estimator_, root/"models/final_compact_mtl_svm.joblib")

config = {
    "features":features,
    "best_params":search.best_params_,
    "cv_balanced_accuracy":search.best_score_
}

with open(root/"models/final_compact_mtl_svm.json","w") as f:
    json.dump(config,f,indent=2)

print("Best:", search.best_params_)
print("CV BA:", round(search.best_score_,3))
print("Saved final SVM")