from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
data = root / "oasis_roi"
rows = []

for p in sorted(data.glob("*/*")):
    files = sorted(p.glob("*.png"))
    for f in files:
        x = np.array(Image.open(f))
        rows.append([p.name, f.name, x.shape, x.min(), x.max(), x.std()])

df = pd.DataFrame(rows, columns=["Subject","File","Shape","Min","Max","Std"])
bad = df[(df["Std"] < 5) | (df["Max"] == df["Min"])]
df.to_csv(root / "roi_audit.csv", index=False)

print("Images:", len(df))
print("Shapes:", df["Shape"].value_counts().to_dict())
print("Blank/Suspicious:", len(bad))
print("Patients:", df["Subject"].nunique())