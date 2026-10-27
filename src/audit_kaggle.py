from pathlib import Path
from PIL import Image
import hashlib
import pandas as pd


project_root = Path(__file__).resolve().parents[1]
raw_root = project_root.parent

folders = {
    "Mild": raw_root / "ALZHEIMER/archive/Data/MildDemented",
    "Moderate": raw_root / "ALZHEIMER/archive 2/Data/ModerateDemented",
    "VeryMild": raw_root / "ALZHEIMER/archive 3/Data/VeryMildDemented",
    "Normal": raw_root / "NORMAL/archive/Data/NonDemented",
}

rows = []

for label, folder in folders.items():
    for f in folder.rglob("*"):
        if not f.is_file():
            continue

        with Image.open(f) as im:
            shape = im.size

        file_hash = hashlib.sha256(f.read_bytes()).hexdigest()
        rows.append([f.name, label, shape, file_hash])

df = pd.DataFrame(
    rows,
    columns=["File", "Class", "Shape", "Hash"]
)

output = project_root / "kaggle_audit.csv"
df.to_csv(output, index=False)

print(df["Class"].value_counts())
print("\nShapes:", df["Shape"].value_counts().head())
print("Exact duplicate images:", df.duplicated("Hash").sum())
print("\nSaved:", output)