from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
raw = root.parent
df = pd.read_csv(root / "kaggle_clean_manifest.csv")

folders = {
    "Normal": raw / "NORMAL/archive/Data/NonDemented",
    "VeryMild": raw / "ALZHEIMER/archive 3/Data/VeryMildDemented",
    "Mild": raw / "ALZHEIMER/archive/Data/MildDemented",
}

df = df[df["Class"] != "Moderate"].copy()
df["Target"] = df["Class"].map(
    {"Normal": "Normal", "VeryMild": "Early_Demented", "Mild": "Early_Demented"}
)
df["Path"] = df.apply(lambda x: str(folders[x["Class"]] / x["File"]), axis=1)
df.to_csv(root / "kaggle_binary_manifest.csv", index=False)

print(df["Target"].value_counts())
print("Total:", len(df))