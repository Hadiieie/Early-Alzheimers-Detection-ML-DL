from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / "oasis_split_manifest.csv")

dev = df[df.Split == "Development"].copy()
train, val = train_test_split(
    dev, test_size=10,
    stratify=dev["Target"], random_state=42
)

df["DevSplit"] = df["Split"]
df.loc[train.index, "DevSplit"] = "Train"
df.loc[val.index, "DevSplit"] = "Validation"

df.to_csv(root / "oasis_model_manifest.csv", index=False)
print(df.groupby(["DevSplit", "Target"]).size())