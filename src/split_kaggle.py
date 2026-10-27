from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / "kaggle_binary_manifest.csv")

train, val = train_test_split(
    df, test_size=.10,
    stratify=df["Target"], random_state=42
)

df["Split"] = "Train"
df.loc[val.index, "Split"] = "Validation"

df.to_csv(root / "kaggle_split_manifest.csv", index=False)
print(df.groupby(["Split", "Target"]).size())
print("\nTotal:", len(df))