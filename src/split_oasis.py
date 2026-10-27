from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / "oasis_final_manifest.csv")

dev, test = train_test_split(
    df, test_size=12, stratify=df["Target"], random_state=42
)

df["Split"] = "Development"
df.loc[test.index, "Split"] = "Final_Test"

df.sort_values("Subject").to_csv(root / "oasis_split_manifest.csv", index=False)

print(df.groupby(["Split", "Target"]).size())
print("\nTotal:", len(df))