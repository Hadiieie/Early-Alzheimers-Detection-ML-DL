from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]

df = pd.read_csv(root / "kaggle_audit.csv")
df = df.sort_values(["Class", "File"])

clean = df.drop_duplicates("Hash", keep="first").copy()
clean["Image_ID"] = range(1, len(clean) + 1)

clean.to_csv(root / "kaggle_clean_manifest.csv", index=False)

print("Raw images:", len(df))
print("Removed duplicates:", len(df) - len(clean))
print("Unique images:", len(clean))
print()
print(clean["Class"].value_counts())