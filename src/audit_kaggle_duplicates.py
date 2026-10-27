from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / "kaggle_audit.csv")

groups = df.groupby("Hash")
dup = groups.filter(lambda x: len(x) > 1)

same = 0
conflict = 0

for _, g in dup.groupby("Hash"):
    if g["Class"].nunique() == 1:
        same += len(g) - 1
    else:
        conflict += len(g)

print("Duplicate rows:", len(dup))
print("Same-class duplicates:", same)
print("Cross-class conflicting rows:", conflict)