from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]

meta = pd.read_csv(root/"oasis_final_manifest.csv")
vol = pd.read_csv(root/"oasis_volume_features.csv")
oof = pd.read_csv(root/"results/three_slice_oof_predictions.csv")
test = pd.read_csv(root/"results/FINAL_TEST/final_test_predictions.csv")

info = meta.merge(vol, on="Subject")

oof = oof.merge(info, on="Subject", suffixes=("","_meta"))
test = test.merge(info, on="Subject", suffixes=("","_meta"))

def show(name, df):
    print("\n", name)
    print("N:", len(df))
    print(df.groupby("Target")[["Age","LeftNorm","RightNorm","Asymmetry"]].mean().round(4))
    print("\nSex:")
    print(pd.crosstab(df.Target, df.Sex))
    print("\nCDR:")
    print(pd.crosstab(df.Target, df.CDR))

show("DEVELOPMENT OOF", oof)
show("EXPOSED TEST", test)

print("\nTEST ERRORS")
print(test.loc[~test.Correct,
 ["Subject","Target","Probability","Age","Sex","CDR",
  "LeftNorm","RightNorm","Asymmetry"]].to_string(index=False))