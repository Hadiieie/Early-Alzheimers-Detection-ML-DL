from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]

df = pd.read_csv(root / "oasis_final_manifest.csv")
meta = pd.read_excel(
    root / "data_audit/oasis_cross-sectional.xlsx"
)[["ID", "eTIV"]]

meta = meta.rename(columns={"ID": "Subject"})
df = df.merge(meta, on="Subject", how="left")

df["LeftNorm"] = (df.Left_mm3 / 1000) / df.eTIV
df["RightNorm"] = (df.Right_mm3 / 1000) / df.eTIV
df["Asymmetry"] = abs(df.Left_mm3-df.Right_mm3) / (df.Left_mm3+df.Right_mm3)

cols = ["Subject","LeftNorm","RightNorm","Asymmetry"]
df[cols].to_csv(root/"oasis_volume_features.csv", index=False)

print("Subjects:", len(df))
print("Missing eTIV:", df.eTIV.isna().sum())
print(df[cols].head())