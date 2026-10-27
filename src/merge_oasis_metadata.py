from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]

mri = pd.read_csv(root / "oasis_manifest.csv")
meta = pd.read_excel(root / "data_audit/oasis_cross-sectional.xlsx")

meta = meta.rename(columns={"ID": "Subject", "M/F": "Sex"})
df = mri.merge(meta[["Subject", "Age", "Sex", "MMSE", "CDR"]], on="Subject", how="left")

def label(cdr):
    if pd.isna(cdr): return "EXCLUDE"
    if cdr == 0: return "Normal"
    if cdr in [0.5, 1]: return "Early_Demented"
    return "EXCLUDE"

df["Target"] = df["CDR"].apply(label)
df.to_csv(root / "oasis_final_manifest.csv", index=False)

print(df["Target"].value_counts(dropna=False))
print("\nMissing CDR:", df["CDR"].isna().sum())
print("Saved subjects:", len(df))