from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]

meta = pd.read_excel(
    root / "data_audit/oasis_cross-sectional.xlsx"
)[["ID", "eTIV"]].rename(columns={"ID": "Subject"})

labels = {
    "HipL":17, "HipR":53,
    "AmyL":18, "AmyR":54,
    "InfVentL":5, "InfVentR":44,
    "LatVentL":4, "LatVentR":43
}

rows = []

for p in sorted((root / "oasis_clean").glob("OAS1_*")):
    seg = nib.load(p / "synthseg.mgz")
    x = seg.get_fdata()
    voxel = abs(np.linalg.det(seg.affine[:3, :3]))
    etiv = float(meta.loc[meta.Subject == p.name, "eTIV"].iloc[0])

    values = {
        name: ((x == label).sum() * voxel / 1000) / etiv
        for name, label in labels.items()
    }

    rows.append({"Subject": p.name, **values})

df = pd.DataFrame(rows)

for name, L, R in [
    ("Hip", "HipL", "HipR"),
    ("Amy", "AmyL", "AmyR"),
    ("InfVent", "InfVentL", "InfVentR"),
    ("LatVent", "LatVentL", "LatVentR")
]:
    df[name + "Asym"] = abs(df[L] - df[R]) / (df[L] + df[R])

df.to_csv(root / "mtl_features.csv", index=False)

print("Subjects:", len(df))
print("Features:", len(df.columns) - 1)
print("Missing:", df.isna().sum().sum())
print(df.head())