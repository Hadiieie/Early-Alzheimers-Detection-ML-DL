from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
meta = pd.read_excel(root/"data_audit/oasis_cross-sectional.xlsx")

meta = meta[meta["CDR"].isin([0, 0.5, 1])].copy()
meta["Target"] = meta["CDR"].map(
    {0:"Normal", 0.5:"Early_Demented", 1:"Early_Demented"}
)

local = {
    p.name for p in
    (root.parent/"OASIS_FREESURFER").glob("OAS1_*")
}

meta["LocalMRI"] = meta["ID"].isin(local)

print("Eligible metadata:", len(meta))
print("\nTargets:")
print(meta["Target"].value_counts())

print("\nAlready local:", meta["LocalMRI"].sum())
print("Missing locally:", (~meta["LocalMRI"]).sum())

missing = meta[~meta["LocalMRI"]]
missing.to_csv(root/"oasis_expansion_candidates.csv", index=False)

print("\nMissing by class:")
print(missing["Target"].value_counts())