from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]

meta = pd.read_excel(
    root / "data_audit/oasis_cross-sectional.xlsx"
)

meta = meta[meta["CDR"].isin([0, 0.5, 1])].copy()

meta["Target"] = meta["CDR"].map({
    0: "Normal",
    0.5: "Early_Demented",
    1: "Early_Demented"
})

excluded = {
    "OAS1_0003_MR1",
    "OAS1_0013_MR1",
    "OAS1_0015_MR1",
    "OAS1_0110_MR1"
}

local = {
    p.name
    for p in (root.parent / "OASIS_FREESURFER").glob("OAS1_*")
}

meta["Excluded"] = meta["ID"].isin(excluded)
meta["Local"] = meta["ID"].isin(local)
meta["Need_MRI"] = ~meta["Local"]

meta.to_csv(
    root / "oasis_expansion_manifest.csv",
    index=False
)

need = meta[(~meta["Excluded"]) & (meta["Need_MRI"])]

need["ID"].to_csv(
    root / "oasis_download_ids.txt",
    index=False,
    header=False
)

print("Eligible:", len(meta))
print("Known excluded:", meta["Excluded"].sum())
print("Need MRI:", len(need))
print("\nNeed MRI by class:")
print(need["Target"].value_counts())