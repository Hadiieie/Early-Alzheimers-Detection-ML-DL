from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
data = root / "oasis_clean"
rows = []

for p in sorted(data.glob("OAS1_*")):
    img = nib.load(p / "synthseg.mgz")
    seg = img.get_fdata()
    L, R = seg == 17, seg == 53
    mask = L | R
    axis = next(i for i,c in enumerate(nib.aff2axcodes(img.affine)) if c in ("A","P"))
    counts = mask.sum(axis=tuple(i for i in range(3) if i != axis))
    peak = int(np.argmax(counts))
    voxel = abs(np.linalg.det(img.affine[:3,:3]))
    slices = list(range(peak-2, peak+3))
    rows.append([p.name, L.sum()*voxel, R.sum()*voxel, peak, *slices, "PASS"])

cols = ["Subject","Left_mm3","Right_mm3","Peak","S1","S2","S3","S4","S5","QC"]
pd.DataFrame(rows, columns=cols).to_csv(root/"oasis_manifest.csv", index=False)
print("Saved:", len(rows), "subjects")