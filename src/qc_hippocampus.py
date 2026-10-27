from pathlib import Path
import nibabel as nib
import numpy as np

root = Path(__file__).resolve().parents[1] / "data_audit"

for p in sorted(root.glob("OAS1_*")):
    img = nib.load(p / "synthseg.mgz")
    seg = img.get_fdata()
    L, R = seg == 17, seg == 53
    mask = L | R

    axis = next(i for i, c in enumerate(nib.aff2axcodes(img.affine)) if c in ("A", "P"))
    counts = mask.sum(axis=tuple(i for i in range(3) if i != axis))
    peak = int(np.argmax(counts))

    voxel = abs(np.linalg.det(img.affine[:3, :3]))
    print(p.name, "| L:", round(L.sum()*voxel), "| R:", round(R.sum()*voxel),
          "| Peak:", peak, "| Slices:", list(range(peak-2, peak+3)))