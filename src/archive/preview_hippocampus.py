from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
p = root / "data_audit/OAS1_0001_MR1"
mri = nib.load(p / "orig.mgz").get_fdata()
seg = nib.load(p / "synthseg_test.mgz").get_fdata()

left, right = seg == 17, seg == 53
coords = np.argwhere(left | right)
cx, cy = coords[:, :2].mean(axis=0).astype(int)
peak = int(np.argmax((left | right).sum(axis=(0, 1))))
box = (slice(cx-64, cx+64), slice(cy-48, cy+48))

fig, ax = plt.subplots(1, 5, figsize=(15, 4))
for a, z in zip(ax, range(peak-2, peak+3)):
    a.imshow(np.rot90(mri[box[0], box[1], z]), cmap="gray")
   L = np.ma.masked_where(~left[box[0], box[1], z], left[box[0], box[1], z])
R = np.ma.masked_where(~right[box[0], box[1], z], right[box[0], box[1], z])

a.imshow(np.rot90(L), cmap="Reds", alpha=.65)
a.imshow(np.rot90(R), cmap="Blues", alpha=.65)
    a.set_title(f"Slice {z}"); a.axis("off")
plt.tight_layout(); plt.show()