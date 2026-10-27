from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
p = root / "data_audit/OAS1_0001_MR1"

mri = nib.load(p / "orig.mgz").get_fdata()
seg = nib.load(p / "synthseg_test.mgz").get_fdata()
mask = np.isin(seg, [17, 53])

x, y, _ = np.where(mask)
box = (slice(x.min()-8, x.max()+9), slice(y.min()-8, y.max()+9))

fig, ax = plt.subplots(1, 5, figsize=(15, 4))
for a, z in zip(ax, range(120, 125)):
    a.imshow(np.rot90(mri[box[0], box[1], z]), cmap="gray")
    a.set_title(f"Slice {z}")
    a.axis("off")
plt.tight_layout()
plt.show()