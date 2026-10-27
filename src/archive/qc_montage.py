from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1] / "data_audit"
subjects = sorted(root.glob("OAS1_*"))
fig, ax = plt.subplots(len(subjects), 5, figsize=(15, 3*len(subjects)))

for r, p in enumerate(subjects):
    mri = nib.load(p/"orig.mgz").get_fdata()
    seg = nib.load(p/"synthseg.mgz").get_fdata()
    L, R = seg == 17, seg == 53
    mask = L | R
    c = np.argwhere(mask)[:, :2].mean(axis=0).astype(int)
    peak = int(np.argmax(mask.sum(axis=(0,1))))
    box = (slice(c[0]-64,c[0]+64), slice(c[1]-48,c[1]+48))
    for j, z in enumerate(range(peak-2, peak+3)):
        ax[r,j].imshow(np.rot90(mri[box[0],box[1],z]), cmap="gray")
        ax[r,j].contour(np.rot90(L[box[0],box[1],z]), levels=[.5], linewidths=1)
        ax[r,j].contour(np.rot90(R[box[0],box[1],z]), levels=[.5], linewidths=1)
        ax[r,j].set_title(f"{p.name}\nSlice {z}", fontsize=8); ax[r,j].axis("off")

plt.tight_layout()
plt.show()