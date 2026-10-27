from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image

root = Path(__file__).resolve().parents[1]
data = root / "oasis_clean"
out = root / "oasis_roi"
df = pd.read_csv(root / "oasis_final_manifest.csv")

def to_u8(x):
    lo, hi = np.percentile(x, [1, 99])
    x = np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
    return (x * 255).astype(np.uint8)

def save_subject(row):
    p = data / row.Subject
    mri = nib.load(p / "orig.mgz").get_fdata()
    seg = nib.load(p / "synthseg.mgz").get_fdata()
    c = np.argwhere(np.isin(seg, [17, 53]))[:, :2].mean(0).astype(int)
    box = (slice(c[0]-64, c[0]+64), slice(c[1]-48, c[1]+48))
    dst = out / row.Target / row.Subject
    dst.mkdir(parents=True, exist_ok=True)
    for i, z in enumerate([row.S1,row.S2,row.S3,row.S4,row.S5], 1):
        roi = np.rot90(mri[box[0], box[1], int(z)])
        Image.fromarray(to_u8(roi)).save(dst / f"slice_{i}.png")

for row in df.itertuples():
    save_subject(row)

print("Patients:", len(df))
print("Images:", len(list(out.rglob("*.png"))))