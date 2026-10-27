from pathlib import Path
import nibabel as nib
import numpy as np

root = Path(__file__).resolve().parents[1]
file = root / "data_audit/OAS1_0001_MR1/synthseg_test.mgz"

img = nib.load(file)
seg = img.get_fdata()
mask = np.isin(seg, [17, 53])

codes = nib.aff2axcodes(img.affine)
coronal_axis = next(i for i, c in enumerate(codes) if c in ("A", "P"))

counts = mask.sum(axis=tuple(i for i in range(3) if i != coronal_axis))
peak = int(np.argmax(counts))

print("Orientation :", codes)
print("Coronal axis:", coronal_axis)
print("Peak slice  :", peak)
print("5 slices    :", list(range(peak - 2, peak + 3)))