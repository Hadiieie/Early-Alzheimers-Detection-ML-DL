from pathlib import Path
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image

from patient_dataset import PatientDataset
from patient_model import PatientModel
from patient_transforms import patient_transform
from gradcam import GradCAM

root = Path(__file__).resolve().parents[1]
subject = "OAS1_0001_MR1"

df = pd.read_csv(root/"oasis_model_manifest.csv")
row = df[df.Subject == subject].copy()

tf = lambda x: patient_transform(x, train=False)
ds = PatientDataset(row, root/"oasis_roi", tf, n_slices=3)

x, y = ds[0]

model = PatientModel(root/"models/kaggle_pretrained_resnet18.pt")
model.load_state_dict(torch.load(
    root/"models/final_three_slice_model.pt",
    map_location="cpu"
))
model.eval()

with torch.no_grad():
    p = torch.softmax(model(x.unsqueeze(0)),1)[0,1].item()

target = 1 if p >= 0.42 else 0

cam = GradCAM(model)
_, maps = cam(x.unsqueeze(0), target)

folder = root/"oasis_roi"/row.iloc[0].Target/subject
files = ["slice_2.png","slice_3.png","slice_4.png"]

fig, ax = plt.subplots(1,3,figsize=(12,4))

for i,f in enumerate(files):
    img = Image.open(folder/f).resize((224,224))
    ax[i].imshow(img, cmap="gray")
    ax[i].imshow(maps[i], cmap="jet", alpha=.40)
    ax[i].set_title(f"Slice {i+1}")
    ax[i].axis("off")

fig.suptitle(
    f"{subject} | Early score={p:.3f} | "
    f"Decision={'Early Demented' if target else 'Normal'}"
)

out = root/"results/EXPLAINABILITY"
out.mkdir(parents=True, exist_ok=True)

plt.tight_layout()
plt.savefig(out/f"gradcam_{subject}.png", dpi=200)
plt.show()