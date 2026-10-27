from pathlib import Path
import pandas as pd
from patient_dataset import PatientDataset
from patient_model import PatientModel
from patient_transforms import patient_transform

root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / "oasis_model_manifest.csv")
df = df[df.DevSplit == "Train"]

tf = lambda x: patient_transform(x, train=False)
ds = PatientDataset(df, root / "oasis_roi", tf)

x, y = ds[0]
model = PatientModel(root / "models/kaggle_pretrained_resnet18.pt")
out = model(x.unsqueeze(0))

print("Patients:", len(ds))
print("Input:", x.shape)
print("Label:", y)
print("Output:", out.shape)