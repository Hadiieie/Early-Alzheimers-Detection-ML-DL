from pathlib import Path
import pandas as pd
from hybrid_dataset import HybridDataset
from hybrid_model import HybridModel
from patient_transforms import patient_transform

root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root/"oasis_model_manifest.csv")
vol = pd.read_csv(root/"oasis_volume_features.csv")
df = df.merge(vol, on="Subject")
df = df[df.DevSplit=="Train"]

tf = lambda x: patient_transform(x, train=False)
ds = HybridDataset(df, root/"oasis_roi", tf)

x, f, y = ds[0]
model = HybridModel(root/"models/kaggle_pretrained_resnet18.pt")
out = model(x.unsqueeze(0), f.unsqueeze(0))

print("Images:", x.shape)
print("Features:", f)
print("Label:", y)
print("Output:", out.shape)