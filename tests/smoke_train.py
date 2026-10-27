import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import MRIDataset
from transforms import train_tf, val_tf
from model import build_model
from train_utils import run_epoch

df = pd.read_csv("../kaggle_split_manifest.csv")
tr = df[df.Split == "Train"].sample(256, random_state=42)
va = df[df.Split == "Validation"].sample(128, random_state=42)

train = DataLoader(MRIDataset(tr, train_tf), batch_size=8, shuffle=True)
val = DataLoader(MRIDataset(va, val_tf), batch_size=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model().to(device)

for p in model.parameters(): p.requires_grad = False
for p in model.fc.parameters(): p.requires_grad = True

loss_fn = torch.nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)

print("Device:", device)
print("Train:", run_epoch(model, train, loss_fn, device, opt))
print("Val:", run_epoch(model, val, loss_fn, device))