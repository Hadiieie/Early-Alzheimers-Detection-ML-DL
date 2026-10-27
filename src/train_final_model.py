from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from patient_dataset import PatientDataset
from patient_model import PatientModel
from patient_transforms import patient_transform

root = Path(__file__).resolve().parents[1]

df = pd.read_csv(root / "oasis_model_manifest.csv")
train_df = df[df["Split"] == "Development"].copy()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(2027)
np.random.seed(2027)
torch.manual_seed(2027)


def loader(train=True):
    tf = lambda x: patient_transform(x, train=train)

    ds = PatientDataset(
        train_df,
        root / "oasis_roi",
        tf,
        n_slices=3
    )

    return DataLoader(
        ds,
        batch_size=2,
        shuffle=train
    )


def fit(model, dl, loss_fn, params, lr, epochs):
    opt = torch.optim.AdamW(params, lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0

        for x, y in dl:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

            total += loss.item() * len(y)

        print(
            f"Epoch {epoch}/{epochs} "
            f"| Loss {total / len(dl.dataset):.4f}"
        )


model = PatientModel(
    root / "models/kaggle_pretrained_resnet18.pt"
).to(device)

n0 = (train_df.Target == "Normal").sum()
n1 = (train_df.Target == "Early_Demented").sum()

weights = torch.tensor(
    [
        len(train_df) / (2 * n0),
        len(train_df) / (2 * n1)
    ],
    dtype=torch.float32,
    device=device
)

loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

print("Device:", device)
print("Development patients:", len(train_df))
print("Stage 1")

for p in model.parameters():
    p.requires_grad = False

for p in model.classifier.parameters():
    p.requires_grad = True

fit(
    model,
    loader(True),
    loss_fn,
    model.classifier.parameters(),
    1e-3,
    3
)

print("Stage 2")

for p in model.encoder.layer4.parameters():
    p.requires_grad = True

params = filter(
    lambda p: p.requires_grad,
    model.parameters()
)

fit(
    model,
    loader(True),
    loss_fn,
    params,
    1e-4,
    5
)

save = root / "models/final_three_slice_model.pt"

torch.save(model.state_dict(), save)

print("\nFINAL MODEL SAVED:")
print(save)