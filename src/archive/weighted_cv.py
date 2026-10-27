from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

from patient_dataset import PatientDataset
from weighted_patient_model import WeightedPatientModel
from patient_transforms import patient_transform

root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / "oasis_model_manifest.csv")
dev = df[df["Split"] == "Development"].reset_index(drop=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(data, train):
    tf = lambda x: patient_transform(x, train=train)
    ds = PatientDataset(data, root / "oasis_roi", tf)
    return DataLoader(ds, batch_size=2, shuffle=train)


def train_stage(model, loader, loss_fn, params, lr, epochs):
    optimizer = torch.optim.AdamW(params, lr=lr)

    for _ in range(epochs):
        model.train()

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()


def predict(model, loader):
    model.eval()
    probabilities = []

    with torch.no_grad():
        for images, _ in loader:
            p = torch.softmax(model(images.to(device)), 1)[:, 1]
            probabilities.extend(p.cpu().tolist())

    return probabilities


skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

oof = [None] * len(dev)


for fold, (tr, va) in enumerate(
    skf.split(dev, dev["Target"]),
    start=1
):

    random.seed(100 + fold)
    np.random.seed(100 + fold)
    torch.manual_seed(100 + fold)

    train_df = dev.iloc[tr].copy()
    val_df = dev.iloc[va].copy()

    model = WeightedPatientModel(
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

    # Stage 1
    for p in model.parameters():
        p.requires_grad = False

    for p in model.classifier.parameters():
        p.requires_grad = True

    model.slice_logits.requires_grad = True

    params = list(model.classifier.parameters()) + [model.slice_logits]

    train_stage(
        model,
        make_loader(train_df, True),
        loss_fn,
        params,
        lr=1e-3,
        epochs=3
    )

    # Stage 2
    for p in model.encoder.layer4.parameters():
        p.requires_grad = True

    params = filter(
        lambda p: p.requires_grad,
        model.parameters()
    )

    train_stage(
        model,
        make_loader(train_df, True),
        loss_fn,
        params,
        lr=1e-4,
        epochs=5
    )

    probabilities = predict(
        model,
        make_loader(val_df, False)
    )

    for i, p in zip(va, probabilities):
        oof[i] = p

    predictions = [int(p >= .5) for p in probabilities]
    true = (val_df.Target == "Early_Demented").astype(int)

    ba = balanced_accuracy_score(true, predictions)

    slice_weights = torch.softmax(
        model.slice_logits.detach().cpu(), 0
    ).numpy()

    print(
        f"Fold {fold}: BA={ba:.3f} | "
        f"Weights={np.round(slice_weights,3)}"
    )


dev["Probability"] = oof

out = root / "results/weighted_oof_predictions.csv"
dev.to_csv(out, index=False)

print("\nSaved 44 weighted OOF predictions")