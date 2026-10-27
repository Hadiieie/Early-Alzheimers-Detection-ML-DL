from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from patient_dataset import PatientDataset
from patient_model import PatientModel
from patient_transforms import patient_transform

root = Path(__file__).resolve().parents[1]

df = pd.read_csv(root / "oasis_model_manifest.csv")
dev = df[df.Split == "Development"].reset_index(drop=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loader(data, train):
    tf = lambda x: patient_transform(x, train=train)
    ds = PatientDataset(data, root / "oasis_roi", tf, n_slices=3)
    return DataLoader(ds, batch_size=2, shuffle=train)


def fit(model, dl, loss_fn, params, lr, epochs):
    opt = torch.optim.AdamW(params, lr=lr)

    for _ in range(epochs):
        model.train()

        for x, y in dl:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()


def predict(model, dl):
    model.eval()
    probs = []

    with torch.no_grad():
        for x, _ in dl:
            p = torch.softmax(model(x.to(device)), 1)[:, 1]
            probs += p.cpu().tolist()

    return probs


skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

oof = [None] * len(dev)


for fold, (tr, va) in enumerate(
    skf.split(dev, dev.Target),
    start=1
):

    random.seed(100 + fold)
    np.random.seed(100 + fold)
    torch.manual_seed(100 + fold)

    train_df = dev.iloc[tr].copy()
    val_df = dev.iloc[va].copy()

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

    for p in model.parameters():
        p.requires_grad = False

    for p in model.classifier.parameters():
        p.requires_grad = True

    fit(
        model,
        loader(train_df, True),
        loss_fn,
        model.classifier.parameters(),
        1e-3,
        3
    )

    for p in model.encoder.layer4.parameters():
        p.requires_grad = True

    params = filter(
        lambda p: p.requires_grad,
        model.parameters()
    )

    fit(
        model,
        loader(train_df, True),
        loss_fn,
        params,
        1e-4,
        5
    )

    probs = predict(
        model,
        loader(val_df, False)
    )

    for i, p in zip(va, probs):
        oof[i] = p

    pred = [int(p >= .5) for p in probs]
    true = (val_df.Target == "Early_Demented").astype(int)

    ba = balanced_accuracy_score(true, pred)

    print(f"Fold {fold}: BA={ba:.3f}")


dev["Probability"] = oof

out = root / "results/three_slice_oof_predictions.csv"
dev.to_csv(out, index=False)

y = (dev.Target == "Early_Demented").astype(int)
pred = (np.array(oof) >= .5).astype(int)

print()
print("OOF BA:", round(balanced_accuracy_score(y, pred), 3))
print("OOF ROC-AUC:", round(roc_auc_score(y, oof), 3))
print("Saved:", out)