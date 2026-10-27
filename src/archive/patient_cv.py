from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

from patient_dataset import PatientDataset
from patient_model import PatientModel
from patient_transforms import patient_transform


root = Path(__file__).resolve().parents[1]

df = pd.read_csv(root / "oasis_model_manifest.csv")
dev = df[df["Split"] == "Development"].reset_index(drop=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(data, train):
    transform = lambda x: patient_transform(x, train=train)
    dataset = PatientDataset(data, root / "oasis_roi", transform)

    return DataLoader(
        dataset,
        batch_size=2,
        shuffle=train
    )


def train_stage(model, loader, loss_fn, params, lr, epochs):
    optimizer = torch.optim.AdamW(params, lr=lr)

    for _ in range(epochs):
        model.train()

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()


def predict(model, loader):
    model.eval()
    probabilities = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            probabilities.extend(probs.cpu().tolist())

    return probabilities


skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

oof_probabilities = [None] * len(dev)


for fold, (train_idx, val_idx) in enumerate(
    skf.split(dev, dev["Target"]),
    start=1
):

    random.seed(100 + fold)
    np.random.seed(100 + fold)
    torch.manual_seed(100 + fold)

    train_df = dev.iloc[train_idx].copy()
    val_df = dev.iloc[val_idx].copy()

    model = PatientModel(
        root / "models/kaggle_pretrained_resnet18.pt"
    ).to(device)

    normal_n = (train_df["Target"] == "Normal").sum()
    early_n = (train_df["Target"] == "Early_Demented").sum()

    weights = torch.tensor(
        [
            len(train_df) / (2 * normal_n),
            len(train_df) / (2 * early_n)
        ],
        dtype=torch.float32,
        device=device
    )

    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    # Stage 1: train classifier only
    for p in model.parameters():
        p.requires_grad = False

    for p in model.classifier.parameters():
        p.requires_grad = True

    train_stage(
        model,
        make_loader(train_df, True),
        loss_fn,
        model.classifier.parameters(),
        lr=1e-3,
        epochs=3
    )

    # Stage 2: fine-tune layer4 + classifier
    for p in model.encoder.layer4.parameters():
        p.requires_grad = True

    trainable = filter(
        lambda p: p.requires_grad,
        model.parameters()
    )

    train_stage(
        model,
        make_loader(train_df, True),
        loss_fn,
        trainable,
        lr=1e-4,
        epochs=5
    )

    probabilities = predict(
        model,
        make_loader(val_df, False)
    )

    for i, probability in zip(val_idx, probabilities):
        oof_probabilities[i] = probability

    predictions = [
        int(p >= 0.5)
        for p in probabilities
    ]

    true_labels = (
        val_df["Target"] == "Early_Demented"
    ).astype(int)

    ba = balanced_accuracy_score(
        true_labels,
        predictions
    )

    print(
        f"Fold {fold}: "
        f"Train={len(train_df)} "
        f"Val={len(val_df)} "
        f"BA={ba:.3f}"
    )


dev["Probability"] = oof_probabilities

output = root / "results/oasis_oof_predictions.csv"
output.parent.mkdir(exist_ok=True)

dev.to_csv(output, index=False)

print("\nSaved:", len(dev), "OOF predictions")
print("File:", output)