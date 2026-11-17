from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "v2"))

from patient_dataset import PatientDataset
from patient_transforms import patient_transform
from efficientnet_patient_model import EfficientNetPatientModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS_HEAD = 3
EPOCHS_FULL = 3
BATCH_SIZE = 4


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0.0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total += loss.item()

    return total / len(loader)


def predict(model, loader):
    model.eval()
    ys, ps = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)

            pred = logits.argmax(dim=1).cpu().numpy()

            ys.extend(y.numpy())
            ps.extend(pred)

    return np.array(ys), np.array(ps)


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    manifest = pd.read_csv(ROOT / "oasis_model_manifest.csv")
    dev = manifest[manifest["Split"] == "Development"].reset_index(drop=True)

    dataset = PatientDataset(
        dev,
        root=ROOT / "oasis_roi",
        transform=patient_transform,
        n_slices=3
    )

    y = (dev["Target"] == "Early_Demented").astype(int).to_numpy()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(y)), y), 1):

        print(f"\n===== FOLD {fold} =====")

        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        model = EfficientNetPatientModel(pretrained=True).to(DEVICE)

        for p in model.encoder.parameters():
            p.requires_grad = False

        criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            model.classifier.parameters(),
            lr=1e-3
        )

        for epoch in range(EPOCHS_HEAD):
            loss = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion
            )

            print(
                f"Head {epoch+1}/{EPOCHS_HEAD} "
                f"Loss={loss:.4f}"
            )

        for p in model.encoder.parameters():
            p.requires_grad = True

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-4
        )

        for epoch in range(EPOCHS_FULL):
            loss = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion
            )

            print(
                f"Full {epoch+1}/{EPOCHS_FULL} "
                f"Loss={loss:.4f}"
            )

        true, pred = predict(
            model,
            val_loader
        )

        ba = balanced_accuracy_score(
            true,
            pred
        )

        fold_scores.append(ba)

        print(f"Fold BA = {ba:.3f}")

    print("\n==============================")
    print("EfficientNet-B0 3-Slice CV")
    print("==============================")

    for i, score in enumerate(fold_scores, 1):
        print(f"Fold {i}: {score:.3f}")

    print(
        f"\nMean BA = "
        f"{np.mean(fold_scores):.3f} "
        f"+/- {np.std(fold_scores):.3f}"
    )


if __name__ == "__main__":
    main()