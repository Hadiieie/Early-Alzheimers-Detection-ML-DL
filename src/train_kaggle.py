from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import MRIDataset
from transforms import train_tf, val_tf
from model import build_model
from train_utils import run_epoch

root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / "kaggle_split_manifest.csv")
tr, va = df[df.Split == "Train"], df[df.Split == "Validation"]

train = DataLoader(MRIDataset(tr, train_tf), batch_size=16, shuffle=True)
val = DataLoader(MRIDataset(va, val_tf), batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model().to(device)
counts = tr.Target.value_counts()

weights = torch.tensor([
    len(tr) / (2 * counts["Normal"]),
    len(tr) / (2 * counts["Early_Demented"])
], dtype=torch.float32, device=device)

loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
save = root / "models/kaggle_pretrained_resnet18.pt"
save.parent.mkdir(exist_ok=True)
logs = []

def stage(name, params, lr, epochs, best):
    opt = torch.optim.AdamW(params, lr=lr)
    for e in range(1, epochs + 1):
        tl, tb = run_epoch(model, train, loss_fn, device, opt)
        vl, vb = run_epoch(model, val, loss_fn, device)
        logs.append([name, e, tl, tb, vl, vb])
        print(f"{name} {e}/{epochs} | Train BA {tb:.4f} | Val BA {vb:.4f}")
        if vb > best:
            best = vb
            torch.save(model.state_dict(), save)
    return best

for p in model.parameters():
    p.requires_grad = False
for p in model.fc.parameters():
    p.requires_grad = True

print("Device:", device)
print("Class weights:", weights.tolist())

best = stage("HEAD", model.fc.parameters(), 1e-3, 3, -1)
model.load_state_dict(torch.load(save, map_location=device))

for p in model.layer4.parameters():
    p.requires_grad = True

params = filter(lambda p: p.requires_grad, model.parameters())
best = stage("LAYER4", params, 1e-4, 3, best)

model.load_state_dict(torch.load(save, map_location=device))

pd.DataFrame(
    logs,
    columns=["Stage","Epoch","TrainLoss","TrainBA","ValLoss","ValBA"]
).to_csv(root / "results/kaggle_training_log.csv", index=False)

print(f"\nBest Validation Balanced Accuracy: {best:.4f}")
print("Saved:", save)