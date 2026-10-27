from pathlib import Path
import json
import torch
from PIL import Image

from patient_model import PatientModel
from patient_transforms import patient_transform

ROOT = Path(__file__).resolve().parents[1]


def load_final():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PatientModel(ROOT / "models/kaggle_pretrained_resnet18.pt")
    state = torch.load(
        ROOT / "models/final_three_slice_model.pt",
        map_location=device
    )

    model.load_state_dict(state)
    model.to(device).eval()

    with open(ROOT / "three_slice_threshold.json") as f:
        threshold = float(json.load(f)["threshold"])

    return model, device, threshold


def predict_files(files, model, device, threshold):
    images = [Image.open(f).convert("RGB") for f in files]
    x = patient_transform(images, train=False).unsqueeze(0).to(device)

    with torch.no_grad():
        early = torch.softmax(model(x), dim=1)[0, 1].item()

    label = "Early_Demented" if early >= threshold else "Normal"
    score = early if label == "Early_Demented" else 1 - early

    return label, early, score