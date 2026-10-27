import torch
from torch import nn
from model import build_model

class PatientModel(nn.Module):
    def __init__(self, weights):
        super().__init__()

        self.encoder = build_model()
        self.encoder.load_state_dict(
            torch.load(weights, map_location="cpu")
        )

        self.classifier = self.encoder.fc
        self.encoder.fc = nn.Identity()

    def forward(self, x):
        b, s, c, h, w = x.shape

        features = self.encoder(
            x.reshape(b*s, c, h, w)
        )

        features = features.reshape(
            b, s, -1
        ).mean(dim=1)

        return self.classifier(features)