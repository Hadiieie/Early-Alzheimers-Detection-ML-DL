import torch
from torch import nn
from model import build_model

class WeightedPatientModel(nn.Module):
    def __init__(self, weights):
        super().__init__()
        base = build_model()
        base.load_state_dict(torch.load(weights, map_location="cpu"))

        self.classifier = base.fc
        base.fc = nn.Identity()
        self.encoder = base

        self.slice_logits = nn.Parameter(torch.zeros(5))

    def forward(self, x):
        b, s, c, h, w = x.shape
        features = self.encoder(x.reshape(b*s, c, h, w))
        features = features.reshape(b, s, -1)

        weights = torch.softmax(self.slice_logits, dim=0)
        patient = (features * weights.view(1, 5, 1)).sum(dim=1)

        return self.classifier(patient)