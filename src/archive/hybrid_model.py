import torch
from torch import nn
from model import build_model

class HybridModel(nn.Module):
    def __init__(self, weights):
        super().__init__()
        base = build_model()
        base.load_state_dict(torch.load(weights, map_location="cpu"))

        old_head = base.fc
        base.fc = nn.Identity()
        self.encoder = base
        self.classifier = nn.Linear(515, 2)

        with torch.no_grad():
            self.classifier.weight[:, :512].copy_(old_head.weight)
            self.classifier.weight[:, 512:].zero_()
            self.classifier.bias.copy_(old_head.bias)

    def forward(self, images, anatomy):
        b,s,c,h,w = images.shape
        x = self.encoder(images.reshape(b*s,c,h,w))
        x = x.reshape(b,s,-1).mean(1)
        return self.classifier(torch.cat([x, anatomy], 1))