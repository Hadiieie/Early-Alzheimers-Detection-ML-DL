import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetPatientModel(nn.Module):
    def __init__(self, pretrained=True, n_classes=2):
        super().__init__()

        weights = (
            EfficientNet_B0_Weights.IMAGENET1K_V1
            if pretrained else None
        )

        net = efficientnet_b0(weights=weights)

        self.encoder = net.features
        self.pool = net.avgpool

        feature_dim = net.classifier[1].in_features

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, n_classes)
        )

    def encode_slice(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        # x: [batch, slices, channels, height, width]

        slice_features = []

        for i in range(x.shape[1]):
            features = self.encode_slice(x[:, i])
            slice_features.append(features)

        features = torch.stack(slice_features, dim=1)

        patient_features = features.mean(dim=1)

        return self.classifier(patient_features)