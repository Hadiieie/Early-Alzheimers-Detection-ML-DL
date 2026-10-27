from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

def build_model(classes=2):
    model = resnet18(
        weights=ResNet18_Weights.IMAGENET1K_V1
    )
    model.fc = nn.Linear(
        model.fc.in_features,
        classes
    )
    return model