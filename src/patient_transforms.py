import random
import torch
from torchvision.transforms import functional as F

MEAN = [.485, .456, .406]
STD = [.229, .224, .225]

def patient_transform(images, train=False):
    angle = random.uniform(-5, 5) if train else 0
    shift = [random.randint(-5, 5) for _ in range(2)] if train else [0, 0]
    output = []

    for image in images:
        image = F.resize(image, [224, 224])
        image = F.affine(image, angle, shift, 1, [0, 0])
        image = F.normalize(F.to_tensor(image), MEAN, STD)
        output.append(image)

    return torch.stack(output)