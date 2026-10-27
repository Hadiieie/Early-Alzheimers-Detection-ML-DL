from torchvision import transforms

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(5),
    transforms.RandomAffine(0, translate=(.03, .03)),
    transforms.ToTensor(),
    transforms.Normalize([.485,.456,.406], [.229,.224,.225])
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([.485,.456,.406], [.229,.224,.225])
])