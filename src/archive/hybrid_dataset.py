from PIL import Image
from torch.utils.data import Dataset
import torch

class HybridDataset(Dataset):
    def __init__(self, df, root, transform):
        self.df, self.root, self.transform = df.reset_index(drop=True), root, transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        folder = self.root / row.Target / row.Subject
        images = [Image.open(folder/f"slice_{n}.png").convert("RGB") for n in range(1,6)]
        features = torch.tensor([row.LeftNorm, row.RightNorm, row.Asymmetry], dtype=torch.float32)
        label = 1 if row.Target == "Early_Demented" else 0
        return self.transform(images), features, label