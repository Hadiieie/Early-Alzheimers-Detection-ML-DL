from PIL import Image
from torch.utils.data import Dataset

class MRIDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        image = Image.open(row.Path).convert("RGB")
        image = self.transform(image)
        label = 1 if row.Target == "Early_Demented" else 0
        return image, label