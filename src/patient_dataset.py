from PIL import Image
from torch.utils.data import Dataset

class PatientDataset(Dataset):
    def __init__(self, df, root, transform, n_slices=5):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.transform = transform
        self.n_slices = n_slices

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        folder = self.root / row.Target / row.Subject

        ids = {
            1: [3],
            3: [2, 3, 4],
            5: [1, 2, 3, 4, 5]
        }[self.n_slices]

        images = [
            Image.open(folder / f"slice_{n}.png").convert("RGB")
            for n in ids
        ]

        label = 1 if row.Target == "Early_Demented" else 0
        return self.transform(images), label