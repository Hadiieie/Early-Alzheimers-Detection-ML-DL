import pandas as pd
from dataset import MRIDataset
from transforms import train_tf

df = pd.read_csv("../kaggle_split_manifest.csv")
df = df[df.Split == "Train"]

ds = MRIDataset(df, train_tf)

x, y = ds[0]

print("Images:", len(ds))
print("Tensor:", x.shape)
print("Label:", y)