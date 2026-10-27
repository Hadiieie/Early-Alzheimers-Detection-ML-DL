from pathlib import Path
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / "kaggle_binary_manifest.csv")

classes = ["Normal", "VeryMild", "Mild"]
fig, ax = plt.subplots(3, 4, figsize=(12, 9))

for r, cls in enumerate(classes):
    sample = df[df["Class"] == cls].sample(4, random_state=42)
    for c, row in enumerate(sample.itertuples()):
        img = Image.open(row.Path).convert("L")
        ax[r, c].imshow(img, cmap="gray")
        ax[r, c].set_title(f"{cls}\n{img.size}")
        ax[r, c].axis("off")

plt.tight_layout()
plt.show()