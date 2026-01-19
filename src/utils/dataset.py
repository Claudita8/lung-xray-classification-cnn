import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from src.utils.paths import PROJECT_ROOT

CLASS_TO_IDX = {
    "COVID": 0,
    "Pneumonia": 1,
    "Normal": 2,
}


class XRayDataset(Dataset):
    def __init__(self, csv_path: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        if "path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError(f"CSV must contain columns: path,label. Got: {self.df.columns.tolist()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = PROJECT_ROOT / row["path"]
        label_str = row["label"]

        image = Image.open(img_path).convert("RGB")
        label = CLASS_TO_IDX[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label
