import sys

sys.path.append("..")
from preprocessing.tiff_handling import handle_tiff
import pandas as pd
from torch.utils.data import Dataset


class ImagePathDataset(Dataset):
    def __init__(self, data: pd.DataFrame, transform=None, target_transform=None):
        self.file_paths = data["file_names"].tolist()  # List of file paths
        self.labels = data["label"].tolist()  # List of labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = handle_tiff(self.file_paths[idx])

        # Convert the numpy array to a PIL Image for compatibility with transforms
        label = 1 if self.labels[idx] == "Singlet" else 0

        # Apply transformations if provided
        if self.transform:
            image = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
