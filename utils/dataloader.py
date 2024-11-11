import sys

sys.path.append("..")
from preprocessing.tiff_handling import handle_tiff
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class ImagePathDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths  # List of file paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = handle_tiff(self.file_paths[idx])

        # Convert the numpy array to a PIL Image for compatibility with transforms
        image = Image.fromarray(img)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image
