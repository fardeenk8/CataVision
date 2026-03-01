"""
PyTorch Dataset for cataract binary classification.
Loads images by path, applies train-time augmentation; val/test only resize + normalize.
"""
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

# Add project root
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD

try:
    from torchvision import transforms
except ImportError:
    transforms = None


def get_train_transform():
    """Train: resize, random flip, small rotation, color jitter, normalize."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])


def get_val_test_transform():
    """Val/Test: resize and normalize only."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])


class CataractDataset(Dataset):
    """Dataset of (image_path, label) for binary cataract."""

    def __init__(self, samples, transform=None):
        """
        Args:
            samples: list of dicts with keys "image_path", "label"
            transform: torchvision transform (train or val/test)
        """
        self.samples = samples
        self.transform = transform or get_val_test_transform()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        path = item["image_path"]
        label = item["label"]
        if isinstance(label, (list, np.ndarray)):
            label = int(label[0]) if len(label) else 0
        label = int(label)
        # Load image
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)
