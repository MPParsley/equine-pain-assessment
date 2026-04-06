"""Dataset loading and preprocessing for equine pain assessment."""

import logging
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision import transforms

from equine_pain.config import CLASS_NAMES, TrainConfig

logger = logging.getLogger(__name__)


def get_train_transforms(config: TrainConfig) -> transforms.Compose:
    """Data augmentation transforms for training."""
    return transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.imagenet_mean, std=config.imagenet_std),
    ])


def get_eval_transforms(config: TrainConfig) -> transforms.Compose:
    """Transforms for validation and inference (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.imagenet_mean, std=config.imagenet_std),
    ])


class EquinePainDataset(Dataset):
    """Dataset for horse pain classification from images.

    Supports a simple directory layout where images are organized by class:
        data_dir/
            no_pain/
                image1.jpg
                image2.jpg
            moderate_pain/
                image3.jpg
            severe_pain/
                image4.jpg

    Also supports a flat directory with a labels CSV file:
        data_dir/
            images/
                image1.jpg
            labels.csv  (columns: filename, pain_level)
    """

    FOLDER_TO_LABEL = {
        "no_pain": 0,
        "moderate_pain": 1,
        "severe_pain": 2,
    }

    def __init__(self, data_dir: str, transform: transforms.Compose | None = None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        self._load_samples()
        logger.info("Loaded %d samples from %s", len(self.samples), data_dir)

    def _load_samples(self):
        """Load image paths and labels from the data directory."""
        csv_path = self.data_dir / "labels.csv"
        if csv_path.exists():
            self._load_from_csv(csv_path)
        else:
            self._load_from_folders()

    def _load_from_folders(self):
        """Load from class-organized folder structure."""
        for folder_name, label in self.FOLDER_TO_LABEL.items():
            folder = self.data_dir / folder_name
            if not folder.exists():
                continue
            for img_path in sorted(folder.iterdir()):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
                    self.samples.append((img_path, label))

    def _load_from_csv(self, csv_path: Path):
        """Load from a CSV file with columns: filename, pain_level."""
        import csv

        images_dir = self.data_dir / "images"
        label_map = {name.lower().replace(" ", "_"): i for i, name in enumerate(CLASS_NAMES)}
        # Also support numeric labels
        label_map.update({"0": 0, "1": 1, "2": 2})

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["filename"]
                pain_level = row["pain_level"].strip().lower().replace(" ", "_")
                label = label_map.get(pain_level)
                if label is None:
                    logger.warning("Unknown pain level '%s' for %s, skipping", pain_level, filename)
                    continue
                img_path = images_dir / filename
                if img_path.exists():
                    self.samples.append((img_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"pixel_values": image, "labels": torch.tensor(label, dtype=torch.long)}


def create_splits(
    data_dir: str, config: TrainConfig
) -> tuple[EquinePainDataset, EquinePainDataset, EquinePainDataset]:
    """Create train/val/test splits from a dataset directory."""
    full_dataset = EquinePainDataset(data_dir)

    total = len(full_dataset)
    train_size = int(total * config.train_split)
    val_size = int(total * config.val_split)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(config.seed)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    train_transform = get_train_transforms(config)
    eval_transform = get_eval_transforms(config)

    train_ds = _TransformedSubset(train_subset, train_transform)
    val_ds = _TransformedSubset(val_subset, eval_transform)
    test_ds = _TransformedSubset(test_subset, eval_transform)

    logger.info("Split sizes - train: %d, val: %d, test: %d", train_size, val_size, test_size)
    return train_ds, val_ds, test_ds


class _TransformedSubset(Dataset):
    """Wraps a Subset to apply transforms at access time."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"pixel_values": image, "labels": torch.tensor(label, dtype=torch.long)}


def download_dataset(output_dir: str = "data") -> str:
    """Download the equine pain dataset from Mendeley Data.

    Dataset: 'Automatic Pain Assessment in Horses'
    DOI: 10.17632/t8rtzcgwxm.3

    Returns the path to the downloaded data directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    marker = output_path / ".downloaded"
    if marker.exists():
        logger.info("Dataset already downloaded at %s", output_path)
        return str(output_path)

    try:
        import urllib.request
        import zipfile

        dataset_url = (
            "https://data.mendeley.com/public-files/datasets/t8rtzcgwxm/files/"
            "dataset.zip"
        )
        zip_path = output_path / "dataset.zip"

        logger.info("Downloading dataset from Mendeley Data...")
        logger.info("URL: %s", dataset_url)
        logger.info(
            "If automatic download fails, manually download from: "
            "https://data.mendeley.com/datasets/t8rtzcgwxm/3"
        )

        urllib.request.urlretrieve(dataset_url, zip_path)

        logger.info("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_path)
        os.remove(zip_path)

        marker.touch()
        logger.info("Dataset downloaded and extracted to %s", output_path)
    except Exception:
        logger.exception(
            "Automatic download failed. Please manually download from: "
            "https://data.mendeley.com/datasets/t8rtzcgwxm/3 "
            "and extract to %s",
            output_path,
        )
        logger.info(
            "Organize images into subfolders: no_pain/, moderate_pain/, severe_pain/ "
            "OR provide a labels.csv with columns: filename, pain_level"
        )
        raise

    return str(output_path)
