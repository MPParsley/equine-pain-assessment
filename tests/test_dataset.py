"""Tests for the equine pain dataset."""

import csv
from pathlib import Path

from PIL import Image

from equine_pain.config import TrainConfig
from equine_pain.dataset import (
    EquinePainDataset,
    get_eval_transforms,
    get_train_transforms,
)


def _create_synthetic_images(data_dir: Path, num_per_class: int = 3):
    """Create synthetic test images in the expected folder structure."""
    for folder in ["no_pain", "moderate_pain", "severe_pain"]:
        folder_path = data_dir / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        for i in range(num_per_class):
            img = Image.new("RGB", (256, 256), color=(i * 30, 100, 200))
            img.save(folder_path / f"test_{i}.jpg")


def _create_csv_dataset(data_dir: Path, num_per_class: int = 3):
    """Create a CSV-based dataset."""
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label_name in ["no_pain", "moderate_pain", "severe_pain"]:
        for i in range(num_per_class):
            filename = f"{label_name}_{i}.jpg"
            img = Image.new("RGB", (256, 256), color=(i * 30, 100, 200))
            img.save(images_dir / filename)
            rows.append({"filename": filename, "pain_level": label_name})

    with open(data_dir / "labels.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "pain_level"])
        writer.writeheader()
        writer.writerows(rows)


def test_dataset_from_folders(tmp_path):
    """Test loading dataset from folder structure."""
    _create_synthetic_images(tmp_path)
    dataset = EquinePainDataset(str(tmp_path))

    assert len(dataset) == 9  # 3 classes * 3 images
    sample = dataset[0]
    assert "pixel_values" in sample
    assert "labels" in sample
    assert sample["labels"].item() in [0, 1, 2]


def test_dataset_from_csv(tmp_path):
    """Test loading dataset from CSV file."""
    _create_csv_dataset(tmp_path)
    dataset = EquinePainDataset(str(tmp_path))

    assert len(dataset) == 9
    sample = dataset[0]
    assert "pixel_values" in sample
    assert "labels" in sample


def test_dataset_with_transforms(tmp_path):
    """Test that transforms are applied correctly."""
    _create_synthetic_images(tmp_path)
    config = TrainConfig()

    train_transform = get_train_transforms(config)
    dataset = EquinePainDataset(str(tmp_path), transform=train_transform)

    sample = dataset[0]
    pixel_values = sample["pixel_values"]

    assert pixel_values.shape == (3, config.image_size, config.image_size)
    assert pixel_values.dtype.is_floating_point


def test_eval_transforms(tmp_path):
    """Test evaluation transforms produce deterministic output."""
    _create_synthetic_images(tmp_path)
    config = TrainConfig()
    eval_transform = get_eval_transforms(config)

    dataset = EquinePainDataset(str(tmp_path), transform=eval_transform)
    sample1 = dataset[0]["pixel_values"]
    sample2 = dataset[0]["pixel_values"]

    # Eval transforms should be deterministic
    assert (sample1 == sample2).all()


def test_empty_dataset(tmp_path):
    """Test that an empty directory produces an empty dataset."""
    dataset = EquinePainDataset(str(tmp_path))
    assert len(dataset) == 0
