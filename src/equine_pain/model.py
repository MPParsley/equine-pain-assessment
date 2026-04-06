"""Model architecture for equine pain classification."""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTImageProcessor

from equine_pain.config import CLASS_NAMES, NUM_CLASSES, TrainConfig

logger = logging.getLogger(__name__)


def create_model(config: TrainConfig) -> ViTForImageClassification:
    """Create a ViT model fine-tuned for pain classification.

    Uses HuggingFace's ViTForImageClassification with a custom classification head
    for 3-class pain level prediction (No Pain / Moderate / Severe).
    """
    model = ViTForImageClassification.from_pretrained(
        config.model_name,
        num_labels=NUM_CLASSES,
        id2label={i: name for i, name in enumerate(CLASS_NAMES)},
        label2id={name: i for i, name in enumerate(CLASS_NAMES)},
        ignore_mismatched_sizes=True,
    )
    logger.info("Created model from %s with %d classes", config.model_name, NUM_CLASSES)
    return model


def load_model(
    model_path: str | None = None,
    hub_model_id: str | None = None,
    device: str | None = None,
) -> tuple[ViTForImageClassification, ViTImageProcessor]:
    """Load a trained model from a local checkpoint or HuggingFace Hub.

    Args:
        model_path: Path to a local model directory or .pt checkpoint file.
        hub_model_id: HuggingFace Hub model ID (e.g., 'username/equine-pain-vit').
        device: Device to load the model on. Auto-detected if None.

    Returns:
        Tuple of (model, processor).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if hub_model_id:
        logger.info("Loading model from HuggingFace Hub: %s", hub_model_id)
        model = ViTForImageClassification.from_pretrained(hub_model_id)
        processor = ViTImageProcessor.from_pretrained(hub_model_id)
    elif model_path:
        path = Path(model_path)
        if path.is_dir():
            logger.info("Loading model from directory: %s", model_path)
            model = ViTForImageClassification.from_pretrained(model_path)
            processor = ViTImageProcessor.from_pretrained(model_path)
        elif path.suffix in (".pt", ".pth"):
            logger.info("Loading model from checkpoint: %s", model_path)
            config = TrainConfig()
            model = create_model(config)
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            processor = ViTImageProcessor.from_pretrained(config.model_name)
        else:
            raise ValueError(f"Unsupported model path format: {model_path}")
    else:
        raise ValueError("Either model_path or hub_model_id must be provided")

    model = model.to(device)
    model.eval()
    return model, processor


class EquinePainModelWrapper(nn.Module):
    """Wrapper for custom training loops (alternative to HuggingFace Trainer)."""

    def __init__(self, config: TrainConfig):
        super().__init__()
        self.model = create_model(config)
        self.config = config

    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor | None = None):
        return self.model(pixel_values=pixel_values, labels=labels)

    def save(self, output_dir: str):
        """Save model and processor to a directory (HuggingFace format)."""
        self.model.save_pretrained(output_dir)
        processor = ViTImageProcessor.from_pretrained(self.config.model_name)
        processor.save_pretrained(output_dir)
        logger.info("Model saved to %s", output_dir)

    def push_to_hub(self, hub_model_id: str):
        """Push model to HuggingFace Hub."""
        self.model.push_to_hub(hub_model_id)
        processor = ViTImageProcessor.from_pretrained(self.config.model_name)
        processor.push_to_hub(hub_model_id)
        logger.info("Model pushed to HuggingFace Hub: %s", hub_model_id)
