"""Configuration and constants for the Equine Pain Assessment model."""

from dataclasses import dataclass, field

# Horse Grimace Scale Facial Action Units
FAU_NAMES = [
    "Stiffly backwards ears",
    "Orbital tightening",
    "Tension above the eye area",
    "Prominent strained chewing muscles",
    "Mouth strained and pronounced chin",
    "Strained nostrils and flattening of the profile",
]

FAU_DESCRIPTIONS = [
    "Ears held stiffly and turned backwards, space between ears appears wider.",
    "Eyelid partially or completely closed, reducing the eye size by more than half.",
    "Contraction of muscles above the eye increases visibility of underlying bone surfaces.",
    "Chewing muscles clearly visible as increased tension above the mouth.",
    "Upper lip drawn back, lower lip causes a pronounced chin.",
    "Nostrils strained and slightly dilated, profile of nose flattens and lips elongate.",
]

# Pain level classification based on total HGS score (0-12)
PAIN_LEVELS = {
    "No Pain": (0, 3),
    "Moderate Pain": (4, 7),
    "Severe Pain": (8, 12),
}

NUM_CLASSES = 3
CLASS_NAMES = list(PAIN_LEVELS.keys())


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    model_name: str = "google/vit-base-patch16-224"
    image_size: int = 224
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 3
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42
    num_workers: int = 2
    output_dir: str = "models"
    data_dir: str = "data"
    hub_model_id: str = ""
    imagenet_mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    imagenet_std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
