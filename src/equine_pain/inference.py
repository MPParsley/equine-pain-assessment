"""Inference pipeline for equine pain assessment."""

import logging
from pathlib import Path

import torch
from PIL import Image

from equine_pain.config import CLASS_NAMES, FAU_DESCRIPTIONS, FAU_NAMES, PAIN_LEVELS

logger = logging.getLogger(__name__)


class PainPredictor:
    """Predicts horse pain level from a single image using a trained ViT model."""

    def __init__(
        self,
        model_path: str | None = None,
        hub_model_id: str | None = None,
        device: str | None = None,
    ):
        from equine_pain.model import load_model

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.processor = load_model(
            model_path=model_path, hub_model_id=hub_model_id, device=self.device
        )

    @torch.no_grad()
    def predict(self, image: str | Path | Image.Image) -> dict:
        """Predict pain level from an image.

        Args:
            image: File path or PIL Image.

        Returns:
            Dictionary with:
                - pain_level: "No Pain", "Moderate Pain", or "Severe Pain"
                - confidence: float 0-1
                - class_probabilities: dict mapping class names to probabilities
                - predicted_class: int (0, 1, or 2)
                - hgs_score_range: tuple (min, max) of HGS score range
                - fau_guide: list of FAU descriptions for reference
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Expected str, Path, or PIL Image, got {type(image)}")

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze()

        predicted_class = probs.argmax().item()
        pain_level = CLASS_NAMES[predicted_class]
        confidence = probs[predicted_class].item()

        return {
            "pain_level": pain_level,
            "confidence": confidence,
            "class_probabilities": {
                name: probs[i].item() for i, name in enumerate(CLASS_NAMES)
            },
            "predicted_class": predicted_class,
            "hgs_score_range": PAIN_LEVELS[pain_level],
            "fau_guide": [
                {"name": name, "description": desc}
                for name, desc in zip(FAU_NAMES, FAU_DESCRIPTIONS)
            ],
        }


def predict_from_cli():
    """CLI entry point for single image prediction."""
    import argparse

    parser = argparse.ArgumentParser(description="Predict horse pain level from an image")
    parser.add_argument("image", type=str, help="Path to horse face image")
    parser.add_argument("--model-path", type=str, help="Path to trained model directory")
    parser.add_argument("--hub-model-id", type=str, help="HuggingFace Hub model ID")
    args = parser.parse_args()

    if not args.model_path and not args.hub_model_id:
        print("Error: Provide either --model-path or --hub-model-id")
        return

    predictor = PainPredictor(model_path=args.model_path, hub_model_id=args.hub_model_id)
    result = predictor.predict(args.image)

    print(f"\nPain Level: {result['pain_level']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"HGS Score Range: {result['hgs_score_range'][0]}-{result['hgs_score_range'][1]}")
    print("\nClass Probabilities:")
    for name, prob in result["class_probabilities"].items():
        bar = "#" * int(prob * 40)
        print(f"  {name:>15}: {prob:.1%} {bar}")


if __name__ == "__main__":
    predict_from_cli()
