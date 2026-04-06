"""Tests for the inference pipeline."""

import pytest
from PIL import Image

from equine_pain.config import CLASS_NAMES, TrainConfig


def _can_download_model():
    """Check if we can download the pre-trained model from HuggingFace Hub."""
    try:
        from equine_pain.model import create_model

        config = TrainConfig()
        create_model(config)
        return True
    except Exception:
        return False


requires_network = pytest.mark.skipif(
    not _can_download_model(),
    reason="Cannot download pre-trained model (no network access or HuggingFace Hub unavailable)",
)


def _create_test_model(tmp_path):
    """Create and save a test model, return the path."""
    from equine_pain.model import EquinePainModelWrapper

    config = TrainConfig()
    wrapper = EquinePainModelWrapper(config)
    model_dir = str(tmp_path / "test_model")
    wrapper.save(model_dir)
    return model_dir


@requires_network
def test_predictor_from_local_model(tmp_path):
    """Test prediction from a locally saved model."""
    from equine_pain.inference import PainPredictor

    model_dir = _create_test_model(tmp_path)
    predictor = PainPredictor(model_path=model_dir)

    test_image = Image.new("RGB", (256, 256), color=(128, 128, 128))
    result = predictor.predict(test_image)

    assert "pain_level" in result
    assert result["pain_level"] in CLASS_NAMES
    assert 0.0 <= result["confidence"] <= 1.0
    assert "class_probabilities" in result
    assert len(result["class_probabilities"]) == 3
    assert abs(sum(result["class_probabilities"].values()) - 1.0) < 1e-4
    assert "hgs_score_range" in result
    assert "fau_guide" in result
    assert len(result["fau_guide"]) == 6


@requires_network
def test_predictor_from_file_path(tmp_path):
    """Test prediction from an image file path."""
    from equine_pain.inference import PainPredictor

    model_dir = _create_test_model(tmp_path)
    predictor = PainPredictor(model_path=model_dir)

    img_path = tmp_path / "test_horse.jpg"
    Image.new("RGB", (224, 224), color=(100, 150, 200)).save(img_path)

    result = predictor.predict(str(img_path))
    assert result["pain_level"] in CLASS_NAMES


@requires_network
def test_predictor_deterministic(tmp_path):
    """Test that predictions are deterministic for the same input."""
    from equine_pain.inference import PainPredictor

    model_dir = _create_test_model(tmp_path)
    predictor = PainPredictor(model_path=model_dir)

    test_image = Image.new("RGB", (256, 256), color=(128, 128, 128))
    result1 = predictor.predict(test_image)
    result2 = predictor.predict(test_image)

    assert result1["pain_level"] == result2["pain_level"]
    assert abs(result1["confidence"] - result2["confidence"]) < 1e-6


def test_pain_levels_config():
    """Test pain level configuration (no network needed)."""
    from equine_pain.config import FAU_NAMES, PAIN_LEVELS

    assert len(CLASS_NAMES) == 3
    assert "No Pain" in CLASS_NAMES
    assert "Moderate Pain" in CLASS_NAMES
    assert "Severe Pain" in CLASS_NAMES
    assert len(FAU_NAMES) == 6
    assert PAIN_LEVELS["No Pain"] == (0, 3)
    assert PAIN_LEVELS["Severe Pain"] == (8, 12)
