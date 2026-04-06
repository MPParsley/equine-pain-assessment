"""Tests for the equine pain model."""

import pytest
import torch

from equine_pain.config import NUM_CLASSES, TrainConfig


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


@requires_network
def test_create_model():
    """Test model creation with default config."""
    from equine_pain.model import create_model

    config = TrainConfig()
    model = create_model(config)
    assert model is not None
    assert model.config.num_labels == NUM_CLASSES


@requires_network
def test_model_forward_pass():
    """Test model forward pass produces correct output shape."""
    from equine_pain.model import EquinePainModelWrapper

    config = TrainConfig()
    wrapper = EquinePainModelWrapper(config)
    wrapper.eval()

    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, config.image_size, config.image_size)
    labels = torch.tensor([0, 1])

    with torch.no_grad():
        outputs = wrapper(pixel_values=dummy_input, labels=labels)

    assert outputs.logits.shape == (batch_size, NUM_CLASSES)
    assert outputs.loss is not None
    assert outputs.loss.item() > 0


@requires_network
def test_model_output_probabilities():
    """Test that softmax of logits sums to 1."""
    from equine_pain.model import EquinePainModelWrapper

    config = TrainConfig()
    wrapper = EquinePainModelWrapper(config)
    wrapper.eval()

    dummy_input = torch.randn(1, 3, config.image_size, config.image_size)

    with torch.no_grad():
        outputs = wrapper(pixel_values=dummy_input)

    probs = torch.softmax(outputs.logits, dim=-1)
    assert abs(probs.sum().item() - 1.0) < 1e-5
    assert probs.shape == (1, NUM_CLASSES)


@requires_network
def test_model_save_and_load(tmp_path):
    """Test saving and loading a model."""
    from equine_pain.model import EquinePainModelWrapper, load_model

    config = TrainConfig()
    wrapper = EquinePainModelWrapper(config)

    save_dir = str(tmp_path / "test_model")
    wrapper.save(save_dir)

    loaded_model, processor = load_model(model_path=save_dir)
    assert loaded_model is not None
    assert processor is not None

    # Verify loaded model produces same outputs
    dummy_input = torch.randn(1, 3, config.image_size, config.image_size)
    wrapper.eval()
    loaded_model.eval()

    with torch.no_grad():
        original_out = wrapper(pixel_values=dummy_input).logits
        loaded_out = loaded_model(pixel_values=dummy_input).logits

    assert torch.allclose(original_out, loaded_out, atol=1e-5)


def test_config_defaults():
    """Test that TrainConfig has sensible defaults (no network needed)."""
    config = TrainConfig()
    assert config.model_name == "google/vit-base-patch16-224"
    assert config.image_size == 224
    assert config.num_epochs == 10
    assert config.batch_size == 16
    assert config.learning_rate == 2e-5
    assert NUM_CLASSES == 3
