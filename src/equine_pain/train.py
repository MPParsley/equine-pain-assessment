"""Training script for equine pain classification model."""

import argparse
import logging
from pathlib import Path

import torch
from sklearn.metrics import classification_report, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from equine_pain.config import CLASS_NAMES, TrainConfig
from equine_pain.dataset import create_splits
from equine_pain.model import EquinePainModelWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_epoch(model, dataloader, optimizer, device):
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(pixel_values=pixel_values, labels=labels)
        total_loss += outputs.loss.item() * labels.size(0)

        preds = outputs.logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(all_labels) if all_labels else 0.0
    correct = sum(p == t for p, t in zip(all_preds, all_labels))
    accuracy = correct / len(all_labels) if all_labels else 0.0
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, accuracy, f1, all_preds, all_labels


def train(config: TrainConfig):
    """Full training pipeline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading dataset from %s", config.data_dir)
    train_ds, val_ds, test_ds = create_splits(config.data_dir, config)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
    )

    # Create model
    logger.info("Creating model: %s", config.model_name)
    wrapper = EquinePainModelWrapper(config)
    wrapper = wrapper.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(
        wrapper.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    # Training loop
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(config.num_epochs):
        train_loss, train_acc = train_epoch(wrapper, train_loader, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(wrapper, val_loader, device)
        scheduler.step()

        logger.info(
            "Epoch %d/%d - train_loss: %.4f, train_acc: %.4f, "
            "val_loss: %.4f, val_acc: %.4f, val_f1: %.4f",
            epoch + 1, config.num_epochs,
            train_loss, train_acc, val_loss, val_acc, val_f1,
        )

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_dir = str(output_dir / "best_model")
            wrapper.save(best_dir)
            logger.info("New best model saved (val_f1=%.4f)", val_f1)
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    # Test evaluation
    logger.info("Evaluating on test set...")
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        wrapper, test_loader, device
    )
    logger.info("Test - loss: %.4f, accuracy: %.4f, f1: %.4f", test_loss, test_acc, test_f1)
    logger.info(
        "\n%s",
        classification_report(test_labels, test_preds, target_names=CLASS_NAMES, zero_division=0),
    )

    # Push to hub if requested
    if config.hub_model_id:
        logger.info("Pushing model to HuggingFace Hub: %s", config.hub_model_id)
        wrapper.push_to_hub(config.hub_model_id)

    return best_val_f1


def main():
    parser = argparse.ArgumentParser(description="Train equine pain classification model")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to dataset directory")
    parser.add_argument(
        "--output-dir", type=str, default="models", help="Output directory for checkpoints"
    )
    parser.add_argument("--model-name", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--push-to-hub", type=str, default="", help="HuggingFace Hub model ID")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hub_model_id=args.push_to_hub,
        seed=args.seed,
    )

    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    train(config)


if __name__ == "__main__":
    main()
