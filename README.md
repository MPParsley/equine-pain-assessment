# Equine Pain Assessment

Automated horse pain assessment using the **Horse Grimace Pain Scale (HGS)** with a fine-tuned Vision Transformer (ViT). Upload a photo of a horse's face and get a pain level classification: **No Pain**, **Moderate Pain**, or **Severe Pain**.

## Horse Grimace Pain Scale

The HGS evaluates 6 Facial Action Units (FAUs), each scored 0-2:

| # | Facial Action Unit | Score |
|---|---|---|
| 1 | Stiffly backwards ears | 0-2 |
| 2 | Orbital tightening | 0-2 |
| 3 | Tension above the eye area | 0-2 |
| 4 | Prominent strained chewing muscles | 0-2 |
| 5 | Mouth strained and pronounced chin | 0-2 |
| 6 | Strained nostrils and flattening of the profile | 0-2 |

**Total score: 0-12** mapped to three pain levels:
- **No Pain**: 0-3
- **Moderate Pain**: 4-7
- **Severe Pain**: 8-12

## Installation

```bash
# Clone the repository
git clone https://github.com/MPParsley/equine-pain-assessment.git
cd equine-pain-assessment

# Install dependencies
pip install -e ".[dev]"
```

## Quick Start

### Web Interface (Gradio)

```bash
python app.py
# Opens at http://localhost:7860
```

The app includes a demo mode when no trained model is available.

### Training

1. **Prepare the dataset** — organize images into folders:
   ```
   data/
   ├── no_pain/
   │   ├── image1.jpg
   │   └── ...
   ├── moderate_pain/
   │   └── ...
   └── severe_pain/
       └── ...
   ```

   Or use a CSV format:
   ```
   data/
   ├── images/
   │   ├── image1.jpg
   │   └── ...
   └── labels.csv  (columns: filename, pain_level)
   ```

   The recommended dataset is from Mendeley Data:
   [Automatic Pain Assessment in Horses](https://data.mendeley.com/datasets/t8rtzcgwxm/3) (DOI: 10.17632/t8rtzcgwxm.3)

2. **Run training:**
   ```bash
   python -m equine_pain.train --data-dir ./data --epochs 10 --batch-size 16
   ```

   Options:
   ```
   --data-dir       Path to dataset directory (default: data)
   --output-dir     Output directory for checkpoints (default: models)
   --model-name     Pre-trained model (default: google/vit-base-patch16-224)
   --epochs         Number of epochs (default: 10)
   --batch-size     Batch size (default: 16)
   --learning-rate  Learning rate (default: 2e-5)
   --push-to-hub    HuggingFace Hub model ID to push to
   --seed           Random seed (default: 42)
   ```

3. **Push to HuggingFace Hub** (optional):
   ```bash
   python -m equine_pain.train --data-dir ./data --push-to-hub username/equine-pain-vit
   ```

### Inference

```bash
python -m equine_pain.inference photo.jpg --model-path models/best_model
```

Or from Python:
```python
from equine_pain.inference import PainPredictor

predictor = PainPredictor(model_path="models/best_model")
result = predictor.predict("horse_photo.jpg")
print(result["pain_level"])      # "No Pain" / "Moderate Pain" / "Severe Pain"
print(result["confidence"])       # 0.0 - 1.0
print(result["class_probabilities"])
```

## GitHub Actions

### CI (automatic on push/PR)
Runs linting (ruff) and tests (pytest) on every push and pull request.

### Train Model (manual)
Trigger via Actions tab → "Train Model" → Run workflow. Configure epochs, batch size, and optionally push to HuggingFace Hub.

### Deploy to HuggingFace Spaces (manual / on release)
Deploys the Gradio app to HuggingFace Spaces. Requires `HF_TOKEN` secret.

## Project Structure

```
├── src/equine_pain/
│   ├── config.py      # HGS constants and training hyperparameters
│   ├── dataset.py     # Dataset loading, augmentation, splits
│   ├── model.py       # ViT model creation, loading, saving
│   ├── train.py       # Training pipeline with early stopping
│   └── inference.py   # Single-image prediction
├── app.py             # Gradio web interface
├── tests/             # pytest test suite
└── .github/workflows/ # CI, training, deployment
```

## Hardware Requirements

### Training

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **GPU VRAM** | 4 GB (batch size 8) | 8+ GB (batch size 16) |
| **System RAM** | 8 GB | 16 GB |
| **Disk** | 2 GB (model + dataset) | 5 GB (with checkpoints) |
| **GPU** | NVIDIA GTX 1650 / T4 | NVIDIA RTX 3060 / A10G |

**Estimated training time** (10 epochs, ~1,500 images, batch size 16):

| Hardware | Time per epoch | Total |
|----------|---------------|-------|
| NVIDIA T4 (16 GB) | ~2 min | ~20 min |
| NVIDIA RTX 3060 (12 GB) | ~1.5 min | ~15 min |
| NVIDIA A100 (40 GB) | ~30 sec | ~5 min |
| CPU only (8-core) | ~15 min | ~2.5 hrs |

- The ViT-Base model has **86M parameters** (~330 MB in FP32). During training, optimizer states and gradients bring peak memory to roughly **3x model size** (~1 GB) plus batch activations.
- If you run out of GPU memory, reduce `--batch-size` (halving it roughly halves VRAM for activations) or use gradient accumulation.
- **CPU-only training** is supported but significantly slower. Set `num_workers=0` if you encounter memory issues on low-RAM machines.
- **Free GPU options**: Google Colab (T4), Kaggle Notebooks (P100/T4), or the GitHub Actions workflow (CPU, suitable for small datasets).

### Inference

| Resource | Minimum |
|----------|---------|
| **GPU VRAM** | Not required (CPU works fine) |
| **System RAM** | 4 GB |
| **Disk** | 500 MB (model weights + dependencies) |

Inference runs in <1 second per image on CPU; a GPU is not required but speeds up batch processing.

## Architecture

- **Backbone**: Vision Transformer (ViT-Base, patch size 16, 224x224) from Google, pre-trained on ImageNet
- **Head**: Linear classification layer (768 → 3 classes)
- **Training**: AdamW optimizer, cosine LR schedule, early stopping on validation F1
- **Augmentation**: Random flip, rotation (15°), color jitter, affine transforms

## References

- Dalla Costa et al. (2014). [Development of the Horse Grimace Scale (HGS) as a Pain Assessment Tool in Horses Undergoing Routine Castration](https://pmc.ncbi.nlm.nih.gov/articles/PMC4312484/). *PLOS ONE*.
- Lundblad et al. (2021). [Pain assessment in horses using automatic facial expression recognition through deep learning-based modeling](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0258672). *PLOS ONE*.
- Broome et al. (2019). [Dynamics Are Important for the Recognition of Equine Pain in Video](https://arxiv.org/abs/1901.02106). *CVPR*.
- Rashid et al. (2022). [Equine Pain Assessment — Mendeley Data](https://data.mendeley.com/datasets/t8rtzcgwxm/3).

## Disclaimer

This tool is for **research and educational purposes only**. Always consult a qualified veterinarian for clinical pain assessment in horses.

## License

MIT
