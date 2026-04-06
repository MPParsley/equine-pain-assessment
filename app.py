"""Gradio web interface for Horse Grimace Pain Scale assessment.

This app can be deployed directly to HuggingFace Spaces.
Run locally: python app.py
"""

import logging
from pathlib import Path

import gradio as gr

from equine_pain.config import FAU_DESCRIPTIONS, FAU_NAMES, PAIN_LEVELS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to load a trained model; fall back to demo mode
predictor = None
MODEL_DIR = Path("models/best_model")
HUB_MODEL_ID = None  # Set to "username/equine-pain-vit" when published


def load_predictor():
    global predictor
    if predictor is not None:
        return True

    try:
        from equine_pain.inference import PainPredictor

        if MODEL_DIR.exists():
            predictor = PainPredictor(model_path=str(MODEL_DIR))
            logger.info("Loaded model from %s", MODEL_DIR)
            return True
        elif HUB_MODEL_ID:
            predictor = PainPredictor(hub_model_id=HUB_MODEL_ID)
            logger.info("Loaded model from HuggingFace Hub: %s", HUB_MODEL_ID)
            return True
    except Exception:
        logger.exception("Failed to load model")

    return False


def predict_pain(image):
    """Main prediction function for the Gradio interface."""
    if image is None:
        return "Please upload an image.", {}, ""

    model_loaded = load_predictor()

    if model_loaded and predictor is not None:
        result = predictor.predict(image)
        pain_level = result["pain_level"]
        confidence = result["confidence"]
        probs = result["class_probabilities"]
        score_range = result["hgs_score_range"]
    else:
        # Demo mode: return informative placeholder
        import random

        random.seed(hash(str(image)) % 2**32)
        probs = {
            "No Pain": random.uniform(0.1, 0.5),
            "Moderate Pain": random.uniform(0.2, 0.5),
            "Severe Pain": random.uniform(0.05, 0.3),
        }
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}
        pain_level = max(probs, key=probs.get)
        confidence = probs[pain_level]
        score_range = PAIN_LEVELS[pain_level]

    # Format label output
    label_output = {name: prob for name, prob in probs.items()}

    # Build detailed report
    if pain_level == "No Pain":
        emoji = "&#9989;"
    elif pain_level == "Moderate Pain":
        emoji = "&#9888;&#65039;"
    else:
        emoji = "&#10060;"

    report = f"""## {emoji} Assessment: {pain_level}

**Confidence:** {confidence:.1%}
**Estimated HGS Score Range:** {score_range[0]} - {score_range[1]} (out of 12)

### Facial Action Units (FAUs) Evaluated

The Horse Grimace Scale assesses these 6 indicators, each scored 0-2:

| # | Facial Action Unit | Description |
|---|---|---|
"""
    for i, (name, desc) in enumerate(zip(FAU_NAMES, FAU_DESCRIPTIONS), 1):
        report += f"| {i} | {name} | {desc} |\n"

    if not model_loaded:
        report += """
---
**Note:** Running in demo mode (no trained model loaded). Results are placeholders.
To get real predictions, train the model first:
```bash
python -m equine_pain.train --data-dir ./data
```
Or download a pre-trained model from HuggingFace Hub.
"""

    return pain_level, label_output, report


# Build Gradio interface
with gr.Blocks(
    title="Horse Pain Assessment (HGS)",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown("""
    # Horse Grimace Pain Scale (HGS) Assessment

    Upload a photo of a horse's face to assess its pain level using the
    Horse Grimace Pain Scale. The model evaluates 6 Facial Action Units
    to classify pain as **No Pain**, **Moderate Pain**, or **Severe Pain**.

    Based on the validated [Horse Grimace Scale](https://pmc.ncbi.nlm.nih.gov/articles/PMC4312484/)
    using a fine-tuned Vision Transformer (ViT).
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Horse Face Image")
            predict_btn = gr.Button("Assess Pain Level", variant="primary", size="lg")

        with gr.Column(scale=1):
            pain_label = gr.Textbox(label="Pain Level", interactive=False)
            confidence_chart = gr.Label(label="Class Probabilities", num_top_classes=3)

    report_output = gr.Markdown(label="Detailed Report")

    predict_btn.click(
        fn=predict_pain,
        inputs=[image_input],
        outputs=[pain_label, confidence_chart, report_output],
    )

    gr.Markdown("""
    ---
    ### About the Horse Grimace Scale

    The HGS was developed as a pain assessment tool for horses, evaluating 6 facial
    action units (FAUs). Each FAU is scored from 0 (not present) to 2 (obviously present),
    giving a total score of 0-12.

    **References:**
    - Dalla Costa et al. (2014) - [Development of the HGS](https://pmc.ncbi.nlm.nih.gov/articles/PMC4312484/)
    - Lundblad et al. (2021) - [Automated pain assessment using deep learning](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0258672)

    **Disclaimer:** This tool is for research and educational purposes. Always consult
    a veterinarian for clinical pain assessment.
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
