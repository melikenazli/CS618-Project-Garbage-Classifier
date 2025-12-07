"""
Gradio interface for live garbage classification

Demonstrates Baseline_CNN and EfficientNet Performance
"""

import torch
from torchvision import transforms
from PIL import Image
import gradio as gr
import numpy as np
from pathlib import Path
from src.config import IMG_SIZE
from src.models.init import get_model


baseline_ckpt = Path("results/best_baseline.pth")
effnet_ckpt = Path("results/best_efficientnet_b0_phase2.pth")

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Model1: EfficientNet-B0
def load_model_efficientnet():
    model = get_model("efficientnet_b0", len(CLASS_NAMES))

    assert effnet_ckpt.exists(), f"Checkpoint not found: {effnet_ckpt}"
    state_dict = torch.load(effnet_ckpt, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model

# Model2: Baseline CNN
def load_model_baseline():
    model = get_model("baseline", len(CLASS_NAMES))

    assert baseline_ckpt.exists(), f"Checkpoint not found: {baseline_ckpt}"
    state_dict = torch.load(baseline_ckpt, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model

# Comment out the unused model
# model = load_model_efficientnet()
model = load_model_baseline()

# Image Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def predict(image):
    if image is None:
        return {}

    # If coming from ImageEditor, it is an EditorValue dict
    if isinstance(image, dict):
        # Prefer the edited/composited image if available
        if image.get("composite", None) is not None:
            image = image["composite"]
        elif image.get("background", None) is not None:
            image = image["background"]
        else:
            return {}
        
    # In some gradio versions, image["image"] can be a numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # To ensure we now have a PIL.Image
    if not isinstance(image, Image.Image):
        return {}

    x = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    # Build a {class: prob} dict for Gradio’s Label component
    return {cls: float(prob) for cls, prob in zip(CLASS_NAMES, probs)}

# Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.ImageEditor(type="pil", sources=["upload", "webcam", "clipboard"], label="Upload / capture and CROP the garbage object", image_mode="RGB"),
    outputs=gr.Label(num_top_classes=3, label="Predicted class (top-3)"),
    title="Garbage Classification Demo",
    description="Upload a picture of garbage (cardboard, glass, metal, paper, plastic, trash) and let the model classify it.",
    examples=None,
)

if __name__ == "__main__":
    demo.launch()
