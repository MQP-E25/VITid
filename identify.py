import os
import sys
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# Set all relevant seeds for reproducibility
import random
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Path to exported model
MODEL_PATH = "vit_model_export"

def identify_image(image_path, model, image_processor, device):
    """
    Identify the class of a single image using a PyTorch ViT model.
    Args:
        image_path: Path to the image file
        model: Loaded ViT model
        image_processor: Loaded image processor
        device: torch.device
    Returns:
        Predicted label (str)
    """
    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits.cpu().numpy()[0]
    pred_id = int(np.argmax(logits))
    id2label = model.config.id2label
    # Build label mapping
    label_map = {}
    if isinstance(id2label, dict):
        for k, v in id2label.items():
            label_map[int(k) if str(k).isdigit() else k] = v
    else:
        for i, v in enumerate(id2label):
            label_map[i] = v
    # Get predicted label
    pred_label = label_map[pred_id]
    # Get softmax confidences
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    # Get top 5 most confident predictions
    top_indices = np.argsort(probs)[::-1][:5]
    top_confidences = {label_map[i]: float(probs[i]) for i in top_indices}
    result = {
        "prediction": pred_label,
        "confidence": float(probs[pred_id]),
        "top_5": top_confidences
    }
    return result