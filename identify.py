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
    pred_id = np.argmax(logits)
    id2label = model.config.id2label
    # Robustly handle both int and str keys
    if isinstance(id2label, dict):
        if pred_id in id2label:
            pred_label = id2label[pred_id]
        elif str(pred_id) in id2label:
            pred_label = id2label[str(pred_id)]
        else:
            raise KeyError(f"Predicted class {pred_id} not found in id2label mapping: {id2label}")
    else:
        pred_label = id2label[pred_id]
    return pred_label