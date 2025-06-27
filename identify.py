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

def batch_identify(image_dir, model, image_processor, device):
    """
    Identify all images in a directory (recursively), print their predicted labels,
    and save a Markdown report of per-species accuracy to batch_out.md.
    """
    results = []
    species_totals = {}
    species_correct = {}
    for root, _, files in os.walk(image_dir):
        # Assume root directory name is the species label
        species = os.path.basename(root)
        if not files:
            continue
        if species not in species_totals:
            species_totals[species] = 0
            species_correct[species] = 0
        for fname in files:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(root, fname)
                label = identify_image(img_path, model, image_processor, device)
                print(f"{img_path}: {label}")
                results.append((img_path, label, species))
                species_totals[species] += 1
                if label == species:
                    species_correct[species] += 1
    # Write Markdown table for per-species accuracy
    total = sum(species_totals.values())
    correct = sum(species_correct.values())
    md_lines = []
    md_lines.append(f'**Overall Accuracy:** {correct}/{total} ({100.0 * correct / total:.2f}%)\n')
    md_lines.append('| Species | Correct/Total | Accuracy |')
    md_lines.append('|---|---|---|')
    for species in sorted(species_totals):
        n = species_totals[species]
        c = species_correct[species]
        pct = 100.0 * c / n if n > 0 else 0.0
        md_lines.append(f"| {species} | {c}/{n} | {pct:.2f}% |")
    with open("batch_out.md", "w") as f:
        for line in md_lines:
            f.write(line + "\n")
    return results

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForImageClassification.from_pretrained(MODEL_PATH).to(device)

    if len(sys.argv) == 1:
        # No arguments: run batch mode on imgs/TEST
        test_dir = os.path.join('imgs', 'TEST')
        print(f"Batch identifying all images in {test_dir}...")
        batch_identify(test_dir, model, image_processor, device)
    elif len(sys.argv) == 2:
        image_path = sys.argv[1]
        if os.path.isdir(image_path):
            batch_identify(image_path, model, image_processor, device)
        else:
            label = identify_image(image_path, model, image_processor, device)
            print(f"Predicted label: {label}")
    else:
        print("Usage: python identify.py [<image_path>|<image_dir>]")
        sys.exit(1)
