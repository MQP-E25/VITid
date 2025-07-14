import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

# Hyperparameters
batch_size = 32
num_epochs = 50
learning_rate = 0.0001821846699164341	
weight_decay_rate = 0.0013365987879961877
image_size = 768
SEED = 42
# Set all relevant seeds for reproducibility
import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model checkpoint
checkpoint = "google/vit-base-patch16-224-in21k"


def data_to_img(curve, size=None):
    """
    Converts a 1D curve to a resized RGB image.
    Args:
        curve: 1D np.ndarray
        size: (width, height) tuple
    Returns:
        PIL.Image (RGB)
    """
    if size is None:
        size = (image_size, image_size)
    fig, ax = plt.subplots()
    ax.plot(curve, color='black')
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    return image.resize(size)

def save_images(x, y, out_dir):
    """
    Saves all curves as images in a directory, grouped by label.
    """
    os.makedirs(out_dir, exist_ok=True)
    for i, (curve, label) in enumerate(zip(x, y)):
        label_dir = os.path.join(out_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        img = data_to_img(curve)
        img.save(os.path.join(label_dir, f"sample_{i}.png"))

def extract_species_name(csv_file):
    """
    Extracts the species name from a CSV filename, handling augmented files (e.g., *_augN.csv).
    """
    base = os.path.splitext(csv_file)[0]
    # Remove _augN if present
    if '_aug' in base:
        base = base[:base.rfind('_aug')]
    try:
        species_name = base.split('_', 1)[1]
    except IndexError:
        species_name = base
    return species_name

def export_curves_to_images(data_dir, out_dir, image_size):
    """
    Converts all CSVs in data_dir to images in out_dir, grouped by species.
    Augmented CSVs are exported as 'AUG#_<speciesname>.png' in the same species folder.
    """
    os.makedirs(out_dir, exist_ok=True)
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
    for csv_file in csv_files:
        csv_path = os.path.join(data_dir, csv_file)
        species_name = extract_species_name(csv_file)
        # All images (original and augmented) go in the same species folder
        out_label_dir = os.path.join(out_dir, species_name)
        os.makedirs(out_label_dir, exist_ok=True)
        df = pd.read_csv(csv_path)
        # Get curve data from CSV
        if 'value' in df.columns:
            curve = df['value'].values.astype(np.float32)
        else:
            curve = df.values.squeeze().astype(np.float32)
        img = data_to_img(curve, size=(image_size, image_size))
        # Determine image filename
        base = os.path.splitext(csv_file)[0]
        if '_aug' in base:
            # Augmented: extract number after '_aug' and use 'AUG#_<speciesname>.png'
            aug_idx = base[base.rfind('_aug')+4:]
            img_name = f"AUG{aug_idx}_{species_name}.png"
        else:
            # Original: use base name
            img_name = f"{base}.png"
        img.save(os.path.join(out_label_dir, img_name))
