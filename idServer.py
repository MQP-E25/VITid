from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import shutil
from identify import identify_image
import torch
import torchvision
from transformers import AutoImageProcessor, AutoModelForImageClassification
from vitHelper import data_to_img, export_curves_to_images, image_size
import os
from io import StringIO
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(
    app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
)
MODEL_PATH = "./MODEL"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_processor = AutoImageProcessor.from_pretrained(MODEL_PATH, use_fast=True)
model = AutoModelForImageClassification.from_pretrained(MODEL_PATH).to(device)


@app.route('/analyzeCSV', methods=['POST'])
def analyze_csv():
    # Ensure /tmp directory exists
    tmp_csv = './tmp/csv'
    tmp_img = './tmp/img'
    os.makedirs(tmp_csv, exist_ok=True)
    os.makedirs(tmp_img, exist_ok=True)
    csv = request.files['csv']
    csv_path = os.path.join(tmp_csv, csv.filename)
    csv.save(csv_path)

    # convert data to img
    export_curves_to_images(tmp_csv, tmp_img, image_size)

    # Identify species using the vit model    
    results = {}

    for dir in sorted(os.listdir(tmp_img)):
        dir_path = os.path.join(tmp_img, dir)
        for img in sorted(os.listdir(dir_path)):
            img_path = os.path.join(dir_path, img)
            pred = identify_image(img_path, model, image_processor, device)
            # If pred is a dict, return all confidences; else wrap in a dict
            # Prevent 'Unnamed' results from being passed back
            if isinstance(pred, dict) and ('Unnamed' in pred or 'unnamed' in pred):
                continue
            if isinstance(pred, str) and pred.lower() == 'unnamed':
                continue
            results[img] = pred

    # Clean up the temporary file(s)
    shutil.rmtree('./tmp')

    # Remove 'Unnamed' keys from results before returning
    filtered_results = {k: v for k, v in results.items() if not (str(k).lower().startswith('unnamed'))}
    return jsonify(filtered_results)

@app.route('/analyzeIMG', methods=['POST'])
def analyze_img():
    # Ensure /tmp directory exists
    sourceimg = './tmp/sourceimg'
    os.makedirs(sourceimg, exist_ok=True)
    img = request.files['img']
    img_path = os.path.join(sourceimg, img.filename)
    img.save(img_path)

    results = {}

    for img in sorted(os.listdir(sourceimg)):
        img_path = os.path.join(sourceimg, img)
        pred = identify_image(img_path, model, image_processor, device)
        results[img] = pred

    # Clean up the temporary file(s)
    shutil.rmtree('./tmp')

    # Remove 'Unnamed' keys from results before returning
    filtered_results = {k: v for k, v in results.items() if not (str(k).lower().startswith('unnamed'))}
    return jsonify(filtered_results)


@app.route('/analyzeNotebook', methods=['POST'])
def analyze_notebook():
    tmp_csv = './tmp/nb'
    tmp_processed = './tmp/processed'
    tmp_img = './tmp/img'

    os.makedirs(tmp_csv, exist_ok=True)
    os.makedirs(tmp_processed, exist_ok=True)
    os.makedirs(tmp_img, exist_ok=True)

    csv = request.files['csv']
    csv_path = os.path.join(tmp_csv, csv.filename)
    csv.save(csv_path)
    
    saved_csvs = open_raw_file_and_save_columns(csv_path, tmp_processed)

    # Convert each sample CSV to image(s) using export_curves_to_images
    export_curves_to_images(tmp_processed, tmp_img, image_size)

    # Identify each image and collect all confidence levels for each sample
    results = {}
    for sample_csv in saved_csvs:
        sample_name = os.path.splitext(os.path.basename(sample_csv))[0]
        found = False
        for subdir in os.listdir(tmp_img):
            subdir_path = os.path.join(tmp_img, subdir)
            if not os.path.isdir(subdir_path):
                continue
            for img_file in os.listdir(subdir_path):
                if sample_name in img_file:
                    img_path = os.path.join(subdir_path, img_file)
                    pred = identify_image(img_path, model, image_processor, device)
                    # Prevent 'Unnamed' results from being passed back
                    if isinstance(pred, dict) and ('Unnamed' in pred or 'unnamed' in pred):
                        continue
                    if isinstance(pred, str) and pred.lower() == 'unnamed':
                        continue
                    results[sample_name] = pred
                    found = True
                    break
            if found:
                break
        if not found:
            results[sample_name] = None

    # Clean up the temporary file(s)
    shutil.rmtree('./tmp')

    # Remove 'Unnamed' keys from results before returning
    filtered_results = {k: v for k, v in results.items() if not (str(k).lower().startswith('unnamed'))}
    return jsonify(filtered_results)


# Process the uploaded file and save each sample column as a separate CSV
def open_raw_file_and_save_columns(f, out_dir):
    with open(f, 'r', encoding='utf-8') as file_obj:
        lines = []
        found_marker = False
        for line in file_obj:
            if found_marker:
                if "End Worksheet" in line:
                    break
                lines.append(line)
            elif "Start Worksheet - Analysis - Melt DNAID-Elasmo Data" in line:
                found_marker = True
        if not lines:
            return []
        lines = lines[1:-1]
        data_str = ''.join(lines)
        df = pd.read_csv(StringIO(data_str), sep=None, engine='python')
        saved_files = []
        for col in df.columns:
            sample_name = str(col)
            if  "Temperature" in sample_name: 
                continue
            out_path = os.path.join(out_dir, f'{sample_name}.csv')
            df[[col]].to_csv(out_path, index=False, header=True)
            saved_files.append(out_path)
        return saved_files
        

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)