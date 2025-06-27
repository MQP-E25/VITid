from flask import Flask, request, jsonify
import os
import shutil
from identify import identify_image
import torch
import torchvision
from transformers import AutoImageProcessor, AutoModelForImageClassification
from vitHelper import data_to_img, export_curves_to_images, image_size

app = Flask(__name__)
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
    results = ''

    for dir in sorted(os.listdir(tmp_img)):
        for img in os.listdir(os.path.join(tmp_img, dir)):
            # print(img)
            results = identify_image(os.path.join(tmp_img, dir, img), model, image_processor, device)
            print(results)

    # Clean up the temporary file(s)
    shutil.rmtree('./tmp')

    return jsonify(results)

@app.route('/analyzeIMG', methods=['POST'])
def analyze_img():
    # Ensure /tmp directory exists
    sourceimg = './tmp/sourceimg'
    os.makedirs(sourceimg, exist_ok=True)
    img = request.files['img']
    img_path = os.path.join(sourceimg, img.filename)
    img.save(img_path)

    results = ''

    for img in sorted(os.listdir(sourceimg)):
        print(img)
        results = identify_image(os.path.join(sourceimg, img), model, image_processor, device)
        print(results)

    # Clean up the temporary file(s)
    shutil.rmtree('./tmp')

    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)