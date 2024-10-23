from flask import Flask, request, jsonify
import os
from flask_cors import CORS
import requests
import numpy as np
import cv2
from PIL import Image

app = Flask(__name__)
CORS(app)

@app.route('/api/virtual-tryon', methods=['POST'])
def virtual_tryon():
    data = request.get_json()
    product_id = data.get('product_id')
    image_url = data.get('image_url')

    if not product_id or not image_url:
        return jsonify({"error": "product_id or image_url missing in request"}), 400

    # Download the image
    image_path = download_image(image_url)

    # Process the image to remove background and face
    processed_image_path = process_image(image_path)

    return jsonify({
        "message": "Virtual try-on successful",
        "product_id": product_id,
        "processed_image_url": processed_image_path
    }), 200

# Function to download the image from a URL and save it locally
def download_image(image_url):
    response = requests.get(image_url)
    
    if response.status_code == 200:
        upload_folder = "uploads/"
        os.makedirs(upload_folder, exist_ok=True)
        image_name = os.path.basename(image_url)
        file_path = os.path.join(upload_folder, image_name)
        
        with open(file_path, 'wb') as file:
            file.write(response.content)
        
        return file_path
    else:
        raise Exception(f"Failed to download image from {image_url}")

# Function to process the image: remove background and face
def process_image(image_path):
    # Load image using OpenCV
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use a simple threshold to create a mask (you can adjust this)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Perform morphological operations to improve mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Invert the mask to get the foreground
    mask_inv = cv2.bitwise_not(mask)

    # Create a blank image with the same dimensions as the original
    background = np.ones_like(image, dtype=np.uint8) * 255  # White background

    # Combine the images: where the mask is white (foreground), we keep the original image
    foreground = cv2.bitwise_and(image, image, mask=mask_inv)
    result_image = cv2.add(foreground, background)

    # Save the result
    output_folder = "processed/"
    os.makedirs(output_folder, exist_ok=True)
    processed_image_name = "processed_" + os.path.basename(image_path)
    processed_image_path = os.path.join(output_folder, processed_image_name)

    # Save the result image
    cv2.imwrite(processed_image_path, result_image)

    return processed_image_path

if __name__ == '__main__':
    app.run(debug=True)
