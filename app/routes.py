from flask import Blueprint, Response, request, jsonify
from .utils import (
    load_clothing_image,
    remove_background_from_clothing_image,
    generate_frames,
)

main = Blueprint('main', __name__)

@main.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@main.route('/api/virtual-tryon', methods=['POST'])
def virtual_tryon():
    data = request.get_json()

    # Validate incoming data
    required_fields = ['product_id', 'image_url', 'product_category', 'product_subcategory']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'No {field} provided'}), 400

    try:
        load_clothing_image(data['image_url'])  # Load the image from the URL
        remove_background_from_clothing_image()  # Process the image
        return jsonify({'success': 'Clothing image processed successfully.'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/capture', methods=['POST'])
def capture_image():
    global captured_frame
    if captured_frame is None:
        return jsonify({"error": "No frame captured"}), 500

    cv2.imwrite('captured_image.png', captured_frame)
    return jsonify({"message": "Image captured successfully"}), 200

@main.route('/get-captured-image', methods=['GET'])
def get_captured_image():
    try:
        with open('captured_image.png', 'rb') as img_file:
            return Response(img_file.read(), mimetype='image/png')
    except FileNotFoundError:
        return jsonify({"error": "Captured image not found"}), 404
