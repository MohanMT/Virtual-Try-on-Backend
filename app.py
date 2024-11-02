from flask import Flask, Response, request, jsonify, render_template
import cv2
import numpy as np
from utils.image_processing import download_clothing_image, remove_background_from_clothing_image
from utils.pose_overlay import overlay_clothes
import mediapipe as mp
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

clothing_image = None  # Placeholder for the clothing image
captured_frame = None  # Store the captured frame

def generate_frames():
    global captured_frame
    cap = cv2.VideoCapture(0)  # Use the default webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Check if pose landmarks are detected
        if results.pose_landmarks:
            overlay_clothes(frame, results.pose_landmarks.landmark, clothing_image)

        captured_frame = frame  # Store the current frame for capturing

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Return the frame to be displayed in the web feed
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/virtual-tryon', methods=['POST'])
def virtual_tryon():
    data = request.json
    product_id = data.get('product_id')
    image_url = data.get('image_url')

    if not product_id or not image_url:
        return jsonify({"error": "Product ID and image URL are required"}), 400

    try:
        # Download and process the clothing image
        global clothing_image
        clothing_image = download_clothing_image(image_url)
        clothing_image = remove_background_from_clothing_image(clothing_image)
        return jsonify({"message": "Clothing image processed successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/capture', methods=['POST'])
def capture_image():
    global captured_frame
    if captured_frame is None:
        return jsonify({"error": "No frame captured"}), 500

    _, buffer = cv2.imencode('.jpg', captured_frame)
    response_data = buffer.tobytes()

    return Response(response_data, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)