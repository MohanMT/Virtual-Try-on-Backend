import cv2
import numpy as np
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import mediapipe as mp
import os

app = Flask(__name__)
CORS(app)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

clothing_image = None  # Global variable to store the uploaded clothing image

# Function to load clothing image and remove background using color threshold
def load_clothing_image_with_bg_removal(path):
    clothing_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    if clothing_image is None:
        raise FileNotFoundError(f"Could not load clothing image from {path}")
    
    # Remove background using HSV color space
    if clothing_image.shape[2] == 3:  # If no alpha channel exists, create one
        hsv = cv2.cvtColor(clothing_image, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([0, 0, 200])
        upper_bound = np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        mask_inv = cv2.bitwise_not(mask)
        
        b_channel, g_channel, r_channel = cv2.split(clothing_image)
        alpha_channel = mask_inv
        clothing_image = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    
    return clothing_image

# Function to overlay clothing on the detected body landmarks
def overlay_clothes(frame, landmarks, x_offset_adjust=0, y_offset_adjust=-100, scale_factor=1.7):
    global clothing_image

    if clothing_image is None:
        print("Clothing image is not loaded correctly.")
        return

    if len(landmarks) > 24:
        shoulder_left = landmarks[11]
        shoulder_right = landmarks[12]
        hip_left = landmarks[23]
        hip_right = landmarks[24]
    else:
        print("Insufficient pose landmarks detected.")
        return

    shoulder_width = int(abs((shoulder_right.x - shoulder_left.x) * frame.shape[1]))
    torso_height = int(abs((hip_left.y - shoulder_left.y) * frame.shape[0]))

    shoulder_width = int(shoulder_width * scale_factor)
    torso_height = int(torso_height * scale_factor)

    if shoulder_width < 50 or torso_height < 50:
        print("Detected body size is too small for proper overlay.")
        return

    resized_clothes = cv2.resize(clothing_image, (shoulder_width, torso_height))

    center_x = int((shoulder_left.x + shoulder_right.x) / 2 * frame.shape[1])
    center_y = int(shoulder_left.y * frame.shape[0])

    y_offset = center_y + y_offset_adjust
    x_offset = center_x - resized_clothes.shape[1] // 2 + x_offset_adjust

    y_offset = max(0, y_offset)
    x_offset = max(0, x_offset)

    available_height = frame.shape[0] - y_offset
    available_width = frame.shape[1] - x_offset

    clothing_height = min(resized_clothes.shape[0], available_height)
    clothing_width = min(resized_clothes.shape[1], available_width)

    cropped_clothes = resized_clothes[:clothing_height, :clothing_width]

    alpha_channel = cropped_clothes[:, :, 3] / 255.0
    for c in range(0, 3):
        frame[y_offset:y_offset + clothing_height, x_offset:x_offset + clothing_width, c] = (
            frame[y_offset:y_offset + clothing_height, x_offset:x_offset + clothing_width, c] * (1 - alpha_channel) +
            cropped_clothes[:, :, c] * alpha_channel
        )

# Function to generate frames from the webcam feed
def generate_frames():
    cap = cv2.VideoCapture(0)  # Use the default webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            overlay_clothes(frame, results.pose_landmarks.landmark)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to serve the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to upload the clothing image
@app.route('/upload-clothing-image', methods=['POST'])
def upload_clothing_image():
    global clothing_image
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        upload_folder = "uploads/"
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        # Process the uploaded image (remove background, etc.)
        clothing_image = load_clothing_image_with_bg_removal(file_path)

        return jsonify({"message": "Clothing image uploaded and processed successfully!"}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
