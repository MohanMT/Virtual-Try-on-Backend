import cv2
import numpy as np
import requests
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import mediapipe as mp

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

clothing_image = None  # Placeholder for the clothing image
captured_frame = None  # Store the captured frame

# Function to download the clothing image from the given URL
def download_clothing_image(image_url):
    global clothing_image
    response = requests.get(image_url)
    image_array = np.frombuffer(response.content, np.uint8)
    clothing_image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)

    if clothing_image is None:
        raise FileNotFoundError(f"Could not download clothing image from {image_url}")

# Function to load clothing image and remove background using color threshold
def remove_background_from_clothing_image():
    global clothing_image

    if clothing_image is None:
        print("No clothing image loaded.")
        return

    if clothing_image.shape[2] == 3:
        print("Removing background from the clothing image...")

        # Convert image to HSV to remove background by color
        hsv = cv2.cvtColor(clothing_image, cv2.COLOR_BGR2HSV)

        # Define the range for background color (e.g., white)
        lower_bound = np.array([0, 0, 200])
        upper_bound = np.array([180, 50, 255])

        # Create mask for the background
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Invert the mask to keep the clothing part
        mask_inv = cv2.bitwise_not(mask)

        # Convert the clothing to have an alpha channel (transparency)
        b_channel, g_channel, r_channel = cv2.split(clothing_image)
        alpha_channel = mask_inv

        # Merge the BGR channels with the alpha channel
        clothing_image = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

# Function to overlay clothing on the detected upper body landmarks
def overlay_clothes(frame, landmarks):
    global clothing_image

    if clothing_image is None:
        print("Clothing image is not loaded correctly.")
        return

    # Ensure the necessary landmarks are present (shoulders and hips)
    if len(landmarks) < 25:
        print("Insufficient pose landmarks detected.")
        return

    shoulder_left = landmarks[11]  # Left shoulder
    shoulder_right = landmarks[12]  # Right shoulder
    hip_left = landmarks[23]  # Left hip
    hip_right = landmarks[24]  # Right hip

    # Calculate shoulder width and torso height for dynamic scaling
    shoulder_width = int(abs((shoulder_right.x - shoulder_left.x) * frame.shape[1]))
    torso_height = int(abs((hip_left.y - shoulder_left.y) * frame.shape[0]))

    # Scale up the clothing by a factor
    scale_factor = 1.5
    shoulder_width = int(shoulder_width * scale_factor)
    torso_height = int(torso_height * scale_factor)

    # Resize clothing to fit between shoulders and hips
    resized_clothes = cv2.resize(clothing_image, (shoulder_width, torso_height))

    # Calculate center point for clothing placement
    center_x = int((shoulder_left.x + shoulder_right.x) / 2 * frame.shape[1])
    center_y = int(shoulder_left.y * frame.shape[0])

    # Position the clothing above the shoulders
    y_offset = center_y - int(torso_height / 2)
    x_offset = center_x - int(shoulder_width / 2)

    # Ensure the offsets are within the frame bounds
    y_offset = max(0, y_offset)
    x_offset = max(0, x_offset)

    # Crop the clothing image if needed to fit within the frame
    available_height = frame.shape[0] - y_offset
    available_width = frame.shape[1] - x_offset

    clothing_height = min(resized_clothes.shape[0], available_height)
    clothing_width = min(resized_clothes.shape[1], available_width)

    cropped_clothes = resized_clothes[:clothing_height, :clothing_width]

    # Create a mask from the alpha channel
    alpha_channel = cropped_clothes[:, :, 3] / 255.0
    for c in range(3):  # Assuming clothing image has 3 channels (RGB)
        frame[y_offset:y_offset + clothing_height, x_offset:x_offset + clothing_width, c] = (
            frame[y_offset:y_offset + clothing_height, x_offset:x_offset + clothing_width, c] * (1 - alpha_channel) +
            cropped_clothes[:, :, c] * alpha_channel
        )

# Function to generate frames from the webcam feed
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
            overlay_clothes(frame, results.pose_landmarks.landmark)

        captured_frame = frame  # Store the current frame for capturing

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Return the frame to be displayed in the web feed
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to serve the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to receive product ID and image URL
@app.route('/api/virtual-tryon', methods=['POST'])
def virtual_tryon():
    data = request.json
    product_id = data.get('product_id')
    image_url = data.get('image_url')

    if not product_id or not image_url:
        return jsonify({"error": "Product ID and image URL are required"}), 400

    try:
        # Download and process the clothing image
        download_clothing_image(image_url)
        remove_background_from_clothing_image()
        return jsonify({"message": "Clothing image processed successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to capture the current frame and save as an image
@app.route('/capture', methods=['POST'])
def capture_image():
    global captured_frame
    if captured_frame is None:
        return jsonify({"error": "No frame captured"}), 500

    # Save the current frame as an image
    cv2.imwrite('captured_image.png', captured_frame)
    return jsonify({"message": "Image captured successfully"}), 200

# Route to send the captured image to frontend
@app.route('/get-captured-image', methods=['GET'])
def get_captured_image():
    try:
        # Load the captured image
        with open('captured_image.png', 'rb') as img_file:
            return Response(img_file.read(), mimetype='image/png')
    except FileNotFoundError:
        return jsonify({"error": "Captured image not found"}), 404


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
