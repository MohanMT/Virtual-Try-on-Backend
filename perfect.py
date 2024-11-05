import cv2
import numpy as np
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import mediapipe as mp
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

clothing_image = None  # Placeholder for the clothing image
captured_frame = None  # Store the captured frame

# Function to load clothing image from a local path
def load_clothing_image(image_path):
    global clothing_image
    if os.path.exists(image_path):
        clothing_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if clothing_image is None:
            raise FileNotFoundError(f"Could not load clothing image from {image_path}")
    else:
        raise FileNotFoundError(f"Clothing image path does not exist: {image_path}")

# Function to load clothing image and remove background using color threshold
def remove_background_from_clothing_image():
    global clothing_image

    if clothing_image is None:
        print("No clothing image loaded.")
        return

    if clothing_image.shape[2] == 3 or clothing_image.shape[2] == 4:  # Check for 3 or 4 channels (with alpha)
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

# Function to overlay clothes for male landmarks
def overlay_male_clothes(frame, landmarks):
    shoulder_left = landmarks[11]  # Left shoulder
    shoulder_right = landmarks[12]  # Right shoulder
    hip_left = landmarks[23]  # Left hip
    hip_right = landmarks[24]  # Right hip

    shoulder_width = int(abs((shoulder_right.x - shoulder_left.x) * frame.shape[1]))
    torso_height = int(abs((hip_left.y - shoulder_left.y) * frame.shape[0]))

    scale_factor = 1.5
    shoulder_width = int(shoulder_width * scale_factor)
    torso_height = int(torso_height * scale_factor)

    resized_clothes = cv2.resize(clothing_image, (shoulder_width, torso_height))

    center_x = int((shoulder_left.x + shoulder_right.x) / 2 * frame.shape[1])
    center_y = int(shoulder_left.y * frame.shape[0])

    y_offset = center_y - int(torso_height / 2)
    x_offset = center_x - int(shoulder_width / 2)

    y_offset = max(0, y_offset)
    x_offset = max(0, x_offset)

    available_height = frame.shape[0] - y_offset
    available_width = frame.shape[1] - x_offset

    clothing_height = min(resized_clothes.shape[0], available_height)
    clothing_width = min(resized_clothes.shape[1], available_width)

    cropped_clothes = resized_clothes[:clothing_height, :clothing_width]

    alpha_channel = cropped_clothes[:, :, 3] / 255.0
    for c in range(3):  # Assuming clothing image has 3 channels (RGB)
        frame[y_offset:y_offset + clothing_height, x_offset:x_offset + clothing_width, c] = (
            frame[y_offset:y_offset + clothing_height, x_offset:x_offset + clothing_width, c] * (1 - alpha_channel) +
            cropped_clothes[:, :, c] * alpha_channel
        )

# Function to overlay upper body clothes for female landmarks (like t-shirts)
def overlay_female_upper_body_clothes(frame, landmarks):
    shoulder_left = landmarks[11]  # Left shoulder
    shoulder_right = landmarks[12]  # Right shoulder
    hip_left = landmarks[23]  # Left hip

    shoulder_width = int(abs((shoulder_right.x - shoulder_left.x) * frame.shape[1]))
    torso_height = int(abs((hip_left.y - shoulder_left.y) * frame.shape[0]))

    scale_factor = 1.5
    shoulder_width = int(shoulder_width * scale_factor)
    torso_height = int(torso_height * scale_factor)

    resized_clothes = cv2.resize(clothing_image, (shoulder_width, torso_height))

    center_x = int((shoulder_left.x + shoulder_right.x) / 2 * frame.shape[1])
    center_y = int(shoulder_left.y * frame.shape[0])

    y_offset = center_y - int(torso_height / 2)
    x_offset = center_x - int(shoulder_width / 2)

    y_offset = max(0, y_offset)
    x_offset = max(0, x_offset)

    available_height = frame.shape[0] - y_offset
    available_width = frame.shape[1] - x_offset

    clothing_height = min(resized_clothes.shape[0], available_height)
    clothing_width = min(resized_clothes.shape[1], available_width)

    cropped_clothes = resized_clothes[:clothing_height, :clothing_width]

    alpha_channel = cropped_clothes[:, :, 3] / 255.0
    for c in range(3):  # Assuming clothing image has 3 channels (RGB)
        frame[y_offset:y_offset + clothing_height, x_offset:x_offset + clothing_width, c] = (
            frame[y_offset:y_offset + clothing_height, x_offset:x_offset + clothing_width, c] * (1 - alpha_channel) +
            cropped_clothes[:, :, c] * alpha_channel
        )

# Function to overlay full body clothes for female landmarks (like frocks)
def overlay_female_full_body_clothes(frame, landmarks):
    shoulder_left = landmarks[11]  # Left shoulder
    shoulder_right = landmarks[12]  # Right shoulder
    hip_left = landmarks[24]  # Left hip
    hip_right = landmarks[23]  # Right hip

    torso_height = int(abs((hip_left.y - shoulder_left.y) * frame.shape[0]))

    scale_factor = 1.5
    torso_height = int(torso_height * scale_factor)

    # Assuming the clothing image is designed for full-body
    resized_clothes = cv2.resize(clothing_image, (frame.shape[1], torso_height))

    center_x = int((shoulder_left.x + shoulder_right.x) / 2 * frame.shape[1])
    center_y = int(hip_left.y * frame.shape[0])

    y_offset = center_y - int(torso_height)
    x_offset = center_x - int(frame.shape[1] / 2)

    y_offset = max(0, y_offset)
    x_offset = max(0, x_offset)

    available_height = frame.shape[0] - y_offset
    available_width = frame.shape[1] - x_offset

    clothing_height = min(resized_clothes.shape[0], available_height)
    clothing_width = min(resized_clothes.shape[1], available_width)

    cropped_clothes = resized_clothes[:clothing_height, :clothing_width]

    alpha_channel = cropped_clothes[:, :, 3] / 255.0
    for c in range(3):  # Assuming clothing image has 3 channels (RGB)
        frame[y_offset:y_offset + clothing_height, x_offset:x_offset + clothing_width, c] = (
            frame[y_offset:y_offset + clothing_height, x_offset:x_offset + clothing_width, c] * (1 - alpha_channel) +
            cropped_clothes[:, :, c] * alpha_channel
        )

# Function to overlay clothes based on product category and subcategory
def overlay_clothes(frame, landmarks, category, subcategory):
    global clothing_image

    if clothing_image is None:
        print("Clothing image is not loaded correctly.")
        return

    # Use male landmarks for "Infants" and "kids"
    if category in ["Infants", "Kids"]:
        overlay_male_clothes(frame, landmarks)
    elif category == "Women":
        if subcategory == "tshirt":
            overlay_female_upper_body_clothes(frame, landmarks)
        elif subcategory == "frock":
            overlay_female_full_body_clothes(frame, landmarks)
        else:
            print("Unknown subcategory:", subcategory)
    else:
        print("Unknown category:", category)

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
            category = "female"  # Replace this with your actual logic to determine the category
            subcategory = "tshirt"  # Replace this with your actual logic to determine the subcategory
            overlay_clothes(frame, results.pose_landmarks.landmark, category, subcategory)

        captured_frame = frame  # Store the current frame for capturing

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:  # Check if encoding was successful
            print("Failed to encode frame.")
            continue

        frame = buffer.tobytes()

        # Return the frame to be displayed in the web feed
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to serve the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to receive clothing image path
@app.route('/api/virtual-tryon', methods=['POST'])
def virtual_tryon():
    global clothing_image

    # Get the JSON data from the request
    data = request.get_json()

    if 'image_path' not in data:
        return jsonify({'error': 'No image path provided'}), 400

    image_path = data['image_path']

    try:
        load_clothing_image(image_path)
        remove_background_from_clothing_image()  # Clean the background after loading
        return jsonify({'success': 'Clothing image processed successfully.'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

# Start the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
