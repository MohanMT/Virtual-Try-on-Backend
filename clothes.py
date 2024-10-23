import cv2
import numpy as np
from flask import Flask, Response, send_file, jsonify
from flask_cors import CORS
import mediapipe as mp
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to load clothing image and remove background using color threshold
def load_clothing_image_with_bg_removal(path):
    clothing_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    if clothing_image is None:
        raise FileNotFoundError(f"Could not load clothing image from {path}")
    
    # Check if the image has 3 channels (BGR)
    if clothing_image.shape[2] == 3:
        print(f"Removing background from {path}")
        
        # Convert image to HSV to remove background by color
        hsv = cv2.cvtColor(clothing_image, cv2.COLOR_BGR2HSV)
        
        # Define the range for background color (white in this case)
        lower_bound = np.array([0, 0, 200])  # Adjust as needed
        upper_bound = np.array([180, 50, 255])  # Adjust as needed
        
        # Create mask for the background
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Invert the mask to keep the clothing part
        mask_inv = cv2.bitwise_not(mask)
        
        # Convert the clothing to have an alpha channel (transparency)
        b_channel, g_channel, r_channel = cv2.split(clothing_image)
        alpha_channel = mask_inv
        
        # Merge the BGR channels with the alpha channel
        clothing_image = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

    return clothing_image

# Load clothing image
# clothing_image_path = r"C:\Users\Nilesh\Virtual_try on\Tryon\Backend\clothes\tshirt.png"
clothing_image_path = r"C:\Users\HP\Virtual_try on\Virtual_try on\Tryon\Backend\clothes\tshirt.png"

clothing_image = load_clothing_image_with_bg_removal(clothing_image_path)

# Function to overlay clothing on the detected body landmarks
def overlay_clothes(frame, landmarks, x_offset_adjust=0, y_offset_adjust=-80, scale_factor=1.7):
    global clothing_image

    if clothing_image is None:
        print("Clothing image is not loaded correctly.")
        return

    # Get shoulder and hip landmarks (checking validity)
    if len(landmarks) > 24:
        shoulder_left = landmarks[11]  # Left shoulder
        shoulder_right = landmarks[12]  # Right shoulder
        hip_left = landmarks[23]  # Left hip
        hip_right = landmarks[24]  # Right hip
    else:
        print("Insufficient pose landmarks detected.")
        return

    # Calculate shoulder width and torso height for dynamic scaling
    shoulder_width = int(abs((shoulder_right.x - shoulder_left.x) * frame.shape[1]))
    torso_height = int(abs((hip_left.y - shoulder_left.y) * frame.shape[0]))

    # Scale up the clothing by the scaling factor
    shoulder_width = int(shoulder_width * scale_factor)
    torso_height = int(torso_height * scale_factor)

    # Ensure minimum size to prevent extremely small or incorrect sizing
    if shoulder_width < 50 or torso_height < 50:
        print("Detected body size is too small for proper overlay.")
        return

    # Resize clothing to fit between shoulders and hips
    resized_clothes = cv2.resize(clothing_image, (shoulder_width, torso_height))

    # Calculate center point for clothing placement
    center_x = int((shoulder_left.x + shoulder_right.x) / 2 * frame.shape[1])
    center_y = int(shoulder_left.y * frame.shape[0])

    # Adjust the Y-offset to position the clothing more accurately over the upper torso
    y_offset = center_y + y_offset_adjust
    x_offset = center_x - resized_clothes.shape[1] // 2 + x_offset_adjust

    # Ensure the offsets are within the frame bounds
    y_offset = max(0, y_offset)
    x_offset = max(0, x_offset)

    # Adjust the size if the clothing exceeds the frame boundaries
    available_height = frame.shape[0] - y_offset
    available_width = frame.shape[1] - x_offset

    clothing_height = min(resized_clothes.shape[0], available_height)
    clothing_width = min(resized_clothes.shape[1], available_width)

    # Crop the clothing image if needed to fit within the frame
    cropped_clothes = resized_clothes[:clothing_height, :clothing_width]

    # Create a mask from the alpha channel
    alpha_channel = cropped_clothes[:, :, 3] / 255.0  # Normalize the alpha channel
    for c in range(0, 3):  # Assuming clothing image has 3 channels (RGB)
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

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Check if pose landmarks are detected
        if results.pose_landmarks:
            overlay_clothes(frame, results.pose_landmarks.landmark)

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

# Route to capture the image and store it
@app.route('/capture', methods=['POST'])
def capture_image():
    cap = cv2.VideoCapture(0)  # Use the default webcam
    ret, frame = cap.read()
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            overlay_clothes(frame, results.pose_landmarks.landmark)

        image_path = "captured_image.png"  # Path where the image will be stored
        cv2.imwrite(image_path, frame)  # Save the captured frame

        return jsonify({"message": "Image captured and stored successfully!"})
    else:
        return jsonify({"error": "Failed to capture image"}), 500

# Route to serve the captured image
@app.route('/get-captured-image')
def get_captured_image():
    image_path = "captured_image.png"  # Path to the stored image
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/png')
    else:
        return jsonify({"error": "Captured image not found"}), 404

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
