import cv2
import numpy as np
from flask import Flask, Response, request, jsonify
import mediapipe as mp
import os

app = Flask(__name__)

# Initialize MediaPipe Pose and Selfie Segmentation
mp_pose = mp.solutions.pose
mp_segmentation = mp.solutions.selfie_segmentation
pose = mp_pose.Pose()
segmentation = mp_segmentation.SelfieSegmentation(model_selection=1)

# Folder containing all clothing images
clothing_folder_path = "C:/Users/Nilesh/Virtual_try on/Tryon/Backend/clothes"  # Update with your folder path

# Load all clothing images from the folder
clothing_images = {}
for filename in os.listdir(clothing_folder_path):
    if filename.endswith(".png"):  # Assuming all clothing images are PNG with transparency
        clothing_image_path = os.path.join(clothing_folder_path, filename)
        clothing_image = cv2.imread(clothing_image_path, cv2.IMREAD_UNCHANGED)
        if clothing_image is not None:
            clothing_images[filename] = clothing_image
            print(f"Loaded clothing image: {filename}")  # Debug statement

if len(clothing_images) == 0:
    raise Exception(f"No clothing images found in folder {clothing_folder_path}")

# Set initial clothing to the first one in the folder
current_clothing = list(clothing_images.values())[0]

# Function to overlay clothing on the detected body landmarks
def overlay_clothes(frame, landmarks, x_offset_adjust=0, y_offset_adjust=-120, scale_factor=1.5):
    global current_clothing

    if current_clothing is None:
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
    resized_clothes = cv2.resize(current_clothing, (shoulder_width, torso_height))

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

    # Check if the clothing image has 4 channels (RGBA) or 3 channels (RGB)
    if cropped_clothes.shape[2] == 4:
        # If there is an alpha channel, blend it with the background
        alpha_channel = cropped_clothes[:, :, 3] / 255.0  # Normalize the alpha channel
        for c in range(0, 3):  # Assuming clothing image has 3 channels (RGB)
            frame[y_offset:y_offset + clothing_height, x_offset:x_offset + clothing_width, c] = (
                frame[y_offset:y_offset + clothing_height, x_offset:x_offset + clothing_width, c] * (1 - alpha_channel) +
                cropped_clothes[:, :, c] * alpha_channel
            )
    else:
        # If the image doesn't have an alpha channel, directly overlay it
        frame[y_offset:y_offset + clothing_height, x_offset:x_offset + clothing_width] = cropped_clothes

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

        # Get the segmentation mask
        mask_results = segmentation.process(rgb_frame)
        mask = mask_results.segmentation_mask

        # Convert mask to binary (foreground and background)
        mask = np.where(mask > 0.5, 1, 0).astype(np.uint8)

        # Apply the mask to the frame to remove background
        frame_bg_removed = frame * mask[:, :, np.newaxis]

        # Check if pose landmarks are detected
        if results.pose_landmarks:
            overlay_clothes(frame_bg_removed, results.pose_landmarks.landmark)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Return the frame to be displayed in the web feed
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Route to serve the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to handle clothing selection
@app.route('/select_clothes', methods=['POST'])
def select_clothes():
    global current_clothing

    data = request.get_json()
    clothing_name = data.get('clothing_type')

    if clothing_name in clothing_images:
        current_clothing = clothing_images[clothing_name]
        print(f"Clothing changed to: {clothing_name}")  # Debug statement
        return jsonify({'message': f'Clothing changed to {clothing_name}'}), 200
    else:
        print("Clothing not found:", clothing_name)  # Debug statement
        return jsonify({'message': 'Clothing not found'}), 400

# Route to get clothing options
@app.route('/clothing_options', methods=['GET'])
def clothing_options():
    return jsonify(list(clothing_images.keys())), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
