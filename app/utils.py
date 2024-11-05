import os
import cv2
import numpy as np
import mediapipe as mp
import requests

# Placeholder for the clothing image
clothing_image = None

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to load clothing image from a local path or URL
def load_clothing_image(image_path):
    global clothing_image
    
    if image_path.startswith('http://') or image_path.startswith('https://'):
        response = requests.get(image_path)
        if response.status_code == 200:
            nparr = np.frombuffer(response.content, np.uint8)
            clothing_image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if clothing_image is None:
                raise ValueError(f"Could not load clothing image from URL: {image_path}")
        else:
            raise ValueError(f"Failed to retrieve image from URL: {image_path}")
    else:
        if os.path.exists(image_path):
            clothing_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if clothing_image is None:
                raise FileNotFoundError(f"Could not load clothing image from {image_path}")
        else:
            raise FileNotFoundError(f"Clothing image path does not exist: {image_path}")

# Function to remove background from clothing image using color threshold
def remove_background_from_clothing_image():
    global clothing_image

    if clothing_image is None:
        print("No clothing image loaded.")
        return

    if clothing_image.shape[2] in [3, 4]:  # Check for 3 or 4 channels
        print("Removing background from the clothing image...")
        hsv = cv2.cvtColor(clothing_image, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([0, 0, 200])
        upper_bound = np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        mask_inv = cv2.bitwise_not(mask)
        b_channel, g_channel, r_channel = cv2.split(clothing_image)
        alpha_channel = mask_inv
        clothing_image = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

# Function to overlay clothes based on category and subcategory
def overlay_clothes(frame, landmarks, category, subcategory):
    global clothing_image

    if clothing_image is None:
        print("Clothing image is not loaded correctly.")
        return

    # Ensure the necessary landmarks are present (shoulders and hips)
    if len(landmarks) < 25:
        print("Insufficient pose landmarks detected.")
        return

    # Overlay clothing based on the category
    if category in ["Kids", "Men"]:
        overlay_male_clothes(frame, landmarks)
    elif category == "Women":
        if subcategory == "Bottom wear":
            overlay_female_upper_body_clothes(frame, landmarks)
        elif subcategory == "Full body":
            overlay_female_full_body_clothes(frame, landmarks)
        else:
            print("Unknown subcategory for Women:", subcategory)
    else:
        print("Unknown category:", category)

# Male upper body overlay function
def overlay_male_clothes(frame, landmarks):
    global clothing_image

    shoulder_left = landmarks[11]  # Left shoulder
    shoulder_right = landmarks[12]  # Right shoulder
    hip_left = landmarks[23]  # Left hip

    # Calculate shoulder width and torso height for dynamic scaling
    shoulder_width = int(abs((shoulder_right.x - shoulder_left.x) * frame.shape[1]))
    torso_height = int(abs((hip_left.y - shoulder_left.y) * frame.shape[0]) * 1.5)  # Adjusted height

    # Resize clothing to fit between shoulders and the bottom
    resized_clothes = cv2.resize(clothing_image, (shoulder_width, torso_height))

    # Calculate center point for clothing placement
    center_x = int((shoulder_left.x + shoulder_right.x) / 2 * frame.shape[1])
    center_y = int(shoulder_left.y * frame.shape[0])

    # Position the clothing
    y_offset = center_y - torso_height  # Position it starting from the shoulder down
    x_offset = center_x - int(shoulder_width / 2)

    # Ensure the offsets are within the frame bounds
    y_offset = max(0, y_offset)
    x_offset = max(0, x_offset)

    # Overlay the clothing on the frame
    overlay_image_with_alpha(frame, resized_clothes, x_offset, y_offset)

# Female upper body overlay function
def overlay_female_upper_body_clothes(frame, landmarks):
    global clothing_image

    shoulder_left = landmarks[11]  # Left shoulder
    shoulder_right = landmarks[12]  # Right shoulder
    hip_left = landmarks[23]  # Left hip

    shoulder_width = int(abs((shoulder_right.x - shoulder_left.x) * frame.shape[1]) * 1.5)
    torso_height = int(abs((hip_left.y - shoulder_left.y) * frame.shape[0]) * 1.5)

    resized_clothes = cv2.resize(clothing_image, (shoulder_width, torso_height))

    center_x = int((shoulder_left.x + shoulder_right.x) / 2 * frame.shape[1])
    center_y = int(shoulder_left.y * frame.shape[0])

    y_offset = center_y - int(torso_height / 2)
    x_offset = center_x - int(shoulder_width / 2)

    overlay_image_with_alpha(frame, resized_clothes, x_offset, y_offset)

# Female full body overlay function
def overlay_female_full_body_clothes(frame, landmarks):
    global clothing_image

    shoulder_left = landmarks[11]  # Left shoulder
    shoulder_right = landmarks[12]  # Right shoulder
    hip_left = landmarks[23]  # Left hip
    hip_right = landmarks[24]  # Right hip

    torso_width = int(abs((shoulder_right.x - shoulder_left.x) * frame.shape[1]))
    torso_height = int(abs((hip_left.y - shoulder_left.y) * frame.shape[0]) * 1.8)

    resized_clothes = cv2.resize(clothing_image, (torso_width, torso_height))

    center_x = int((shoulder_left.x + shoulder_right.x) / 2 * frame.shape[1])
    center_y = int(hip_left.y * frame.shape[0])

    y_offset = center_y - torso_height
    x_offset = center_x - int(torso_width / 2)

    overlay_image_with_alpha(frame, resized_clothes, x_offset, y_offset)

# Helper function to overlay an image with alpha channel
def overlay_image_with_alpha(frame, overlay, x, y):
    h, w = overlay.shape[:2]

    # Check if overlay is within frame bounds
    if y < 0 or x < 0 or y + h > frame.shape[0] or x + w > frame.shape[1]:
        print(f"Overlay position out of bounds: x={x}, y={y}, w={w}, h={h}, frame_shape={frame.shape}")
        return

    alpha_channel = overlay[:, :, 3] / 255.0
    for c in range(3):
        frame[y:y + h, x:x + w, c] = (
            frame[y:y + h, x:x + w, c] * (1 - alpha_channel) +
            overlay[:, :, c] * alpha_channel
        )

# Function to generate frames from the webcam feed
def generate_frames(category, subcategory):
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
            overlay_clothes(frame, results.pose_landmarks.landmark, category, subcategory)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Example usage
# load_clothing_image("path/to/your/clothing_image.png")  # Load the clothing image
# remove_background_from_clothing_image()  # Remove background from the loaded clothing image
# category = "Men"
# subcategory = "Upper Body"
# for frame in generate_frames(category, subcategory):
#     # Handle the frame (for example, display it in a window or send it to a web client)


# Male full body overlay function
# def overlay_male_full_body_clothes(frame, landmarks):
    # shoulder_left = landmarks[11]
    # shoulder_right = landmarks[12]
    # hip_left = landmarks[23]
    # hip_right = landmarks[24]

    # shoulder_width = int(abs((shoulder_right.x - shoulder_left.x) * frame.shape[1]))
    # torso_height = int(abs((hip_right.y - shoulder_left.y) * frame.shape[0]) * 1.8)

    # resized_clothes = cv2.resize(clothing_image, (shoulder_width, torso_height))

    # center_x = int((shoulder_left.x + shoulder_right.x) / 2 * frame.shape[1])
    # center_y = int(hip_left.y * frame.shape[0])

    # y_offset = center_y - torso_height
    # x_offset = center_x - int(shoulder_width / 2)

    # overlay_image_with_alpha(frame, resized_clothes, x_offset, y_offset)
