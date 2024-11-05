import os
import cv2
import numpy as np
import mediapipe as mp
import requests



clothing_image = None  # Placeholder for the clothing image
# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


# Function to load clothing image from a local path
def load_clothing_image(image_path):
    global clothing_image
    
    # Check if the path is a URL
    if image_path.startswith('http://') or image_path.startswith('https://'):
        response = requests.get(image_path)
        if response.status_code == 200:
            # Convert response content to a numpy array and read it as an image
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


def overlay_clothes(frame, landmarks, category, subcategory):
    global clothing_image

    if clothing_image is None:
        print("Clothing image is not loaded correctly.")
        return

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
