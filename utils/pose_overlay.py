import cv2
import numpy as np

def overlay_clothes(frame, landmarks, clothing_image):
    if clothing_image is None:
        print("Clothing image is not loaded correctly.")
        return

    if len(landmarks) < 25:
        print("Insufficient pose landmarks detected.")
        return

    shoulder_left = landmarks[11]
    shoulder_right = landmarks[12]
    hip_left = landmarks[23]
    hip_right = landmarks[24]

    shoulder_width = int(abs((shoulder_right.x - shoulder_left.x) * frame.shape[1]))
    torso_height = int(abs((hip_left.y - shoulder_left.y) * frame.shape[0]))

    scale_factor = 1.5
    shoulder_width = int(shoulder_width * scale_factor)
    torso_height = int(torso_height * scale_factor)

    resized_clothes = cv2.resize(clothing_image, (shoulder_width, torso_height))
    center_x = int((shoulder_left.x + shoulder_right.x) / 2 * frame.shape[1])
    center_y = int(shoulder_left.y * frame.shape[0])

    y_offset = max(0, center_y - int(torso_height / 2))
    x_offset = max(0, center_x - int(shoulder_width / 2))

    available_height = frame.shape[0] - y_offset
    available_width = frame.shape[1] - x_offset

    clothing_height = min(resized_clothes.shape[0], available_height)
    clothing_width = min(resized_clothes.shape[1], available_width)

    cropped_clothes = resized_clothes[:clothing_height, :clothing_width]
    alpha_channel = cropped_clothes[:, :, 3] / 255.0
    for c in range(3):
        frame[y_offset:y_offset + clothing_height, x_offset:x_offset + clothing_width, c] = (
            frame[y_offset:y_offset + clothing_height, x_offset:x_offset + clothing_width, c] * (1 - alpha_channel) +
            cropped_clothes[:, :, c] * alpha_channel
        )