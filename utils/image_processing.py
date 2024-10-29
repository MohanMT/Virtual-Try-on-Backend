import numpy as np
import cv2
import requests

def download_clothing_image(image_url):
    response = requests.get(image_url)
    image_array = np.frombuffer(response.content, np.uint8)
    clothing_image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)

    if clothing_image is None:
        raise FileNotFoundError(f"Could not download clothing image from {image_url}")
    return clothing_image

def remove_background_from_clothing_image(clothing_image):
    if clothing_image.shape[2] == 3:
        hsv = cv2.cvtColor(clothing_image, cv2.COLOR_BGR2HSV)

        lower_bound = np.array([0, 0, 200])
        upper_bound = np.array([180, 50, 255])
        
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        mask_inv = cv2.bitwise_not(mask)

        b_channel, g_channel, r_channel = cv2.split(clothing_image)
        alpha_channel = mask_inv

        clothing_image = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    return clothing_image
