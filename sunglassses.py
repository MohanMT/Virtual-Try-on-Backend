import cv2
import numpy as np
from flask import Flask, Response, request, jsonify

app = Flask(__name__)

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Dictionary to hold multiple sunglasses options
sunglasses_dict = {
    "sporty": cv2.imread('sunglasses_sporty.png', cv2.IMREAD_UNCHANGED),
    "classic": cv2.imread('sunglasses_classic.png', cv2.IMREAD_UNCHANGED),
    "futuristic": cv2.imread('sunglasses_futuristic.png', cv2.IMREAD_UNCHANGED)
}

# Default selection
selected_sunglasses = sunglasses_dict["classic"]

@app.route('/select_sunglasses', methods=['POST'])
def select_sunglasses():
    global selected_sunglasses
    choice = request.json.get('sunglass_type', 'classic')  # Default to 'classic'
    
    print(f"Received request to select {choice} sunglasses")  # Debugging
    
    if choice in sunglasses_dict:
        selected_sunglasses = sunglasses_dict[choice]
        
        if selected_sunglasses is None:
            print(f"Failed to load {choice} sunglasses image")
            return jsonify({"status": "error", "message": f"Failed to load {choice} sunglasses image."}), 500
        
        print(f"Sunglasses updated to {choice}")  # Debugging
        return jsonify({"status": "success", "message": f"Selected {choice} sunglasses."}), 200
    else:
        print("Invalid sunglasses type")  # Debugging
        return jsonify({"status": "error", "message": "Invalid sunglasses type."}), 400

def overlay_sunglasses(frame, face_coordinates):
    global selected_sunglasses
    if selected_sunglasses is None:
        print("Sunglasses image not loaded, skipping overlay.")
        return

    (x, y, w, h) = face_coordinates
    sunglasses = selected_sunglasses  # Use the selected sunglasses

    # Resize the selected sunglasses
    sunglasses_width = int(w * 1.2)
    sunglasses_height = int(sunglasses_width * sunglasses.shape[0] / sunglasses.shape[1])
    sunglasses_resized = cv2.resize(sunglasses, (sunglasses_width, sunglasses_height))

    # Calculate position to overlay sunglasses
    y_offset = int(y + h / 4)
    x_offset = x - int((sunglasses_width - w) / 2)

    if y_offset < 0 or x_offset < 0 or y_offset + sunglasses_resized.shape[0] > frame.shape[0] or x_offset + sunglasses_resized.shape[1] > frame.shape[1]:
        return  # Skip overlay if out of bounds

    roi = frame[y_offset:y_offset + sunglasses_resized.shape[0], x_offset:x_offset + sunglasses_resized.shape[1]]

    # Mask for sunglasses
    alpha_s = sunglasses_resized[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        roi[:, :, c] = (alpha_s * sunglasses_resized[:, :, c] + alpha_l * roi[:, :, c])

    frame[y_offset:y_offset + sunglasses_resized.shape[0], x_offset:x_offset + sunglasses_resized.shape[1]] = roi

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            overlay_sunglasses(frame, (x, y, w, h))

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
    