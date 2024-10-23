import cv2
import numpy as np
from flask import Flask, Response
import mediapipe as mp
from panda3d.core import *
from direct.showbase.ShowBase import ShowBase

app = Flask(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

class VirtualTryOnApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # Load the 3D model of the t-shirt
        self.tshirt_model = self.loader.loadModel("3dmodel/oversized-tshirt.obj")
        self.tshirt_model.reparentTo(self.render)

        # Set up camera view
        self.cam.setPos(0, -5, 2)  # Adjust camera position as needed
        self.cam.lookAt(0, 0, 0)

        self.model_scale_factor = 0.2  # Scale factor for the t-shirt model

    def update_model_position(self, landmarks):
        # Get shoulder and hip landmarks (checking validity)
        if len(landmarks) > 24:
            shoulder_left = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            shoulder_right = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            hip_left = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        else:
            print("Insufficient pose landmarks detected.")
            return

        # Calculate shoulder width and torso height for dynamic scaling
        shoulder_width = int(abs((shoulder_right.x - shoulder_left.x) * 640))  # Webcam width
        torso_height = int(abs((hip_left.y - shoulder_left.y) * 480))  # Webcam height

        # Position and scale the t-shirt model accordingly
        shoulder_width_scaled = shoulder_width * self.model_scale_factor
        torso_height_scaled = torso_height * self.model_scale_factor

        # Center the t-shirt over the shoulders
        x_pos = (shoulder_left.x + shoulder_right.x) / 2 * 640
        y_pos = shoulder_left.y * 480

        # Set position and scale
        self.tshirt_model.setPos(x_pos / 100, -1.5, -5)  # Position in front of the camera
        self.tshirt_model.setScale(shoulder_width_scaled / 100, torso_height_scaled / 100, 1)

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            self.update_model_position(results.pose_landmarks.landmark)

def generate_frames():
    cap = cv2.VideoCapture(0)  # Use the default webcam

    app = VirtualTryOnApp()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        app.process_frame(frame)

        # Update the Panda3D window
        app.taskMgr.step()

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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
