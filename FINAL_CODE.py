# Import necessary libraries
import cv2
import os
import datetime
import pyaudio
import numpy as np
import mediapipe as mp
import threading
from fpdf import FPDF
import imutils
from flask import Flask, render_template, request, Response, redirect, url_for
from flask_sqlalchemy import SQLAlchemy

# Create a Flask app
app = Flask(__name__)

# Configure the database URI
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:postgres@localhost:5432/multistreaming'  # Replace with your database URI

# Create a SQLAlchemy database instance
db = SQLAlchemy(app)

# Define a SQLAlchemy model for the camera configuration
class Cameraangle1(db.Model):
    tablename = "cameraangle1"
    print("@@@@@@",)
    camera_id = db.Column(db.Integer, primary_key=True)
    camera_url = db.Column(db.String(255), nullable=False)
    min_angle = db.Column(db.Integer, nullable=False)
    max_angle = db.Column(db.Integer, nullable=False)
    view = db.Column(db.String(50), nullable=False)

# Function to get camera URLs from the database
def get_camera_urls():
    with app.app_context():
        global cameras  
        cameras = Cameraangle1.query.all()
        return [(camera.camera_id, camera.camera_url, camera.min_angle, camera.max_angle, camera.view) for camera in cameras]

# Get camera URLs and initialize cameras
camera_urls = get_camera_urls()
print("@@@@@@@@@@@@@@",camera_urls)

# Create a list to store error messages for each camera (initialized with None)
error_messages = [None] * len(camera_urls)  # Define error_messages globally
print("@@@@@@",error_messages)


# Route to display the index page
@app.route('/')
def index():
    cameras = Cameraangle1.query.all()
    print("@@@@@",cameras)
    return render_template('index.html', cameras=cameras)

# Route to add a new camera
@app.route('/add_camera', methods=['GET', 'POST'])
def add_camera():
    if request.method == 'POST':
        camera_url = request.form.get('camera_url')
        min_angle = request.form.get('min_angle')
        max_angle = request.form.get('max_angle')
        view = request.form.get('view')

        camera = Cameraangle1(camera_url=camera_url, min_angle=min_angle, max_angle=max_angle, view=view)

        # Insert the camera record within the app context
        db.session.add(camera)
        db.session.commit()

    cameras = Cameraangle1.query.all()
    return render_template('add_camera.html', cameras=cameras)

# Route to edit an existing camera
@app.route('/edit_camera/<int:camera_id>', methods=['GET', 'POST'])
def edit_camera(camera_id):
    camera = Cameraangle1.query.get(camera_id)

    if request.method == 'POST':
        camera.camera_url = request.form.get('camera_url')
        camera.min_angle = request.form.get('min_angle')
        camera.max_angle = request.form.get('max_angle')
        camera.view = request.form.get('view')

        # Update the camera record within the app context
        db.session.commit()

        # Redirect back to the index page after editing
        return redirect(url_for('index'))

    return render_template('edit_camera.html', camera=camera)

# Route to delete an existing camera
@app.route('/delete_camera/<int:camera_id>', methods=['GET', 'POST'])
def delete_camera(camera_id):
    camera = Cameraangle1.query.get(camera_id)

    if request.method == 'POST':
        # Delete the camera record within the app context
        db.session.delete(camera)
        db.session.commit()

        # Redirect back to the index page after deleting
        return redirect(url_for('index'))

    return render_template('delete_camera.html', camera=camera)

# Dictionary to track the pose estimation status for each camera
pose_estimation_status = {camera_id: None for camera_id in range(len(camera_urls))}

class PoseEstimationThread(threading.Thread):
    def __init__(self, camera_id, camera_url, min_angle, max_angle, output_path):
        threading.Thread.__init__(self)
        self.camera_id = camera_id
        self.camera_url = camera_url
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.output_path = output_path
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.recording = False
        self.buffer = []  # Buffer to store frames before shooting position
        self.buffer_duration = 10  # Duration (in seconds) to store frames before shooting
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.sound_threshold = 0.5  # Adjust this threshold for sound detection

    def run(self):
        pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        cap = cv2.VideoCapture(self.camera_url)  # Pass the video URL as a string

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = 0
        start_recording_frame = 0  # Frame at which recording started

        # Sound detection setup
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        sound_frames = []

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Apply MOG2 background subtraction
            fg_mask = self.background_subtractor.apply(frame)
            # print(fg_mask)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                if landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST] and landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]:
                    right_wrist_landmark = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                    right_shoulder_landmark = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

                    shooting_rifle_angle = np.arctan2(right_wrist_landmark.y - right_shoulder_landmark.y,
                                                       right_wrist_landmark.x - right_shoulder_landmark.x)
                    min_angle = self.min_angle
                    max_angle = self.max_angle

                    if min_angle <= np.degrees(shooting_rifle_angle) <= max_angle:
                        if not self.recording:
                            self.recording = True
                            start_recording_frame = frame_count
                            current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                            video_filename = os.path.join(self.output_path, f"camera{self.camera_id}_user_shots_target_{current_time}.avi")
                            fourcc = cv2.VideoWriter_fourcc(*'XVID')
                            self.correct_pose_writer = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height), isColor=True)

                        self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                                      landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0),
                                                                                                    thickness=2,
                                                                                                    circle_radius=2))

                        cv2.putText(frame, "Correct", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                                      landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255),
                                                                                                    thickness=2,
                                                                                                    circle_radius=2))
                        cv2.putText(frame, "Wrong", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    if self.recording:
                        self.correct_pose_writer.write(frame)


            if self.recording and frame_count - start_recording_frame >= self.buffer_duration * fps:
                self.recording = False
                self.correct_pose_writer.release()

            # Sound detection
            sound_data = stream.read(1024)
            sound_frames.append(sound_data)

            if len(sound_frames) >= fps * self.buffer_duration:
                sound_frames.pop(0)

            if self.recording and self.detect_sound(sound_frames):
                self.save_previous_frames(frame_count - start_recording_frame)

            frame_count += 1

            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()

    def detect_sound(self, sound_frames):
        audio_data = b''.join(sound_frames)
        audio_amplitude = np.frombuffer(audio_data, dtype=np.int16)
        sound_volume = np.max(audio_amplitude) / 32767.0
        return sound_volume > self.sound_threshold

    def save_previous_frames(self, frame_count):
        frames_to_save = self.buffer[-frame_count:]
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        video_filename = os.path.join(self.output_path, f"camera{self.camera_id}_user_shots_target_{current_time}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_filename, fourcc, 30.0, (frames_to_save[0].shape[1], frames_to_save[0].shape[0]), isColor=True)

        for frame in frames_to_save:
            out.write(frame)

        out.release()
        print(f"Camera {self.camera_id} - Video Saved")

# Specify the output directory for saved videos
output_dir = 'output_videos'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    print("!!!!!!!!!!!!!!!!!,", camera_id)
    
    # Check if camera_id is within the valid range
    if 0 <= camera_id < len(camera_urls):
        camera_info = camera_urls[camera_id]
        print("camera details are:", camera_info)
        camera_url = camera_info[1]
        print("Camera url are",camera_url)
        min_angle = camera_info[2]
        max_angle = camera_info[3]
        view = camera_info[4]
        print('!!!!!!!!!!!!!!', camera_id)
        print('@@@@@@@@@@@@1', camera_url)
        return Response(PoseEstimationThread(camera_id, camera_url, min_angle, max_angle, view).run(), mimetype='multipart/x-mixed-replace; boundary=frame')
       
    else:
        return "Invalid camera ID"

# Add route to view a specific camera

@app.route('/view_camera/<int:camera_id>')
def view_camera(camera_id):
    if 0 <= camera_id < len(camera_urls):
        camera_info = camera_urls[camera_id]
        print("@@@@@@@@",camera_info)
        camera_url = camera_info[1]
        print("@@@@@@",camera_url)
        min_angle = camera_info[2]
        print("@@@@@@@",min_angle)
        max_angle = camera_info[3]
        print("@@@@@@@@@@",max_angle)
        view = camera_info[4]
        print("@@@@@@",view)

        return render_template('view.html', camera_id=camera_id, camera_url=camera_url, min_angle=min_angle, max_angle=max_angle, view=view)
    else:
        return "Invalid camera ID"
    
#Route for cropping
@app.route('/start_crop/<int:camera_id>')
def start_crop(camera_id):
    global static_ROI
    static_ROI.display_window = False
    cv2.namedWindow('cropped_image')
    # Set OpenCV window to be topmost
    cv2.setWindowProperty('cropped_image', cv2.WND_PROP_TOPMOST, 1)
    cv2.setMouseCallback('cropped_image', static_ROI.extract_coordinates)

    # Check if camera_id is within the valid range
    if 0 <= camera_id < len(camera_urls):
        # Open the specified camera for cropping
        camera_info = camera_urls[camera_id]
        print("@@@@@@@@",camera_info)
        camera_url = camera_info[1]
        print("@@@@@@@@",camera_url)
        cap = cv2.VideoCapture(camera_url)

        while True:
            if cap.isOpened():
                (status, static_ROI.frame) = cap.read()
                print(status)
                cv2.imshow('cropped_image', static_ROI.frame)

                key = cv2.waitKey(2)

                if key == ord('q'):
                    cv2.destroyAllWindows()
                    break
            else:
                break

        cv2.destroyAllWindows()
        static_ROI.display_window = True
        cap.release()

        return redirect(url_for('streaming'))
    else:
        return "Invalid camera ID"

#Route for refresh camera
@app.route('/refresh_camera/<int:camera_id>', methods=['GET', 'POST'])
def refresh_camera(camera_id):
    if 0 <= camera_id < len(camera_urls):
        camera_info = camera_urls[camera_id]
        camera_url = camera_info[1]

        cap = cv2.VideoCapture(camera_url)

        if not cap.isOpened():
            error_messages[camera_id] = "Error: Camera is not available or cannot be connected."
        else:
            # Camera is connected, clear any previous error message for this camera
            error_messages[camera_id] = None

            # Restart the thread for the camera
            pose_estimation_status[camera_id] = None  # Reset pose estimation status
            threading.Thread(target=PoseEstimationThread(camera_id, camera_url, camera_info[2], camera_info[3], output_dir).run).start()

        # Redirect back to the streaming page
        return redirect(url_for('streaming'))
    else:
        return "Invalid camera ID"

# Route for streaming
@app.route('/streaming')
def streaming():
    # Pass the number of cameras to the template
    num_cameras = len(camera_urls)
    return render_template('video_feed.html', num_cameras=num_cameras, camera_urls=camera_urls,error_messages=error_messages)

# StaticROI class and related routes
class StaticROI(object):
    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        self.image_coordinates = []
        self.extract = False
        self.selected_ROI = False
        self.cropped_image = None
        self.captured_image_path = None
        self.cropped_image_path = None
        self.pdf_path = None
        self.total_shots = 0
        self.total_score = 0
        self.frame = None  # To store the current frame
        self.display_window = False  # Flag to determine whether to display the OpenCV window

    def extract_coordinates(self, event, x, y, flags, parameters):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x, y)]
            self.extract = True

        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x, y))
            self.extract = False
            self.selected_ROI = True
            cv2.rectangle(self.frame, self.image_coordinates[0], self.image_coordinates[1], (0, 255, 0), 2)
            self.crop_ROI()
            self.save_captured_image()
            self.save_cropped_image()
            self.process_and_create_pdf()

    def crop_ROI(self):
        if self.selected_ROI:
            x1, y1 = self.image_coordinates[0]
            x2, y2 = self.image_coordinates[1]
            self.cropped_image = self.frame[y1:y2, x1:x2]

    def save_captured_image(self):
        output_folder = "output_images"
        os.makedirs(output_folder, exist_ok=True)
        self.captured_image_path = os.path.join(output_folder, "captured_image.png")
        cv2.imwrite(self.captured_image_path, self.frame)

    def save_cropped_image(self):
        if self.selected_ROI and self.cropped_image is not None:
            output_folder = "output_images"
            os.makedirs(output_folder, exist_ok=True)
            self.cropped_image_path = os.path.join(output_folder, "cropped_image.png")
            cv2.imwrite(self.cropped_image_path, self.cropped_image)

    def process_and_create_pdf(self):
        if self.captured_image_path and self.cropped_image_path:
            default = cv2.imread(self.captured_image_path)
            img = cv2.resize(default, (640, 640))

            # Image processing logic
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)

            v_mask = cv2.inRange(v, 0, 155)

            cnts = cv2.findContours(v_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            for c in cnts:
                if cv2.contourArea(c) > 10000:
                    cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
                    area_max = cv2.contourArea(c)

            radius_max = np.sqrt(area_max / np.pi)
            section_size = radius_max / 9

            centre_v_mask = cv2.inRange(v, 215, 255)
            cnts = cv2.findContours(centre_v_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            for c in cnts:
                if cv2.contourArea(c) > 10:
                    centre_coords = self.centroid(c)

            h_mask = cv2.inRange(h, 0, 30)
            h_mask = cv2.medianBlur(h_mask, 11)
            cnts = cv2.findContours(h_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            holes = []
            print(holes)
            HoleDists = []

            scoreboundaries = []
            for i in range(1, 10):
                cv2.circle(img, centre_coords, int(i * section_size), (255, 0, 0), 1)
                scoreboundaries.append(int(i * section_size))

            scores = {'pure': {}, 'cut': {}}

            for c in cnts:
                if cv2.contourArea(c) > 1:
                    x, y, w, h = cv2.boundingRect(c)
                    pts = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
                    print(pts)

                    centre_holes = self.centroid(c)

                    pointscore = 0
                    for pt in c:
                        pt = pt[0]
                        X = pt[0]
                        Y = pt[1]

                        HoleDist = np.sqrt((X - centre_coords[0]) * 2 + (Y - centre_coords[1]) * 2)
                        HoleDists.append(HoleDist)
                        score = self.getScore(scoreboundaries, HoleDist)

                        if score > pointscore:
                            pointscore = score

                    cv2.circle(img, centre_holes, 1, (0, 0, 255), -1)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.drawContours(img, [c], -1, (0, 255, 0), 1)

                    cv2.putText(img, "Score: " + str(pointscore), (centre_holes[0] - 20, centre_holes[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    if centre_holes[1] < centre_coords[1]:
                        # Pure Shot
                        if pointscore in scores['pure']:
                            scores['pure'][pointscore] += 1
                        else:
                            scores['pure'][pointscore] = 1
                    else:
                        # Cut Shot
                        if pointscore in scores['cut']:
                            scores['cut'][pointscore] += 1
                        else:
                            scores['cut'][pointscore] = 1

                    self.total_shots += 1
                    self.total_score += pointscore

            # Create PDF with captured, cropped images, and results table
            pdf = FPDF()
            # Add the first page with the cropped image
            pdf.add_page()
            pdf.image(self.cropped_image_path, x=10, y=10, w=190)

            # Add the second page with the results table
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, "Number of Pure Shots and Cut Shots", ln=True, align='C')
            for i in range(1, 11):
                pdf.cell(30, 10, str(i), 1, 0, 'C')
                if i in scores['pure']:
                    pdf.cell(60, 10, str(scores['pure'][i]), 1, 0, 'C')
                else:
                    pdf.cell(60, 10, '0', 1, 0, 'C')
                if i in scores['cut']:
                    pdf.cell(60, 10, str(scores['cut'][i]), 1, 1, 'C')
                else:
                    pdf.cell(60, 10, '0', 1, 1, 'C')

            pdf.cell(0, 10, "", ln=True)  # Add an empty line

            # Add total shots and total score information
            pdf.cell(0, 10, f"Total Shots: {self.total_shots}", ln=True)
            pdf.cell(0, 10, f"Total Score: {self.total_score}", ln=True)
            self.pdf_path = "results.pdf"

            pdf.output(self.pdf_path)

    def centroid(self, contour):
        M = cv2.moments(contour)
        cx = int(round(M['m10'] / M['m00']))
        cy = int(round(M['m01'] / M['m00']))
        centre = (cx, cy)
        return centre

    def getScore(self, scoreboundaries, HoleDist):
        score = 0
        if scoreboundaries[0] > HoleDist:
            score = 10
        for i in range(1, len(scoreboundaries)):
            if scoreboundaries[i - 1] <= HoleDist < scoreboundaries[i]:
                score = len(scoreboundaries) - i
        return score


# Create a StaticROI instance
static_ROI = StaticROI()

if __name__ == '__main__':
    app.run(debug=True)

