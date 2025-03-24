from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory, Response, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from config import Config
import cv2
import numpy as np
import os
import torch
from PIL import Image
from datetime import datetime
import uuid
from pathlib import Path
from flask_migrate import Migrate
import mimetypes
import re
import mediapipe as mp
from torchvision import transforms
import torch.nn.functional as F
from inference_sdk import InferenceHTTPClient
import base64
import io

app = Flask(__name__)
app.config.from_object(Config)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Replace with a secure secret key
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///traffic.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize YOLO models with better confidence thresholds and class configurations
try:
    # Model for vehicle detection
    vehicle_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    vehicle_model.conf = 0.25  # Lower confidence threshold for better detection
    # Don't restrict classes for vehicle model to detect all types
    
    # Model for helmet and phone detection
    violation_model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
    violation_model.conf = 0.25  # Lower base confidence
    violation_model.iou = 0.45
    violation_model.classes = [0, 27, 67]  # person, cell phone, helmet
except Exception as e:
    print(f"Error loading models: {e}")
    vehicle_model = None
    violation_model = None

# Add new global variables after other global variables
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'

# Ensure upload and processed directories exist
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(PROCESSED_FOLDER).mkdir(parents=True, exist_ok=True)

db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Global variables for signal status and traffic data
signal_status = {
    'north': 'red',
    'south': 'red',
    'east': 'red',
    'west': 'red'
}

wait_times = {
    'north': 0,
    'south': 0,
    'east': 0,
    'west': 0
}

traffic_counts = {
    'north': 0,
    'south': 0,
    'east': 0,
    'west': 0
}

# Timer control status
timer_active = False

# Add after other global variables
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# Add after other global variables
ROBOFLOW_CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="YvsbdvKc1p5Jtb82NBTd"
)

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Model for vehicle counts
class VehicleCount(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False)
    direction = db.Column(db.String(10), nullable=False)
    count = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f'<VehicleCount {self.direction}: {self.count} at {self.timestamp}>'

@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('signup'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('signup'))

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash('Registration successful!')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        
        flash('Invalid username or password')
        return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', 
                         signal_status=signal_status,
                         wait_times=wait_times)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/timer_control', methods=['POST'])
@login_required
def timer_control():
    global timer_active
    data = request.get_json()
    action = data.get('action')

    if action == 'start':
        timer_active = True
        return jsonify({
            'status': 'success',
            'message': 'Timer control started'
        })
    elif action == 'stop':
        timer_active = False
        # Reset all signals to red
        for direction in signal_status:
            signal_status[direction] = 'red'
            wait_times[direction] = 0
        
        return jsonify({
            'signal_status': signal_status,
            'wait_times': wait_times
        })
    
    return jsonify({'error': 'Invalid action'}), 400

@app.route('/update_signals', methods=['POST'])
@login_required
def update_signals():
    data = request.get_json()
    current_direction = data.get('current_direction')
    
    if current_direction not in ['north', 'south', 'east', 'west']:
        return jsonify({'error': 'Invalid direction'}), 400
    
    # Update signal status based on current direction
    for direction in signal_status:
        if direction == current_direction:
            signal_status[direction] = 'green'
            wait_times[direction] = 0
        else:
            signal_status[direction] = 'red'
            # Calculate wait time based on position in cycle
            directions = ['north', 'south', 'east', 'west']
            current_index = directions.index(current_direction)
            direction_index = directions.index(direction)
            positions_away = (direction_index - current_index) % len(directions)
            wait_times[direction] = positions_away * 30  # 30 seconds per cycle
    
    return jsonify({
        'signal_status': signal_status,
        'wait_times': wait_times
    })

@app.route('/emergency_stop', methods=['POST'])
@login_required
def emergency_stop():
    global timer_active
    timer_active = False  # Stop timer control if it's running
    
    data = request.get_json()
    allowed_direction = data.get('direction')
    
    if allowed_direction not in ['north', 'south', 'east', 'west']:
        return jsonify({'error': 'Invalid direction'}), 400
    
    # Update signal status
    for direction in signal_status:
        if direction == allowed_direction:
            signal_status[direction] = 'green'
            wait_times[direction] = 0
        else:
            signal_status[direction] = 'red'
            wait_times[direction] = 60  # Set 60 seconds wait time for stopped directions
    
    return jsonify({
        'signal_status': signal_status,
        'wait_times': wait_times
    })

@app.route('/manual_override', methods=['POST'])
@login_required
def manual_override():
    data = request.get_json()
    selected_direction = data.get('direction')
    
    if selected_direction not in ['north', 'south', 'east', 'west']:
        return jsonify({'error': 'Invalid direction'}), 400
    
    # Update signal status
    for direction in signal_status:
        if direction == selected_direction:
            signal_status[direction] = 'green'
            wait_times[direction] = 0
        else:
            signal_status[direction] = 'red'
            wait_times[direction] = 30  # Set 30 seconds wait time for stopped directions
    
    return jsonify({
        'signal_status': signal_status,
        'wait_times': wait_times
    })

@app.route('/analyze_traffic', methods=['POST'])
@login_required
def analyze_traffic():
    if 'photo' not in request.files:
        print("No photo in request")
        return jsonify({'error': 'No photo uploaded'}), 400
    
    photo = request.files['photo']
    direction = request.form.get('direction')
    
    print(f"Analyzing traffic for {direction} direction")
    
    if photo.filename == '':
        print("Empty filename")
        return jsonify({'error': 'No photo selected'}), 400
    
    if direction not in ['north', 'south', 'east', 'west']:
        print(f"Invalid direction: {direction}")
        return jsonify({'error': 'Invalid direction'}), 400
    
    try:
        # Save the uploaded file
        filename = secure_filename(photo.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        photo.save(filepath)
        print(f"Saved photo to {filepath}")
        
        # Analyze the image using YOLOv5
        print("Starting vehicle detection")
        vehicle_count = detect_vehicles(filepath)
        print(f"Vehicle detection completed. Count: {vehicle_count}")
        
        # Update traffic counts
        traffic_counts[direction] = vehicle_count
        print(f"Updated traffic counts: {traffic_counts}")
        
        # Store count in database
        try:
            vehicle_record = VehicleCount(
                timestamp=datetime.utcnow(),
                direction=direction,
                count=vehicle_count
            )
            db.session.add(vehicle_record)
            db.session.commit()
            print(f"Saved count to database: {direction} = {vehicle_count}")
        except Exception as db_error:
            print(f"Database error: {db_error}")
            db.session.rollback()
            # Continue even if database save fails
        
        # Clean up
        try:
            os.remove(filepath)
            print(f"Cleaned up file: {filepath}")
        except Exception as cleanup_error:
            print(f"Cleanup error: {cleanup_error}")
        
        return jsonify({
            'status': 'success',
            'vehicle_count': vehicle_count,
            'message': f'Analysis completed successfully. Detected {vehicle_count} vehicles.'
        })
    
    except Exception as e:
        print(f"Error in analyze_traffic: {str(e)}")
        # Try to clean up file if it exists
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass
        
        return jsonify({
            'status': 'error',
            'error': f'Error processing image: {str(e)}'
        }), 500

def detect_vehicles(image_path):
    """
    Detect vehicles in an image using YOLOv5.
    Returns the count of vehicles detected.
    """
    try:
        print("Starting vehicle detection...")
        # Read image
        img = Image.open(image_path)
        
        # Perform detection
        results = vehicle_model(img)
        
        # COCO dataset indices for vehicles:
        # 2: car, 3: motorcycle, 5: bus, 7: truck
        vehicle_classes = [2, 3, 5, 7]
        
        # Filter detections by confidence and class
        vehicle_count = 0
        for detection in results.xyxy[0]:
            cls = int(detection[5])  # Class index
            conf = float(detection[4])  # Confidence
            
            # Lower confidence threshold for better detection
            if cls in vehicle_classes and conf > 0.25:
                vehicle_count += 1
                print(f"Detected vehicle of class {cls} with confidence {conf:.2f}")
        
        print(f"Total vehicles detected: {vehicle_count}")
        return vehicle_count

    except Exception as e:
        print(f"Error in vehicle detection: {e}")
        # Try fallback method
        return fallback_detect_vehicles(image_path)

def fallback_detect_vehicles(image_path):
    """
    Fallback method using basic OpenCV detection if YOLO fails.
    """
    try:
        print("Using fallback detection method")
        image = cv2.imread(image_path)
        if image is None:
            print("Failed to read image in fallback detection")
            return 0
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')
        vehicles = car_cascade.detectMultiScale(gray, 1.1, 3)
        count = len(vehicles)
        print(f"Fallback detection found {count} vehicles")
        return count
    except Exception as e:
        print(f"Error in fallback detection: {e}")
        return 0

@app.route('/start_auto_control', methods=['POST'])
@login_required
def start_auto_control():
    try:
        data = request.json
        vehicle_counts = data.get('vehicle_counts', {})
        
        # Check if we have counts for all directions
        required_directions = ['north', 'south', 'east', 'west']
        missing_directions = [dir for dir in required_directions if dir not in vehicle_counts or vehicle_counts[dir] is None]
        
        if missing_directions:
            return jsonify({
                'status': 'error',
                'message': f'Missing traffic data for directions: {", ".join(missing_directions)}'
            }), 400

        # Store counts in the database
        timestamp = datetime.utcnow()
        for direction, count in vehicle_counts.items():
            vehicle_count = VehicleCount(
                timestamp=timestamp,
                direction=direction,
                count=count
            )
            db.session.add(vehicle_count)
        
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"Database error: {e}")
        
        # Find direction with highest traffic
        max_direction = max(vehicle_counts.items(), key=lambda x: x[1])[0]
        max_count = vehicle_counts[max_direction]
        
        print(f"Max direction: {max_direction} with count: {max_count}")
        
        # Update signal status based on traffic density
        global signal_status, wait_times
        for direction in signal_status:
            if direction == max_direction:
                signal_status[direction] = 'green'
                wait_times[direction] = 0
                print(f"Setting {direction} to green")
            else:
                signal_status[direction] = 'red'
                # Calculate wait time based on traffic difference
                count_diff = max_count - vehicle_counts[direction]
                wait_times[direction] = min(60, max(30, count_diff * 5))
                print(f"Setting {direction} to red with wait time {wait_times[direction]}")
        
        print("Final signal status:", signal_status)
        print("Final wait times:", wait_times)
        
        response_data = {
            'status': 'success',
            'message': 'Automatic control started successfully',
            'signal_status': signal_status,
            'wait_times': wait_times,
            'max_direction': max_direction,
            'traffic_counts': vehicle_counts
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in start_auto_control: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/get_total_vehicles', methods=['GET'])
def get_total_vehicles():
    try:
        # Get the sum of all vehicle counts from the database
        total = db.session.query(db.func.sum(VehicleCount.count)).scalar() or 0
        return jsonify({'total_vehicles': total})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/violation-detection')
@login_required
def violation_detection():
    # Get results from session if they exist
    results = session.pop('detection_results', None)
    show_results = request.args.get('show_results', False)
    
    if show_results and results:
        print("Displaying results:", results)
        return render_template('violation_detection.html', results=results)
    return render_template('violation_detection.html')

@app.route('/upload-video', methods=['POST'])
@login_required
def upload_video():
    if 'video' not in request.files:
        flash('No video file uploaded')
        return redirect(url_for('violation_detection'))
    
    video = request.files['video']
    if video.filename == '':
        flash('No selected file')
        return redirect(url_for('violation_detection'))
    
    if video and allowed_file(video.filename):
        # Generate unique filename
        filename = f"{uuid.uuid4()}.{video.filename.rsplit('.', 1)[1].lower()}"
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        video.save(video_path)
        
        # Process video
        results = process_video(video_path)
        
        if results:
            # Save results to database
            violation_record = ViolationRecord(
                video_path=video_path,
                processed_path=results['processed_video'],
                total_bikers=results['total_bikers'],
                no_helmet=results['no_helmet'],
                using_phone=results['using_phone'],
                compliant=results['compliant']
            )
            db.session.add(violation_record)
            db.session.commit()
            
            return render_template('violation_detection.html', results=results)
    
    flash('Error processing video')
    return redirect(url_for('violation_detection'))

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def process_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Failed to open video file")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video writer with H.264 codec
        output_filename = f"processed_{os.path.basename(video_path)}"
        temp_output_path = os.path.join(PROCESSED_FOLDER, f"temp_{output_filename}")
        final_output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        
        # First write to a temporary file using MJPG codec
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
        
        # Initialize counters and tracking variables
        total_bikers = 0  # Unique motorcycles counted
        no_helmet = 0
        using_phone = 0
        frame_count = 0
        
        # Tracking variables
        tracked_motorcycles = {}  # Dictionary to store tracked motorcycles with their last seen frame
        motorcycle_violations = {}  # Dictionary to store violations for each tracked motorcycle
        iou_threshold = 0.5  # IoU threshold for matching
        next_bike_id = 1  # Counter for assigning unique IDs
        
        # Font settings for text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            # Process every 2nd frame for better accuracy
            if frame_count % 2 != 0:
                continue
            
            # Detect motorcycles using vehicle_model
            vehicle_results = vehicle_model(frame)
            current_detections = []
            active_tracks = set()  # Keep track of which IDs are seen in current frame
            
            # Process current frame detections
            for vehicle_det in vehicle_results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = vehicle_det.cpu().numpy()
                
                if cls == 3 and conf > 0.35:  # motorcycle with good confidence
                    current_box = [x1, y1, x2, y2]
                    matched_id = None
                    best_iou = 0
                    
                    # Try to match with existing tracked motorcycles
                    for bike_id, (tracked_box, last_seen, _) in tracked_motorcycles.items():
                        if frame_count - last_seen < fps * 3:  # Only match with recently seen bikes (within 3 seconds)
                            iou = calculate_iou(current_box, tracked_box)
                            if iou > iou_threshold and iou > best_iou:
                                matched_id = bike_id
                                best_iou = iou
                    
                    if matched_id is not None:
                        # Update existing track
                        tracked_motorcycles[matched_id] = (current_box, frame_count, conf)
                        active_tracks.add(matched_id)
                        bike_id = matched_id
                    else:
                        # New motorcycle detected
                        bike_id = next_bike_id
                        next_bike_id += 1
                        tracked_motorcycles[bike_id] = (current_box, frame_count, conf)
                        motorcycle_violations[bike_id] = {'no_helmet': False, 'phone': False}
                        active_tracks.add(bike_id)
                        total_bikers += 1
                    
                    # Draw red box around motorcycle
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    # Add motorcycle ID and confidence
                    text = f"Bike #{bike_id} ({conf:.2f})"
                    cv2.putText(frame, text, 
                              (int(x1), int(y1) - 10),
                              font, font_scale, (0, 0, 255), font_thickness)
                    
                    # First detect all persons in the frame for driver identification
                    person_results = violation_model(frame)
                    driver_detected = False
                    driver_head_box = None
                    
                    # Calculate motorcycle regions
                    bike_front = x1 + (x2 - x1) * 0.4  # Front 40% of motorcycle
                    bike_top = y1
                    bike_height = y2 - y1
                    
                    # Look for person detections that could be the driver
                    for person_det in person_results.xyxy[0]:
                        px1, py1, px2, py2, pconf, pcls = person_det.cpu().numpy()
                        
                        if pcls == 0 and pconf > 0.4:  # If it's a person with good confidence
                            # Check if person is in the front part of the motorcycle
                            person_center_x = (px1 + px2) / 2
                            person_bottom = py2
                            
                            # Person should be in front part of bike and properly positioned
                            if (person_center_x < bike_front and 
                                person_bottom > y1 + bike_height * 0.3 and 
                                py1 < y1 + bike_height * 0.7):
                                
                                # Define head region as top 30% of detected person
                                head_y1 = py1
                                head_y2 = py1 + (py2 - py1) * 0.3
                                head_x1 = px1
                                head_x2 = px2
                                
                                driver_detected = True
                                driver_head_box = [head_x1, head_y1, head_x2, head_y2]
                                
                                # Draw driver detection
                                cv2.rectangle(frame, (int(px1), int(py1)), (int(px2), int(py2)), 
                                            (0, 255, 0), 1)
                                cv2.putText(frame, "Driver", (int(px1), int(py1) - 5),
                                          font, font_scale, (0, 255, 0), font_thickness)
                                break
                    
                    # If driver detected, analyze head region for helmet and phone
                    if driver_detected and driver_head_box:
                        head_x1, head_y1, head_x2, head_y2 = driver_head_box
                        
                        # Expand head region for better detection
                        head_width = head_x2 - head_x1
                        head_height = head_y2 - head_y1
                        
                        # Make detection region larger
                        expanded_head_box = [
                            max(0, head_x1 - head_width * 0.5),
                            max(0, head_y1 - head_height * 0.5),
                            min(frame.shape[1], head_x2 + head_width * 0.5),
                            min(frame.shape[0], head_y2 + head_height * 0.5)
                        ]
                        
                        # Extract region for helmet detection
                        head_region = frame[
                            int(expanded_head_box[1]):int(expanded_head_box[3]),
                            int(expanded_head_box[0]):int(expanded_head_box[2])
                        ]
                        
                        if head_region.size > 0:
                            print(f"\nProcessing frame {frame_count} for bike #{bike_id}")
                            print(f"Head region size: {head_region.shape}")
                            # Use Roboflow for helmet detection
                            has_helmet, confidence = detect_helmet_using_roboflow(head_region)
                            print(f"🎯 Frame {frame_count}, Bike #{bike_id}: Helmet={has_helmet}, Confidence={confidence}\n")
                            # Draw detection box
                            color = (0, 255, 0) if has_helmet else (0, 0, 255)
                            cv2.rectangle(frame, 
                                        (int(expanded_head_box[0]), int(expanded_head_box[1])),
                                        (int(expanded_head_box[2]), int(expanded_head_box[3])),
                                        color, 2)
                            
                            # Update violations and draw indicators
                            if has_helmet:
                                cv2.putText(frame, f"Helmet Detected ({confidence:.2f})", 
                                          (int(expanded_head_box[0]), int(expanded_head_box[1]) - 10),
                                          font, font_scale, (0, 255, 0), font_thickness)
                            else:
                                if not motorcycle_violations[bike_id]['no_helmet']:
                                    motorcycle_violations[bike_id]['no_helmet'] = True
                                    no_helmet += 1
                                cv2.putText(frame, "No Helmet!", 
                                          (int(expanded_head_box[0]), int(expanded_head_box[1]) - 10),
                                          font, font_scale, (0, 0, 255), font_thickness)
                    else:
                        # If no driver detected, show searching region
                        search_x1 = x1 + (x2 - x1) * 0.3
                        search_x2 = x1 + (x2 - x1) * 0.7
                        search_y1 = y1
                        search_y2 = y1 + (y2 - y1) * 0.4
                        cv2.rectangle(frame, (int(search_x1), int(search_y1)), 
                                    (int(search_x2), int(search_y2)), 
                                    (128, 128, 128), 1)
                        cv2.putText(frame, "Searching for driver", 
                                  (int(search_x1), int(search_y1) - 5),
                                  font, font_scale, (128, 128, 128), font_thickness)
            
            # Clean up old tracks (remove tracks not seen for more than 3 seconds)
            current_time = frame_count
            tracked_motorcycles = {
                bike_id: (box, last_seen, conf) 
                for bike_id, (box, last_seen, conf) in tracked_motorcycles.items()
                if current_time - last_seen < fps * 3
            }
            
            # Add frame statistics
            stats_text = f"Unique Bikes: {total_bikers} | Drivers without Helmet: {no_helmet} | Phone Usage: {using_phone}"
            cv2.putText(frame, stats_text, (10, 30), font, font_scale, (255, 255, 255), font_thickness)
            
            # Write processed frame
            out.write(frame)
        
        # Release resources
        cap.release()
        out.release()
        
        # Convert the video to H.264 MP4 using ffmpeg
        try:
            import subprocess
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', temp_output_path,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p',
                final_output_path
            ]
            subprocess.run(ffmpeg_cmd, check=True)
            # Remove temporary file
            os.remove(temp_output_path)
        except Exception as e:
            print(f"Error converting video: {e}")
            # If conversion fails, try to use the temporary file
            if os.path.exists(temp_output_path):
                os.rename(temp_output_path, final_output_path)
        
        # Calculate compliant riders
        compliant = total_bikers - (no_helmet + using_phone)
        
        return {
            'total_bikers': total_bikers,
            'no_helmet': no_helmet,
            'using_phone': using_phone,
            'compliant': compliant,
            'processed_video': f"processed/{output_filename}"
        }
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class ViolationRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    video_path = db.Column(db.String(255), nullable=False)
    processed_path = db.Column(db.String(255), nullable=False)
    total_bikers = db.Column(db.Integer, default=0)
    no_helmet = db.Column(db.Integer, default=0)
    using_phone = db.Column(db.Integer, default=0)
    compliant = db.Column(db.Integer, default=0)

@app.route('/static/processed/<path:filename>')
def serve_video(filename):
    try:
        video_path = os.path.join(app.root_path, 'static', 'processed', filename)
        
        # Check if file exists
        if not os.path.exists(video_path):
            return "Video not found", 404

        # Get file size
        file_size = os.path.getsize(video_path)
        
        # Handle range requests
        range_header = request.headers.get('Range', None)
        if range_header:
            try:
                byte1, byte2 = 0, None
                match = re.search('bytes=(\d+)-(\d*)', range_header)
                if not match:
                    return "Invalid range header", 400
                
                groups = match.groups()
                if groups[0]:
                    byte1 = int(groups[0])
                if groups[1]:
                    byte2 = int(groups[1])
                
                if byte2 is None:
                    byte2 = min(byte1 + (10 * 1024 * 1024), file_size - 1)  # Limit chunk size to 10MB
                
                length = byte2 - byte1 + 1
                
                resp = Response(
                    partial_video_stream(video_path, byte1, byte2),
                    206,
                    mimetype='video/mp4',
                    direct_passthrough=True
                )
                
                resp.headers.add('Content-Range', f'bytes {byte1}-{byte2}/{file_size}')
                resp.headers.add('Accept-Ranges', 'bytes')
                resp.headers.add('Content-Length', str(length))
                resp.headers.add('Cache-Control', 'no-cache')
                return resp
            except Exception as e:
                print(f"Error processing range request: {e}")
                return str(e), 500

        # If no range header or error in range processing, serve entire file
        response = send_from_directory(
            os.path.join(app.root_path, 'static', 'processed'),
            filename,
            mimetype='video/mp4'
        )
        response.headers.add('Accept-Ranges', 'bytes')
        response.headers.add('Content-Length', str(file_size))
        response.headers.add('Cache-Control', 'no-cache')
        return response

    except Exception as e:
        print(f"Error serving video: {e}")
        return str(e), 500

def partial_video_stream(video_path, byte1=None, byte2=None):
    """Generator to stream video file in chunks."""
    chunk_size = 10 * 1024 * 1024  # 10MB chunks
    
    with open(video_path, 'rb') as video:
        if byte1 is not None:
            video.seek(byte1)
        while True:
            if byte2 is not None:
                chunk = video.read(min(chunk_size, byte2 - byte1 + 1))
            else:
                chunk = video.read(chunk_size)
            if not chunk:
                break
            yield chunk

def is_hand_near_ear(hand_landmarks, frame_height):
    """Check if hand position indicates phone usage"""
    if hand_landmarks:
        # Get index finger tip and thumb tip positions
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        
        # Check if hand is raised (near ear level)
        is_raised = wrist.y < 0.6  # Hand is in upper 60% of body
        
        # Check if fingers are close together (holding something)
        fingers_close = abs(index_tip.x - thumb_tip.x) < 0.1
        
        return is_raised and fingers_close
    return False

def process_image(image_array):
    """
    Process a single image to detect helmet and phone usage.
    Returns detection results and processed image.
    """
    try:
        # Convert image array to PIL Image if needed
        if isinstance(image_array, np.ndarray):
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_array)
        else:
            pil_image = image_array

        # Convert back to OpenCV format for drawing
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Initialize results
        has_helmet = False
        using_phone = False
        person_detected = False
        helmet_confidence = 0
        
        # Detect person first
        person_results = violation_model(frame)
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Initialize MediaPipe Hands with improved settings
        with mp_hands.Hands(
            static_image_mode=False,  # Set to False for better tracking
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            # Process the frame for hand detection
            hand_results = hands.process(frame_rgb)
        
        for person_det in person_results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = person_det.cpu().numpy()
            
            if cls == 0 and conf > 0.4:  # person detection
                person_detected = True
                # Draw person detection
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                            (0, 255, 0), 2)
                
                # Define regions for detection
                person_height = y2 - y1
                person_width = x2 - x1
                
                # Create a larger region for helmet detection (top half of person plus margin)
                margin = person_width * 0.3  # 30% margin
                head_y1 = int(max(0, y1 - margin))
                head_y2 = int(y1 + person_height * 0.6)  # Include upper 60% of person
                head_x1 = int(max(0, x1 - margin))
                head_x2 = int(min(frame.shape[1], x2 + margin))
                
                # Extract region for helmet detection
                head_region = frame[head_y1:head_y2, head_x1:head_x2]
                
                # Only process if region is large enough
                min_size = 100  # Minimum dimension size
                if head_region.size > 0 and head_region.shape[0] > min_size and head_region.shape[1] > min_size:
                    print(f"\nProcessing head region with size: {head_region.shape}")
                    # Use Roboflow for helmet detection
                    has_helmet, helmet_confidence = detect_helmet_using_roboflow(head_region)
                    
                    # Draw detection box
                    color = (0, 255, 0) if has_helmet else (0, 0, 255)
                    cv2.rectangle(frame, (head_x1, head_y1), (head_x2, head_y2), color, 2)
                    
                    if has_helmet:
                        cv2.putText(frame, f"Helmet Detected ({helmet_confidence:.2f})", 
                                  (head_x1, head_y1 - 10),
                                  font, font_scale, (0, 255, 0), font_thickness)
                    else:
                        cv2.putText(frame, "No Helmet!", 
                                  (head_x1, head_y1 - 10),
                                  font, font_scale, (0, 0, 255), font_thickness)
                
                # Check for phone usage using hand pose
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        # Get hand position relative to the person's bounding box
                        hand_x = min([lm.x * frame.shape[1] for lm in hand_landmarks.landmark])
                        hand_y = min([lm.y * frame.shape[0] for lm in hand_landmarks.landmark])
                        
                        # Check if hand is near the head region
                        if (head_x1 <= hand_x <= head_x2 and 
                            head_y1 <= hand_y <= head_y2):
                            
                            # Draw hand landmarks
                            mp_drawing.draw_landmarks(
                                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                            )
                            
                            # Check hand gesture for phone usage
                            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                            
                            # Calculate distance between thumb and index finger
                            distance = np.sqrt(
                                (thumb_tip.x - index_tip.x)**2 + 
                                (thumb_tip.y - index_tip.y)**2
                            )
                            
                            # If fingers are close together near head, likely holding phone
                            if distance < 0.1:
                                using_phone = True
                                cv2.putText(frame, "Phone Usage Detected!", 
                                          (int(hand_x), int(hand_y) - 10),
                                          font, font_scale, (0, 165, 255), font_thickness)
        
        if not person_detected:
            cv2.putText(frame, "No person detected", 
                      (10, 30), font, font_scale, (0, 0, 255), font_thickness)
        
        # Save processed image
        output_filename = f"processed_{uuid.uuid4()}.jpg"
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        cv2.imwrite(output_path, frame)
        
        return {
            'has_helmet': has_helmet,
            'helmet_confidence': helmet_confidence,
            'using_phone': using_phone,
            'person_detected': person_detected,
            'processed_image': f"processed/{output_filename}"
        }
    
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/upload-photo', methods=['POST'])
@login_required
def upload_photo():
    if 'photo' not in request.files:
        flash('No photo uploaded')
        return redirect(url_for('violation_detection'))
    
    photo = request.files['photo']
    if photo.filename == '':
        flash('No selected file')
        return redirect(url_for('violation_detection'))
    
    if photo and allowed_file(photo.filename):
        try:
            # Read image
            image = Image.open(photo)
            # Process image
            results = process_image(image)
            
            if results:
                return render_template('violation_detection.html', results=results)
            else:
                flash('Error processing image')
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
    else:
        flash('Invalid file type')
    
    return redirect(url_for('violation_detection'))

@app.route('/webcam-photo', methods=['POST'])
@login_required
def webcam_photo():
    if 'photo' not in request.files:
        return jsonify({'error': 'No photo uploaded'}), 400
    
    photo = request.files['photo']
    if photo.filename == '':
        return jsonify({'error': 'No photo data'}), 400
    
    try:
        print("Processing webcam photo...")
        # Read image from webcam capture
        image_array = cv2.imdecode(
            np.frombuffer(photo.read(), np.uint8),
            cv2.IMREAD_COLOR
        )
        
        if image_array is None:
            print("Failed to decode image")
            return jsonify({'error': 'Failed to decode image'}), 400
            
        print(f"Image shape: {image_array.shape}")
        
        # Process image
        print("Processing image through detection model...")
        results = process_image(image_array)
        print(f"Detection results: {results}")
        
        if results:
            # Return the results in the template without redirecting
            return render_template('violation_detection.html', results=results)
        else:
            print("No results from process_image")
            return jsonify({'error': 'Error processing image'}), 500
    
    except Exception as e:
        print(f"Error in webcam_photo: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Add new helper function for helmet detection
def detect_helmet_using_roboflow(frame):
    """
    Detect helmets using Roboflow's pre-trained model
    """
    try:
        print("Starting helmet detection with Roboflow...")
        
        # Convert frame to base64 string
        _, buffer = cv2.imencode('.jpg', frame)
        img_str = base64.b64encode(buffer).decode('utf-8')
        print("Image converted to base64 successfully")
        
        # Get predictions from Roboflow
        print("Sending request to Roboflow API...")
        result = ROBOFLOW_CLIENT.infer(
            img_str,
            model_id="helmet-detection-ar0n2/1"
        )
        print(f"Received response from Roboflow: {result}")
        
        # Process predictions
        if isinstance(result, dict) and 'predictions' in result:
            print(f"Found {len(result['predictions'])} predictions")
            for prediction in result['predictions']:
                confidence = float(prediction.get('confidence', 0))
                class_name = prediction.get('class', '')
                print(f"Prediction: class={class_name}, confidence={confidence}")
                
                # Check for "With Helmet" class name
                if class_name == "With Helmet" and confidence > 0.5:
                    print(f"✅ Helmet detected with confidence: {confidence}")
                    return True, confidence
                # Also check for "helmet" in case of model version differences
                elif class_name.lower() == "helmet" and confidence > 0.5:
                    print(f"✅ Helmet detected with confidence: {confidence}")
                    return True, confidence
                    
            print("❌ No helmet detected in predictions")
        else:
            print(f"❌ Unexpected result format: {result}")
        
        return False, 0
    except Exception as e:
        print(f"❌ Error in Roboflow detection: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, 0

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    # Run with HTTPS using the generated certificates
    app.run(debug=True, 
            port=5001, 
            host='0.0.0.0',
            ssl_context=('cert.pem', 'key.pem')) 