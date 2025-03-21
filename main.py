import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import time
import pyttsx3
import os

# Check if MPS (Metal Performance Shader) is available
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load YOLOv8 model for person detection
model = YOLO("yolov8n.pt").to(device)

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Load InsightFace (ArcFace) for face recognition
app = FaceAnalysis(name="buffalo_l")  # Uses a high-accuracy model
app.prepare(ctx_id=0, det_size=(640, 640))
engine = pyttsx3.init()

# Known faces database (embeddings) loaded from subdirectories in the 'faces' folder
known_face_encodings = []
known_face_names = []

def get_face_embedding(image_path):
    image = cv2.imread(image_path)
    faces = app.get(image)
    return faces[0].embedding if faces else None

faces_dir = "faces"
for person_name in os.listdir(faces_dir):
    person_path = os.path.join(faces_dir, person_name)
    if os.path.isdir(person_path):
        embeddings = []
        for filename in os.listdir(person_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(person_path, filename)
                embedding = get_face_embedding(image_path)
                if embedding is not None:
                    embeddings.append(embedding)
        if embeddings:
            # Compute the average embedding for this person
            avg_embedding = np.mean(embeddings, axis=0)
            known_face_encodings.append(avg_embedding)
            known_face_names.append(person_name.capitalize())

# Open webcam
cap = cv2.VideoCapture(0)  # 0 for webcam

# Set video resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Store tracked people
tracked_people = defaultdict(int)
greeted = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w = frame.shape[:2]  # Get original frame size
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize frame to 640x640 for YOLO inference
    resized_frame = cv2.resize(rgb_frame, (640, 640))

    # Convert to tensor
    frame_tensor = torch.from_numpy(resized_frame).float().to(device)
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0  # Normalize

    # Run YOLOv8 inference on MPS
    results = model(frame_tensor)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        if cls == 0 and conf > 0.5:  # Only detect people
            # Scale bounding boxes back to original size
            x1 = int(x1 * orig_w / 640)
            y1 = int(y1 * orig_h / 640)
            x2 = int(x2 * orig_w / 640)
            y2 = int(y2 * orig_h / 640)

            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

    # Update tracker
    tracked_objects = tracker.update_tracks(detections, frame=frame)

    for obj in tracked_objects:
        if not obj.is_confirmed():
            continue

        track_id = obj.track_id
        bbox = obj.to_ltrb()
        x1, y1, x2, y2 = map(int, bbox)

        # Extract face from detected person
        face_frame = frame[y1:y2, x1:x2]
        face_name = "Unknown"

        if face_frame.size > 0:
            faces = app.get(face_frame)
            if faces:
                face_embedding = faces[0].embedding

                # Compute similarity with known faces
                similarities = cosine_similarity([face_embedding], known_face_encodings)
                best_match = np.argmax(similarities)
                if similarities[0][best_match] > 0.6:  # Threshold for recognition
                    face_name = known_face_names[best_match]

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track_id} | {face_name}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        tracked_people[track_id] += 1

        if face_name != "Unknown" and track_id not in greeted:
            greeting_text = f"Hello, {face_name}!"
            # Display greeting below the bounding box
            cv2.putText(frame, greeting_text, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Ensure the screenshots directory exists
            screenshot_dir = "screenshots"
            if not os.path.exists(screenshot_dir):
                os.makedirs(screenshot_dir)

            # Save a screenshot of the current frame with a timestamp
            screenshot_filename = os.path.join(screenshot_dir, f"greeting_{face_name}_{time.strftime('%Y%m%d_%H%M%S')}.png")
            cv2.imwrite(screenshot_filename, frame)

            # Play audio greeting
            greeting_audio = f"Hello {face_name}, smile for a picture!"
            engine.say(greeting_audio)
            engine.runAndWait()

            greeted[track_id] = True

    # Display person count
    cv2.putText(frame, f'Persons Count: {len(tracked_people)}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Face ID + Person Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
