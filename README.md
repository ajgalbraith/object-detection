# 🧠 Real-Time Person Tracking and Face Recognition

This project combines **YOLOv8** (for real-time person detection), **DeepSORT** (for multi-object tracking), and **InsightFace** (for face recognition) to identify and track known individuals in a live webcam feed.

## 🚀 Features

- Real-time webcam object detection (people only)
- Unique ID assignment per person using DeepSORT
- Face recognition with preloaded known face embeddings
- GPU acceleration using **Apple M1/M2 Metal backend** (`mps`)

---

## 📦 Requirements

Python 3.9+ (tested on macOS)

### Recommended: Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

---

### 📥 Installation

1. Install dependencies

```bash
pip install -r requirements.txt
```
If you encounter any issues, install manually:

```bash
pip install opencv-python torch torchvision numpy ultralytics deep_sort_realtime insightface scikit-learn
```
💡 If you’re on a Mac with Apple Silicon, PyTorch should automatically use the mps backend.

2. Download the YOLOv8 model

This project uses the nano version of YOLOv8 for fast inference.

```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```
Or use curl if wget isn’t installed:

```bash
curl -L -o yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```
🔁 You can swap this model with any YOLOv8 .pt model (e.g., yolov8s.pt, yolov8m.pt).

⸻

### 🖼️ Add Known Faces

Place reference face images in a subfolder inside a `faces/` directory, named after each person:

```
faces/
├── james/
│   └── james1.jpg
├── david/
│   └── david1.png
```

Each subfolder name (e.g., "james") will be used as the label when recognizing that person's face.

---

### ▶️ Run the App

```bash
python main.py
```

Press `Q` to exit the window.

- When a known face is recognized, the app:
  - Greets the person using text-to-speech
  - Saves a screenshot in the `screenshots/` directory with a timestamp
  - Displays bounding boxes and labels with tracking ID and recognized name

---

### 📁 Output

When a known face is detected and greeted, a screenshot is saved in the `screenshots/` folder with a filename like:

```
greeting_James_20250321_183745.png
```

---

### 📌 Notes
- The model uses a cosine similarity threshold of **0.6** to match unknown faces with known identities.
- Each tracked person is assigned a unique ID using DeepSORT.
- The `tracked_people` dictionary keeps count of how many times each person is seen.
- The system uses text-to-speech (TTS) to greet recognized individuals.
- On first recognition, a screenshot is saved in the `screenshots/` folder.

---

### 📄 License

MIT License
