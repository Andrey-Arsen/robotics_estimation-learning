# 🚗 YOLOv8 + Kalman Filter Object Tracking

This project demonstrates real-time object detection using a custom-trained YOLOv8 model and motion prediction using a Kalman filter. It can predict the object's trajectory even when it's temporarily occluded (e.g., a car going under a bridge).

## 📌 Features

- ✅ Custom YOLOv8 object detection (trained with Roboflow)
- ✅ Real-time webcam input
- ✅ Kalman filter tracking with velocity and acceleration
- ✅ Continues prediction even if the object disappears
- ✅ Trajectory visualization

---

## 📁 Project Structure

```bash
.
├── train_yolo_colab.ipynb   # Google Colab notebook for training YOLOv8
├── track_with_kalman.py     # Object detection and tracking with Kalman filter
├── best_my_car.pt           # Trained YOLOv8 weights
├── README.md                # Project documentation
```

---

## 🚀 How to Use

### 1. Train YOLOv8 in Google Colab

```python
from google.colab import userdata
from roboflow import Roboflow

ROBOFLOW_API_KEY = userdata.get('ROBOFLOW_API_KEY')
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace("arsen-flytr").project("my_car-pjbkr")
version = project.version(1)
dataset = version.download("yolov8")
```

Then train the model:

```bash
!yolo task=detect mode=train model=yolo11s.pt data={dataset.location}/data.yaml epochs=50 imgsz=640 plots=True
```

---

### 2. Run Kalman Tracker Locally

**Install dependencies:**

```bash
pip install ultralytics opencv-python filterpy
```

**Run the script:**

```bash
python track_with_kalman.py
```

The script will:

- Open the webcam
- Detect the object using the trained YOLOv8 model
- Use a Kalman filter to predict and smooth the object's position
- Draw a bounding box and predicted points
- Keep tracking even if the object disappears for a while

---

## 📷 Example Output

![Kalman Tracking in Action](tracking.mp4)

---

## 🧠 Tech Stack

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [FilterPy – Kalman Filter](https://github.com/rlabbe/filterpy)
- [OpenCV](https://opencv.org/) – Real-time video processing


