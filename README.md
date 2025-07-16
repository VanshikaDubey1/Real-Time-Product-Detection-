# 🎯 YOLOv5 + OpenCV Real-Time Object Detection with Audio Alerts

This project performs real-time object detection from a webcam using **YOLOv5** and **OpenCV**, and plays custom audio alerts for specific detected objects (e.g. cell phone or person).

---

## 🚀 Features

- 🧠 **YOLOv5s Pretrained Model** (COCO dataset via PyTorch Hub)
- 🎥 **Live Webcam Detection**
- 🖼️ Bounding boxes with labels and confidence scores
- 📊 Frame-per-second (FPS) counter and object count overlay
- 🔊 **Custom sound alerts**:
  - `cellphone_alert.mp3` when a **cell phone** is detected
  - `person_alert.mp3` when a **person** is detected
- 💾 Output video saved as `output_yolo.avi`

---

## 📦 Requirements
- Python 3.7+
  OpenCV
  PyTorch
  yolov5s.pt model file
 - See [`requirements.txt`](./requirements.txt)

### Python Packages

pip install opencv-python torch torchvision torchaudio playsound

🧠 Model Info
Uses YOLOv5s from ultralytics/yolov5

Trained on COCO dataset (80 common object classes)
🛠️ Setup Instructions
Place the following files in the project folder:

yolo_opencv_webcam.py

cellphone_alert.mp3 – custom audio alert for phone detection

person_alert.mp3 – custom audio alert for person detection

Run the script:
python yolo_opencv_webcam.py
Press q to quit the detection window.
