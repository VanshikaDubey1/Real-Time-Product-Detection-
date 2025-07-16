# file: yolo_opencv_webcam.py

import cv2
import torch
import time
import os
from collections import Counter
from playsound import playsound

# Load pretrained YOLOv5s model from Ultralytics
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# Specify product-related object names from COCO
INTERESTED_CLASSES = {'bottle', 'cell phone', 'laptop', 'book', 'person'}
ALERT_SOUNDS = {
    'cell phone': 'cellphone_alert.mp3',
    'person': 'person_alert.mp3'
    # Add more class-specific sound files here if needed
}

# Open webcam stream
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Failed to open webcam. Trying camera index 1...")
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    raise IOError("❌ Cannot open webcam on index 0 or 1. Please check your camera.")

# Define video writer for saving output
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output_yolo.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))

prev_time = time.time()
total_frames = 0
alert_flags = {label: False for label in ALERT_SOUNDS}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()
    total_frames += 1

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)
    boxes = results.xyxy[0]  # (x1, y1, x2, y2, conf, cls)

    label_counter = Counter()
    tracked_objects = []
    current_frame_labels = set()

    for *xyxy, conf, cls in boxes:
        label = model.names[int(cls)]
        if label not in INTERESTED_CLASSES:
            continue

        conf_text = f'{conf:.2f}'
        label_counter[label] += 1
        tracked_objects.append(label)
        current_frame_labels.add(label)

        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {conf_text}', (int(xyxy[0]), int(xyxy[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for alert_label, sound_path in ALERT_SOUNDS.items():
        if alert_label in current_frame_labels and not alert_flags[alert_label] and os.path.exists(sound_path):
            playsound(sound_path)
            alert_flags[alert_label] = True
        elif alert_label not in current_frame_labels:
            alert_flags[alert_label] = False

    # Calculate and display FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time + 1e-6)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Display object counts
    y_offset = 50
    for label, count in label_counter.items():
        cv2.putText(frame, f'{label}: {count}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25

    # Display all tracked object names in one line
    product_list = ', '.join(sorted(set(tracked_objects)))
    cv2.putText(frame, f'Tracked: {product_list}', (10, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show frame and write to video
    cv2.imshow('YOLOv5 Webcam', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
