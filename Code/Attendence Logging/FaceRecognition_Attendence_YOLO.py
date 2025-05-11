from collections import defaultdict
import imutils
import pickle
import time
import cv2
import os
import numpy as np
import torch
from ultralytics import YOLO

# Paths to model files and recognizer
embeddingModel = r"D:\PythonProject\Face_Recognition_DL\openface.nn4.small2.v1.t7"
recognizerFile = r"D:\PythonProject\Face_Recognition_DL\Output_Models\recognizer_Openface06.pickle"
labelEncFile = r"D:\PythonProject\Face_Recognition_DL\Output_Models\LE_Openface06.pickle"
conf = 0.8  # Confidence threshold

# Load the YOLOv8 face detection model
print("[INFO] Loading YOLOv8 face detector...")
yolo_model_path = r"D:\PythonProject\Face_Recognition_DL\model\yolov8n-face.pt"
yolo = YOLO(yolo_model_path)

# Load face recognizer and label encoder
print("[INFO] Loading face recognizer and label encoder...")
embedder = cv2.dnn.readNetFromTorch(embeddingModel)
recognizer = pickle.loads(open(recognizerFile, "rb").read())
le = pickle.loads(open(labelEncFile, "rb").read())

# Start video stream
print("[INFO] Starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(2.0)

# List to store recognized names
recognized_names = []

while True:
    frame_counter = 0
    scores = defaultdict(float)

    while frame_counter < 100:
        ret, frame = cam.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=600)

        # YOLO face detection
        results = yolo.predict(frame, conf=0.4, verbose=False)
        boxes = results[0].boxes if results else []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face = frame[y1:y2, x1:x2]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            # Prepare face for embedding
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0),
                                             swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # Recognize face
            preds = recognizer.predict_proba([vec.flatten()])[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            if proba * 100 < 50:
                name = "Unknown"
            else:
                scores[name] += proba

            # Draw results
            text = f"{name} : {proba * 100:.2f}%"
            y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(frame, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        cv2.imshow("Frame", frame)
        frame_counter += 1

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    if scores:
        name = max(scores.items(), key=lambda x: x[1])[0]
        print(f"Attendance for person: {name}")
        recognized_names.append(name)
    else:
        print("No valid recognition for this set of frames.")

    print("Press 'c' to continue recognizing another person, or 'q' to quit and generate report.")
    action = cv2.waitKey(0) & 0xFF

    if action == ord("q"):
        break
    elif action == ord("c"):
        print("[INFO] Continuing for next person...")

# Cleanup
cam.release()
cv2.destroyAllWindows()

# Save recognized names
with open("recognized_names01.txt", "w") as f:
    for item in recognized_names:
        f.write(f"{item}\n")

print("[INFO] Names saved to recognized_names01.txt")
