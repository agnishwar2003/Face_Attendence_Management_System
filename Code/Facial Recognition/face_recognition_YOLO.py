import numpy as np
import imutils
import pickle
import time
import cv2
import os
import torch
from ultralytics import YOLO

# Paths to model files and recognizer
embeddingModel = r"D:\PythonProject\Face_Recognition_DL\openface.nn4.small2.v1.t7"
recognizerFile = r"D:\PythonProject\Face_Recognition_DL\Output_Models\recognizer_Openface06.pickle"
labelEncFile = r"D:\PythonProject\Face_Recognition_DL\Output_Models\LE_Openface06.pickle"
conf = 0.8  # Confidence threshold

# Load the YOLOv8 model
print("[INFO] loading YOLO model...")
yolo_model_path = r"D:\PythonProject\Face_Recognition_DL\model\yolov8n-face.pt"
yolo = YOLO(yolo_model_path)

# Load the face recognizer
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embeddingModel)
recognizer = pickle.loads(open(recognizerFile, "rb").read())
le = pickle.loads(open(labelEncFile, "rb").read())

# Start video stream
print("[INFO] starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    _, frame = cam.read()
    frame = imutils.resize(frame, width=600)

    # Perform YOLO-based face detection
    results = yolo.predict(frame, conf=0.4)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            face = frame[y1:y2, x1:x2]
            (fH, fW) = face.shape[:2]

            # Skip small face regions
            if fW < 20 or fH < 20:
                continue

            # Prepare the face ROI for embedding extraction
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # Predict probabilities using recognizer
            preds = recognizer.predict_proba([vec.flatten()])[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # Classify as "Unknown" if below confidence threshold
            if proba * 100 < 65:
                name = "Unknown"

            # Display the result
            text = "{} : {:.2f}%".format(name, proba * 100)
            y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(frame, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            if name != "Unknown":
                print(f"Recognized: {name}, Probability: {proba * 100:.2f}%")
            else:
                print(f"Unknown face detected with probability: {proba * 100:.2f}%")

    # Show the output frame
    cv2.imshow("Frame", frame)

    if cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1:
        break

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

# Cleanup
cam.release()
cv2.destroyAllWindows()
