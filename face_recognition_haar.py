import numpy as np
import imutils
import pickle
import time
import cv2
import os

# Paths to model files and recognizer
embeddingModel = r"D:\PythonProject\Face_Detection_DL\openface.nn4.small2.v1.t7"
recognizerFile = r"D:\PythonProject\Face_Detection_DL\output\recognizer.pickle"
labelEncFile = r"D:\PythonProject\Face_Detection_DL\output\le.pickle"
conf = 0.8  # Confidence threshold

# Load Haar Cascade for face detection
print("Loading face detector...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load the face recognizer
print("Loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

recognizer = pickle.loads(open(recognizerFile, "rb").read())
le = pickle.loads(open(labelEncFile, "rb").read())

# Start face recognition
print("Starting face recognition...")
cam = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    _, frame = cam.read()
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for Haar Cascade

    # Detect faces using Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        print(f"Detections found: {len(faces)} face(s) detected.")
    else:
        print("No faces detected.")

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        (fH, fW) = face.shape[:2]

        # Skip small faces that can't be detected properly
        if fW < 20 or fH < 20:
            continue

        # Prepare the face for recognition
        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)

        # Perform forward pass on the embedder to get the feature vector
        embedder.setInput(faceBlob)
        vec = embedder.forward()

        # Perform recognition with the pre-trained recognizer
        preds = recognizer.predict_proba([vec.flatten()])[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]

        # If the recognition probability is low, classify as "Unknown"
        if proba * 100 < 80:
            name = "Unknown"

        # Draw rectangle and put text for recognition
        text = "{} : {:.2f}%".format(name, proba * 100)
        y = y - 10 if y - 10 > 10 else y + 10
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        if name != "Unknown":
            print(f"Recognized: {name}, Probability: {proba * 100:.2f}%")
        else:
            print(f"Unknown face detected with probability: {proba * 100:.2f}%")

    # Show the frame with detected faces and recognized names
    cv2.imshow("Frame", frame)

    if cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1:  # Close button clicked
        break

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key to quit
        break

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
