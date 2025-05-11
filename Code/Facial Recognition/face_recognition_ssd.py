import numpy as np
import imutils
import pickle
import time
import cv2
import os

# Paths to model files and recognizer
embeddingModel = r"D:\PythonProject\Face_Recognition_DL\openface.nn4.small2.v1.t7"
recognizerFile = r"D:\PythonProject\Face_Recognition_DL\Output_Models\recognizer_Openface02.pickle"
labelEncFile = r"D:\PythonProject\Face_Recognition_DL\Output_Models\LE_Openface02.pickle"
conf = 0.8  # Confidence threshold

# Paths to the prototxt file and pre-trained model
prototxt_path = r"D:\PythonProject\Face_Recognition_DL\model\deploy.prototxt.txt"
model_path = r"D:\PythonProject\Face_Recognition_DL\model\res10_300x300_ssd_iter_140000.caffemodel"
confidence_threshold = 0.5

# Load the DNN model for face detection
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Load the face recognizer
print("Loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

recognizer = pickle.loads(open(recognizerFile, "rb").read())
le = pickle.loads(open(labelEncFile, "rb").read())

# Initialize camera stream
print("[INFO] starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    _, frame = cam.read()
    frame = imutils.resize(frame, width=600)
    
    # Grab the frame dimensions and convert it to a blob for face detection
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # Filter out detections by confidence
        if confidence < confidence_threshold:
            continue

        # Compute the (x, y)-coordinates of the bounding box for the face
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        face = frame[startY:endY, startX:endX]
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
        if proba * 100 < 75:
            name = "Unknown"

        # Draw rectangle and put text for recognition
        text = "{} : {:.2f}%".format(name, proba * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 1)
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        if name != "Unknown":
            print(f"Recognized: {name}, Probability: {proba * 100:.2f}%")
        else:
            print(f"Unknown face detected with probability: {proba * 100:.2f}%")

    # Show the output frame
    cv2.imshow("Frame", frame)

    if cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1:  # Close button clicked
        break

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key to quit
        break

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
