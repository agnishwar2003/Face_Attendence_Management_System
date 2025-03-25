# Openface

import numpy as np
import pickle
import time
import cv2
import torch
from ultralytics import YOLO
import imutils

# Paths to model files and recognizer
embeddingModel = r"D:\PythonProject\Face_Recognition_DL\openface.nn4.small2.v1.t7"
recognizerFile = r"D:\PythonProject\Face_Recognition_DL\output\recognizer_updated_Openface01.pickle"
labelEncFile = r"D:\PythonProject\Face_Recognition_DL\output\le_updated_Openface01.pickle"
conf_threshold = 0.8  # Confidence threshold

yolo_model_path = r"D:\PythonProject\Face_Recognition_DL\model\yolov8n-face.pt"

# Load YOLO model for face detection
print("[INFO] Loading YOLO model...")
yolo = YOLO(yolo_model_path)

# Load the face recognizer
print("[INFO] Loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embeddingModel)
recognizer = pickle.loads(open(recognizerFile, "rb").read())
le = pickle.loads(open(labelEncFile, "rb").read())

# Initialize camera stream
print("[INFO] Starting video stream...")
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
            
            # Skip small faces
            if fW < 20 or fH < 20:
                continue

            # Prepare the face for recognition
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            
            # Extract embeddings
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            
            # Perform recognition
            preds = recognizer.predict_proba([vec.flatten()])[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            
            # If recognition confidence is low, classify as "Unknown"
            if proba * 100 < 70:
                name = "Unknown"
            
            # Draw bounding box and recognition text
            text = f"{name} : {proba * 100:.2f}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            print(f"Recognized: {name}, Probability: {proba * 100:.2f}%")

    # Show the output frame
    cv2.imshow("YOLO Face Recognition", frame)
    
    if cv2.getWindowProperty("YOLO Face Recognition", cv2.WND_PROP_VISIBLE) < 1:  # Close button clicked
        break
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key to quit
        break

# Cleanup
cam.release()
cv2.destroyAllWindows()



# #ArcFace
# import numpy as np
# import pickle
# import time
# import cv2
# import torch
# from ultralytics import YOLO
# import imutils

# # Paths to model files and recognizer
# recognizerFile = r"D:\PythonProject\Face_Recognition_DL\output\recognizer_updated_arc.pickle"
# labelEncFile = r"D:\PythonProject\Face_Recognition_DL\output\le_updated_arc.pickle"
# embeddingFile = r"D:\PythonProject\Face_Recognition_DL\output\embeddings_arcface.pickle"
# yolo_model_path = r"D:\PythonProject\Face_Recognition_DL\model\yolov8n-face.pt"
# conf = 0.8  # Confidence threshold

# # Load YOLO model for face detection
# print("[INFO] Loading YOLO model...")
# yolo = YOLO(yolo_model_path)

# # Load the face recognizer and label encoder
# print("[INFO] Loading face recognizer...")
# recognizer = pickle.loads(open(recognizerFile, "rb").read())
# le = pickle.loads(open(labelEncFile, "rb").read())

# # Load saved embeddings
# print("[INFO] Loading stored face embeddings...")
# data = pickle.loads(open(embeddingFile, "rb").read())
# known_embeddings = np.array(data["embeddings"])
# known_names = np.array(data["names"])

# # Initialize camera stream
# print("[INFO] Starting video stream...")
# cam = cv2.VideoCapture(0)
# time.sleep(2.0)

# def get_embedding(face_img):
#     """Extract embedding using DeepFace ArcFace."""
#     from deepface import DeepFace
#     try:
#         embedding = DeepFace.represent(face_img, model_name='ArcFace', detector_backend='retinaface', enforce_detection=False)
#         return np.array(embedding[0]['embedding'])
#     except Exception as e:
#         print(f"Error extracting embedding: {e}")
#         return None

# while True:
#     _, frame = cam.read()
#     frame = imutils.resize(frame, width=600)

#     # Perform YOLO-based face detection
#     results = yolo.predict(frame, conf=0.4)

#     for result in results:
#         boxes = result.boxes
#         for box in boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])

#             face = frame[y1:y2, x1:x2]
#             (fH, fW) = face.shape[:2]

#             # Skip small faces
#             if fW < 20 or fH < 20:
#                 continue

#             # Extract embedding
#             embedding = get_embedding(face)
#             if embedding is None:
#                 continue  # Skip if embedding extraction fails

#             # Perform recognition
#             preds = recognizer.predict_proba([embedding])[0]
#             j = np.argmax(preds)
#             proba = preds[j]
#             name = le.classes_[j]

#             # If recognition confidence is low, classify as "Unknown"
#             if proba * 100 < 70:
#                 name = "Unknown"

#             # Draw bounding box and recognition text
#             text = f"{name} : {proba * 100:.2f}%"
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#             cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

#             print(f"Recognized: {name}, Probability: {proba * 100:.2f}%")

#     # Show the output frame
#     cv2.imshow("YOLO Face Recognition", frame)

#     if cv2.getWindowProperty("YOLO Face Recognition", cv2.WND_PROP_VISIBLE) < 1:  # Close button clicked
#         break

#     key = cv2.waitKey(1) & 0xFF
#     if key == 27:  # ESC key to quit
#         break

# # Cleanup
# cam.release()
# cv2.destroyAllWindows()
