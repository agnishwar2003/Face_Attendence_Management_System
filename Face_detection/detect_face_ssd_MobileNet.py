
# # Load the pre-trained MobileNet-SSD model and configuration files
# model_path = "D:\\PythonProject\\Face_Detection_DL\\mobilenet_iter_73000.caffemodel"  # Pre-trained model weights
# config_path = "D:\\PythonProject\\Face_Detection_DL\\deploy.prototxt.txt"  # Network configuration
# net = cv2.dnn.readNetFromCaffe(config_path, model_path)

import cv2
import numpy as np

# Load the pre-trained MobileNet-SSD model and configuration files
model_path = 'D:/PythonProject/Face_Detection_DL/Face_detection/res10_300x300_ssd_iter_140000.caffemodel'
confidence_threshold = 0.5  # Pre-trained model weights
config_path = 'D:/PythonProject/Face_Detection_DL/Face_detection/deploy.prototxt.txt'  # Network configuration
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Starting facial detection...")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop over the faces detected
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Preprocess the face region for MobileNet-SSD
        face_region = frame[y:y + h, x:x + w]

        # Resize to 300x300 and preprocess image for the model
        blob = cv2.dnn.blobFromImage(face_region, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=True, crop=False)

        # Set input for the network
        net.setInput(blob)

        # Forward pass to get detections
        detections = net.forward()

        # Placeholder for detection visualization
        cv2.putText(frame, "Face detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame with rectangles and labels
    cv2.imshow("Facial Detection", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
