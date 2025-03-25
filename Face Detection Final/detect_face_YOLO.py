import cvzone
from ultralytics import YOLO
import cv2

# Access the default laptop camera
cap = cv2.VideoCapture(0)  # 0 is the default camera index

# Load the YOLO model pretrained for face detection
facemodel = YOLO(r"D:\PythonProject\Face_Recognition_DL\Face_detection\yolov8n-face.pt")  # Ensure you have the YOLOv8 face model file

# Loop to continuously read frames from the camera
while cap.isOpened():
    ret, frame = cap.read()  # Capture a frame from the camera
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Resize the frame for consistent processing (optional)
    frame = cv2.resize(frame, (700, 500))

    # Perform face detection
    face_results = facemodel.predict(frame, conf=0.40)  # Confidence threshold

    # Process the detection results
    for result in face_results:
        boxes = result.boxes
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Draw a fancy rectangle around the detected face
            cvzone.cornerRect(frame, [x1, y1, w, h], l=9, rt=3)

    # Display the frame with detected faces
    cv2.imshow('Face Detection - YOLOv8', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
