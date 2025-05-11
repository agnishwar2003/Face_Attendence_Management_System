from retinaface import RetinaFace
import cv2
import matplotlib.pyplot as plt

# Open a connection to the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

if not cap.isOpened():
    print("Error: Unable to access the camera")
    exit()

print("Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame")
        break

    # Flip the frame horizontally for a mirror effect (optional)
    frame = cv2.flip(frame, 1)

    # Detect faces in the current frame
    detections = RetinaFace.detect_faces(frame, threshold=0.5)

    # Draw bounding boxes around detected faces
    for key in detections.keys():
        identity = detections[key]
        facial_area = identity["facial_area"]
        cv2.rectangle(frame,
                      (facial_area[0], facial_area[1]),  # Top-left corner
                      (facial_area[2], facial_area[3]),  # Bottom-right corner
                      (255, 0, 0),                      # Rectangle color (blue)
                      2)                                # Line thickness

    # Display the frame with bounding boxes
    cv2.imshow("Live Face Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
