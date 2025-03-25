# Import necessary packages
import dlib
import cv2
import time

# Function to convert and trim bounding box
def convert_and_trim_bb(image, rect):
    # Extract the starting and ending (x, y)-coordinates of the bounding box
    startX = rect.left()
    startY = rect.top()
    endX = rect.right()
    endY = rect.bottom()
    # Ensure the bounding box coordinates fall within the spatial dimensions of the image
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])
    # Compute the width and height of the bounding box
    w = endX - startX
    h = endY - startY
    # Return the bounding box coordinates
    return (startX, startY, w, h)

# Load dlib's HOG + Linear SVM face detector
print("[INFO] Loading HOG + Linear SVM face detector...")
detector = dlib.get_frontal_face_detector()

# Open webcam for live video
cap = cv2.VideoCapture(0)  # Default camera (ID = 0)
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

print("[INFO] Starting live face detection. Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    # Resize the frame to a width of 600 pixels
    frame = cv2.resize(frame, (600, int(frame.shape[0] * (600 / frame.shape[1]))))

    # Convert BGR to RGB (as dlib expects RGB images)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform face detection using dlib's face detector
    start = time.time()
    rects = detector(rgb, 1)  # `1` is the upsample parameter
    end = time.time()

    print("[INFO] Face detection took {:.4f} seconds".format(end - start))

    # Convert dlib rectangles to bounding boxes
    boxes = [convert_and_trim_bb(frame, r) for r in rects]

    # Draw bounding boxes on the frame
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Live Face Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
