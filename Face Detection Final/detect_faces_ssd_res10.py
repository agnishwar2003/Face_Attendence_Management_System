# Import the necessary packages
import numpy as np
import imutils
import cv2
import time

# Define the paths to the prototxt file and pre-trained model
prototxt_path = r"D:\PythonProject\Face_Recognition_DL\Face_detection\deploy.prototxt.txt"
model_path = r"D:\PythonProject\Face_Recognition_DL\Face_detection\res10_300x300_ssd_iter_140000.caffemodel"
confidence_threshold = 0.5

# Resize OpenCV window size
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

# Load model architecture and weights
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Initialize camera stream
print("[INFO] starting video stream...")

vid = cv2.VideoCapture(0)

# Loop over the frames from the video stream
while True:
    # Start the timer to calculate inference time
    start_time = time.time()

    # Read frame from camera and resize to 400 pixels
    ret, frame = vid.read()
    frame = imutils.resize(frame, width=400)
 
    # Grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
 
    # Pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    # Calculate inference time
    inference_time = time.time() - start_time

    # Add inference time text on top of the frame
    cv2.putText(frame, "Inference Time: {:.4f} sec".format(inference_time), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # Filter out detections by confidence
        if confidence < confidence_threshold:
            continue

        # Compute the (x, y)-coordinates of the bounding box for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
 
        # Draw the bounding box of the face along with the associated probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (0, 0, 255), 1)
        cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# After the loop, release the cap object and destroy all windows
vid.release()
cv2.destroyAllWindows()
