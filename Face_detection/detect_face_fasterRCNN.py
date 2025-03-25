import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import cv2
import numpy as np

# Load the pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define a function for live face detection
def live_face_detection(confidence_threshold=0.8):
    # Open the webcam (0 for default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to exit the live feed.")
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Convert the frame to RGB (as PIL expects RGB format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert the frame to a PyTorch tensor
        image_tensor = F.to_tensor(rgb_frame).unsqueeze(0)  # Add batch dimension
        
        # Perform inference
        with torch.no_grad():
            predictions = model(image_tensor)

        # Process the predictions
        boxes = predictions[0]['boxes']
        scores = predictions[0]['scores']

        # Filter boxes based on the confidence threshold
        for box, score in zip(boxes, scores):
            if score > confidence_threshold:
                x1, y1, x2, y2 = map(int, box)
                # Draw bounding boxes on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Add confidence score as text
                cv2.putText(frame, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Display the resulting frame
        cv2.imshow('Live Face Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

# Run the live face detection
live_face_detection(confidence_threshold=0.8)
