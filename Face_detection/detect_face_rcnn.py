import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# Load the pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Function to filter detections based on a confidence threshold
def filter_detections(predictions, threshold=0.8):
    boxes = predictions["boxes"].detach().numpy()
    scores = predictions["scores"].detach().numpy()
    filtered_boxes = [box for box, score in zip(boxes, scores) if score > threshold]
    return filtered_boxes

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert the frame to a tensor
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = F.to_tensor(frame_rgb).unsqueeze(0)

    # Perform object detection
    with torch.no_grad():
        predictions = model(frame_tensor)[0]

    # Filter the detections
    boxes = filter_detections(predictions)

    # Draw the bounding boxes on the frame
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Real-Time Face Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()